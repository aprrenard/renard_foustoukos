"""
Reactivation Participation and LMI Prediction Analysis

This script analyzes whether a neuron's participation in reactivation events
predicts its Learning Modulation Index (LMI). It computes participation rates
across days (baseline, learning, post-learning) and correlates with LMI values.

**IMPORTANT**: This script loads pre-computed reactivation events from
reactivation.py to ensure consistency. It does NOT detect events independently.
Make sure to run reactivation.py with mode='compute' first.

Approach:
1. Load pre-computed reactivation events from reactivation_results.pkl
2. For each reactivation event, extract cell responses in ±150ms window
3. Define participation: average response ≥ 5% dF/F
4. Compute participation rates per cell, per day
5. Aggregate: Baseline (days -2,-1), Learning (day 0), Post (days +1,+2)
6. Correlate participation with LMI, separately by LMI sign and reward group

Output:
- CSV files with participation rates and correlation statistics
- Multi-page PDF with scatter plots, temporal evolution, and distributions
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, pearsonr, ttest_rel
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
from joblib import Parallel, delayed
import warnings

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.core_analysis.reactivations.reactivation import (
    create_whisker_template,
    compute_template_correlation,
    detect_reactivation_events,
    load_surrogate_thresholds,
    get_threshold_for_mouse_day
)

# =============================================================================
# PARAMETERS
# =============================================================================

sampling_rate = 30  # Hz
win = (0, 0.300)  # Window for template (matches reactivation.py)
days = [-2, -1, 0, 1, 2]
days_str = ['-2', '-1', '0', '+1', '+2']
n_map_trials = 40  # Number of mapping trials for template
threshold_dff = None  # dF/F threshold for cells included in template (None = all cells)
min_event_distance_ms = 150  # Minimum distance between events (ms)
min_event_distance_frames = int(min_event_distance_ms / 1000 * sampling_rate)

# Participation parameters
event_window_ms = 150  # ±150ms around event (total 300ms)
event_window_frames = int(event_window_ms / 1000 * sampling_rate)  # ±5 frames
participation_threshold = 0.10  # 5% dF/F for participation
min_events_for_reliability = 5  # Flag estimates based on < 5 events

# LMI thresholds (matches characterize_LMI_cells.py)
LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025

# Parallel processing
n_jobs = 35

# Statistical parameters
alpha_fdr = 0.05  # FDR correction level

# Visualization
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

# Path to saved reactivation results — must match what was used in reactivation.py
# percentile_to_use: 95, 99, or 99.9  (selects the corresponding pkl file)
percentile_to_use = 99

# Surrogate threshold parameters (only used as fallback if preloaded events are missing)
use_surrogate_thresholds = True
threshold_mode  = 'day'      # 'mouse' or 'day'
threshold_type  = 'percentile'
threshold_corr  = 0.45       # fallback fixed threshold

save_dir = os.path.join(io.results_dir, 'reactivation')
_p_str = str(int(percentile_to_use)) if percentile_to_use == int(percentile_to_use) else str(int(percentile_to_use * 10))
reactivation_results_file = os.path.join(save_dir, f'reactivation_results_p{_p_str}.pkl')

# Load database
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes'
)

# Separate mice by reward group
r_plus_mice = []
r_minus_mice = []
for mouse in all_mice:
    try:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse, db=db)
        if reward_group == 'R+':
            r_plus_mice.append(mouse)
        elif reward_group == 'R-':
            r_minus_mice.append(mouse)
    except:
        continue

print(f"Found {len(r_plus_mice)} R+ mice: {r_plus_mice}")
print(f"Found {len(r_minus_mice)} R- mice: {r_minus_mice}")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_reactivation_results(results_file):
    """
    Load pre-computed reactivation results from reactivation.py.

    Parameters
    ----------
    results_file : str
        Path to the reactivation_results.pkl file

    Returns
    -------
    dict
        Dictionary with 'r_plus_results' and 'r_minus_results'
    """
    import pickle

    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Reactivation results file not found: {results_file}\n"
            f"Please run reactivation.py with mode='compute' first."
        )

    print(f"\nLoading pre-computed reactivation events from: {results_file}")
    with open(results_file, 'rb') as f:
        results_data = pickle.load(f)

    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']

    print(f"✓ Loaded results for {len(r_plus_results)} R+ mice and {len(r_minus_results)} R- mice")

    return r_plus_results, r_minus_results


def extract_event_responses(mouse, day, verbose=True, threshold_dict=None, inflection_points=None, preloaded_events=None):
    """
    Extract cell responses around reactivation events for a single mouse and day.

    Parameters
    ----------
    mouse : str
        Mouse ID
    day : int
        Day number (-2, -1, 0, 1, 2)
    verbose : bool
        Print progress information
    threshold_dict : dict, optional
        Per-mouse, per-day thresholds from surrogate analysis.
        If None, uses global threshold_corr parameter.
        NOTE: Only used if preloaded_events is None (for backwards compatibility).
    inflection_points : dict, optional
        Dictionary mapping (mouse_id, roi) to inflection trial_w value (1-indexed).
        If provided and day==0, adds 'inflection_phase' column to responses_df
        with values: 'before', 'after', 'at_inflection', or 'unknown'.
    preloaded_events : np.ndarray, optional
        Pre-computed event indices from reactivation.py.
        If provided, these events are used instead of detecting new ones.
        This ensures consistency with the main reactivation analysis.

    Returns
    -------
    responses_df : pd.DataFrame
        Columns: mouse_id, day, roi, event_idx, avg_response, participates
        (and inflection_phase if inflection_points provided and day==0)
        Each row is one cell's response to one event
    n_events : int
        Total number of events detected
    """
    if verbose:
        print(f"\n  Extracting event responses for {mouse}, Day {day}")

    try:
        # Step 1: Create template (reuse from reactivation.py)
        template, cells_mask = create_whisker_template(mouse, day, threshold_dff, verbose=False)

        # Step 2: Load learning data
        folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        file_name = 'tensor_xarray_learning_data.nc'
        xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)

        # Select this day
        xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)

        # Select no_stim trials only
        nostim_trials = xarray_day.sel(trial=xarray_day['no_stim'] == 1)

        # For day 0 with inflection points, load whisker trials to map inflection to global trial_id
        whisker_trial_ids = None
        if day == 0 and inflection_points is not None:
            whisker_trials = xarray_day.sel(trial=xarray_day['whisker_stim'] == 1)
            whisker_trial_ids = whisker_trials['trial_id'].values

        n_nostim_trials = len(nostim_trials.trial)
        if verbose:
            print(f"    No-stim trials: {n_nostim_trials}")

        if n_nostim_trials < 10:
            if verbose:
                print(f"    Warning: Insufficient no-stim trials ({n_nostim_trials} < 10), skipping...")
            return None, 0

        # Step 3: Prepare data and compute correlations
        n_cells, n_trials, n_timepoints = nostim_trials.shape
        data = nostim_trials.values.reshape(n_cells, -1)
        data = np.nan_to_num(data, nan=0.0)

        correlations = compute_template_correlation(data, template)

        # Step 4: Get events - use preloaded if available, otherwise detect
        if preloaded_events is not None:
            events = preloaded_events
            if verbose:
                print(f"    Using pre-computed events from reactivation.py: {len(events)} events")
        else:
            # Fallback: detect events (for backwards compatibility)
            current_threshold = get_threshold_for_mouse_day(threshold_dict, mouse, day, threshold_corr)
            if verbose:
                if threshold_dict is not None:
                    print(f"    Using threshold: {current_threshold:.4f} (surrogate-based, {threshold_mode} mode)")
                print(f"    WARNING: Detecting events independently (not using reactivation.py results)")
            events = detect_reactivation_events(correlations, current_threshold, min_event_distance_frames)
            if verbose:
                print(f"    Events detected: {len(events)}")

        if len(events) == 0:
            if verbose:
                print(f"    Warning: No events detected")
            return None, 0

        # Step 5: Extract responses around each event
        # Events are indices in the flattened array, need to map back to trials
        responses_list = []

        # Reconstruct 3D data for windowing
        data_3d = nostim_trials.values  # (n_cells, n_trials, n_timepoints)
        roi_list = nostim_trials['roi'].values

        for event_idx in events:
            # Map event index to trial and time
            trial_idx = event_idx // n_timepoints
            time_idx = event_idx % n_timepoints

            # Skip events too close to trial boundaries
            if time_idx < event_window_frames or time_idx >= n_timepoints - event_window_frames:
                continue

            # Extract ±event_window_frames around event
            # event_window_frames = 5, so we extract [time_idx-5 : time_idx+6] = 11 frames
            window_start = time_idx - event_window_frames
            window_end = time_idx + event_window_frames + 1

            window_data = data_3d[:, trial_idx, window_start:window_end]  # (n_cells, 11)

            # Average over time window
            avg_response = np.mean(window_data, axis=1)  # (n_cells,)

            # Determine participation (response >= threshold)
            participates = avg_response >= participation_threshold

            # Get global trial_id for this event (for inflection phase determination)
            event_trial_id = nostim_trials['trial_id'].values[trial_idx]

            # Store results for each cell
            for icell in range(n_cells):
                roi = roi_list[icell]

                # Determine inflection phase
                inflection_phase = 'unknown'
                if whisker_trial_ids is not None and (mouse, roi) in inflection_points:
                    inflection_trial_w = inflection_points[(mouse, roi)]

                    # Get global trial_id of the Nth whisker trial (trial_w is 1-indexed)
                    if 1 <= inflection_trial_w <= len(whisker_trial_ids):
                        inflection_trial_id = whisker_trial_ids[int(inflection_trial_w - 1)]

                        if event_trial_id < inflection_trial_id:
                            inflection_phase = 'before'
                        elif event_trial_id > inflection_trial_id:
                            inflection_phase = 'after'
                        else:
                            inflection_phase = 'at_inflection'

                responses_list.append({
                    'mouse_id': mouse,
                    'day': day,
                    'roi': roi,
                    'event_idx': event_idx,
                    'avg_response': avg_response[icell],
                    'participates': participates[icell],
                    'inflection_phase': inflection_phase
                })

        responses_df = pd.DataFrame(responses_list)

        if verbose:
            print(f"    Extracted responses for {len(responses_df)} cell-event pairs")
            print(f"    Participation rate: {responses_df['participates'].mean()*100:.1f}%")

        return responses_df, len(events)

    except Exception as e:
        if verbose:
            print(f"    Error extracting event responses: {str(e)}")
        return None, 0


def compute_participation_rate(responses_df):
    """
    Compute participation rate per cell.

    Parameters
    ----------
    responses_df : pd.DataFrame
        Output from extract_event_responses

    Returns
    -------
    participation_df : pd.DataFrame
        Columns: mouse_id, day, roi, participation_rate, n_events, reliable
    """
    if responses_df is None or len(responses_df) == 0:
        return None

    # Group by mouse, day, roi and compute statistics
    grouped = responses_df.groupby(['mouse_id', 'day', 'roi']).agg(
        n_participations=('participates', 'sum'),
        n_events=('participates', 'count'),
        mean_response=('avg_response', 'mean')
    ).reset_index()

    # Compute participation rate
    grouped['participation_rate'] = grouped['n_participations'] / grouped['n_events']

    # Flag unreliable estimates (< min_events_for_reliability events)
    grouped['reliable'] = grouped['n_events'] >= min_events_for_reliability

    return grouped


def compute_participation_rate_by_inflection(responses_df):
    """
    Compute participation rates separately for before and after inflection.

    Requires responses_df to have 'inflection_phase' column.

    Returns
    -------
    pd.DataFrame
        Columns: mouse_id, roi, participation_rate_before, participation_rate_after,
                 n_events_before, n_events_after, reliable_before, reliable_after
    """
    if responses_df is None or len(responses_df) == 0:
        return None

    results = []

    for (mouse_id, roi), group in responses_df.groupby(['mouse_id', 'roi']):
        before = group[group['inflection_phase'] == 'before']
        after = group[group['inflection_phase'] == 'after']

        # Compute rates and counts
        n_before = len(before)
        n_after = len(after)

        rate_before = before['participates'].mean() if n_before > 0 else np.nan
        rate_after = after['participates'].mean() if n_after > 0 else np.nan

        results.append({
            'mouse_id': mouse_id,
            'roi': roi,
            'participation_rate_before': rate_before,
            'participation_rate_after': rate_after,
            'n_events_before': n_before,
            'n_events_after': n_after,
            'reliable_before': n_before >= min_events_for_reliability,
            'reliable_after': n_after >= min_events_for_reliability
        })

    return pd.DataFrame(results)


def analyze_mouse_participation_by_inflection(mouse, inflection_points, threshold_dict=None, verbose=False):
    """
    Analyze participation rates before and after inflection on day 0.

    Parameters
    ----------
    mouse : str
        Mouse ID
    inflection_points : dict
        Mapping (mouse_id, roi) -> inflection_trial_w (1-indexed whisker trial number)
    threshold_dict : dict, optional
        Thresholds for event detection (from surrogate analysis)
    verbose : bool

    Returns
    -------
    pd.DataFrame or None
        Participation rates before/after inflection for this mouse
    """
    if verbose:
        print(f"\n  Processing {mouse} for before/after inflection analysis...")

    # Extract responses for day 0 with inflection phase tracking
    responses_df, n_events = extract_event_responses(
        mouse, day=0, threshold_dict=threshold_dict, verbose=verbose,
        inflection_points=inflection_points
    )

    if responses_df is None or len(responses_df) == 0:
        if verbose:
            print(f"    No events detected for {mouse}")
        return None

    # Compute participation rates by inflection phase
    participation_df = compute_participation_rate_by_inflection(responses_df)

    if verbose and participation_df is not None:
        print(f"    Computed before/after rates for {len(participation_df)} cells")

    return participation_df


def aggregate_across_days(participation_df_all):
    """
    Aggregate participation rates across day periods.

    Parameters
    ----------
    participation_df_all : pd.DataFrame
        Combined participation data across all days

    Returns
    -------
    aggregated_df : pd.DataFrame
        Columns: mouse_id, roi, baseline_rate, learning_rate, post_rate,
                 delta_learning, delta_post, reliable_baseline, reliable_learning, reliable_post
    """
    if participation_df_all is None or len(participation_df_all) == 0:
        return None

    # Define day groups
    baseline_days = [-2, -1]
    learning_day = [0]
    post_days = [1, 2]

    results = []

    # Group by mouse and roi
    for (mouse_id, roi), group in participation_df_all.groupby(['mouse_id', 'roi']):
        # Baseline: mean of days -2, -1
        baseline_data = group[group['day'].isin(baseline_days)]
        if len(baseline_data) > 0:
            baseline_rate = baseline_data['participation_rate'].mean()
            reliable_baseline = baseline_data['reliable'].all()
        else:
            baseline_rate = np.nan
            reliable_baseline = False

        # Learning: day 0
        learning_data = group[group['day'] == 0]
        if len(learning_data) > 0:
            learning_rate = learning_data['participation_rate'].iloc[0]
            reliable_learning = learning_data['reliable'].iloc[0]
        else:
            learning_rate = np.nan
            reliable_learning = False

        # Post-learning: mean of days +1, +2
        post_data = group[group['day'].isin(post_days)]
        if len(post_data) > 0:
            post_rate = post_data['participation_rate'].mean()
            reliable_post = post_data['reliable'].all()
        else:
            post_rate = np.nan
            reliable_post = False

        # Compute change scores
        delta_learning = learning_rate - baseline_rate if not np.isnan(baseline_rate) else np.nan
        delta_post = post_rate - baseline_rate if not np.isnan(baseline_rate) else np.nan

        results.append({
            'mouse_id': mouse_id,
            'roi': roi,
            'baseline_rate': baseline_rate,
            'learning_rate': learning_rate,
            'post_rate': post_rate,
            'delta_learning': delta_learning,
            'delta_post': delta_post,
            'reliable_baseline': reliable_baseline,
            'reliable_learning': reliable_learning,
            'reliable_post': reliable_post
        })

    aggregated_df = pd.DataFrame(results)
    return aggregated_df


def load_and_match_lmi_data(participation_df):
    """
    Load LMI data and match with participation data.

    Parameters
    ----------
    participation_df : pd.DataFrame
        Aggregated participation data

    Returns
    -------
    merged_df : pd.DataFrame
        Combined data with LMI and participation metrics
    """
    # Load LMI results
    lmi_df_path = os.path.join(io.processed_dir, 'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df_path)

    # Add reward group if not present
    if 'reward_group' not in lmi_df.columns:
        mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()
        lmi_df['reward_group'] = lmi_df['mouse_id'].map(
            dict(mice_count[['mouse_id', 'reward_group']].values)
        )

    # Categorize LMI cells
    lmi_df['lmi_category'] = 'neutral'
    lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD, 'lmi_category'] = 'positive'
    lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD, 'lmi_category'] = 'negative'

    # Merge with participation data
    merged_df = pd.merge(
        participation_df,
        lmi_df[['mouse_id', 'roi', 'lmi', 'lmi_p', 'lmi_category', 'reward_group', 'cell_type']],
        on=['mouse_id', 'roi'],
        how='inner'
    )

    print(f"\n  Merged data:")
    print(f"    Total cells with both participation and LMI: {len(merged_df)}")
    print(f"    Positive LMI cells: {(merged_df['lmi_category'] == 'positive').sum()}")
    print(f"    Negative LMI cells: {(merged_df['lmi_category'] == 'negative').sum()}")
    print(f"    Neutral LMI cells: {(merged_df['lmi_category'] == 'neutral').sum()}")
    print(f"    R+ cells: {(merged_df['reward_group'] == 'R+').sum()}")
    print(f"    R- cells: {(merged_df['reward_group'] == 'R-').sum()}")

    return merged_df


def correlate_with_lmi(merged_df, participation_metric='learning_rate', reliable_filter=True):
    """
    Compute correlations between participation and LMI for different groups.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Combined data with LMI and participation
    participation_metric : str
        Which participation metric to correlate ('baseline_rate', 'learning_rate',
        'post_rate', 'delta_learning', 'delta_post')
    reliable_filter : bool
        Whether to filter for reliable estimates

    Returns
    -------
    results_df : pd.DataFrame
        Correlation statistics for each group
    """
    results = []

    # Define groups to analyze
    reward_groups = ['R+', 'R-']
    lmi_categories = ['positive', 'negative']

    for reward_group in reward_groups:
        for lmi_category in lmi_categories:
            # Filter data
            mask = (merged_df['reward_group'] == reward_group) & \
                   (merged_df['lmi_category'] == lmi_category)

            if reliable_filter:
                # Add reliability filter based on metric
                if participation_metric == 'baseline_rate':
                    mask &= merged_df['reliable_baseline']
                elif participation_metric == 'learning_rate':
                    mask &= merged_df['reliable_learning']
                elif participation_metric == 'post_rate':
                    mask &= merged_df['reliable_post']
                elif participation_metric in ['delta_learning', 'delta_post']:
                    mask &= merged_df['reliable_baseline']  # Need reliable baseline for deltas

            group_data = merged_df[mask]

            if len(group_data) < 3:
                # Not enough data for correlation
                results.append({
                    'reward_group': reward_group,
                    'lmi_category': lmi_category,
                    'participation_metric': participation_metric,
                    'n_cells': len(group_data),
                    'spearman_r': np.nan,
                    'spearman_p': np.nan,
                    'pearson_r': np.nan,
                    'pearson_p': np.nan
                })
                continue

            # Remove NaN values
            valid_mask = ~(group_data[participation_metric].isna() | group_data['lmi'].isna())
            x = group_data.loc[valid_mask, participation_metric].values
            y = group_data.loc[valid_mask, 'lmi'].values

            if len(x) < 3:
                results.append({
                    'reward_group': reward_group,
                    'lmi_category': lmi_category,
                    'participation_metric': participation_metric,
                    'n_cells': len(x),
                    'spearman_r': np.nan,
                    'spearman_p': np.nan,
                    'pearson_r': np.nan,
                    'pearson_p': np.nan
                })
                continue

            # Compute correlations
            spearman_r, spearman_p = spearmanr(x, y)
            pearson_r, pearson_p = pearsonr(x, y)

            results.append({
                'reward_group': reward_group,
                'lmi_category': lmi_category,
                'participation_metric': participation_metric,
                'n_cells': len(x),
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p
            })

    results_df = pd.DataFrame(results)

    # Apply FDR correction
    if len(results_df) > 0:
        valid_p = ~results_df['spearman_p'].isna()
        if valid_p.sum() > 0:
            _, p_corrected, _, _ = multipletests(
                results_df.loc[valid_p, 'spearman_p'].values,
                alpha=alpha_fdr,
                method='fdr_bh'
            )
            results_df.loc[valid_p, 'spearman_p_fdr'] = p_corrected
        else:
            results_df['spearman_p_fdr'] = np.nan

    return results_df


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_mouse_participation(mouse, days=[-2, -1, 0, 1, 2], verbose=True, threshold_dict=None, preloaded_results=None):
    """
    Analyze reactivation participation for a single mouse across multiple days.

    Parameters
    ----------
    mouse : str
        Mouse ID
    days : list
        List of days to analyze
    verbose : bool
        Print progress information
    threshold_dict : dict, optional
        Per-mouse, per-day thresholds from surrogate analysis.
        If None, uses global threshold_corr parameter.
        NOTE: Only used if preloaded_results is None (for backwards compatibility).
    preloaded_results : dict, optional
        Pre-computed reactivation results from reactivation.py.
        Expected structure: results['days'][day]['events']

    Returns
    -------
    participation_df : pd.DataFrame
        Participation data for this mouse
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING MOUSE: {mouse}")
        print(f"{'='*60}")

    all_responses = []

    for day in days:
        if verbose:
            print(f"\nProcessing Day {day}...")

        # Get preloaded events if available
        preloaded_events = None
        if preloaded_results is not None and day in preloaded_results.get('days', {}):
            preloaded_events = preloaded_results['days'][day]['events']

        responses_df, n_events = extract_event_responses(
            mouse, day, verbose=verbose, threshold_dict=threshold_dict,
            preloaded_events=preloaded_events
        )

        if responses_df is not None and len(responses_df) > 0:
            all_responses.append(responses_df)

    if len(all_responses) == 0:
        if verbose:
            print(f"\nNo valid data for mouse {mouse}")
        return None

    # Combine all responses
    all_responses_df = pd.concat(all_responses, ignore_index=True)

    # Compute participation rates
    participation_df = compute_participation_rate(all_responses_df)

    if verbose:
        print(f"\nCompleted mouse {mouse}: {len(participation_df)} cell-day records")

    return participation_df


def process_single_mouse(mouse, days, verbose=False, threshold_dict=None, preloaded_results=None):
    """
    Wrapper for parallel processing.

    Parameters
    ----------
    mouse : str
        Mouse ID
    days : list
        List of days to analyze
    verbose : bool
        Print progress information
    threshold_dict : dict, optional
        Per-mouse, per-day thresholds from surrogate analysis
    preloaded_results : dict, optional
        Pre-computed reactivation results for this mouse from reactivation.py

    Returns
    -------
    tuple
        (mouse_id, participation_df)
    """
    participation_df = analyze_mouse_participation(
        mouse, days=days, verbose=verbose, threshold_dict=threshold_dict,
        preloaded_results=preloaded_results
    )
    return (mouse, participation_df)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_participation_rate_vs_lmi(merged_df, participation_metric='learning_rate',
                                    save_path=None):
    """
    Create scatter plots of participation rate (y) vs LMI (x) for R+ and R-.

    Simple 1x2 panel layout showing how reactivation participation relates to LMI.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged participation and LMI data
    participation_metric : str
        Which participation metric to use
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reward_groups = ['R+', 'R-']

    for i, reward_group in enumerate(reward_groups):
        ax = axes[i]

        # Filter data for this reward group
        mask = (merged_df['reward_group'] == reward_group)
        group_data = merged_df[mask]

        # Remove NaN values
        valid_mask = ~(group_data['lmi'].isna() | group_data[participation_metric].isna())
        x = group_data.loc[valid_mask, 'lmi'].values
        y = group_data.loc[valid_mask, participation_metric].values

        if len(x) < 3:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={len(x)})',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('LMI', fontweight='bold')
            ax.set_ylabel('Participation Rate (Day 0)', fontweight='bold')
            ax.set_title(f'{reward_group}', fontweight='bold', fontsize=14)
            continue

        # Scatter plot
        ax.scatter(x, y, s=40, alpha=0.6, edgecolors='black', linewidths=0.5, color='#2ca02c')

        # Reference lines
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.4)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.4)

        # Regression line and statistics
        if len(x) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7)

            # Compute Pearson correlation
            pearson_r, pearson_p = pearsonr(x, y)

            # Add statistics text
            stats_text = f'n = {len(x)} cells\n'
            stats_text += f'Pearson r = {pearson_r:.3f}\n'
            if pearson_p < 0.001:
                stats_text += f'p < 0.001 ***'
            elif pearson_p < 0.01:
                stats_text += f'p = {pearson_p:.3f} **'
            elif pearson_p < 0.05:
                stats_text += f'p = {pearson_p:.3f} *'
            else:
                stats_text += f'p = {pearson_p:.3f} ns'

            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax.set_xlabel('LMI', fontweight='bold', fontsize=12)
        ax.set_ylabel('Participation Rate (Day 0)', fontweight='bold', fontsize=12)
        ax.set_title(f'{reward_group}', fontweight='bold', fontsize=14)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-.1, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")

        # Save data CSV
        data_csv = save_path.replace('.svg', '_data.csv')
        data_export = merged_df[['mouse_id', 'roi', 'reward_group', 'lmi', 'lmi_p',
                                  'lmi_category', participation_metric]].copy()
        data_export = data_export.dropna(subset=['lmi', participation_metric])
        data_export.to_csv(data_csv, index=False)
        print(f"  Data CSV saved: {data_csv}")

        # Save statistics CSV (correlation stats for each group)
        stats_csv = save_path.replace('.svg', '_stats.csv')
        stats_list = []

        for reward_group in ['R+', 'R-']:
            mask = (merged_df['reward_group'] == reward_group)
            group_data = merged_df[mask]
            valid_mask = ~(group_data['lmi'].isna() | group_data[participation_metric].isna())
            x = group_data.loc[valid_mask, 'lmi'].values
            y = group_data.loc[valid_mask, participation_metric].values

            if len(x) >= 3:
                pearson_r, pearson_p = pearsonr(x, y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                significance = ""
                if pearson_p < 0.001:
                    significance = "***"
                elif pearson_p < 0.01:
                    significance = "**"
                elif pearson_p < 0.05:
                    significance = "*"
                else:
                    significance = "ns"

                stats_list.append({
                    'reward_group': reward_group,
                    'n_cells': len(x),
                    'pearson_r': pearson_r,
                    'p_value': pearson_p,
                    'significance': significance,
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'std_err': std_err,
                    'test': 'Pearson correlation',
                })
            else:
                stats_list.append({
                    'reward_group': reward_group,
                    'n_cells': len(x),
                    'pearson_r': np.nan,
                    'p_value': np.nan,
                    'significance': 'insufficient data',
                    'slope': np.nan,
                    'intercept': np.nan,
                    'r_squared': np.nan,
                    'std_err': np.nan,
                    'test': 'insufficient data',
                })

        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(stats_csv, index=False)
        print(f"  Stats CSV saved: {stats_csv}")

        plt.close()
    else:
        return fig


def plot_participation_rate_vs_lmi_histogram(merged_df, participation_metric='learning_rate',
                                               save_path=None):
    """
    Create histograms of participation rate (y) vs LMI (x) for R+ and R-.

    Bins LMI into 0.2-width bins and shows mean participation rate per bin.
    Alternative visualization to the scatter plot.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged participation and LMI data
    participation_metric : str
        Which participation metric to use
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reward_groups = ['R+', 'R-']
    group_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}

    # Define LMI bins with 0.2 width
    lmi_bins = np.arange(-1.0, 1.2, 0.2)
    bin_centers = lmi_bins[:-1] + 0.1
    bin_labels = [f'{bc:.1f}' for bc in bin_centers]

    for i, reward_group in enumerate(reward_groups):
        ax = axes[i]

        mask = (merged_df['reward_group'] == reward_group)
        group_data = merged_df[mask].dropna(subset=['lmi', participation_metric]).copy()

        if len(group_data) < 3:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={len(group_data)})',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('LMI', fontweight='bold')
            ax.set_ylabel('Mean Participation Rate (Day 0)', fontweight='bold')
            ax.set_title(f'{reward_group}', fontweight='bold', fontsize=14)
            continue

        group_data['lmi_bin'] = pd.cut(
            group_data['lmi'], bins=lmi_bins, labels=bin_labels, include_lowest=True
        )

        sns.barplot(
            data=group_data, x='lmi_bin', y=participation_metric,
            color=group_colors[reward_group], errorbar='ci', ax=ax,
            alpha=0.8, edgecolor='black', linewidth=0.8
        )

        # Pearson correlation on unbinned data
        x = group_data['lmi'].values
        y = group_data[participation_metric].values
        pearson_r, pearson_p = pearsonr(x, y)

        if pearson_p < 0.001:
            p_str = 'p < 0.001 ***'
        elif pearson_p < 0.01:
            p_str = f'p = {pearson_p:.3f} **'
        elif pearson_p < 0.05:
            p_str = f'p = {pearson_p:.3f} *'
        else:
            p_str = f'p = {pearson_p:.3f} ns'

        stats_text = f'n = {len(x)} cells\nPearson r = {pearson_r:.3f}\n{p_str}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax.set_xlabel('LMI', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Participation Rate (Day 0)', fontweight='bold', fontsize=12)
        ax.set_title(f'{reward_group}', fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")

        # Save binned data CSV
        data_csv = save_path.replace('.svg', '_data.csv')
        binned_data_list = []

        for reward_group in ['R+', 'R-']:
            mask = (merged_df['reward_group'] == reward_group)
            gd = merged_df[mask].dropna(subset=['lmi', participation_metric]).copy()
            gd['lmi_bin'] = pd.cut(gd['lmi'], bins=lmi_bins, labels=bin_labels, include_lowest=True)

            for bin_idx, (bin_label, bin_center) in enumerate(zip(bin_labels, bin_centers)):
                bin_values = gd.loc[gd['lmi_bin'] == bin_label, participation_metric].values
                if len(bin_values) > 0:
                    binned_data_list.append({
                        'reward_group': reward_group,
                        'lmi_bin_min': lmi_bins[bin_idx],
                        'lmi_bin_max': lmi_bins[bin_idx + 1],
                        'lmi_bin_center': bin_center,
                        'mean_participation_rate': np.mean(bin_values),
                        'sem_participation_rate': np.std(bin_values, ddof=1) / np.sqrt(len(bin_values)),
                        'n_cells': len(bin_values)
                    })

        binned_data_df = pd.DataFrame(binned_data_list)
        binned_data_df.to_csv(data_csv, index=False)
        print(f"  Data CSV saved: {data_csv}")

        # Save statistics CSV (overall Pearson correlation on unbinned data)
        stats_csv = save_path.replace('.svg', '_stats.csv')
        stats_list = []

        for reward_group in ['R+', 'R-']:
            mask = (merged_df['reward_group'] == reward_group)
            group_data = merged_df[mask]
            valid_mask = ~(group_data['lmi'].isna() | group_data[participation_metric].isna())
            x = group_data.loc[valid_mask, 'lmi'].values
            y = group_data.loc[valid_mask, participation_metric].values

            if len(x) >= 3:
                pearson_r, pearson_p = pearsonr(x, y)

                significance = ""
                if pearson_p < 0.001:
                    significance = "***"
                elif pearson_p < 0.01:
                    significance = "**"
                elif pearson_p < 0.05:
                    significance = "*"
                else:
                    significance = "ns"

                stats_list.append({
                    'reward_group': reward_group,
                    'n_cells': len(x),
                    'pearson_r': pearson_r,
                    'p_value': pearson_p,
                    'significance': significance,
                    'test': 'Pearson correlation',
                    'note': 'correlation on unbinned data'
                })
            else:
                stats_list.append({
                    'reward_group': reward_group,
                    'n_cells': len(x),
                    'pearson_r': np.nan,
                    'p_value': np.nan,
                    'significance': 'insufficient data',
                    'test': 'insufficient data',
                    'note': 'correlation on unbinned data'
                })

        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(stats_csv, index=False)
        print(f"  Stats CSV saved: {stats_csv}")

        plt.close()
    else:
        return fig


def plot_temporal_evolution(merged_df, participation_df_all, save_path=None):
    """
    Plot participation rates across days for LMI+ vs LMI- cells.

    Barplot showing per-mouse averages across days, with lines connecting
    individual mice. Separate panels for R+ and R-. Includes 2-way repeated
    measures ANOVA (days × LMI category) with FDR-corrected posthoc tests.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    reward_groups = ['R+', 'R-']
    lmi_categories = ['positive', 'negative']
    colors = {'positive': '#d62728', 'negative': '#1f77b4'}

    for i, reward_group in enumerate(reward_groups):
        ax = axes[i]

        # Prepare data structure for barplot
        plot_data_lmi_pos = []
        plot_data_lmi_neg = []

        for lmi_category in lmi_categories:
            # Filter cells by reward group and LMI category
            mask = (merged_df['reward_group'] == reward_group) & \
                   (merged_df['lmi_category'] == lmi_category)
            cells_df = merged_df[mask][['mouse_id', 'roi']]

            if len(cells_df) == 0:
                continue

            # Get participation data for these cells and compute per-mouse averages
            for mouse_id in cells_df['mouse_id'].unique():
                mouse_cells = cells_df[cells_df['mouse_id'] == mouse_id]

                # Get participation data for this mouse's cells
                for day in days:
                    participation_rates = []
                    for _, row in mouse_cells.iterrows():
                        cell_mask = (participation_df_all['mouse_id'] == mouse_id) & \
                                   (participation_df_all['roi'] == row['roi']) & \
                                   (participation_df_all['day'] == day)
                        cell_data = participation_df_all[cell_mask]
                        if len(cell_data) > 0:
                            participation_rates.append(cell_data['participation_rate'].values[0])

                    # If no data for this day, use 0 (no events detected = 0% participation)
                    # Otherwise compute mean across cells
                    if len(participation_rates) > 0:
                        mean_rate = np.mean(participation_rates)
                    else:
                        mean_rate = 0.0

                    data_dict = {
                        'mouse_id': mouse_id,
                        'day': day,
                        'participation_rate': mean_rate,
                        'lmi_category': lmi_category
                    }

                    if lmi_category == 'positive':
                        plot_data_lmi_pos.append(data_dict)
                    else:
                        plot_data_lmi_neg.append(data_dict)

        # Convert to DataFrames
        df_pos = pd.DataFrame(plot_data_lmi_pos) if plot_data_lmi_pos else pd.DataFrame()
        df_neg = pd.DataFrame(plot_data_lmi_neg) if plot_data_lmi_neg else pd.DataFrame()

        # Combine for plotting
        all_data = []
        if not df_pos.empty:
            all_data.append(df_pos)
        if not df_neg.empty:
            all_data.append(df_neg)

        if len(all_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            continue

        plot_df = pd.concat(all_data, ignore_index=True)

        # =====================================================================
        # 2-WAY REPEATED MEASURES ANOVA: days × LMI category
        # =====================================================================
        print(f"\n  {reward_group} - Statistical Analysis:")
        print("  " + "-" * 56)

        # Check if we have data for both LMI categories
        has_both_categories = (not df_pos.empty) and (not df_neg.empty)

        if has_both_categories:
            # Find mice that have data for both LMI+ and LMI- cells
            mice_pos = set(df_pos['mouse_id'].unique())
            mice_neg = set(df_neg['mouse_id'].unique())
            mice_both = mice_pos & mice_neg

            if len(mice_both) >= 3:
                print(f"  Running 2-way repeated measures ANOVA (n={len(mice_both)} mice)")

                # Prepare data for ANOVA: need long format with factors
                # Subject: mouse_id, Within-subject factors: day, lmi_category
                anova_data = []
                for mouse_id in sorted(mice_both):
                    for day in days:
                        for lmi_cat in ['positive', 'negative']:
                            # Get data for this mouse-day-category
                            if lmi_cat == 'positive':
                                cat_df = df_pos
                            else:
                                cat_df = df_neg

                            mouse_day_data = cat_df[(cat_df['mouse_id'] == mouse_id) &
                                                     (cat_df['day'] == day)]

                            if len(mouse_day_data) > 0:
                                rate = mouse_day_data['participation_rate'].values[0]
                                anova_data.append({
                                    'mouse_id': mouse_id,
                                    'day': day,
                                    'lmi_category': lmi_cat,
                                    'participation_rate': rate
                                })

                df_anova = pd.DataFrame(anova_data)

                # Run 2-way repeated measures ANOVA
                try:
                    aovrm = AnovaRM(df_anova, 'participation_rate', 'mouse_id',
                                   within=['day', 'lmi_category'])
                    anova_results = aovrm.fit()

                    print("\n  ANOVA Results:")
                    print(anova_results.summary())

                    # Extract F-statistics and p-values
                    anova_table = anova_results.anova_table
                    day_effect_p = anova_table.loc['day', 'Pr > F']
                    lmi_effect_p = anova_table.loc['lmi_category', 'Pr > F']
                    interaction_p = anova_table.loc['day:lmi_category', 'Pr > F']

                    print(f"\n  Main effect of Day: p = {day_effect_p:.4f}")
                    print(f"  Main effect of LMI category: p = {lmi_effect_p:.4f}")
                    print(f"  Interaction (Day × LMI): p = {interaction_p:.4f}")

                    # Save ANOVA results to CSV
                    if save_path:
                        anova_csv = save_path.replace('.svg', f'_{reward_group.replace("+", "plus").replace("-", "minus")}_anova.csv')
                        anova_table.to_csv(anova_csv)
                        print(f"\n  Saved ANOVA table: {anova_csv}")

                except Exception as e:
                    print(f"\n  ANOVA failed: {e}")
                    print("  Falling back to paired t-tests per day")
                    anova_results = None
                    interaction_p = np.nan
            else:
                print(f"  Insufficient mice with both LMI+ and LMI- cells (n={len(mice_both)})")
                anova_results = None
                interaction_p = np.nan
        else:
            print("  Only one LMI category present - skipping ANOVA")
            anova_results = None
            interaction_p = np.nan

        # =====================================================================
        # POSTHOC TESTS: Paired t-tests per day with FDR correction
        # =====================================================================
        print(f"\n  Posthoc tests (LMI+ vs LMI- per day, FDR-corrected):")

        posthoc_results = {}
        p_values_all_days = []
        days_with_tests = []

        for day in days:
            # Get data for both LMI categories for this day
            pos_day = df_pos[df_pos['day'] == day]
            neg_day = df_neg[df_neg['day'] == day]

            # Find mice present in both groups
            mice_pos = set(pos_day['mouse_id'].unique())
            mice_neg = set(neg_day['mouse_id'].unique())
            mice_both = mice_pos & mice_neg

            if len(mice_both) >= 3:  # Need at least 3 pairs for meaningful test
                # Get paired data
                pos_values = []
                neg_values = []
                for mouse_id in sorted(mice_both):
                    pos_val = pos_day[pos_day['mouse_id'] == mouse_id]['participation_rate'].values[0]
                    neg_val = neg_day[neg_day['mouse_id'] == mouse_id]['participation_rate'].values[0]
                    pos_values.append(pos_val)
                    neg_values.append(neg_val)

                # Paired t-test
                try:
                    t_stat, p_val = ttest_rel(pos_values, neg_values)
                    p_values_all_days.append(p_val)
                    days_with_tests.append(day)
                    posthoc_results[day] = {'p_uncorrected': p_val, 'n_mice': len(mice_both)}
                except Exception as e:
                    print(f"    Day {day:+2d}: Test failed - {e}")
                    posthoc_results[day] = {'p_uncorrected': np.nan, 'n_mice': len(mice_both)}
            else:
                posthoc_results[day] = {'p_uncorrected': np.nan, 'n_mice': len(mice_both)}

        # Apply FDR correction (Benjamini-Hochberg)
        if len(p_values_all_days) > 0:
            reject, p_corrected, _, _ = multipletests(p_values_all_days, method='fdr_bh')

            # Update results with corrected p-values
            for idx, day in enumerate(days_with_tests):
                posthoc_results[day]['p_corrected'] = p_corrected[idx]
                posthoc_results[day]['significant_fdr'] = reject[idx]

            # Print results
            for day in days:
                if day in posthoc_results and not np.isnan(posthoc_results[day]['p_uncorrected']):
                    p_unc = posthoc_results[day]['p_uncorrected']
                    p_corr = posthoc_results[day].get('p_corrected', np.nan)
                    sig = posthoc_results[day].get('significant_fdr', False)
                    n = posthoc_results[day]['n_mice']

                    sig_marker = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
                    print(f"    Day {day:+2d}: n={n} mice, p_uncorr={p_unc:.4f}, p_FDR={p_corr:.4f} {sig_marker}")
                elif day in posthoc_results:
                    n = posthoc_results[day]['n_mice']
                    print(f"    Day {day:+2d}: Insufficient paired data (n={n} mice)")

            # Save posthoc results to CSV
            if save_path:
                posthoc_csv = save_path.replace('.svg', f'_{reward_group.replace("+", "plus").replace("-", "minus")}_posthoc.csv')
                posthoc_df = pd.DataFrame([
                    {'day': day, **posthoc_results[day]}
                    for day in days if day in posthoc_results
                ])
                posthoc_df.to_csv(posthoc_csv, index=False)
                print(f"\n  Saved posthoc results: {posthoc_csv}")
        else:
            print("    No valid paired comparisons available")
            posthoc_results = {day: {'p_corrected': np.nan, 'significant_fdr': False} for day in days}

        # Create barplot with individual mouse data points connected by lines
        x_positions = {day: idx for idx, day in enumerate(days)}
        bar_width = 0.35

        # Plot bars (mean across mice with 95% CI)
        for j, lmi_category in enumerate(lmi_categories):
            cat_data = plot_df[plot_df['lmi_category'] == lmi_category]
            if len(cat_data) == 0:
                continue

            # Compute stats per day
            day_stats_list = []
            for day in days:
                day_data = cat_data[cat_data['day'] == day]['participation_rate'].values
                if len(day_data) > 0:
                    mean_val = np.mean(day_data)
                    # 95% confidence interval
                    ci = 1.96 * np.std(day_data, ddof=1) / np.sqrt(len(day_data))
                    day_stats_list.append({
                        'day': day,
                        'mean': mean_val,
                        'ci': ci
                    })

            day_stats = pd.DataFrame(day_stats_list)

            x_vals = [x_positions[day] + (j - 0.5) * bar_width for day in day_stats['day']]

            ax.bar(x_vals, day_stats['mean'],
                  width=bar_width, label=f'{lmi_category.capitalize()} LMI',
                  color=colors[lmi_category], alpha=0.7, edgecolor='black', linewidth=1)

            # Add seaborn-style error bars (95% CI) - thinner, cleaner style
            ax.errorbar(x_vals, day_stats['mean'],
                       yerr=day_stats['ci'],
                       fmt='none', ecolor='black', capsize=4, linewidth=1.2,
                       capthick=1.2, alpha=0.9, zorder=10)

            # Plot individual mouse trajectories
            mice = cat_data['mouse_id'].unique()
            for mouse_id in mice:
                mouse_data = cat_data[cat_data['mouse_id'] == mouse_id].sort_values('day')
                mouse_x = [x_positions[day] + (j - 0.5) * bar_width for day in mouse_data['day']]
                mouse_y = mouse_data['participation_rate'].values

                # Lines only (no markers), colored by LMI category
                ax.plot(mouse_x, mouse_y, '-', color=colors[lmi_category],
                       linewidth=0.8, alpha=0.4, zorder=5)

        # Add significance markers for FDR-corrected posthoc tests
        y_max = 0.57
        for day in days:
            if day in posthoc_results:
                p_val = posthoc_results[day].get('p_corrected', np.nan)
                x_center = x_positions[day]

                if not np.isnan(p_val) and p_val < 0.05:
                    sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                    ax.text(x_center, y_max, sig_marker, ha='center', va='bottom',
                           fontsize=12, fontweight='bold')

        ax.set_xlabel('Day', fontweight='bold', fontsize=13)
        ax.set_ylabel('Participation Rate', fontweight='bold', fontsize=13)
        ax.set_title(f'{reward_group} Mice', fontweight='bold', fontsize=14)
        ax.set_xticks([x_positions[day] for day in days])
        ax.set_xticklabels([str(d) for d in days])
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 0.6)
        sns.despine(ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")

        # Save data CSV (per-mouse participation rates for each day and LMI category)
        data_csv = save_path.replace('.svg', '_data.csv')

        # Combine all data from both reward groups
        all_plot_data = []
        for reward_group in ['R+', 'R-']:
            # Get the merged_df for this reward group
            group_merged = merged_df[merged_df['reward_group'] == reward_group]
            lmi_categories = ['positive', 'negative']

            for lmi_category in lmi_categories:
                # Filter cells by LMI category
                mask = group_merged['lmi_category'] == lmi_category
                cells_df = group_merged[mask][['mouse_id', 'roi']]

                if len(cells_df) == 0:
                    continue

                # Get participation data for these cells
                for mouse_id in cells_df['mouse_id'].unique():
                    mouse_cells = cells_df[cells_df['mouse_id'] == mouse_id]

                    for day in days:
                        participation_rates = []
                        for _, row in mouse_cells.iterrows():
                            cell_mask = (participation_df_all['mouse_id'] == mouse_id) & \
                                       (participation_df_all['roi'] == row['roi']) & \
                                       (participation_df_all['day'] == day)
                            cell_data = participation_df_all[cell_mask]
                            if len(cell_data) > 0:
                                participation_rates.append(cell_data['participation_rate'].values[0])

                        # Mean rate for this mouse-day-category
                        if len(participation_rates) > 0:
                            mean_rate = np.mean(participation_rates)
                        else:
                            mean_rate = 0.0

                        all_plot_data.append({
                            'reward_group': reward_group,
                            'mouse_id': mouse_id,
                            'day': day,
                            'lmi_category': lmi_category,
                            'participation_rate': mean_rate,
                            'n_cells': len(participation_rates)
                        })

        plot_data_df = pd.DataFrame(all_plot_data)
        plot_data_df.to_csv(data_csv, index=False)
        print(f"  Data CSV saved: {data_csv}")

        plt.close()
    else:
        return fig


def generate_visualizations(merged_df, participation_df_all, output_dir):
    """
    Generate visualization figures as SVG.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged participation and LMI data
    participation_df_all : pd.DataFrame
        Per-day participation data
    output_dir : str
        Directory to save figures
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Save figures as SVG
    print("\n  Figure 1a: Participation rate vs LMI scatter plot (all cells)")
    plot_participation_rate_vs_lmi(
        merged_df,
        participation_metric='learning_rate',
        save_path=os.path.join(output_dir, 'participation_rate_vs_lmi_all.svg')
    )

    print("\n  Figure 1b: Participation rate vs LMI histogram (all cells)")
    plot_participation_rate_vs_lmi_histogram(
        merged_df,
        participation_metric='learning_rate',
        save_path=os.path.join(output_dir, 'participation_rate_vs_lmi_histogram_all.svg')
    )

    print("\n  Figure 2: Temporal evolution across days (barplot per mouse)")
    plot_temporal_evolution(
        merged_df,
        participation_df_all,
        save_path=os.path.join(output_dir, 'temporal_evolution_per_mouse.svg')
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("REACTIVATION PARTICIPATION AND LMI PREDICTION ANALYSIS")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Event window: ±{event_window_ms}ms (±{event_window_frames} frames)")
    print(f"  Participation threshold: {participation_threshold*100}% dF/F")
    print(f"  LMI thresholds: {LMI_NEGATIVE_THRESHOLD} (negative), {LMI_POSITIVE_THRESHOLD} (positive)")
    print(f"  Days: {days}")
    print(f"  Parallel jobs: {n_jobs}")

    # Create output directory
    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/reactivation_lmi'
    output_dir = io.adjust_path_to_host(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Load surrogate thresholds if available
    # Choose file based on threshold_mode
    if threshold_mode == 'mouse':
        surrogate_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates', 'surrogate_thresholds.csv')
    elif threshold_mode == 'day':
        surrogate_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates_per_day', 'surrogate_thresholds_per_day.csv')
    else:
        raise ValueError(f"Invalid threshold_mode '{threshold_mode}'. Must be 'mouse' or 'day'.")

    threshold_dict = None
    if use_surrogate_thresholds and os.path.exists(surrogate_csv_path):
        print(f"\n{'='*60}")
        print("LOADING SURROGATE-BASED THRESHOLDS")
        print(f"{'='*60}")
        threshold_dict = load_surrogate_thresholds(surrogate_csv_path, threshold_type=threshold_type, threshold_mode=threshold_mode)
        print(f"Loaded thresholds from: {surrogate_csv_path}")
        print(f"Threshold type: {threshold_type} ({'percentile-based' if threshold_type in ['percentile', '95'] else 'FWER/maximum'})")
        print(f"Threshold mode: {threshold_mode} ({'one per mouse' if threshold_mode == 'mouse' else 'per mouse-day'})")
        if threshold_mode == 'mouse':
            print(f"Loaded thresholds for {len(threshold_dict)} mice")
        else:
            n_mice = len(threshold_dict)
            n_mouse_days = sum(len(days_dict) for days_dict in threshold_dict.values())
            print(f"Loaded thresholds for {n_mice} mice, {n_mouse_days} mouse-day combinations")
    else:
        if not use_surrogate_thresholds:
            print(f"\nuse_surrogate_thresholds=False: Using common threshold instead of surrogate-based thresholds")
        elif not os.path.exists(surrogate_csv_path):
            print(f"\nSurrogate threshold file not found: {surrogate_csv_path}")
        print(f"Using common threshold: {threshold_corr}")

    # Load pre-computed reactivation events from reactivation.py
    print("\n" + "="*60)
    print("LOADING PRE-COMPUTED REACTIVATION EVENTS")
    print("="*60)

    r_plus_reactivations, r_minus_reactivations = load_reactivation_results(reactivation_results_file)

    # Create a combined dictionary for easy lookup
    all_reactivation_results = {}
    all_reactivation_results.update(r_plus_reactivations)
    all_reactivation_results.update(r_minus_reactivations)

    # Process all mice in parallel
    print("\n" + "="*60)
    print("PROCESSING ALL MICE")
    print("="*60)

    all_mice_to_process = r_plus_mice + r_minus_mice
    print(f"Processing {len(all_mice_to_process)} mice in parallel...")
    print(f"Using pre-computed events from reactivation.py for consistency")

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_mouse)(
            mouse, days, verbose=False, threshold_dict=threshold_dict,
            preloaded_results=all_reactivation_results.get(mouse, None)
        )
        for mouse in all_mice_to_process
    )

    # Combine results
    all_participation_data = []
    for mouse, participation_df in results_list:
        if participation_df is not None:
            all_participation_data.append(participation_df)

    if len(all_participation_data) == 0:
        print("\nERROR: No valid participation data collected!")
        sys.exit(1)

    participation_df_all = pd.concat(all_participation_data, ignore_index=True)
    print(f"\nCollected participation data: {len(participation_df_all)} cell-day records")

    # Save per-day participation rates
    csv_path = os.path.join(output_dir, 'cell_participation_rates_per_day.csv')
    participation_df_all.to_csv(csv_path, index=False)
    print(f"Saved per-day participation rates to: {csv_path}")

    # Aggregate across days
    print("\n" + "="*60)
    print("AGGREGATING ACROSS DAYS")
    print("="*60)

    aggregated_df = aggregate_across_days(participation_df_all)

    # Save aggregated participation
    csv_path = os.path.join(output_dir, 'cell_participation_rates_aggregated.csv')
    aggregated_df.to_csv(csv_path, index=False)
    print(f"Saved aggregated participation rates to: {csv_path}")

    # Load and match LMI data
    print("\n" + "="*60)
    print("MATCHING WITH LMI DATA")
    print("="*60)

    merged_df = load_and_match_lmi_data(aggregated_df)

    # Save merged data
    csv_path = os.path.join(output_dir, 'participation_lmi_merged.csv')
    merged_df.to_csv(csv_path, index=False)
    print(f"Saved merged data to: {csv_path}")

    # === DIAGNOSTIC: DATA AVAILABILITY ANALYSIS ===
    print("\n" + "="*60)
    print("DIAGNOSTIC: DATA AVAILABILITY PER MOUSE AND DAY")
    print("="*60)

    # Check how many cells have participation data for each day
    for reward_group in ['R+', 'R-']:
        print(f"\n{reward_group} mice:")
        for lmi_cat in ['positive', 'negative']:
            mask = (merged_df['reward_group'] == reward_group) & \
                   (merged_df['lmi_category'] == lmi_cat)
            cells = merged_df[mask][['mouse_id', 'roi']]

            print(f"  {lmi_cat.capitalize()} LMI cells: {len(cells)} total")

            # Count cells with data for each day
            for day in days:
                n_with_data = 0
                for _, cell in cells.iterrows():
                    cell_mask = (participation_df_all['mouse_id'] == cell['mouse_id']) & \
                               (participation_df_all['roi'] == cell['roi']) & \
                               (participation_df_all['day'] == day)
                    if len(participation_df_all[cell_mask]) > 0:
                        n_with_data += 1

                pct = 100 * n_with_data / len(cells) if len(cells) > 0 else 0
                print(f"    Day {day:+2d}: {n_with_data}/{len(cells)} cells ({pct:.1f}%)")

    # Check reasons for missing data
    print("\n  Possible reasons for missing data:")
    print("  1. No reactivation events detected on that day for that mouse")
    print("  2. Cell was not active/detected in the imaging on that day")
    print("  3. Data quality issues for that specific day")
    print(f"  4. Previously filtered out by reliability criterion (now removed)")

    # Count events per mouse-day
    print("\n  Checking event counts per mouse-day from participation data:")
    event_summary = participation_df_all.groupby(['mouse_id', 'day'])['n_events'].agg(['mean', 'min', 'max']).reset_index()
    print(f"\n  Event count summary (across all cells):")
    print(event_summary.to_string(index=False))

    # === BEFORE/AFTER INFLECTION ANALYSIS (DAY 0 ONLY) ===
    print("\n" + "="*60)
    print("PARTICIPATION RATES: BEFORE vs AFTER INFLECTION (DAY 0)")
    print("="*60)

    # Load inflection points from plasticity results (LMI cells)
    plasticity_csv = os.path.join(
        io.adjust_path_to_host(
            '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/plasticity'
        ),
        'plasticity_results_lmi_cells.csv'
    )

    if not os.path.exists(plasticity_csv):
        print(f"\n  WARNING: Plasticity results not found at {plasticity_csv}")
        print("  Skipping before/after inflection analysis. Run lmi_plasticity.py first.")
    else:
        plasticity_df = pd.read_csv(plasticity_csv)

        # Create inflection dict: (mouse_id, roi) -> inflection_trial_w
        inflection_dict = {
            (row['mouse_id'], row['roi']): row['inflection']
            for _, row in plasticity_df.iterrows()
            if not pd.isna(row['inflection'])
        }

        print(f"\n  Loaded {len(inflection_dict)} cells with inflection points")

        # Process all mice for before/after inflection analysis
        print(f"\n  Processing {len(all_mice_to_process)} mice in parallel...")

        results_inflection = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(analyze_mouse_participation_by_inflection)(
                mouse, inflection_dict, threshold_dict=threshold_dict, verbose=False
            )
            for mouse in all_mice_to_process
        )

        # Combine results
        inflection_data = [r for r in results_inflection if r is not None and len(r) > 0]

        if len(inflection_data) == 0:
            print("\n  ERROR: No valid before/after inflection data collected!")
        else:
            inflection_participation_df = pd.concat(inflection_data, ignore_index=True)
            print(f"\n  Collected before/after data: {len(inflection_participation_df)} cells")

            # Print summary
            n_both_reliable = (
                (inflection_participation_df['reliable_before']) &
                (inflection_participation_df['reliable_after'])
            ).sum()
            print(f"  Cells with reliable data both before and after: {n_both_reliable}")

            # Save
            csv_path = os.path.join(output_dir, 'cell_participation_rates_by_inflection.csv')
            inflection_participation_df.to_csv(csv_path, index=False)
            print(f"\n  Saved before/after inflection rates to: {csv_path}")

    # Generate visualizations
    generate_visualizations(merged_df, participation_df_all, output_dir)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(all_mice_to_process)} mice")
    print(f"Total cells analyzed: {merged_df['roi'].nunique()}")
    print(f"Results saved to: {output_dir}")
