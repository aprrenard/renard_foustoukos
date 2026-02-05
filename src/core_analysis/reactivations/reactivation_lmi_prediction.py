"""
Reactivation Participation and LMI Prediction Analysis

This script analyzes whether a neuron's participation in reactivation events
predicts its Learning Modulation Index (LMI). It computes participation rates
across days (baseline, learning, post-learning) and correlates with LMI values.

Approach:
1. For each reactivation event, extract cell responses in ±150ms window
2. Define participation: average response ≥ 5% dF/F
3. Compute participation rates per cell, per day
4. Aggregate: Baseline (days -2,-1), Learning (day 0), Post (days +1,+2)
5. Correlate participation with LMI, separately by LMI sign and reward group

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
from scipy.stats import spearmanr, pearsonr
from scipy import stats
from statsmodels.stats.multitest import multipletests
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

# Event detection parameters (must match reactivation.py)
threshold_type = 'percentile'  # Options: 'percentile' or 'max' (FWER)
threshold_mode = 'mouse'  # Options: 'mouse' (baseline-derived, same for all days) or 'day' (per-day thresholds)
threshold_dff = None  # 5% dff threshold for template cells (use None for all cells)
threshold_corr = 0.45  # Default correlation threshold for event detection (if no surrogate thresholds available)
min_event_distance_ms = 500
min_event_distance_frames = int(min_event_distance_ms / 1000 * sampling_rate)
prominence = 0.15  # Minimum prominence of peaks for event detection (vertical distance to contour line)

# NOTE: Surrogate-based thresholds
# If reactivation_surrogates.py or reactivation_surrogates_per_day.py has been run,
# the script will automatically load and use thresholds instead of the fixed threshold_corr value.
#
# threshold_type options:
#   - 'percentile': Uses the percentile specified in surrogate scripts (default: 99.9th)
#   - 'max' (FWER/maximum): More conservative, controls family-wise error rate
#
# threshold_mode options:
#   - 'mouse': One threshold per mouse (from baseline days -2 and -1), applied to all days
#              Loads from: reactivation_surrogates/surrogate_thresholds.csv
#   - 'day': Separate threshold per mouse-day combination, computed from each day's data
#            Loads from: reactivation_surrogates_per_day/surrogate_thresholds_per_day.csv

# Participation parameters
event_window_ms = 150  # ±150ms around event (total 300ms)
event_window_frames = int(event_window_ms / 1000 * sampling_rate)  # ±5 frames
participation_threshold = 0.1  # 5% dF/F for participation
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

def extract_event_responses(mouse, day, verbose=True, threshold_dict=None, inflection_points=None):
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
    inflection_points : dict, optional
        Dictionary mapping (mouse_id, roi) to inflection trial_w value (1-indexed).
        If provided and day==0, adds 'inflection_phase' column to responses_df
        with values: 'before', 'after', 'at_inflection', or 'unknown'.

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

        # Step 4: Detect events - use per-mouse, per-day threshold if available
        current_threshold = get_threshold_for_mouse_day(threshold_dict, mouse, day, threshold_corr, threshold_mode)
        if verbose and threshold_dict is not None:
            print(f"    Using threshold: {current_threshold:.4f} (surrogate-based, {threshold_mode} mode)")
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

def analyze_mouse_participation(mouse, days=[-2, -1, 0, 1, 2], verbose=True, threshold_dict=None):
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

        responses_df, n_events = extract_event_responses(mouse, day, verbose=verbose, threshold_dict=threshold_dict)

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


def process_single_mouse(mouse, days, verbose=False, threshold_dict=None):
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

    Returns
    -------
    tuple
        (mouse_id, participation_df)
    """
    participation_df = analyze_mouse_participation(mouse, days=days, verbose=verbose, threshold_dict=threshold_dict)
    return (mouse, participation_df)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_scatter_participation_vs_lmi(merged_df, participation_metric='learning_rate',
                                       save_path=None):
    """
    Create scatter plots of participation vs LMI for different groups.

    Page 1: 2×2 panels [R+, R-] × [LMI+, LMI-]
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    reward_groups = ['R+', 'R-']
    lmi_categories = ['positive', 'negative']

    for i, reward_group in enumerate(reward_groups):
        for j, lmi_category in enumerate(lmi_categories):
            ax = axes[j, i]

            # Filter data
            mask = (merged_df['reward_group'] == reward_group) & \
                   (merged_df['lmi_category'] == lmi_category)

            # Add reliability filter
            if participation_metric in ['baseline_rate']:
                mask &= merged_df['reliable_baseline']
            elif participation_metric in ['learning_rate']:
                mask &= merged_df['reliable_learning']
            elif participation_metric in ['post_rate']:
                mask &= merged_df['reliable_post']
            elif participation_metric in ['delta_learning', 'delta_post']:
                mask &= merged_df['reliable_baseline']

            group_data = merged_df[mask]

            # Remove NaN values
            valid_mask = ~(group_data[participation_metric].isna() | group_data['lmi'].isna())
            x = group_data.loc[valid_mask, participation_metric].values
            y = group_data.loc[valid_mask, 'lmi'].values

            if len(x) < 3:
                ax.text(0.5, 0.5, f'Insufficient data\n(n={len(x)})',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xlabel(participation_metric.replace('_', ' ').title())
                ax.set_ylabel('LMI')
                ax.set_title(f'{reward_group} {lmi_category.capitalize()} LMI')
                continue

            # Scatter plot
            ax.scatter(x, y, s=40, alpha=0.6, edgecolors='black', linewidths=0.5)

            # Regression line
            if len(x) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7)

                # Compute Spearman correlation
                spearman_r, spearman_p = spearmanr(x, y)

                # Add statistics text
                stats_text = f'n = {len(x)} cells\n'
                stats_text += f'Spearman r = {spearman_r:.3f}\n'
                if spearman_p < 0.001:
                    stats_text += f'p < 0.001 ***'
                elif spearman_p < 0.01:
                    stats_text += f'p = {spearman_p:.3f} **'
                elif spearman_p < 0.05:
                    stats_text += f'p = {spearman_p:.3f} *'
                else:
                    stats_text += f'p = {spearman_p:.3f} ns'

                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            ax.set_xlabel(participation_metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('LMI', fontweight='bold')
            ax.set_title(f'{reward_group} {lmi_category.capitalize()} LMI', fontweight='bold')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_temporal_evolution(merged_df, participation_df_all, save_path=None):
    """
    Plot participation rates across days for LMI+ vs LMI- cells.

    Page 2: Line plots [R+ left, R- right]
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    reward_groups = ['R+', 'R-']
    lmi_categories = ['positive', 'negative']
    colors = {'positive': '#d62728', 'negative': '#1f77b4'}

    for i, reward_group in enumerate(reward_groups):
        ax = axes[i]

        for lmi_category in lmi_categories:
            # Filter cells
            mask = (merged_df['reward_group'] == reward_group) & \
                   (merged_df['lmi_category'] == lmi_category)
            cell_list = merged_df[mask][['mouse_id', 'roi']].values

            if len(cell_list) == 0:
                continue

            # Get participation data for these cells
            participation_data = []
            for mouse_id, roi in cell_list:
                cell_mask = (participation_df_all['mouse_id'] == mouse_id) & \
                           (participation_df_all['roi'] == roi)
                cell_data = participation_df_all[cell_mask]
                participation_data.append(cell_data)

            if len(participation_data) == 0:
                continue

            participation_combined = pd.concat(participation_data, ignore_index=True)

            # Compute mean and SEM per day
            day_stats = participation_combined.groupby('day')['participation_rate'].agg(['mean', 'sem']).reset_index()

            ax.plot(day_stats['day'], day_stats['mean'], '-o',
                   color=colors[lmi_category], linewidth=2, markersize=8,
                   label=f'{lmi_category.capitalize()} LMI (n={len(cell_list)})')

            ax.fill_between(day_stats['day'],
                           day_stats['mean'] - day_stats['sem'],
                           day_stats['mean'] + day_stats['sem'],
                           color=colors[lmi_category], alpha=0.3)

        ax.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax.set_ylabel('Participation Rate', fontweight='bold', fontsize=12)
        ax.set_title(f'{reward_group} Mice', fontweight='bold', fontsize=14)
        ax.set_xticks(days)
        ax.set_xticklabels([str(d) for d in days])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_change_scores(merged_df, save_path=None):
    """
    Plot change scores (delta participation) vs LMI.

    Page 3: Same 2×2 layout as Page 1, but for delta_learning
    """
    return plot_scatter_participation_vs_lmi(merged_df, participation_metric='delta_learning',
                                              save_path=save_path)


def plot_distributions(merged_df, participation_df_all, save_path=None):
    """
    Plot distribution comparisons of participation rates.

    Page 4: Violin/box plots grouped by Day × LMI sign
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    reward_groups = ['R+', 'R-']

    for i, reward_group in enumerate(reward_groups):
        ax = axes[i]

        # Prepare data
        plot_data = []

        for lmi_category in ['positive', 'negative']:
            # Filter cells
            mask = (merged_df['reward_group'] == reward_group) & \
                   (merged_df['lmi_category'] == lmi_category)
            cell_list = merged_df[mask][['mouse_id', 'roi']].values

            if len(cell_list) == 0:
                continue

            # Get participation data
            for mouse_id, roi in cell_list:
                cell_mask = (participation_df_all['mouse_id'] == mouse_id) & \
                           (participation_df_all['roi'] == roi)
                cell_data = participation_df_all[cell_mask]

                for _, row in cell_data.iterrows():
                    plot_data.append({
                        'day': row['day'],
                        'participation_rate': row['participation_rate'],
                        'lmi_category': lmi_category
                    })

        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            continue

        plot_df = pd.DataFrame(plot_data)

        # Violin plot
        sns.violinplot(data=plot_df, x='day', y='participation_rate', hue='lmi_category',
                      ax=ax, palette={'positive': '#d62728', 'negative': '#1f77b4'},
                      split=False, inner='quartile')

        ax.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax.set_ylabel('Participation Rate', fontweight='bold', fontsize=12)
        ax.set_title(f'{reward_group} Mice', fontweight='bold', fontsize=14)
        ax.legend(title='LMI Category', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return fig


def generate_visualizations(merged_df, participation_df_all, correlation_results_df, output_dir):
    """
    Generate all visualization PDFs.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged participation and LMI data
    participation_df_all : pd.DataFrame
        Per-day participation data
    correlation_results_df : pd.DataFrame
        Correlation statistics
    output_dir : str
        Directory to save figures
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Create multi-page PDF
    pdf_path = os.path.join(output_dir, 'participation_lmi_analysis.pdf')

    with PdfPages(pdf_path) as pdf:
        # Page 1: Scatter plots - Learning participation vs LMI
        print("  Page 1: Scatter plots (Learning participation vs LMI)")
        fig = plot_scatter_participation_vs_lmi(merged_df, participation_metric='learning_rate')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Temporal evolution
        print("  Page 2: Temporal evolution across days")
        fig = plot_temporal_evolution(merged_df, participation_df_all)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 3: Change scores vs LMI
        print("  Page 3: Scatter plots (Change scores vs LMI)")
        fig = plot_change_scores(merged_df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 4: Distribution comparisons
        print("  Page 4: Distribution comparisons")
        fig = plot_distributions(merged_df, participation_df_all)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"\nSaved visualization PDF to: {pdf_path}")

    # Also save individual plots as SVG
    svg_dir = os.path.join(output_dir, 'svg_figures')
    os.makedirs(svg_dir, exist_ok=True)

    print("\n  Saving individual SVG figures...")
    plot_scatter_participation_vs_lmi(merged_df, participation_metric='learning_rate',
                                      save_path=os.path.join(svg_dir, 'scatter_learning_vs_lmi.svg'.replace('.svg', '.pdf')))
    plot_temporal_evolution(merged_df, participation_df_all,
                           save_path=os.path.join(svg_dir, 'temporal_evolution.svg'.replace('.svg', '.pdf')))
    plot_change_scores(merged_df,
                      save_path=os.path.join(svg_dir, 'scatter_delta_vs_lmi.svg'.replace('.svg', '.pdf')))
    plot_distributions(merged_df, participation_df_all,
                      save_path=os.path.join(svg_dir, 'distributions.svg'.replace('.svg', '.pdf')))

    print(f"  Saved individual figures to: {svg_dir}")


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
    if os.path.exists(surrogate_csv_path):
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
        print(f"\nSurrogate threshold file not found: {surrogate_csv_path}")
        print(f"Using default threshold: {threshold_corr}")

    # Process all mice in parallel
    print("\n" + "="*60)
    print("PROCESSING ALL MICE")
    print("="*60)

    all_mice_to_process = r_plus_mice + r_minus_mice
    print(f"Processing {len(all_mice_to_process)} mice in parallel...")

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_mouse)(mouse, days, verbose=False, threshold_dict=threshold_dict)
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

    # Compute correlations for different participation metrics
    print("\n" + "="*60)
    print("COMPUTING CORRELATIONS")
    print("="*60)

    participation_metrics = [
        'baseline_rate', 'learning_rate', 'post_rate',
        'delta_learning', 'delta_post'
    ]

    all_correlation_results = []
    for metric in participation_metrics:
        print(f"\n  Computing correlations for: {metric}")
        corr_results = correlate_with_lmi(merged_df, participation_metric=metric, reliable_filter=True)
        all_correlation_results.append(corr_results)

    correlation_results_df = pd.concat(all_correlation_results, ignore_index=True)

    # Save correlation results
    csv_path = os.path.join(output_dir, 'participation_lmi_correlations.csv')
    correlation_results_df.to_csv(csv_path, index=False)
    print(f"\nSaved correlation results to: {csv_path}")

    # Print summary
    print("\n" + "="*60)
    print("CORRELATION SUMMARY")
    print("="*60)
    print(correlation_results_df.to_string(index=False))

    # Generate visualizations
    generate_visualizations(merged_df, participation_df_all, correlation_results_df, output_dir)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(all_mice_to_process)} mice")
    print(f"Total cells analyzed: {merged_df['roi'].nunique()}")
    print(f"Results saved to: {output_dir}")
