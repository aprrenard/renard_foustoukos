"""
Decoding analysis restricted to high-participation cells.

Reproduces the decoding analysis from decoding.py but filters to cells with
high reactivation participation rates during day 0.

Strategy:
- Train decoder on FULL population (all cells with participation data)
- Compute decision values twice:
  1. Using all cells (baseline)
  2. Using only high-participation cells (participation_rate ≥ threshold)

This tests whether high-participation cells (reactivating neurons) carry most
discriminative information about learning state transitions.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *


# ============================================================================
# PARAMETERS
# ============================================================================

# Analysis parameters
sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = False
projection_type = None  # 'wS2', 'wM1' or None

# Participation filtering parameters
PARTICIPATION_THRESHOLD = 0.5  # Minimum participation rate to include cell
RUN_BOTH_POPULATIONS = True  # Run both full and filtered analyses

# Get mice
_, _, mice, db = io.select_sessions_from_db(io.db_path, io.nwb_dir, two_p_imaging='yes')
print(f"Found {len(mice)} mice: {mice}")


# ============================================================================
# LOAD PARTICIPATION RATES
# ============================================================================

participation_csv = io.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/reactivation_lmi/'
    'cell_participation_rates_aggregated.csv'
)

if not os.path.exists(participation_csv):
    raise FileNotFoundError(
        f"Participation rates CSV not found at {participation_csv}. "
        "Run reactivation_lmi_prediction.py first."
    )

print(f"\nLoading participation rates from: {participation_csv}")
participation_df = pd.read_csv(participation_csv)
print(f"  Loaded {len(participation_df)} cells with participation data")

# Create dictionary for fast lookup: (mouse_id, roi) -> learning_rate
participation_dict = {}
for _, row in participation_df.iterrows():
    participation_dict[(row['mouse_id'], row['roi'])] = row['learning_rate']

# Print statistics
n_high_part = (participation_df['learning_rate'] >= PARTICIPATION_THRESHOLD).sum()
print(f"  Cells with participation ≥ {PARTICIPATION_THRESHOLD}: {n_high_part}/{len(participation_df)} ({100*n_high_part/len(participation_df):.1f}%)")


# ============================================================================
# LOAD DATA
# ============================================================================

vectors_rew_mapping = []
vectors_nonrew_mapping = []
mice_rew = []
mice_nonrew = []
vectors_nonrew_day0_learning = []
vectors_rew_day0_learning = []

# Load behaviour table with learning trials
path = io.adjust_path_to_host(
    r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
    r'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
)
table = pd.read_csv(path)
# Select day 0 performance for whisker trials
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1)]

print("\nLoading imaging data for all mice...")
for mouse in mice:
    print(f"  Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days
    xarray = xarray.sel(trial=xarray['day'].isin(days))

    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"    No cells of type {projection_type} for mouse {mouse}.")
            continue

    # Check that each day has at least n_map_trials mapping trials
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'    Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    # Average bins
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')

    # Remove artefacts by setting them at 0
    print(f"    {np.isnan(d.values).sum()} NaN values in mapping data")
    d = d.fillna(0)

    if rew_gp == 'R-':
        vectors_nonrew_mapping.append(d)
        mice_nonrew.append(mouse)
    elif rew_gp == 'R+':
        vectors_rew_mapping.append(d)
        mice_rew.append(mouse)

    # Load learning data for day 0
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days
    xarray = xarray.sel(trial=xarray['day'].isin([0]))
    # Select whisker trials
    xarray = xarray.sel(trial=xarray['whisker_stim'] == 1)

    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"    No cells of type {projection_type} for mouse {mouse}.")
            continue

    # Average bins
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')

    # Remove artefacts
    print(f"    {np.isnan(xarray.values).sum()} NaN values in learning data")
    xarray = xarray.fillna(0)

    if rew_gp == 'R-':
        vectors_nonrew_day0_learning.append(xarray)
    elif rew_gp == 'R+':
        vectors_rew_day0_learning.append(xarray)

print(f"\nData loaded:")
print(f"  R+ mice: {len(mice_rew)}")
print(f"  R- mice: {len(mice_nonrew)}")


# ============================================================================
# PROGRESSIVE LEARNING ANALYSIS WITH PARTICIPATION FILTERING
# ============================================================================

def progressive_learning_analysis(
    vectors_mapping, vectors_learning, mice_list, bh_df=None,
    pre_days=[-2, -1], post_days=[1, 2], window_size=10, step_size=5,
    align_to_learning=False, trials_before=50, trials_after=100, seed=42,
    participation_dict=None, participation_threshold=0.5, run_both_populations=True
):
    """
    Analyze progressive learning during Day 0 using a sliding window approach.
    Train decoder on Day -2/-1 vs +1/+2, then apply to Day 0 learning trials.

    NEW: Optionally filter by participation rate during day 0.

    Parameters
    ----------
    vectors_mapping : list
        List of xarrays with mapping data for each mouse
    vectors_learning : list
        List of xarrays with Day 0 learning data for each mouse
    mice_list : list
        List of mouse IDs
    bh_df : pd.DataFrame, optional
        Behavioral dataframe with 'learning_trial' column for alignment
    pre_days : list
        Days to use as "pre-learning" for training (default: [-2, -1])
    post_days : list
        Days to use as "post-learning" for training (default: [1, 2])
    window_size : int
        Number of trials in each sliding window
    step_size : int
        Step size for sliding window
    align_to_learning : bool
        If True, align trials to learning onset (trial 0 = learning_trial)
    trials_before : int
        Number of trials before learning onset to include (only if align_to_learning=True)
    trials_after : int
        Number of trials after learning onset to include (only if align_to_learning=True)
    seed : int
        Random seed for classifier
    participation_dict : dict, optional
        Dictionary mapping (mouse_id, roi) -> participation_rate
    participation_threshold : float
        Minimum participation rate to include cell in filtered analysis
    run_both_populations : bool
        If True, run both full and filtered analyses

    Returns
    -------
    tuple
        (pd.DataFrame with results, dict with classifier weights)
        Results DataFrame has 'population_type' column: 'full' or 'filtered'
    """
    results = []
    classifier_weights = {}

    for i, (d_mapping, d_learning, mouse) in enumerate(zip(vectors_mapping, vectors_learning, mice_list)):
        print(f"\n  {mouse}: mapping shape={d_mapping.shape}, learning shape={d_learning.shape}")
        day_per_trial = d_mapping['day'].values

        # Extract ROI coordinates for this mouse
        if 'roi' in d_mapping.coords:
            roi_coords = d_mapping.coords['roi'].values
        else:
            roi_coords = np.arange(d_mapping.shape[0])

        n_total = len(roi_coords)

        # Create masks for participation filtering
        if participation_dict is not None:
            # Mask for cells with participation data
            valid_cells_mask = np.array([
                (mouse, roi) in participation_dict
                for roi in roi_coords
            ])

            # Mask for high-participation cells (among valid cells)
            high_part_mask = np.array([
                participation_dict.get((mouse, roi), 0) >= participation_threshold
                for roi in roi_coords
            ])

            n_valid = np.sum(valid_cells_mask)
            n_high_part = np.sum(high_part_mask & valid_cells_mask)

            print(f"    Cells with participation data: {n_valid}/{n_total}")
            print(f"    Cells with participation ≥ {participation_threshold}: {n_high_part}/{n_valid}")

            if n_valid == 0:
                print(f"    WARNING: No cells with participation data for {mouse}, skipping")
                continue

            # Filter data to cells with participation data
            d_mapping_filtered = d_mapping.sel(cell=valid_cells_mask)
            d_learning_filtered = d_learning.sel(cell=valid_cells_mask)

            # Update high_part_mask to match filtered data
            high_part_mask_filtered = high_part_mask[valid_cells_mask]
        else:
            # No filtering - use all cells
            d_mapping_filtered = d_mapping
            d_learning_filtered = d_learning
            high_part_mask_filtered = np.ones(n_total, dtype=bool)
            n_valid = n_total
            n_high_part = n_total
            print(f"    No participation filtering - using all {n_total} cells")

        # Get Day -2/-1 and +1/+2 trials for training from mapping data
        train_mask = np.isin(day_per_trial, pre_days + post_days)
        if np.sum(train_mask) < 4:
            print(f"    Not enough training trials for {mouse}, skipping")
            continue

        X_train = d_mapping_filtered.values[:, train_mask].T
        y_train = np.array([0 if day in pre_days else 1 for day in day_per_trial[train_mask]])

        # Train classifier on FULL valid population
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        clf.fit(X_train_scaled, y_train)

        # Store classifier weights
        classifier_weights[mouse] = {
            'coef': clf.coef_[0],
            'roi': d_mapping_filtered.coords['roi'].values if 'roi' in d_mapping_filtered.coords else np.arange(clf.coef_.shape[1]),
            'intercept': clf.intercept_[0],
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'n_features': clf.coef_.shape[1],
            'sign_flip': None
        }

        # Sanity check: Verify sign convention
        pre_mask = np.isin(day_per_trial, pre_days)
        post_mask = np.isin(day_per_trial, post_days)
        if np.sum(pre_mask) > 0 and np.sum(post_mask) > 0:
            X_pre = scaler.transform(d_mapping_filtered.values[:, pre_mask].T)
            X_post = scaler.transform(d_mapping_filtered.values[:, post_mask].T)
            mean_dec_pre = np.mean(clf.decision_function(X_pre))
            mean_dec_post = np.mean(clf.decision_function(X_post))
        else:
            mean_dec_pre, mean_dec_post = 0.0, 0.0

        if mean_dec_pre > mean_dec_post:
            print(f"    WARNING: Flipped decision values! Pre={mean_dec_pre:.3f}, Post={mean_dec_post:.3f}")
            print(f"      Flipping sign for plotting consistency")
            sign_flip = -1
        else:
            sign_flip = 1
            print(f"    Decision values oriented correctly. Pre={mean_dec_pre:.3f}, Post={mean_dec_post:.3f}")

        classifier_weights[mouse]['sign_flip'] = sign_flip

        # Get learning trial for this mouse if aligning
        learning_trial_idx = None
        if align_to_learning and bh_df is not None:
            mouse_bh = bh_df[bh_df['mouse_id'] == mouse]
            if not mouse_bh.empty and 'learning_trial' in mouse_bh.columns:
                learning_trial_val = mouse_bh['learning_trial'].iloc[0]
                if not np.isnan(learning_trial_val):
                    learning_trial_idx = int(learning_trial_val)
                    print(f"    Learning onset at trial_w = {learning_trial_idx}")

        # Get Day 0 learning trials
        n_learning_trials = d_learning_filtered.sizes['trial']

        # Determine trial range to analyze
        if align_to_learning and learning_trial_idx is not None:
            trial_start = max(0, learning_trial_idx - trials_before)
            trial_end = min(n_learning_trials, learning_trial_idx + trials_after)
            trial_offset = learning_trial_idx
        else:
            trial_start = 0
            trial_end = n_learning_trials
            trial_offset = 0

        # Define population types to analyze
        populations = []
        if run_both_populations:
            populations = [
                ('full', None),  # No filtering
                ('filtered', high_part_mask_filtered)  # Apply mask
            ]
        else:
            populations = [('filtered', high_part_mask_filtered)]

        # Sliding window analysis on learning trials
        for pop_type, mask in populations:
            window_results = []

            for start_idx in range(trial_start, max(trial_start, trial_end - window_size + 1), step_size):
                end_idx = start_idx + window_size
                if end_idx > trial_end:
                    break

                # Get window data from learning trials
                X_window = d_learning_filtered.values[:, start_idx:end_idx].T
                if X_window.shape[0] == 0:
                    continue

                # Apply participation filter if specified
                if mask is not None:
                    # Zero out low-participation cells
                    X_window_masked = X_window.copy()
                    X_window_masked[:, ~mask] = 0.0
                    X_window_scaled = scaler.transform(X_window_masked)
                else:
                    X_window_scaled = scaler.transform(X_window)

                # Decision values (distance to hyperplane)
                decision_values = clf.decision_function(X_window_scaled)
                mean_decision_value = np.mean(decision_values) * sign_flip

                # Probability of being "post"
                if hasattr(clf, "predict_proba"):
                    preds = clf.predict(X_window_scaled)
                    mean_proba_post = np.mean(preds == 1)
                else:
                    dv = decision_values * sign_flip
                    mean_proba_post = np.mean(1 / (1 + np.exp(-dv)))

                # Store both absolute and aligned trial indices
                trial_center_abs = start_idx + window_size // 2
                trial_center_aligned = trial_center_abs - trial_offset

                window_results.append({
                    'window_start': start_idx,
                    'window_center': start_idx + window_size // 2,
                    'window_end': end_idx,
                    'trial_start': start_idx,
                    'trial_center': trial_center_abs,
                    'trial_center_aligned': trial_center_aligned,
                    'trial_end': end_idx,
                    'mean_decision_value': mean_decision_value,
                    'mean_proba_post': mean_proba_post,
                    'mouse_idx': i,
                    'mouse_id': mouse,
                    'population_type': pop_type,  # NEW
                    'n_cells': n_valid if pop_type == 'full' else n_high_part  # NEW
                })

            results.extend(window_results)

    return pd.DataFrame(results), classifier_weights


# ============================================================================
# RUN ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("RUNNING PARTICIPATION-FILTERED DECODING ANALYSIS")
print("="*70)

# Analysis parameters
window_size = 10
step_size = 1
align_to_learning = False
trials_before = 50
trials_after = 100

# Try to get good/bad subsets
try:
    mice_good = [m for m in mice_rew if m in mice_groups.get('good_day0', [])]
    mice_bad = [m for m in mice_rew if m in (mice_groups.get('bad_day0', []) + mice_groups.get('meh_day0', []))]
except Exception:
    mice_good, mice_bad = [], []

# Run analysis for R+ and R- groups
print("\n### R+ mice ###")
results_rew, weights_rew = progressive_learning_analysis(
    vectors_rew_mapping, vectors_rew_day0_learning, mice_rew,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after,
    participation_dict=participation_dict, participation_threshold=PARTICIPATION_THRESHOLD,
    run_both_populations=RUN_BOTH_POPULATIONS
)

print("\n### R- mice ###")
results_nonrew, weights_nonrew = progressive_learning_analysis(
    vectors_nonrew_mapping, vectors_nonrew_day0_learning, mice_nonrew,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after,
    participation_dict=participation_dict, participation_threshold=PARTICIPATION_THRESHOLD,
    run_both_populations=RUN_BOTH_POPULATIONS
)

# Add reward group labels
results_rew['reward_group'] = 'R+'
results_nonrew['reward_group'] = 'R-'
results_combined = pd.concat([results_rew, results_nonrew], ignore_index=True)

print(f"\nCollected {len(results_combined)} window results")
print(f"  Full population results: {len(results_combined[results_combined['population_type'] == 'full'])}")
print(f"  Filtered population results: {len(results_combined[results_combined['population_type'] == 'filtered'])}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = io.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
)
os.makedirs(output_dir, exist_ok=True)

# Save results
results_file = os.path.join(output_dir, 'decoder_results_participation_filtered.csv')
results_combined.to_csv(results_file, index=False)
print(f"\nSaved results to: {results_file}")

# Save classifier weights
weights_file = os.path.join(output_dir, 'classifier_weights_participation_filtered.pkl')
all_weights = {
    'R+': weights_rew,
    'R-': weights_nonrew
}
with open(weights_file, 'wb') as f:
    pickle.dump(all_weights, f)
print(f"Saved classifier weights to: {weights_file}")


# ============================================================================
# PLOTTING
# ============================================================================

print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

# Define cut_n_trials for plotting
cut_n_trials = 120


def plot_behavior(ax, data, color, title, cut_n_trials=cut_n_trials, align_to_learning=False):
    """Plot behavioral learning curves."""
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0,
                   cut_n_trials//2 if align_to_learning else cut_n_trials)
        ax.set_ylim(0, 1)
        return

    if align_to_learning and 'learning_trial' in data.columns:
        data_plot = data.copy()
        data_plot['trial_w_aligned'] = data_plot.groupby('mouse_id').apply(
            lambda x: x['trial_w'] - x['learning_trial'].iloc[0] if not pd.isna(x['learning_trial'].iloc[0]) else x['trial_w']
        ).reset_index(level=0, drop=True)
        x_col = 'trial_w_aligned'
        xlabel = 'Trial relative to learning onset'
    else:
        data_plot = data
        x_col = 'trial_w'
        xlabel = 'Trial within Day 0'

    sns.lineplot(data=data_plot, x=x_col, y='learning_curve_w', color=color, errorbar='ci', ax=ax)
    if align_to_learning:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Learning onset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Learning curve (w)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    if cut_n_trials is not None:
        if align_to_learning:
            ax.set_xlim(-cut_n_trials//2, cut_n_trials//2)
        else:
            ax.set_xlim(0, cut_n_trials)


def plot_decision(ax, data, colors, title, cut_n_trials=cut_n_trials, align_to_learning=False, ylim=None):
    """Plot decision values for both full and filtered populations."""
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0,
                   cut_n_trials//2 if align_to_learning else cut_n_trials)
        return

    if align_to_learning and 'trial_center_aligned' in data.columns:
        x_col = 'trial_center_aligned'
        xlabel = 'Trial relative to learning onset'
    else:
        x_col = 'trial_center'
        xlabel = 'Trial within Day 0'

    # Plot full population (solid line)
    data_full = data[data['population_type'] == 'full']
    if len(data_full) > 0:
        sns.lineplot(data=data_full, x=x_col, y='mean_decision_value', estimator=np.mean,
                    errorbar='ci', color=colors[0], ax=ax, label='Full population', linestyle='-')

    # Plot filtered population (dashed line)
    data_filtered = data[data['population_type'] == 'filtered']
    if len(data_filtered) > 0:
        sns.lineplot(data=data_filtered, x=x_col, y='mean_decision_value', estimator=np.mean,
                    errorbar='ci', color=colors[1], ax=ax, label='High participation', linestyle='--')

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    if align_to_learning:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean decision value')
    ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if cut_n_trials is not None:
        if align_to_learning:
            ax.set_xlim(-cut_n_trials//2, cut_n_trials//2)
        else:
            ax.set_xlim(0, cut_n_trials)


def plot_proba(ax, data, colors, title, cut_n_trials=cut_n_trials, align_to_learning=False):
    """Plot probability P(post) for both full and filtered populations."""
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0,
                   cut_n_trials//2 if align_to_learning else cut_n_trials)
        ax.set_ylim(0, 1)
        return

    if align_to_learning and 'trial_center_aligned' in data.columns:
        x_col = 'trial_center_aligned'
        xlabel = 'Trial relative to learning onset'
    else:
        x_col = 'trial_center'
        xlabel = 'Trial within Day 0'

    # Plot full population (solid line)
    data_full = data[data['population_type'] == 'full']
    if len(data_full) > 0:
        sns.lineplot(data=data_full, x=x_col, y='mean_proba_post', estimator=np.mean,
                    errorbar='ci', color=colors[0], ax=ax, label='Full population', linestyle='-')

    # Plot filtered population (dashed line)
    data_filtered = data[data['population_type'] == 'filtered']
    if len(data_filtered) > 0:
        sns.lineplot(data=data_filtered, x=x_col, y='mean_proba_post', estimator=np.mean,
                    errorbar='ci', color=colors[1], ax=ax, label='High participation', linestyle='--')

    ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--')
    if align_to_learning:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('P(post)')
    ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    ax.set_ylim(0, 1)
    if cut_n_trials is not None:
        if align_to_learning:
            ax.set_xlim(-cut_n_trials//2, cut_n_trials//2)
        else:
            ax.set_xlim(0, cut_n_trials)


# Create 3-row x 2-col figure (R+ and R- only)
fig = plt.figure(figsize=(14, 12))

# Define colors for full vs filtered
colors_rew = ['#d62728', '#ff9896']  # Red (full), light red (filtered)
colors_nonrew = ['#1f77b4', '#aec7e8']  # Blue (full), light blue (filtered)

# Top row: Behavioral learning curves
ax1 = plt.subplot(3, 2, 1)
data_rew = bh_df.loc[bh_df['mouse_id'].isin(mice_rew)]
plot_behavior(ax1, data_rew, colors_rew[0], 'R+ mice behavior', align_to_learning=align_to_learning)

ax2 = plt.subplot(3, 2, 2)
data_nonrew = bh_df.loc[bh_df['mouse_id'].isin(mice_nonrew)]
plot_behavior(ax2, data_nonrew, colors_nonrew[0], 'R- mice behavior', align_to_learning=align_to_learning)

# Middle row: Decision values
ax3 = plt.subplot(3, 2, 3)
plot_decision(ax3, results_rew, colors_rew, 'R+ mean decision value',
             align_to_learning=align_to_learning, ylim=(-3, 6))

ax4 = plt.subplot(3, 2, 4)
plot_decision(ax4, results_nonrew, colors_nonrew, 'R- mean decision value',
             align_to_learning=align_to_learning, ylim=(-3, 6))

# Bottom row: Probability P(post)
ax5 = plt.subplot(3, 2, 5)
plot_proba(ax5, results_rew, colors_rew, 'R+ P(post)', align_to_learning=align_to_learning)

ax6 = plt.subplot(3, 2, 6)
plot_proba(ax6, results_nonrew, colors_nonrew, 'R- P(post)', align_to_learning=align_to_learning)

plt.tight_layout()
sns.despine()

# Save figure
fig_file = os.path.join(output_dir, 'decoder_decision_value_participation_filtered.svg')
plt.savefig(fig_file, format='svg', dpi=300)
print(f"\nSaved figure to: {fig_file}")

plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"Results saved to: {output_dir}")
print(f"  - decoder_results_participation_filtered.csv")
print(f"  - classifier_weights_participation_filtered.pkl")
print(f"  - decoder_decision_value_participation_filtered.svg")
