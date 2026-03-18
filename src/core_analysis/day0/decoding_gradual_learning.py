"""
Decoding analysis of gradual learning during Day 0.

This script analyzes progressive learning during Day 0 using a sliding window approach.
It trains a decoder on Day -2 vs +2 mapping trials, then applies it to Day 0 learning trials
to track the decision values and probabilities over time.

The analysis includes:
- Progressive learning analysis with sliding windows
- Individual mouse analysis
- Statistical quantification (slope analysis)
- Correlation analysis (decision values vs behavioral performance)
- Visualization of results
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon, ttest_1samp, pearsonr, linregress
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *


# ============================================================================
# CONFIGURATION
# ============================================================================

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = False
projection_type = None  # 'wS2', 'wM1' or None

# Analysis parameters
window_size = 10
step_size = 1
align_to_learning = False  # Set to True to align to individual learning onset
trials_before = 50  # Number of trials before learning onset to include
trials_after = 100  # Number of trials after learning onset to include
cut_n_trials = 100  # For plotting


# ============================================================================
# LOAD DATA
# ============================================================================

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(f"Found {len(mice)} mice: {mice}")

# Load responsive cells if needed
if select_responsive_cells:
    test_df = os.path.join(io.processed_dir, f'response_test_results_win_180ms.csv')
    test_df = pd.read_csv(test_df)
    test_df = test_df.loc[test_df['day'].isin(days)]
    # Select cells as responsive if they pass the test on at least one day
    selected_cells = test_df.groupby(['mouse_id', 'roi', 'cell_type'])['pval_mapping'].min().reset_index()
    selected_cells = selected_cells.loc[selected_cells['pval_mapping'] <= 0.05/5]

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

# Load data
vectors_rew_mapping = []
vectors_nonrew_mapping = []
mice_rew = []
mice_nonrew = []
vectors_nonrew_day0_learning = []
vectors_rew_day0_learning = []

# Load behaviour table with learning trials
path = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(path)
# Select day 0 performance for whisker trials
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1)]

print("\nLoading data for each mouse...")
for mouse in mice:

    # Load mapping data
    # ------------------

    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days
    xarray = xarray.sel(trial=xarray['day'].isin(days))

    # Select responsive cells
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue

    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    # Average bins
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')

    # Remove artefacts by setting them at 0. To avoid NaN values and mismatches
    print(np.isnan(d.values).sum(), 'NaN values in the data.')
    d = d.fillna(0)

    if rew_gp == 'R-':
        vectors_nonrew_mapping.append(d)
        mice_nonrew.append(mouse)
    elif rew_gp == 'R+':
        vectors_rew_mapping.append(d)
        mice_rew.append(mouse)

    # Load learning data for day 0
    # -----------------------------

    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days
    xarray = xarray.sel(trial=xarray['day'].isin([0]))
    # Select whisker trials
    xarray = xarray.sel(trial=xarray['whisker_stim'] == 1)

    # Select responsive cells
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue

    # Average bins
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')

    # Remove artefacts by setting them at 0. To avoid NaN values and mismatches
    print(np.isnan(xarray.values).sum(), 'NaN values in the data.')
    xarray = xarray.fillna(0)

    if rew_gp == 'R-':
        vectors_nonrew_day0_learning.append(xarray)
    elif rew_gp == 'R+':
        vectors_rew_day0_learning.append(xarray)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def progressive_learning_analysis(vectors_mapping, vectors_learning, mice_list, bh_df=None,
                                  pre_days=[-2, -1], post_days=[1, 2], window_size=10, step_size=5,
                                  align_to_learning=False, trials_before=50, trials_after=100, seed=42):
    """
    Analyze progressive learning during Day 0 using a sliding window approach.
    Train decoder on Day -2 vs +2 mapping trials, then apply to Day 0 learning trials.

    Parameters:
    -----------
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

    Returns:
    --------
    tuple: (pd.DataFrame, dict)
        - pd.DataFrame with window results including trial indices (absolute and relative to learning)
        - dict with classifier weights for each mouse: {'mouse_id': {'coef': weights, 'intercept': bias, 'scaler_mean': mean, 'scaler_scale': scale}}
    """
    results = []
    classifier_weights = {}

    for i, (d_mapping, d_learning, mouse) in enumerate(zip(vectors_mapping, vectors_learning, mice_list)):
        print(d_mapping.shape, d_learning.shape)
        day_per_trial = d_mapping['day'].values

        # Get Day -2/-1 and +1/+2 trials for training from mapping data
        train_mask = np.isin(day_per_trial, pre_days + post_days)
        if np.sum(train_mask) < 4:
            print(f"Not enough training trials for {mouse}, skipping.")
            continue

        X_train = d_mapping.values[:, train_mask].T
        y_train = np.array([0 if day in pre_days else 1 for day in day_per_trial[train_mask]])

        # Train classifier
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        clf.fit(X_train_scaled, y_train)

        # Store classifier weights for this mouse (cell-by-cell contribution)
        # Extract ROI coordinates from the xarray
        if 'roi' in d_mapping.coords:
            roi_coords = d_mapping.coords['roi'].values
        else:
            # Fallback to indices if roi coord not found
            roi_coords = np.arange(clf.coef_.shape[1])

        classifier_weights[mouse] = {
            'coef': clf.coef_[0],  # Weights for each cell/neuron
            'roi': roi_coords,  # ROI identifiers for each cell
            'intercept': clf.intercept_[0],  # Classifier bias term
            'scaler_mean': scaler.mean_,  # Mean values used for scaling
            'scaler_scale': scaler.scale_,  # Scale values used for scaling
            'n_features': clf.coef_.shape[1],  # Number of cells/features
            'sign_flip': None  # Will be set after sign check
        }

        # Sanity check: Verify sign convention is correct (decision_function sign)
        pre_mask = np.isin(day_per_trial, pre_days)
        post_mask = np.isin(day_per_trial, post_days)
        if np.sum(pre_mask) > 0 and np.sum(post_mask) > 0:
            X_pre = scaler.transform(d_mapping.values[:, pre_mask].T)
            X_post = scaler.transform(d_mapping.values[:, post_mask].T)
            mean_dec_pre = np.mean(clf.decision_function(X_pre))
            mean_dec_post = np.mean(clf.decision_function(X_post))
        else:
            mean_dec_pre, mean_dec_post = 0.0, 0.0

        if mean_dec_pre > mean_dec_post:
            print(f"WARNING: {mouse} has flipped decision values! Pre={mean_dec_pre:.3f}, Post={mean_dec_post:.3f}")
            print(f"  Flipping sign of decision values for plotting consistency.")
            sign_flip = -1
        else:
            sign_flip = 1
            print(f"{mouse}: Decision values oriented. Pre={mean_dec_pre:.3f}, Post={mean_dec_post:.3f}")

        # Update sign_flip in classifier weights
        classifier_weights[mouse]['sign_flip'] = sign_flip

        # Get learning trial for this mouse if aligning
        learning_trial_idx = None
        if align_to_learning and bh_df is not None:
            mouse_bh = bh_df[bh_df['mouse_id'] == mouse]
            if not mouse_bh.empty and 'learning_trial' in mouse_bh.columns:
                # Get the learning_trial value (should be same for all rows of this mouse)
                learning_trial_val = mouse_bh['learning_trial'].iloc[0]
                if not np.isnan(learning_trial_val):
                    learning_trial_idx = int(learning_trial_val)
                    print(f"{mouse}: Learning onset at trial_w = {learning_trial_idx}")
                else:
                    print(f"{mouse}: No learning trial defined, using absolute indexing")
            else:
                print(f"{mouse}: No behavioral data found, using absolute indexing")

        # Get Day 0 learning trials
        n_learning_trials = d_learning.sizes['trial']

        # Determine trial range to analyze
        if align_to_learning and learning_trial_idx is not None:
            # Align to learning onset: trial 0 = learning_trial
            trial_start = max(0, learning_trial_idx - trials_before)
            trial_end = min(n_learning_trials, learning_trial_idx + trials_after)
            trial_offset = learning_trial_idx  # For converting to relative indices
        else:
            # Use absolute trial indices
            trial_start = 0
            trial_end = n_learning_trials
            trial_offset = 0

        # Sliding window analysis on learning trials
        window_results = []
        for start_idx in range(trial_start, max(trial_start, trial_end - window_size + 1), step_size):
            end_idx = start_idx + window_size
            if end_idx > trial_end:
                break

            # Get window data from learning trials
            X_window = d_learning.values[:, start_idx:end_idx].T
            if X_window.shape[0] == 0:
                continue
            X_window_scaled = scaler.transform(X_window)

            # Decision values (distance to hyperplane)
            decision_values = clf.decision_function(X_window_scaled)
            mean_decision_value = np.mean(decision_values) * sign_flip

            # Probability of being "post"
            if hasattr(clf, "predict_proba"):
                # Use predicted labels to get the proportion of trials classified as "post"
                # (we trained with labels 0=pre, 1=post), not the classifier's confidence.
                preds = clf.predict(X_window_scaled)
                mean_proba_post = np.mean(preds == 1)
            else:
                # If classifier has no predict_proba (e.g., some SVMs), approximate with sigmoid on decision
                dv = decision_values * sign_flip
                mean_proba_post = np.mean(1 / (1 + np.exp(-dv)))

            # Store both absolute and aligned trial indices
            trial_center_abs = start_idx + window_size // 2
            trial_center_aligned = trial_center_abs - trial_offset  # Relative to learning onset

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
                'mouse_id': mouse
            })

        results.extend(window_results)

    return pd.DataFrame(results), classifier_weights


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_behavior(ax, data, color, title, cut_n_trials=cut_n_trials, align_to_learning=False):
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0, cut_n_trials//2 if align_to_learning else cut_n_trials)
        ax.set_ylim(0, 1)
        return

    if align_to_learning and 'learning_trial' in data.columns:
        # Create aligned trial index
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

def plot_decision(ax, data, color, title, cut_n_trials=cut_n_trials, align_to_learning=False, ylim=None):
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0, cut_n_trials//2 if align_to_learning else cut_n_trials)
        return

    if align_to_learning and 'trial_center_aligned' in data.columns:
        x_col = 'trial_center_aligned'
        xlabel = 'Trial relative to learning onset'
    else:
        x_col = 'trial_center'
        xlabel = 'Trial within Day 0'

    sns.lineplot(data=data, x=x_col, y='mean_decision_value', estimator=np.mean, errorbar='ci', color=color, ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    if align_to_learning:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Learning onset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean decision value')
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if cut_n_trials is not None:
        if align_to_learning:
            ax.set_xlim(-cut_n_trials//2, cut_n_trials//2)
        else:
            ax.set_xlim(0, cut_n_trials)

def plot_proba(ax, data, color, title, cut_n_trials=cut_n_trials, align_to_learning=False):
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0, cut_n_trials//2 if align_to_learning else cut_n_trials)
        ax.set_ylim(0, 1)
        return

    if align_to_learning and 'trial_center_aligned' in data.columns:
        x_col = 'trial_center_aligned'
        xlabel = 'Trial relative to learning onset'
    else:
        x_col = 'trial_center'
        xlabel = 'Trial within Day 0'

    sns.lineplot(data=data, x=x_col, y='mean_proba_post', estimator=np.mean, errorbar='ci', color=color, ax=ax)
    ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--')
    if align_to_learning:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Learning onset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('P(post)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    if cut_n_trials is not None:
        if align_to_learning:
            ax.set_xlim(-cut_n_trials//2, cut_n_trials//2)
        else:
            ax.set_xlim(0, cut_n_trials)


# ============================================================================
# RUN PROGRESSIVE LEARNING ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("RUNNING PROGRESSIVE LEARNING ANALYSIS")
print("="*80 + "\n")

# If mice_groups exist, try to plot good/bad subsets, otherwise plot empty panels
try:
    mice_good = [m for m in mice_rew if m in mice_groups.get('good_day0', [])]
    mice_bad = [m for m in mice_rew if m in (mice_groups.get('bad_day0', []) + mice_groups.get('meh_day0', []))]
except Exception:
    mice_good, mice_bad = [], []

# Build results for R+ and R- (and optionally good/bad subsets)
print("\nAnalyzing R+ mice...")
results_rew, weights_rew = progressive_learning_analysis(
    vectors_rew_mapping, vectors_rew_day0_learning, mice_rew,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)

print("\nAnalyzing R- mice...")
results_nonrew, weights_nonrew = progressive_learning_analysis(
    vectors_nonrew_mapping, vectors_nonrew_day0_learning, mice_nonrew,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)

print("\nAnalyzing good learners...")
results_good, weights_good = progressive_learning_analysis(
    [vectors_rew_mapping[i] for i, m in enumerate(mice_rew) if m in mice_good],
    [vectors_rew_day0_learning[i] for i, m in enumerate(mice_rew) if m in mice_good],
    mice_good,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)

print("\nAnalyzing bad learners...")
results_bad, weights_bad = progressive_learning_analysis(
    [vectors_rew_mapping[i] for i, m in enumerate(mice_rew) if m in mice_bad],
    [vectors_rew_day0_learning[i] for i, m in enumerate(mice_rew) if m in mice_bad],
    mice_bad,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)

results_rew['reward_group'] = 'R+'
results_nonrew['reward_group'] = 'R-'
results_combined = pd.concat([results_rew, results_nonrew], ignore_index=True)


# ============================================================================
# SAVE CLASSIFIER WEIGHTS
# ============================================================================

print("\n" + "="*80)
print("SAVING CLASSIFIER WEIGHTS")
print("="*80 + "\n")

output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Combine all weights dictionaries
all_weights_dict = {
    'R+': weights_rew,
    'R-': weights_nonrew,
    'good_learners': weights_good,
    'bad_learners': weights_bad
}

# Convert to DataFrame format
weights_rows = []
for group_name, weights_dict in all_weights_dict.items():
    for mouse_id, weight_info in weights_dict.items():
        # Get the weights and ROI IDs for each cell
        coefs = weight_info['coef']
        rois = weight_info['roi']
        sign_flip = weight_info['sign_flip']

        # Create a row for each cell
        for roi, weight in zip(rois, coefs):
            weights_rows.append({
                'mouse_id': mouse_id,
                'roi': roi,
                'classifier_weight': weight * sign_flip,  # Apply sign flip for consistency
                'classifier_weight_raw': weight,  # Keep raw weight without sign flip
                'reward_group': 'R+' if group_name in ['R+', 'good_learners'] else 'R-',
                'learner_group': group_name,
                'sign_flip': sign_flip
            })

df_weights = pd.DataFrame(weights_rows)

# Save as CSV
weights_file_csv = os.path.join(output_dir, 'classifier_weights.csv')
df_weights.to_csv(weights_file_csv, index=False)
print(f"Classifier weights saved to: {weights_file_csv}")
print(f"  Format: DataFrame with {len(df_weights)} rows (cells)")
print(f"  Columns: {', '.join(df_weights.columns)}")

# Also save the full weights dictionaries with all metadata as pickle for backward compatibility
weights_file_pkl = os.path.join(output_dir, 'classifier_weights_full.pkl')
with open(weights_file_pkl, 'wb') as f:
    pickle.dump(all_weights_dict, f)
print(f"Full weight metadata saved to: {weights_file_pkl}")


# ============================================================================
# MAIN VISUALIZATION: BEHAVIOR, DECISION VALUES, AND PROBABILITIES
# ============================================================================

print("\n" + "="*80)
print("CREATING MAIN VISUALIZATION")
print("="*80 + "\n")

# Create a 3-row x 4-col figure:
plt.figure(figsize=(16, 12))

# Top row: Behavioral learning curves (4 panels)
ax1 = plt.subplot(3, 4, 1)
data_rew = bh_df.loc[bh_df['mouse_id'].isin(mice_rew)]
plot_behavior(ax1, data_rew, reward_palette[1], 'R+ mice behavior', align_to_learning=align_to_learning)

ax2 = plt.subplot(3, 4, 2)
data_nonrew = bh_df.loc[bh_df['mouse_id'].isin(mice_nonrew)]
plot_behavior(ax2, data_nonrew, reward_palette[0], 'R- mice behavior', align_to_learning=align_to_learning)

ax3 = plt.subplot(3, 4, 3)
data_good = bh_df.loc[bh_df['mouse_id'].isin(mice_good)]
plot_behavior(ax3, data_good, reward_palette[1], 'Good day0 mice behavior', align_to_learning=align_to_learning)

ax4 = plt.subplot(3, 4, 4)
data_bad = bh_df.loc[bh_df['mouse_id'].isin(mice_bad)]
plot_behavior(ax4, data_bad, reward_palette[1], 'Bad day0 mice behavior', align_to_learning=align_to_learning)

# Middle row: Decision values (4 panels)
ax5 = plt.subplot(3, 4, 5)
plot_decision(ax5, results_rew, reward_palette[1], 'R+ mean decision value', align_to_learning=align_to_learning, ylim=(-3, 4))

ax6 = plt.subplot(3, 4, 6)
plot_decision(ax6, results_nonrew, reward_palette[0], 'R- mean decision value', align_to_learning=align_to_learning, ylim=(-3, 4))

ax7 = plt.subplot(3, 4, 7)
plot_decision(ax7, results_good, reward_palette[1], 'Good day0 mean decision value', align_to_learning=align_to_learning, ylim=(-3, 4))

ax8 = plt.subplot(3, 4, 8)
plot_decision(ax8, results_bad, reward_palette[1], 'Bad day0 mean decision value', align_to_learning=align_to_learning, ylim=(-3, 6))

# Bottom row: Probability P(post) (4 panels)
ax9 = plt.subplot(3, 4, 9)
plot_proba(ax9, results_rew, reward_palette[1], 'R+ P(post)', align_to_learning=align_to_learning)

ax10 = plt.subplot(3, 4, 10)
plot_proba(ax10, results_nonrew, reward_palette[0], 'R- P(post)', align_to_learning=align_to_learning)

ax11 = plt.subplot(3, 4, 11)
plot_proba(ax11, results_good, reward_palette[1], 'Good day0 P(post)', align_to_learning=align_to_learning)

ax12 = plt.subplot(3, 4, 12)
plot_proba(ax12, results_bad, reward_palette[1], 'Bad day0 P(post)', align_to_learning=align_to_learning)

plt.tight_layout()
sns.despine()

# Save figure
output_dir_gradual = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/gradual_learning'
output_dir_gradual = io.adjust_path_to_host(output_dir_gradual)
os.makedirs(output_dir_gradual, exist_ok=True)
plt.savefig(os.path.join(output_dir_gradual, 'decoder_decision_value_day0_learning_with_alignment_to_learning.svg'), format='svg', dpi=300)
print(f"Saved: decoder_decision_value_day0_learning_with_alignment_to_learning.svg")


# ============================================================================
# INDIVIDUAL MOUSE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SINGLE MOUSE PROGRESSIVE LEARNING ANALYSIS")
print("="*80 + "\n")

# Create PDF with individual mouse plots
pdf_file = os.path.join(output_dir_gradual, 'individual_mouse_progressive_learning.pdf')

with PdfPages(pdf_file) as pdf:
    # Plot each mouse separately
    all_mice = results_combined['mouse_id'].unique()

    for mouse in all_mice:
        mouse_data = results_combined[results_combined['mouse_id'] == mouse]
        mouse_bh = bh_df[bh_df['mouse_id'] == mouse]
        reward_group = mouse_data['reward_group'].iloc[0]
        color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]

        fig = plt.figure(figsize=(12, 8))

        # Top panel: Behavior
        ax1 = plt.subplot(3, 1, 1)
        plot_behavior(ax1, mouse_bh, color, f'{mouse} ({reward_group}) - Behavior',
                     cut_n_trials=cut_n_trials, align_to_learning=align_to_learning)

        # Middle panel: Decision values
        ax2 = plt.subplot(3, 1, 2)
        plot_decision(ax2, mouse_data, color, f'{mouse} ({reward_group}) - Decision Value',
                     cut_n_trials=cut_n_trials, align_to_learning=align_to_learning)

        # Bottom panel: Probability P(post)
        ax3 = plt.subplot(3, 1, 3)
        plot_proba(ax3, mouse_data, color, f'{mouse} ({reward_group}) - P(post)',
                  cut_n_trials=cut_n_trials, align_to_learning=align_to_learning)

        plt.tight_layout()
        sns.despine()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"Individual mouse plots saved to: {pdf_file}")


# ============================================================================
# STATISTICAL QUANTIFICATION: SLOPE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STATISTICAL TESTS FOR PROGRESSIVE LEARNING EFFECT")
print("="*80 + "\n")

print("METHOD: Linear Trend Analysis")
print("Test if decision values show significant positive slope over time\n")

slopes_per_mouse = []
slopes_pvals = []
slopes_mice = []
slopes_groups = []

for mouse in results_combined['mouse_id'].unique():
    mouse_data = results_combined[results_combined['mouse_id'] == mouse]
    reward_group = mouse_data['reward_group'].iloc[0]

    # Get trial indices and decision values
    if align_to_learning and 'trial_center_aligned' in mouse_data.columns:
        x = mouse_data['trial_center_aligned'].values
        mask = ~np.isnan(x)
        x = x[mask]
        y = mouse_data['mean_decision_value'].values[mask]
    else:
        x = mouse_data['trial_center'].values
        y = mouse_data['mean_decision_value'].values

    if len(x) < 5:
        continue

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    slopes_per_mouse.append(slope)
    slopes_pvals.append(p_value)
    slopes_mice.append(mouse)
    slopes_groups.append(reward_group)

    print(f"{mouse} ({reward_group}): slope={slope:.4f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

# Test if slopes are significantly positive at population level
df_slopes = pd.DataFrame({
    'mouse_id': slopes_mice,
    'reward_group': slopes_groups,
    'slope': slopes_per_mouse,
    'p_value': slopes_pvals
})

print("\nPopulation-level statistics:")
for group in ['R+', 'R-']:
    sub = df_slopes[df_slopes['reward_group'] == group]
    if len(sub) >= 3:
        # One-sample test against 0
        stat_w, p_wilcox = wilcoxon(sub['slope'].values, alternative='greater')
        stat_t, p_ttest = ttest_1samp(sub['slope'].values, 0, alternative='greater')

        # Count significant mice
        n_sig = np.sum(sub['p_value'] < 0.05)
        n_total = len(sub)

        print(f"\n{group} Group (N={n_total}):")
        print(f"  Mean slope: {np.mean(sub['slope'].values):.4f} ± {np.std(sub['slope'].values):.4f}")
        print(f"  Wilcoxon test (H0: median slope ≤ 0): p={p_wilcox:.4f}")
        print(f"  t-test (H0: mean slope ≤ 0): p={p_ttest:.4f}")
        print(f"  Mice with significant positive slope (p<0.05): {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)")


# Save individual mouse results to CSV
print("\n" + "-"*80)
print("SAVING SLOPE ANALYSIS RESULTS")
print("-"*80 + "\n")

df_slopes.to_csv(os.path.join(output_dir_gradual, 'progressive_learning_slopes.csv'), index=False)
print(f"Saved: progressive_learning_slopes.csv")

# Compute and Save Population-Level Statistics
population_stats = []

for group in ['R+', 'R-']:
    # Slopes
    sub_slope = df_slopes[df_slopes['reward_group'] == group]
    if len(sub_slope) >= 3:
        stat_w_slope, p_wilcox_slope = wilcoxon(sub_slope['slope'].values, alternative='greater')
        stat_t_slope, p_ttest_slope = ttest_1samp(sub_slope['slope'].values, 0, alternative='greater')
        n_pos_slope = np.sum(sub_slope['slope'] > 0)
        n_sig_slope = np.sum(sub_slope['p_value'] < 0.05)
        mean_slope = np.mean(sub_slope['slope'].values)
        std_slope = np.std(sub_slope['slope'].values)

        population_stats.append({
            'reward_group': group,
            'method': 'Linear Slope',
            'mean_value': mean_slope,
            'std_value': std_slope,
            'n_positive': n_pos_slope,
            'n_significant': n_sig_slope,
            'n_total': len(sub_slope),
            'p_wilcoxon': p_wilcox_slope,
            'p_ttest': p_ttest_slope
        })

df_population_stats = pd.DataFrame(population_stats)
df_population_stats.to_csv(os.path.join(output_dir_gradual, 'progressive_learning_population_statistics.csv'), index=False)
print(f"Saved: progressive_learning_population_statistics.csv")


# Add analysis for good and bad R+ learners
print("\n" + "-"*80)
print("ADDITIONAL ANALYSIS: GOOD vs BAD R+ LEARNERS")
print("-"*80 + "\n")

# Add learner_group column to df_slopes
df_slopes['learner_group'] = df_slopes['mouse_id'].apply(
    lambda x: 'Good R+' if x in mice_good else ('Bad R+' if x in mice_bad else 'Other')
)

print("Slopes for Good vs Bad R+ learners:")
for group in ['Good R+', 'Bad R+']:
    sub = df_slopes[df_slopes['learner_group'] == group]
    if len(sub) >= 3:
        stat_w, p_wilcox = wilcoxon(sub['slope'].values, alternative='greater')
        stat_t, p_ttest = ttest_1samp(sub['slope'].values, 0, alternative='greater')
        n_sig = np.sum(sub['p_value'] < 0.05)
        n_total = len(sub)

        print(f"\n{group} (N={n_total}):")
        print(f"  Mean slope: {np.mean(sub['slope'].values):.4f} ± {np.std(sub['slope'].values):.4f}")
        print(f"  Wilcoxon test (H0: median slope ≤ 0): p={p_wilcox:.4f}")
        print(f"  t-test (H0: mean slope ≤ 0): p={p_ttest:.4f}")
        print(f"  Mice with significant positive slope (p<0.05): {n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)")

        # Add to population stats
        population_stats.append({
            'reward_group': group,
            'method': 'Linear Slope',
            'mean_value': np.mean(sub['slope'].values),
            'std_value': np.std(sub['slope'].values),
            'n_positive': np.sum(sub['slope'] > 0),
            'n_significant': n_sig,
            'n_total': len(sub),
            'p_wilcoxon': p_wilcox,
            'p_ttest': p_ttest
        })

# Update and re-save population stats with good/bad learners
df_population_stats = pd.DataFrame(population_stats)
df_population_stats.to_csv(os.path.join(output_dir_gradual, 'progressive_learning_population_statistics_with_subgroups.csv'), index=False)
print(f"\nSaved: progressive_learning_population_statistics_with_subgroups.csv")


# Population-Level Statistics Visualization (including good/bad learners)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: R+ vs R-
ax = axes[0]
plot_data = []
for group in ['R+', 'R-']:
    sub = df_slopes[df_slopes['reward_group'] == group]
    for val in sub['slope'].values:
        plot_data.append({'group': group, 'value': val})

df_plot = pd.DataFrame(plot_data)

# Strip plot with individual mice
sns.swarmplot(data=df_plot, x='group', y='value', palette=reward_palette[::-1],
             ax=ax, size=8, alpha=0.6)

# Overlay mean with error bars (CI)
sns.pointplot(data=df_plot, x='group', y='value', palette=reward_palette[::-1],
             ax=ax, errorbar='ci', markersize=10, join=False)

# Add horizontal line at 0
ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)

# Add p-values as text
for i, group in enumerate(['R+', 'R-']):
    pop_stat = df_population_stats[(df_population_stats['reward_group'] == group) &
                                   (df_population_stats['method'] == 'Linear Slope')]
    if not pop_stat.empty:
        p_wilcox = pop_stat['p_wilcoxon'].values[0]
        p_text = f"p={p_wilcox:.4f}" if p_wilcox >= 0.001 else "p<0.001"

        # Add significance stars
        if p_wilcox < 0.001:
            sig_text = "***"
        elif p_wilcox < 0.01:
            sig_text = "**"
        elif p_wilcox < 0.05:
            sig_text = "*"
        else:
            sig_text = "n.s."

        # Position text above the data
        y_max = df_plot[df_plot['group'] == group]['value'].max()
        y_pos = y_max + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(i, y_pos, f"{p_text}\n{sig_text}", ha='center', va='bottom',
               fontsize=9, fontweight='bold')

ax.set_xlabel('Reward Group', fontsize=11)
ax.set_ylabel('Slope (per trial)', fontsize=11)
ax.set_title('R+ vs R-\nPopulation Test (Wilcoxon)', fontsize=12, fontweight='bold')

# Set y-lim to include text
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.15 * y_range)

# Panel 2: Good vs Bad R+ learners
ax = axes[1]
plot_data_learners = []
for group in ['Good R+', 'Bad R+']:
    sub = df_slopes[df_slopes['learner_group'] == group]
    for val in sub['slope'].values:
        plot_data_learners.append({'group': group, 'value': val})

df_plot_learners = pd.DataFrame(plot_data_learners)

if not df_plot_learners.empty:
    # Strip plot with individual mice
    sns.swarmplot(data=df_plot_learners, x='group', y='value',
                 palette=[reward_palette[1], reward_palette[1]], ax=ax, size=8, alpha=0.6)

    # Overlay mean with error bars (CI)
    sns.pointplot(data=df_plot_learners, x='group', y='value',
                 palette=[reward_palette[1], reward_palette[1]], ax=ax, errorbar='ci', markersize=10, join=False)

    # Add horizontal line at 0
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Add p-values as text
    for i, group in enumerate(['Good R+', 'Bad R+']):
        pop_stat = df_population_stats[(df_population_stats['reward_group'] == group) &
                                       (df_population_stats['method'] == 'Linear Slope')]
        if not pop_stat.empty:
            p_wilcox = pop_stat['p_wilcoxon'].values[0]
            p_text = f"p={p_wilcox:.4f}" if p_wilcox >= 0.001 else "p<0.001"

            # Add significance stars
            if p_wilcox < 0.001:
                sig_text = "***"
            elif p_wilcox < 0.01:
                sig_text = "**"
            elif p_wilcox < 0.05:
                sig_text = "*"
            else:
                sig_text = "n.s."

            # Position text above the data
            y_max = df_plot_learners[df_plot_learners['group'] == group]['value'].max()
            y_pos = y_max + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(i, y_pos, f"{p_text}\n{sig_text}", ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    ax.set_xlabel('Learner Group', fontsize=11)
    ax.set_ylabel('Slope (per trial)', fontsize=11)
    ax.set_title('Good vs Bad R+ Learners\nPopulation Test (Wilcoxon)', fontsize=12, fontweight='bold')

    # Set y-lim to include text
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.15 * y_range)
else:
    ax.text(0.5, 0.5, 'No data for good/bad learners', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Good vs Bad R+ Learners', fontsize=12, fontweight='bold')

plt.tight_layout()
sns.despine()

plt.savefig(os.path.join(output_dir_gradual, 'progressive_learning_population_statistics.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir_gradual, 'progressive_learning_population_statistics.png'), format='png', dpi=300)
print(f"Saved: progressive_learning_population_statistics figure")


# Summary Visualization of Individual Mice
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Slopes per mouse
ax = axes[0]
for group, color in zip(['R+', 'R-'], reward_palette[::-1]):
    sub = df_slopes[df_slopes['reward_group'] == group]
    x = np.arange(len(sub))
    ax.bar(x + (0.4 if group == 'R+' else 0), sub['slope'].values,
           width=0.4, color=color, alpha=0.7, label=group)
    # Mark significant ones
    sig_mask = sub['p_value'].values < 0.05
    ax.scatter(x[sig_mask] + (0.4 if group == 'R+' else 0),
              sub['slope'].values[sig_mask],
              marker='*', s=200, color='black', zorder=10)
ax.axhline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Mouse')
ax.set_ylabel('Slope (decision value per trial)')
ax.set_title('Linear Trend per Mouse\n(* = p<0.05)')
ax.legend()
ax.set_xticks([])

# Panel 2: Summary statistics
ax = axes[1]
ax.axis('off')

summary_text = "SUMMARY OF PROGRESSIVE LEARNING\n"
summary_text += "="*40 + "\n\n"

for group in ['R+', 'R-']:
    summary_text += f"{group} Group:\n"
    summary_text += "-"*40 + "\n"

    # Slope analysis
    sub_slope = df_slopes[df_slopes['reward_group'] == group]
    n_pos_slope = np.sum(sub_slope['slope'] > 0)
    n_sig_slope = np.sum(sub_slope['p_value'] < 0.05)
    stat_w, p_slope = wilcoxon(sub_slope['slope'].values, alternative='greater')

    summary_text += f"Linear Trend:\n"
    summary_text += f"  {n_pos_slope}/{len(sub_slope)} positive slopes\n"
    summary_text += f"  {n_sig_slope}/{len(sub_slope)} significant (p<0.05)\n"
    summary_text += f"  Population p = {p_slope:.4f}\n\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
sns.despine()

plt.savefig(os.path.join(output_dir_gradual, 'progressive_learning_individual_mice_summary.svg'), format='svg', dpi=300)
print(f"Saved: progressive_learning_individual_mice_summary figure")


# ============================================================================
# CORRELATION ANALYSIS: DECISION VALUES VS BEHAVIORAL PERFORMANCE
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS: Decision Values vs Behavioral Performance")
print("="*80 + "\n")

mice = results_combined['mouse_id'].unique()

# Method 1: Direct Correlation
print("METHOD 1: Direct Correlation Analysis")
print("Pearson correlation between decision values and behavioral performance\n")

corr_real = []
corr_mice = []
corr_groups = []

for mouse in mice:
    group = results_combined.loc[results_combined['mouse_id'] == mouse, 'reward_group'].iloc[0]
    dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
    bh_mouse = bh_df[bh_df['mouse_id'] == mouse]

    # Align trials appropriately
    if align_to_learning:
        learning_trial = bh_mouse['learning_trial'].iloc[0] if 'learning_trial' in bh_mouse.columns else 0
        bh_mouse_aligned = bh_mouse.copy()
        bh_mouse_aligned['trial_w_aligned'] = bh_mouse_aligned['trial_w'] - learning_trial
        common_trials_aligned = np.intersect1d(
            dec_mouse['trial_center_aligned'].dropna(),
            bh_mouse_aligned['trial_w_aligned']
        )
        if len(common_trials_aligned) < 10:
            continue
        dec_vals = dec_mouse.set_index('trial_center_aligned').loc[common_trials_aligned]['mean_decision_value'].values
        perf_vals = bh_mouse_aligned.set_index('trial_w_aligned').loc[common_trials_aligned]['learning_curve_w'].values
    else:
        common_trials = np.intersect1d(dec_mouse['trial_start'], bh_mouse['trial_w'])
        if len(common_trials) < 10:
            continue
        dec_vals = dec_mouse.set_index('trial_start').loc[common_trials]['mean_decision_value'].values
        perf_vals = bh_mouse.set_index('trial_w').loc[common_trials]['learning_curve_w'].values

    # Compute Pearson correlation
    corr = pearsonr(perf_vals, dec_vals)[0]
    corr_real.append(corr)
    corr_mice.append(mouse)
    corr_groups.append(group)

    print(f"{mouse} ({group}): r={corr:.3f}")

# Save correlation data
df_corr = pd.DataFrame({
    'mouse_id': corr_mice,
    'reward_group': corr_groups,
    'correlation': corr_real
})

df_corr.to_csv(os.path.join(output_dir_gradual, 'correlation_decision_behavior_data.csv'), index=False)
print(f"\nSaved: correlation_decision_behavior_data.csv")

# Population-level statistics
print("\nPopulation-level statistics:")
for group in ['R+', 'R-']:
    sub = df_corr[df_corr['reward_group'] == group]
    if len(sub) >= 3:
        stat_w, p_wilcox = wilcoxon(sub['correlation'].values, alternative='greater')
        stat_t, p_ttest = ttest_1samp(sub['correlation'].values, 0, alternative='greater')

        print(f"\n{group} Group (N={len(sub)}):")
        print(f"  Mean correlation: {np.mean(sub['correlation'].values):.3f} ± {np.std(sub['correlation'].values):.3f}")
        print(f"  Wilcoxon test (H0: median ≤ 0): p={p_wilcox:.4f}")
        print(f"  t-test (H0: mean ≤ 0): p={p_ttest:.4f}")

# Save population statistics
pop_stats_corr = []
for group in ['R+', 'R-']:
    sub = df_corr[df_corr['reward_group'] == group]
    if len(sub) >= 3:
        stat_w, p_wilcox = wilcoxon(sub['correlation'].values, alternative='greater')
        stat_t, p_ttest = ttest_1samp(sub['correlation'].values, 0, alternative='greater')

        pop_stats_corr.append({
            'reward_group': group,
            'method': 'Direct Correlation',
            'mean_value': np.mean(sub['correlation'].values),
            'std_value': np.std(sub['correlation'].values),
            'n_total': len(sub),
            'p_wilcoxon': p_wilcox,
            'p_ttest': p_ttest
        })

df_pop_stats_corr = pd.DataFrame(pop_stats_corr)


# Add analysis for good and bad R+ learners
print("\n" + "-"*80)
print("ADDITIONAL ANALYSIS: GOOD vs BAD R+ LEARNERS (CORRELATION)")
print("-"*80 + "\n")

# Add learner_group column to df_corr
df_corr['learner_group'] = df_corr['mouse_id'].apply(
    lambda x: 'Good R+' if x in mice_good else ('Bad R+' if x in mice_bad else 'Other')
)

print("Correlations for Good vs Bad R+ learners:")
for group in ['Good R+', 'Bad R+']:
    sub = df_corr[df_corr['learner_group'] == group]
    if len(sub) >= 3:
        stat_w, p_wilcox = wilcoxon(sub['correlation'].values, alternative='greater')
        stat_t, p_ttest = ttest_1samp(sub['correlation'].values, 0, alternative='greater')

        print(f"\n{group} (N={len(sub)}):")
        print(f"  Mean correlation: {np.mean(sub['correlation'].values):.3f} ± {np.std(sub['correlation'].values):.3f}")
        print(f"  Wilcoxon test (H0: median ≤ 0): p={p_wilcox:.4f}")
        print(f"  t-test (H0: mean ≤ 0): p={p_ttest:.4f}")

        # Add to population stats
        pop_stats_corr.append({
            'reward_group': group,
            'method': 'Direct Correlation',
            'mean_value': np.mean(sub['correlation'].values),
            'std_value': np.std(sub['correlation'].values),
            'n_total': len(sub),
            'p_wilcoxon': p_wilcox,
            'p_ttest': p_ttest
        })

# Update and re-save population stats with good/bad learners
df_pop_stats_corr = pd.DataFrame(pop_stats_corr)
df_pop_stats_corr.to_csv(os.path.join(output_dir_gradual, 'correlation_population_statistics_with_subgroups.csv'), index=False)
print(f"\nSaved: correlation_population_statistics_with_subgroups.csv")


# Visualization (two panels: R+/R- and Good/Bad)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: R+ vs R-
ax = axes[0]
sns.swarmplot(data=df_corr, x='reward_group', y='correlation', palette=reward_palette[::-1], size=8, alpha=0.6, ax=ax)
sns.pointplot(data=df_corr, x='reward_group', y='correlation', palette=reward_palette[::-1], errorbar='ci', ax=ax, markersize=10, join=False)
ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylabel('Pearson Correlation\n(Decision value vs Performance)', fontsize=11)
ax.set_xlabel('Reward Group', fontsize=11)
ax.set_title('R+ vs R-', fontsize=12, fontweight='bold')
ax.set_ylim(-1, 1)

# Add p-value text for each group
for i, group in enumerate(['R+', 'R-']):
    pop_stat = df_pop_stats_corr[(df_pop_stats_corr['reward_group'] == group) &
                                 (df_pop_stats_corr['method'] == 'Direct Correlation')]
    if not pop_stat.empty:
        p_wilcox = pop_stat['p_wilcoxon'].values[0]
        p_text = f"p={p_wilcox:.4f}" if p_wilcox >= 0.001 else "p<0.001"
        sig_text = "***" if p_wilcox < 0.001 else "**" if p_wilcox < 0.01 else "*" if p_wilcox < 0.05 else "n.s."
        y_pos = ax.get_ylim()[1] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(i, y_pos, f"{p_text}\n{sig_text}", ha='center', va='top', fontsize=10, fontweight='bold')

# Panel 2: Good vs Bad R+ learners
ax = axes[1]
df_corr_learners = df_corr[df_corr['learner_group'].isin(['Good R+', 'Bad R+'])]

if not df_corr_learners.empty:
    sns.swarmplot(data=df_corr_learners, x='learner_group', y='correlation',
                 palette=[reward_palette[1], reward_palette[1]], size=8, alpha=0.6, ax=ax)
    sns.pointplot(data=df_corr_learners, x='learner_group', y='correlation',
                 palette=[reward_palette[1], reward_palette[1]], errorbar='ci', ax=ax, markersize=10, join=False)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Pearson Correlation\n(Decision value vs Performance)', fontsize=11)
    ax.set_xlabel('Learner Group', fontsize=11)
    ax.set_title('Good vs Bad R+ Learners', fontsize=12, fontweight='bold')
    ax.set_ylim(-1, 1)

    # Add p-value text for each group
    for i, group in enumerate(['Good R+', 'Bad R+']):
        pop_stat = df_pop_stats_corr[(df_pop_stats_corr['reward_group'] == group) &
                                     (df_pop_stats_corr['method'] == 'Direct Correlation')]
        if not pop_stat.empty:
            p_wilcox = pop_stat['p_wilcoxon'].values[0]
            p_text = f"p={p_wilcox:.4f}" if p_wilcox >= 0.001 else "p<0.001"
            sig_text = "***" if p_wilcox < 0.001 else "**" if p_wilcox < 0.01 else "*" if p_wilcox < 0.05 else "n.s."
            y_pos = ax.get_ylim()[1] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(i, y_pos, f"{p_text}\n{sig_text}", ha='center', va='top', fontsize=10, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No data for good/bad learners', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Good vs Bad R+ Learners', fontsize=12, fontweight='bold')

plt.tight_layout()
sns.despine()

plt.savefig(os.path.join(output_dir_gradual, 'correlation_decision_behavior.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir_gradual, 'correlation_decision_behavior.png'), format='png', dpi=300)
print(f"\nSaved: correlation_decision_behavior figure")

print("\n" + "="*80)
print("CORRELATION ANALYSIS COMPLETE")
print("="*80 + "\n")


# ============================================================================
# ILLUSTRATION: EXAMPLE MICE
# ============================================================================

print("\n" + "="*80)
print("CREATING ILLUSTRATION OF RESULTS FOR EXAMPLE MICE")
print("="*80 + "\n")

# Find the R+ and R- mouse with highest correlation between behavior and decision value
best_rplus = df_corr[df_corr['reward_group'] == 'R+'].sort_values('correlation', ascending=False).iloc[0]['mouse_id']
best_rminus = df_corr[df_corr['reward_group'] == 'R-'].sort_values('correlation', ascending=False).iloc[10]['mouse_id']
example_mice = [best_rplus, best_rminus]

# Publication-quality figure: one column per mouse
fig, axes = plt.subplots(2, len(example_mice), figsize=(7, 7), sharex=False)

for col, mouse in enumerate(example_mice):
    # Panel 1: Behavioral performance (whisker trials only)
    bh_mouse = bh_df[(bh_df['mouse_id'] == mouse) & (bh_df['whisker_stim'] == 1)]
    color = reward_palette[1] if results_combined[results_combined['mouse_id'] == mouse]['reward_group'].iloc[0] == 'R+' else reward_palette[0]
    ax_beh = axes[0, col]
    # Cut to first 100 trials for reward mouse
    if results_combined[results_combined['mouse_id'] == mouse]['reward_group'].iloc[0] == 'R+':
        bh_mouse = bh_mouse[bh_mouse['trial_w'] < 100]
    if not bh_mouse.empty:
        sns.lineplot(data=bh_mouse, x='trial_w', y='learning_curve_w', ax=ax_beh, color=color, linewidth=2.5)
        ax_beh.set_ylabel('Performance (whisker trials)', fontsize=13)
        ax_beh.set_ylim(0, 1)
        ax_beh.set_xlim(0, 100)
        ax_beh.tick_params(axis='both', labelsize=12)
    else:
        ax_beh.set_title(f"{mouse}: No behavioral data", fontsize=14, fontweight='bold')

    # Panel 2: Decision values (same trials)
    dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
    ax_dec = axes[1, col]
    if not dec_mouse.empty and not bh_mouse.empty:
        common_trials = np.intersect1d(dec_mouse['trial_start'], bh_mouse['trial_w'])
        dec_plot = dec_mouse.set_index('trial_start').loc[common_trials]
        sns.lineplot(x=common_trials, y=dec_plot['mean_decision_value'], ax=ax_dec, color=color, linewidth=2.5)
        ax_dec.set_ylabel('Decoder Decision Value', fontsize=13)
        ax_dec.tick_params(axis='both', labelsize=12)
        ax_dec.set_xlim([0, 100])
        xticks = np.arange(0, 101, 20)
        ax_dec.set_xticks(xticks)
        ax_dec.set_ylim([-5, 3])
        yticks = np.arange(-5, 4, 1)
        ax_dec.set_yticks(yticks)
        ax_dec.set_yticklabels([str(y) for y in yticks], fontsize=12)
        ax_dec.axhline(0, color='gray', linestyle='--', linewidth=1)
    else:
        ax_dec.set_title(f"{mouse}: No decoder data", fontsize=14, fontweight='bold')
    ax_dec.set_xlabel('Trial within Day 0', fontsize=13)

# Shared x-label for bottom row
for ax in axes[1, :]:
    ax.set_xlabel('Trial within Day 0', fontsize=13)

plt.tight_layout(h_pad=2.5)
sns.despine()

# Save to SVG
plt.savefig(os.path.join(output_dir_gradual, 'example_mice_behavior_decision_value.svg'), format='svg', dpi=300)
print(f"Saved: example_mice_behavior_decision_value.svg")


print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")
