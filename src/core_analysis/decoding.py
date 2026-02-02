import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from scipy.stats import bootstrap
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier


# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate



# ##################################################
# Decoding before and after learning.
# ##################################################


sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 180 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = False
projection_type = None  # 'wS2', 'wM1' or None

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data.
vectors_rew = []
vectors_nonrew = []
mice_rew = []
mice_nonrew = []

# Load responsive cells.
# Responsiveness df.
# test_df = os.path.join(io.processed_dir, f'response_test_results_alldaystogether_win_180ms.csv')
# test_df = pd.read_csv(test_df)
# test_df = test_df.loc[test_df['mouse_id'].isin(mice)]
# selected_cells = test_df.loc[test_df['pval_mapping'] <= 0.05]

if select_responsive_cells:
    test_df = os.path.join(io.processed_dir, f'response_test_results_win_180ms.csv')
    test_df = pd.read_csv(test_df)
    test_df = test_df.loc[test_df['day'].isin(days)]
    # Select cells as responsive if they pass the test on at least one day.
    selected_cells = test_df.groupby(['mouse_id', 'roi', 'cell_type'])['pval_mapping'].min().reset_index()
    selected_cells = selected_cells.loc[selected_cells['pval_mapping'] <= 0.05/5]

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]


for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin(days))
    
    # Select responsive cells.
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
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day.
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    # Average bins.
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')

    # Remove artefacts by setting them at 0. To avoid NaN values and
    # mismatches (that concerns a single cell).
    print(np.isnan(d.values).sum(), 'NaN values in the data.')
    d = d.fillna(0)
    
    if rew_gp == 'R-':
        vectors_nonrew.append(d)
        mice_nonrew.append(mouse)
    elif rew_gp == 'R+':
        vectors_rew.append(d)
        mice_rew.append(mouse)


# Decoding accuracy between reward groups.
# ----------------------------------------
# Train a single classifier per mouse and plot average cross-validated accuracy.

def per_mouse_cv_accuracy(vectors, label_encoder, seed=42, n_shuffles=100, return_weights=False, debug=False, n_jobs=20):
    accuracies = []
    chance_accuracies = []
    weights_per_mouse = []
    rng = np.random.default_rng(seed)

    for d in vectors:
        days_per_trial = d['day'].values
        mask = np.isin(days_per_trial, [-2, -1, 1, 2])
        trials = d.values[:, mask].T
        labels = np.array(['pre' if day in [-2, -1] else 'post' for day in days_per_trial[mask]])
        X = trials
        y = labels

        y_enc = label_encoder.transform(y)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        fold_scores = []
        all_true = []
        all_pred = []
        for train_idx, test_idx in cv.split(X, y_enc):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]
            # Scaling
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_test_proc = scaler.transform(X_test)
            clf.fit(X_train_proc, y_train)
            y_pred = clf.predict(X_test_proc)
            fold_scores.append(np.mean(y_pred == y_test))
            all_true.extend(y_test)
            all_pred.extend(y_pred)
        acc = np.mean(fold_scores)
        accuracies.append(acc)

        if debug:
            print("Accuracy:", acc)
            print("True labels:", all_true)
            print("Predicted labels:", all_pred)
            print("Label counts (true):", np.bincount(all_true))
            print("Label counts (pred):", np.bincount(all_pred))

        # Estimate chance level by shuffling labels n_shuffles times (parallelized)
        def shuffle_score(clf, X, y_enc, cv):
            y_shuff = rng.permutation(y_enc)
            fold_scores = []
            for train_idx, test_idx in cv.split(X, y_shuff):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_shuff[train_idx], y_shuff[test_idx]
                # Scaling
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(X_train)
                X_test_proc = scaler.transform(X_test)
                clf.fit(X_train_proc, y_train)
                y_pred = clf.predict(X_test_proc)
                fold_scores.append(np.mean(y_pred == y_test))
            return np.mean(fold_scores)

        shuffle_scores = Parallel(n_jobs=n_jobs)(
            delayed(shuffle_score)(clf, X, y_enc, cv) for _ in range(n_shuffles)
        )
        chance_accuracies.append(np.mean(shuffle_scores))

        if return_weights:
            # Train classifier on full dataset and return weights
            scaler_full = StandardScaler()
            X_full = scaler_full.fit_transform(X)
            clf_full = LogisticRegression(max_iter=5000, random_state=seed)
            clf_full.fit(X_full, y_enc)
            weights_per_mouse.append(clf_full.coef_.flatten())

    if return_weights:
        return np.array(accuracies), np.array(chance_accuracies), weights_per_mouse
    else:
        return np.array(accuracies), np.array(chance_accuracies)


le = LabelEncoder()
le.fit(['pre', 'post'])

accs_rew, chance_rew, weights_rew = per_mouse_cv_accuracy(vectors_rew, le, n_shuffles=10, return_weights=True)
accs_nonrew, chance_nonrew, weights_nonrew = per_mouse_cv_accuracy(vectors_nonrew, le, n_shuffles=10, return_weights=True)

# for w in weights_rew:
#     plt.plot(w, label='R+')

print(f"Mean accuracy R+: {np.nanmean(accs_rew):.3f} +/- {np.nanstd(accs_rew):.3f}")
print(f"Mean accuracy R-: {np.nanmean(accs_nonrew):.3f} +/- {np.nanstd(accs_nonrew):.3f}")

# Plot
plt.figure(figsize=(4, 5))
# Plot actual accuracies
sns.barplot(data=[accs_rew, accs_nonrew], palette=reward_palette[::-1], linestyle=None, estimator=np.nanmean, errorbar='ci')
sns.swarmplot(data=[accs_rew, accs_nonrew], palette=reward_palette[::-1], alpha=0.7)
plt.xticks([0, 1], ['R+', 'R-'])
plt.ylabel('Cross-validated accuracy')
plt.title('Pre vs post learning classification accuracy across mice')
plt.ylim(0, 1)
sns.despine()

# Statistical test: Mann-Whitney U test between R+ and R- accuracies
stat, p_value = mannwhitneyu(accs_rew, accs_nonrew, alternative='two-sided')
print(f"Mann-Whitney U test: stat={stat:.3f}, p-value={p_value:.4f}")

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'decoding_accuracy.svg'
if projection_type is not None:
    svg_file = f'decoding_accuracy_{projection_type}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save results.
results_df = pd.DataFrame({
    'accuracy': np.concatenate([accs_rew, accs_nonrew]),
    'reward_group': ['R+'] * len(accs_rew) + ['R-'] * len(accs_nonrew),
    'chance_accuracy': np.concatenate([chance_rew, chance_nonrew]),
})
results_csv_file = 'decoding_accuracy_data.csv'
if projection_type is not None:
    results_csv_file = f'decoding_accuracy_data_{projection_type}.csv'
results_df.to_csv(os.path.join(output_dir, results_csv_file), index=False)
# Save stats.

stat_file = 'decoding_accuracy_stats.csv'
if projection_type is not None:
    stat_file = f'decoding_accuracy_stats_{projection_type}.csv'
pd.DataFrame({'stat': [stat], 'p_value': [p_value]}).to_csv(os.path.join(output_dir, stat_file), index=False)










# Relationship between classifier weights and learning modulation index.
# ----------------------------------------------------------------------

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)

# Merge classifier weights and LMI for each mouse
weights_all = []
lmi_all = []
mouse_ids = []

for i, mouse in enumerate(mice_rew + mice_nonrew):
    # Get classifier weights for this mouse
    if mouse in mice_rew:
        w = weights_rew[mice_rew.index(mouse)]
    else:
        w = weights_nonrew[mice_nonrew.index(mouse)]
    # Get cell IDs
    d = vectors_rew[mice_rew.index(mouse)] if mouse in mice_rew else vectors_nonrew[mice_nonrew.index(mouse)]
    cell_ids = d['roi'].values if 'roi' in d.coords else d['cell'].values
    # Get LMI for this mouse
    lmi_mouse = lmi_df[(lmi_df['mouse_id'] == mouse) & (lmi_df['roi'].isin(cell_ids))]
    lmi_mouse = lmi_mouse.set_index('roi').reindex(cell_ids)
    lmi_vals = lmi_mouse['lmi'].values
    # Only keep cells with non-nan LMI and weights
    mask = ~np.isnan(lmi_vals) & ~np.isnan(w)
    weights_all.append(w[mask])
    lmi_all.append(lmi_vals[mask])
    mouse_ids.extend([mouse] * np.sum(mask))
    # Flatten lists
    weights_flat = np.concatenate(weights_all)
    lmi_flat = np.concatenate(lmi_all)
    mouse_ids_flat = np.array(mouse_ids)

# Plot scatter and regression for each mouse
plt.figure(figsize=(4, 4))

for i, mouse in enumerate(np.unique(mouse_ids_flat)):
    mask = mouse_ids_flat == mouse
    sns.scatterplot(x=lmi_flat[mask], y=weights_flat[mask], alpha=0.5)
    # # Regression line for each mouse
    # if np.sum(mask) > 1:
    #     reg = LinearRegression().fit(lmi_flat[mask].reshape(-1, 1), weights_flat[mask])
    #     x_vals = np.linspace(np.nanmin(lmi_flat[mask]), np.nanmax(lmi_flat[mask]), 100)
    #     plt.plot(x_vals, reg.predict(x_vals.reshape(-1, 1)), color='grey', alpha=0.5)

# Main regression line for all mice
reg_all = LinearRegression().fit(lmi_flat.reshape(-1, 1), weights_flat)
x_vals = np.linspace(np.nanmin(lmi_flat), np.nanmax(lmi_flat), 100)
y_pred = reg_all.predict(x_vals.reshape(-1, 1))
plt.plot(x_vals, y_pred, color='#2d2d2d', linewidth=2)

# Bootstrap confidence interval for regression line
n_boot = 1000
y_boot = np.zeros((n_boot, len(x_vals)))
for i in range(n_boot):
    Xb, yb = resample(lmi_flat, weights_flat)
    regb = LinearRegression().fit(Xb.reshape(-1, 1), yb)
    y_boot[i] = regb.predict(x_vals.reshape(-1, 1))
ci_low = np.percentile(y_boot, 2.5, axis=0)
ci_high = np.percentile(y_boot, 97.5, axis=0)
plt.fill_between(x_vals, ci_low, ci_high, color='black', alpha=0.2, label='95% CI')

plt.xlabel('Learning Modulation Index (LMI)')
plt.ylabel('Classifier Weight')
plt.title('Classifier Weight vs LMI')
plt.tight_layout()
plt.ylim(-2.5, 2)
sns.despine()

# Save plot
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_mouse.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_mouse.png'), format='png', dpi=300)


# Fit linear regression
reg = LinearRegression()
reg.fit(X, y)
r2 = reg.score(X, y)
print(f"Linear regression R^2: {r2:.3f}")

# Bootstrapped confidence interval for R^2
n_boot = 1000
r2_boot = []
for _ in range(n_boot):
    Xb, yb = resample(X, y)
    regb = LinearRegression().fit(Xb, yb)
    r2_boot.append(regb.score(Xb, yb))
r2_ci = np.percentile(r2_boot, [2.5, 97.5])
print(f"Bootstrapped R^2 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")

# Save plot
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_group.svg'), format='svg', dpi=300)


# Accuracy as a function of percent most modulated cells removed.
# ---------------------------------------------------------------

# This is to show that without the cells modulated on average, no information
# can be decoded. The non-modulated cells could still carry some information
# similarly to non-place cells in the hippocampus.


# Accuracy as a function of percent most modulated cells removed.
# ---------------------------------------------------------------

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)

le = LabelEncoder()
le.fit(['pre', 'post'])
percentiles = np.arange(0, 91, 5)  # 0% to 60% in steps of 5%
accs_rew_curve = []
accs_nonrew_curve = []

for perc in percentiles:
    accs_rew_perc = []
    accs_nonrew_perc = []
    for group, vectors, mice_group in zip(
        ['R+', 'R-'],
        [vectors_rew, vectors_nonrew],
        [mice_rew, mice_nonrew]
    ):
        for i, mouse in enumerate(mice_group):
            # Get vector and cell LMI for this mouse
            d = vectors[i]
            cell_ids = d['roi'].values if 'roi' in d.coords else d['cell'].values
            lmi_mouse = lmi_df[(lmi_df['mouse_id'] == mouse) & (lmi_df['roi'].isin(cell_ids))]
            lmi_mouse = lmi_mouse.set_index('roi').reindex(cell_ids)
            abs_lmi = np.abs(lmi_mouse['lmi'].values)
            # Sort cells by abs(LMI)
            sorted_idx = np.argsort(-abs_lmi)  # descending
            n_cells = len(cell_ids)
            n_remove = int(np.round(n_cells * perc / 100))
            keep_idx = sorted_idx[n_remove:]
            # If less than 2 cells remain, skip
            if len(keep_idx) < 2:
                continue
            # Subset vector
            d_sub = d.isel({d.dims[0]: keep_idx})
            # Classification
            days_per_trial = d_sub['day'].values
            mask = np.isin(days_per_trial, [-2, -1, 1, 2])
            trials = d_sub.values[:, mask].T
            labels = np.array(['pre' if day in [-2, -1] else 'post' for day in days_per_trial[mask]])
            y_enc = le.transform(labels)
            clf = LogisticRegression(max_iter=50000)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scaler = StandardScaler()
            X = scaler.fit_transform(trials)
            scores = cross_val_score(clf, X, y_enc, cv=cv, n_jobs=1)
            acc = np.mean(scores)
            if group == 'R+':
                accs_rew_perc.append(acc)
            else:
                accs_nonrew_perc.append(acc)
    accs_rew_curve.append(accs_rew_perc)
    accs_nonrew_curve.append(accs_nonrew_perc)
# Plot with bootstrap confidence intervals
mean_rew = []
ci_rew = []
mean_nonrew = []
ci_nonrew = []
x_vals = 100 - percentiles  # 100% to 40% retained

for a in accs_rew_curve:
    mean_rew.append(np.nanmean(a))
    if len(a) > 1:
        res = bootstrap((np.array(a),), np.nanmean, confidence_level=0.95, n_resamples=1000, method='basic')
        ci_rew.append(res.confidence_interval)
    else:
        ci_rew.append((np.nan, np.nan))
for a in accs_nonrew_curve:
    mean_nonrew.append(np.nanmean(a))
    if len(a) > 1:
        res = bootstrap((np.array(a),), np.nanmean, confidence_level=0.95, n_resamples=1000, method='basic')
        ci_nonrew.append(res.confidence_interval)
    else:
        ci_nonrew.append((np.nan, np.nan))

# Prepare DataFrame for seaborn
df_plot = pd.DataFrame({
    'percent_cells_retained': np.tile(x_vals, 2),
    'accuracy': np.concatenate([mean_rew, mean_nonrew]),
    'ci_low': np.concatenate([np.array([ci.low for ci in ci_rew]), np.array([ci.low for ci in ci_nonrew])]),
    'ci_high': np.concatenate([np.array([ci.high for ci in ci_rew]), np.array([ci.high for ci in ci_nonrew])]),
    'reward_group': ['R+'] * len(x_vals) + ['R-'] * len(x_vals)
})

plt.figure(figsize=(6, 5))
sns.lineplot(data=df_plot, x='percent_cells_retained', y='accuracy', hue='reward_group', palette=reward_palette[::-1])
for group, color in zip(['R+', 'R-'], reward_palette[::-1]):
    sub = df_plot[df_plot['reward_group'] == group]
    plt.fill_between(sub['percent_cells_retained'], sub['ci_low'], sub['ci_high'], color=color, alpha=0.3)
plt.xlabel('Percent of cells retained')
plt.ylabel('Classification accuracy')
plt.title('Accuracy vs percent modulated cells retained')
plt.legend()
plt.ylim(0, 1)
plt.xlim(100, min(x_vals))  # Flip x-axis: start at 100% and go down
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'accuracy_vs_percent_modulated_cells.svg'), format='svg', dpi=300)

# Save data
curve_df = pd.DataFrame({
    'percent_cells_retained': np.tile(100 - percentiles, 2),
    'accuracy': np.concatenate([mean_rew, mean_nonrew]),
    'reward_group': ['R+'] * len(percentiles) + ['R-'] * len(percentiles)
})
curve_df.to_csv(os.path.join(output_dir, 'accuracy_vs_percent_modulated_cells.csv'), index=False)









# Decoding accuracy between pair of days.
# ---------------------------------------

# The rationale is to use a classifier trained on pre vs post, excluding day 0,
# to assess whether activity after day 0 learning already resembles post-learning activity.

# For each mouse, train a classifier to distinguish activity patterns on day -2, -1 vs +1, +2.
# Then, use this trained classifier to decode every pair of days (including each day against itself).
# Plot accuracy as a matrix for each reward group.
 
all_days = [-2, -1, 0, 1, 2]
day_labels = [str(d) for d in all_days]

def pairwise_day_decoding_fixed_decoder_cv(vectors, days, seed=42, n_splits=10):
    # Returns: list of accuracy matrices (n_mice x n_days x n_days)
    acc_matrices = []
    rng = np.random.default_rng(seed)
    for d in vectors:
        acc_matrix = np.full((len(days), len(days)), np.nan)
        day_per_trial = d['day'].values
        # For each fold
        fold_accs = np.zeros((n_splits, len(days), len(days)))
        # Get indices for each day
        day_indices = {day: np.where(day_per_trial == day)[0] for day in days}
        # For each fold
        for k in range(n_splits):
            # Split indices for each day
            train_idx = []
            test_idx = []
            for day in days:
                idx = day_indices[day]
                if len(idx) < n_splits:
                    continue  # skip if not enough trials
                idx_shuff = rng.permutation(idx)
                fold_size = len(idx) // n_splits
                test_fold = idx_shuff[k*fold_size:(k+1)*fold_size] if k < n_splits-1 else idx_shuff[k*fold_size:]
                train_fold = np.setdiff1d(idx, test_fold)
                train_idx.append((day, train_fold))
                test_idx.append((day, test_fold))
            # Train decoder on pre vs post
            train_days = [-2, -1, 1,2]
            train_mask = np.concatenate([fold for day, fold in train_idx if day in train_days])
            train_labels = np.concatenate([[day]*len(fold) for day, fold in train_idx if day in train_days])
            if len(train_mask) < 2 or len(np.unique(train_labels)) < 2:
                continue
            X_train = d.values[:, train_mask].T
            y_train = np.array([0 if day <= -1 else 1 for day in train_labels])
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=5000, random_state=seed)
            clf.fit(X_train_proc, y_train)
            # Test decoder on all pairs of days (using test data only)
            for i, day_i in enumerate(days):
                for j, day_j in enumerate(days):
                    test_mask_i = [fold for day, fold in test_idx if day == day_i]
                    test_mask_j = [fold for day, fold in test_idx if day == day_j]
                    if not test_mask_i or not test_mask_j:
                        continue
                    test_mask_i = test_mask_i[0]
                    test_mask_j = test_mask_j[0]
                    if len(test_mask_i) < 1 or len(test_mask_j) < 1:
                        continue
                    X_test = np.concatenate([d.values[:, test_mask_i].T, d.values[:, test_mask_j].T], axis=0)
                    y_test = np.array([0] * len(test_mask_i) + [1] * len(test_mask_j))
                    X_test_proc = scaler.transform(X_test)
                    y_pred = clf.predict(X_test_proc)
                    acc = np.mean(y_pred == y_test)
                    fold_accs[k, i, j] = acc
        # Average over folds
        with np.errstate(invalid='ignore'):
            acc_matrix = np.nanmean(fold_accs, axis=0)
        acc_matrices.append(acc_matrix)
    return np.array(acc_matrices)

# Compute accuracy matrices for each group
accs_rew_matrix = pairwise_day_decoding_fixed_decoder_cv(vectors_rew, all_days)
accs_nonrew_matrix = pairwise_day_decoding_fixed_decoder_cv(vectors_nonrew, all_days)
# Average across mice
mean_accs_rew = np.nanmean(accs_rew_matrix, axis=0)
mean_accs_nonrew = np.nanmean(accs_nonrew_matrix, axis=0)

# Make matrices symmetric with filled diagonal for aesthetics
def make_symmetric_with_diag(mat):
    sym_mat = np.full_like(mat, np.nan)
    iu = np.triu_indices_from(mat, k=1)
    sym_mat[iu] = mat[iu]
    il = np.tril_indices_from(mat, k=-1)
    sym_mat[il] = mat.T[il]
    diag = np.diag_indices_from(mat)
    sym_mat[diag] = np.diag(mat)
    return sym_mat

mean_accs_rew_sym = make_symmetric_with_diag(mean_accs_rew)
mean_accs_nonrew_sym = make_symmetric_with_diag(mean_accs_nonrew)

# Plot accuracy matrices and shared colormap
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
vmax = max(np.nanmax(mean_accs_rew_sym), np.nanmax(mean_accs_nonrew_sym))
vmin = 0.5

sns.heatmap(mean_accs_rew_sym, annot=True, fmt=".2f", xticklabels=day_labels, yticklabels=day_labels,
            ax=ax0, cmap="viridis", vmin=vmin, vmax=vmax, mask=np.isnan(mean_accs_rew_sym), cbar=False)
ax0.set_title("R+ group: Fixed decoder (-2 vs +2) pairwise day accuracy (CV)")
ax0.set_xlabel("Day")
ax0.set_ylabel("Day")

sns.heatmap(mean_accs_nonrew_sym, annot=True, fmt=".2f", xticklabels=day_labels, yticklabels=day_labels,
            ax=ax1, cmap="viridis", vmin=vmin, vmax=vmax, mask=np.isnan(mean_accs_nonrew_sym), cbar=False)
ax1.set_title("R- group: Fixed decoder (-2 vs +2) pairwise day accuracy (CV)")
ax1.set_xlabel("Day")
ax1.set_ylabel("Day")

# Shared colorbar
norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=ax2)

plt.tight_layout()
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'pairwise_day_decoding_fixed_decoder_accuracy_matrix.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'pairwise_day_decoding_fixed_decoder_accuracy_matrix.png'), format='png', dpi=300)

# Save data
np.save(os.path.join(output_dir, 'pairwise_day_decoding_fixed_decoder_accuracy_matrix_Rplus.npy'), mean_accs_rew)
np.save(os.path.join(output_dir, 'pairwise_day_decoding_fixed_decoder_accuracy_matrix_Rminus.npy'), mean_accs_nonrew)


# ---------------------------------------------------------------
# Quantification: Does Day 0 look more like pre or post?
# ---------------------------------------------------------------

# For each mouse, compare decoding accuracy for:
#  - Day -2/-1 vs Day 0 (pre vs day0)
#  - Day 0 vs Day +1/+2 (day0 vs post)
# This uses the fixed decoder trained on pre vs post (as above).

def extract_pairwise_decoding_accuracy(acc_matrices, all_days):
    # Returns two arrays: acc_pre_vs_day0, acc_day0_vs_post (n_mice,)
    idx_pre = [0, 1]  # indices for -2, -1
    idx_post = [3, 4] # indices for +1, +2
    idx_day0 = 2      # index for 0
    acc_pre_vs_day0 = []
    acc_day0_vs_post = []
    for mat in acc_matrices:
        # Pre vs Day 0: average over both pre days
        pre_accs = []
        for i in idx_pre:
            pre_accs.append(mat[i, idx_day0])
        acc_pre_vs_day0.append(np.nanmean(pre_accs))
        # Day 0 vs Post: average over both post days
        post_accs = []
        for j in idx_post:
            post_accs.append(mat[idx_day0, j])
        acc_day0_vs_post.append(np.nanmean(post_accs))
    return np.array(acc_pre_vs_day0), np.array(acc_day0_vs_post)

acc_pre_vs_day0_rew, acc_day0_vs_post_rew = extract_pairwise_decoding_accuracy(accs_rew_matrix, all_days)
acc_pre_vs_day0_nonrew, acc_day0_vs_post_nonrew = extract_pairwise_decoding_accuracy(accs_nonrew_matrix, all_days)

# Bar plot: R+ and R- groups, pre vs day0 and day0 vs post
fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True)

for ax, group, acc_pre, acc_post, color in zip(
    axes,
    ['R+', 'R-'],
    [acc_pre_vs_day0_rew, acc_pre_vs_day0_nonrew],
    [acc_day0_vs_post_rew, acc_day0_vs_post_nonrew],
    reward_palette[::-1]
):
    df_plot = pd.DataFrame({
        'Comparison': ['Pre vs Day0'] * len(acc_pre) + ['Day0 vs Post'] * len(acc_post),
        'Accuracy': np.concatenate([acc_pre, acc_post])
    })
    sns.barplot(data=df_plot, x='Comparison', y='Accuracy', errorbar='ci', ax=ax, color=color, alpha=0.7)
    sns.swarmplot(data=df_plot, x='Comparison', y='Accuracy', ax=ax, color=color, alpha=0.5, size=7)
    ax.set_title(f'{group} group')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.5)
    stat, pval = wilcoxon(acc_pre, acc_post, alternative='two-sided')
    ax.text(0.5, 0.95, f'Wilcoxon p={pval:.3f}', ha='center', va='top', transform=ax.transAxes, fontsize=10)
    ax.set_ylabel('Decoding accuracy (CV)')
    ax.set_xlabel('')

plt.suptitle('Does Day 0 look more like pre or post?')
plt.tight_layout()
sns.despine()

plt.savefig(os.path.join(output_dir, 'day0_pre_vs_post_decoding_accuracy_barplot.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'day0_pre_vs_post_decoding_accuracy_barplot.png'), format='png', dpi=300)

# Save quantification data
df_quant = pd.DataFrame({
    'group': ['R+'] * len(acc_pre_vs_day0_rew) + ['R+'] * len(acc_day0_vs_post_rew) +
             ['R-'] * len(acc_pre_vs_day0_nonrew) + ['R-'] * len(acc_day0_vs_post_nonrew),
    'comparison': (['pre_vs_day0'] * len(acc_pre_vs_day0_rew) +
                  ['day0_vs_post'] * len(acc_day0_vs_post_rew) +
                  ['pre_vs_day0'] * len(acc_pre_vs_day0_nonrew) +
                  ['day0_vs_post'] * len(acc_day0_vs_post_nonrew)),
    'accuracy': np.concatenate([acc_pre_vs_day0_rew, acc_day0_vs_post_rew,
                                acc_pre_vs_day0_nonrew, acc_day0_vs_post_nonrew])
})
df_quant.to_csv(os.path.join(output_dir, 'day0_pre_vs_post_decoding_accuracy_quantification.csv'), index=False)



















# ############################################
# Decoding decision value across learning during Day 0.
# #############################################

# The idea is to see if decoding accuracy improves progressively during Day 0 learning trials,
# indicating progressive learning, or if it jumps suddenly at some point,
# indicating an inflexion point, or if it remains flat.
# Use a decoder trained on Day -2 vs Day +2 as before, possibly using a
# sliding window. Look both at the classification of each trial/group of trials
# and at the decision values (distance to the hyperplane) as a continuous measure.
# We want to correlate that with performance across mice.

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 180 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40 
substract_baseline = True
select_responsive_cells = False
select_lmi = False
projection_type = None  # 'wS2', 'wM1' or None

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# mice = [m for m in mice if m in mice_groups['good_day0']]

# Load data.
vectors_rew_mapping = []
vectors_nonrew_mapping = []
mice_rew = []
mice_nonrew = []
vectors_nonrew_day0_learning = []
vectors_rew_day0_learning = []

# Load behaviour table with learning trials.
path = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(path)
# Select day 0 performance for whisker trials.
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1)]

for mouse in mice:
    
    # Load mapping data.
    # ------------------
    
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)
    # xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin(days))
    
    # Select responsive cells.
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
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day.
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    # Average bins.
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Remove artefacts by setting them at 0. To avoid NaN values and
    # mismatches (that concerns a single cell).
    print(np.isnan(d.values).sum(), 'NaN values in the data.')
    d = d.fillna(0)
    
    if rew_gp == 'R-':
        vectors_nonrew_mapping.append(d)
        mice_nonrew.append(mouse)
    elif rew_gp == 'R+':
        vectors_rew_mapping.append(d)
        mice_rew.append(mouse)

    # Load learning data for day 0.
    # -----------------------------
    
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    # xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin([0]))
    # Select whisker trials.
    xarray = xarray.sel(trial=xarray['whisker_stim'] == 1)
    
    # Select responsive cells.
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue
        
    # Average bins.
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Remove artefacts by setting them at 0. To avoid NaN values and
    # mismatches (that concerns a single cell).
    print(np.isnan(xarray.values).sum(), 'NaN values in the data.')
    xarray = xarray.fillna(0)
        
    if rew_gp == 'R-':
        vectors_nonrew_day0_learning.append(xarray)
    elif rew_gp == 'R+':
        vectors_rew_day0_learning.append(xarray)
    

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

# Run analysis for both groups
window_size = 10
step_size = 1

# Set alignment parameters
align_to_learning = False  # Set to True to align to individual learning onset
trials_before = 50  # Number of trials before learning onset to include
trials_after = 100  # Number of trials after learning onset to include

# If mice_groups exist, try to plot good/bad subsets, otherwise plot empty panels
try:
    mice_good = [m for m in mice_rew if m in mice_groups.get('good_day0', [])]
    mice_bad = [m for m in mice_rew if m in (mice_groups.get('bad_day0', []) + mice_groups.get('meh_day0', []))]
except Exception:
    mice_good, mice_bad = [], []

# Build results for R+ and R- (and optionally good/bad subsets)
results_rew, weights_rew = progressive_learning_analysis(
    vectors_rew_mapping, vectors_rew_day0_learning, mice_rew,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)
results_nonrew, weights_nonrew = progressive_learning_analysis(
    vectors_nonrew_mapping, vectors_nonrew_day0_learning, mice_nonrew,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)
results_good, weights_good = progressive_learning_analysis(
    [vectors_rew_mapping[i] for i, m in enumerate(mice_rew) if m in mice_good],
    [vectors_rew_day0_learning[i] for i, m in enumerate(mice_rew) if m in mice_good],
    mice_good,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)
results_bad, weights_bad = progressive_learning_analysis(
    [vectors_rew_mapping[i] for i, m in enumerate(mice_rew) if m in mice_bad],
    [vectors_rew_day0_learning[i] for i, m in enumerate(mice_rew) if m in mice_bad],
    mice_bad,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)
results_rew['reward_group'] = 'R+'
results_nonrew['reward_group'] = 'R-'
results_combined = pd.concat([results_rew, results_nonrew], ignore_index=True)

# Save classifier weights to disk for later use
# Convert weights dictionaries to DataFrame format with columns: mouse_id, roi, classifier_weight

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
[]
cut_n_trials = 120

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
plot_decision(ax5, results_rew, reward_palette[1], 'R+ mean decision value', align_to_learning=align_to_learning, ylim=(-3, 6))

ax6 = plt.subplot(3, 4, 6)
plot_decision(ax6, results_nonrew, reward_palette[0], 'R- mean decision value', align_to_learning=align_to_learning, ylim=(-3, 6))

ax7 = plt.subplot(3, 4, 7)
plot_decision(ax7, results_good, reward_palette[1], 'Good day0 mean decision value', align_to_learning=align_to_learning, ylim=(-3, 6))

ax8 = plt.subplot(3, 4, 8)
plot_decision(ax8, results_bad, reward_palette[1], 'Bad day0 mean decision value', align_to_learning=align_to_learning, ylim=(-3, 6))

# Bottom row: Probability P(post) (4 panels) - separated from decision values
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
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/gradual_learning'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'decoder_decision_value_day0_learning_with_alignment_to_learning.svg'), format='svg', dpi=300)








# ============================================================================
# SINGLE MOUSE ANALYSIS: Robustness of Progressive Learning Effect
# ============================================================================

print("\n" + "="*80)
print("SINGLE MOUSE PROGRESSIVE LEARNING ANALYSIS")
print("="*80 + "\n")

# 1. Individual mouse plots in a multi-page PDF
# ----------------------------------------------

from matplotlib.backends.backend_pdf import PdfPages

# Create PDF with individual mouse plots
pdf_file = os.path.join(output_dir, 'individual_mouse_progressive_learning.pdf')

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


# 2. Statistical Quantification of Progressive Learning
# ------------------------------------------------------

print("\n" + "-"*80)
print("STATISTICAL TESTS FOR PROGRESSIVE LEARNING EFFECT")
print("-"*80 + "\n")

# Method 1: Slope Analysis
# ------------------------
print("METHOD 1: Linear Trend Analysis")
print("Test if decision values show significant positive slope over time\n")

from scipy.stats import linregress

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
# -------------------------------------
print("\n" + "-"*80)
print("SAVING RESULTS TO CSV FILES")
print("-"*80 + "\n")

# Save slope data
df_slopes.to_csv(os.path.join(output_dir, 'progressive_learning_slopes.csv'), index=False)
print(f"Saved: progressive_learning_slopes.csv")


# Compute and save population-level statistics
# ---------------------------------------------

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
df_population_stats.to_csv(os.path.join(output_dir, 'progressive_learning_population_statistics.csv'), index=False)
print(f"Saved: progressive_learning_population_statistics.csv")


# Population-level statistics visualization
# ------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# Prepare data for plotting
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
ax.set_title('Linear Slope\nPopulation Test (Wilcoxon)', fontsize=12, fontweight='bold')

# Set y-lim to include text
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.15 * y_range)

plt.tight_layout()
sns.despine()

plt.savefig(os.path.join(output_dir, 'progressive_learning_population_statistics.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'progressive_learning_population_statistics.png'), format='png', dpi=300)
print(f"Saved: progressive_learning_population_statistics figure")


# Summary visualization of individual mice
# -----------------------------------------

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

plt.savefig(os.path.join(output_dir, 'progressive_learning_individual_mice_summary.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'progressive_learning_individual_mice_summary.png'), format='png', dpi=300)
print(f"Saved: progressive_learning_individual_mice_summary figure")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")








# ============================================================================
# CORRELATION ANALYSIS: Decision Values vs Behavioral Performance
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS: Decision Values vs Behavioral Performance")
print("="*80 + "\n")

mice = results_combined['mouse_id'].unique()
# mice = [m for m in mice if m in mice_groups['good_day0']]

# Method 1: Direct Correlation
# -----------------------------
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

df_corr.to_csv(os.path.join(output_dir, 'correlation_decision_behavior_data.csv'), index=False)
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


# Method 2: Rate of Change Analysis
# ----------------------------------
print("\n" + "-"*80)
print("METHOD 2: Rate of Change Analysis")
print("Correlation between slopes of decision values and behavioral performance\n")

slopes_dec = []
slopes_perf = []
slope_mice = []
slope_groups = []

for mouse in mice:
    group = results_combined.loc[results_combined['mouse_id'] == mouse, 'reward_group'].iloc[0]
    dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
    bh_mouse = bh_df[bh_df['mouse_id'] == mouse]
    
    if align_to_learning:
        learning_trial = bh_mouse['learning_trial'].iloc[0] if 'learning_trial' in bh_mouse.columns else 0
        bh_mouse_aligned = bh_mouse.copy()
        bh_mouse_aligned['trial_w_aligned'] = bh_mouse_aligned['trial_w'] - learning_trial
        common_trials = np.intersect1d(
            dec_mouse['trial_center_aligned'].dropna(), 
            bh_mouse_aligned['trial_w_aligned']
        )
        if len(common_trials) < 10:
            continue
        x_trials = np.array(common_trials)
        dec_vals = dec_mouse.set_index('trial_center_aligned').loc[common_trials]['mean_decision_value'].values
        perf_vals = bh_mouse_aligned.set_index('trial_w_aligned').loc[common_trials]['learning_curve_w'].values
    else:
        common_trials = np.intersect1d(dec_mouse['trial_start'], bh_mouse['trial_w'])
        if len(common_trials) < 10:
            continue
        x_trials = np.array(common_trials)
        dec_vals = dec_mouse.set_index('trial_start').loc[common_trials]['mean_decision_value'].values
        perf_vals = bh_mouse.set_index('trial_w').loc[common_trials]['learning_curve_w'].values
    
    # Fit linear regression to get slopes
    from sklearn.linear_model import LinearRegression
    reg_dec = LinearRegression().fit(x_trials.reshape(-1, 1), dec_vals)
    reg_perf = LinearRegression().fit(x_trials.reshape(-1, 1), perf_vals)
    
    slopes_dec.append(reg_dec.coef_[0])
    slopes_perf.append(reg_perf.coef_[0])
    slope_mice.append(mouse)
    slope_groups.append(group)
    
    print(f"{mouse} ({group}): decoder slope={reg_dec.coef_[0]:.4f}, behavior slope={reg_perf.coef_[0]:.4f}")

# Save slope data
df_slopes_corr = pd.DataFrame({
    'mouse_id': slope_mice,
    'reward_group': slope_groups,
    'slope_decoder': slopes_dec,
    'slope_behavior': slopes_perf
})

df_slopes_corr.to_csv(os.path.join(output_dir, 'correlation_slopes_data.csv'), index=False)
print(f"\nSaved: correlation_slopes_data.csv")

# Test correlation between slopes
print("\nCorrelation between decoder and behavior slopes:")
for group in ['R+', 'R-']:
    sub = df_slopes_corr[df_slopes_corr['reward_group'] == group]
    if len(sub) >= 3:
        corr_slopes, p_corr = pearsonr(sub['slope_decoder'].values, sub['slope_behavior'].values)
        print(f"\n{group} Group:")
        print(f"  Decoder slope: mean={np.mean(sub['slope_decoder'].values):.4f} ± {np.std(sub['slope_decoder'].values):.4f}")
        print(f"  Behavior slope: mean={np.mean(sub['slope_behavior'].values):.4f} ± {np.std(sub['slope_behavior'].values):.4f}")
        print(f"  Correlation of slopes: r={corr_slopes:.3f}, p={p_corr:.4f}")
        
        # Add to population stats
        pop_stats_corr.append({
            'reward_group': group,
            'method': 'Slope Correlation',
            'mean_value': corr_slopes,
            'std_value': np.nan,  # Not applicable for a single correlation value
            'n_total': len(sub),
            'p_wilcoxon': np.nan,
            'p_ttest': p_corr  # Use Pearson p-value here
        })

# Update population stats dataframe
df_pop_stats_corr = pd.DataFrame(pop_stats_corr)
df_pop_stats_corr.to_csv(os.path.join(output_dir, 'correlation_population_statistics.csv'), index=False)
print(f"\nSaved: correlation_population_statistics.csv")


# Visualization
# -------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Direct correlation
ax = axes[0]
sns.swarmplot(data=df_corr, x='reward_group', y='correlation', palette=reward_palette[::-1], size=8, alpha=0.6, ax=ax)
sns.pointplot(data=df_corr, x='reward_group', y='correlation', palette=reward_palette[::-1], errorbar='ci', ax=ax, markersize=10, join=False)
ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylabel('Pearson Correlation\n(Decision value vs Performance)', fontsize=11)
ax.set_title('Direct Correlation', fontsize=12, fontweight='bold')
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

# Panel 2: Slope correlation
ax = axes[1]
sns.scatterplot(data=df_slopes_corr, x='slope_behavior', y='slope_decoder', hue='reward_group', palette=reward_palette[::-1], alpha=0.7, s=80, edgecolor='black', linewidth=0.5, ax=ax)
# Regression lines per group
for group, color in zip(['R+', 'R-'], reward_palette[::-1]):
    sub = df_slopes_corr[df_slopes_corr['reward_group'] == group]
    if len(sub) >= 3:
        from sklearn.linear_model import LinearRegression
        X = sub['slope_behavior'].values.reshape(-1, 1)
        y = sub['slope_decoder'].values
        reg = LinearRegression().fit(X, y)
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, color=color, linewidth=2, alpha=0.8)
        # Add correlation text
        pop_stat = df_pop_stats_corr[(df_pop_stats_corr['reward_group'] == group) & 
                                    (df_pop_stats_corr['method'] == 'Slope Correlation')]
        if not pop_stat.empty:
            r_val = pop_stat['mean_value'].values[0]
            p_val = pop_stat['p_ttest'].values[0]
            p_text = f"p={p_val:.4f}" if p_val >= 0.001 else "p<0.001"
            ax.text(0.05, 0.95 - (0.15 * (0 if group == 'R+' else 1)), f"{group}: r={r_val:.3f}, {p_text}",
                    transform=ax.transAxes, fontsize=10, fontweight='bold',
                    verticalalignment='top', color=color)

ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
ax.set_xlabel('Behavior Slope (per trial)', fontsize=11)
ax.set_ylabel('Decoder Slope (per trial)', fontsize=11)
ax.set_title('Rate of Change Correlation', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')

plt.tight_layout()
sns.despine()

plt.savefig(os.path.join(output_dir, 'correlation_decision_behavior.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'correlation_decision_behavior.png'), format='png', dpi=300)
print(f"\nSaved: correlation_decision_behavior figure")

print("\n" + "="*80)
print("CORRELATION ANALYSIS COMPLETE")
print("="*80 + "\n")



 
# ############################################################################
# Illustration of the result for two example mice.
# ############################################################################
# Find the R+ and R- mouse with highest correlation between behavior and decision value
best_rplus = df_corr[df_corr['reward_group'] == 'R+'].sort_values('correlation', ascending=False).iloc[0]['mouse_id']
best_rminus = df_corr[df_corr['reward_group'] == 'R-'].sort_values('correlation', ascending=False).iloc[1]['mouse_id']
example_mice = [best_rplus, best_rminus]

# Improved publication-quality figure: one figure, one column per mouse, reward color palette 
fig, axes = plt.subplots(2, len(example_mice), figsize=(7 , 7), sharex=False)

for col, mouse in enumerate(example_mice):
    # Panel 1: Behavioral performance (whisker trials only)
    bh_mouse = bh_df[(bh_df['mouse_id'] == mouse) & (bh_df['whisker_stim'] == 1)]
    color = reward_palette[1] if results_combined[results_combined['mouse_id'] == mouse]['reward_group'].iloc[0] == 'R+' else reward_palette[0]
    ax_beh = axes[0, col]
    # Cut to first 120 trials for reward mouse
    if results_combined[results_combined['mouse_id'] == mouse]['reward_group'].iloc[0] == 'R+':
        bh_mouse = bh_mouse[bh_mouse['trial_w'] < 120]
    if not bh_mouse.empty:
        sns.lineplot(data=bh_mouse, x='trial_w', y='learning_curve_w', ax=ax_beh, color=color, linewidth=2.5)
        ax_beh.set_ylabel('Performance (whisker trials)', fontsize=13)
        ax_beh.set_ylim(0, 1)
        ax_beh.tick_params(axis='both', labelsize=12)
        ax_beh.spines['top'].set_visible(False)
        ax_beh.spines['right'].set_visible(False)
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
        ax_dec.spines['top'].set_visible(False)
        ax_dec.spines['right'].set_visible(False)
        # Set different y-limits for R+ and R-
        if results_combined[results_combined['mouse_id'] == mouse]['reward_group'].iloc[0] == 'R+':
            ax_dec.set_ylim([-5, 3])
            yticks = np.arange(-5, 4, 1)
        else:
            ax_dec.set_ylim([-4, 3])
            yticks = np.arange(-4, 4, 1)
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
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/gradual_learning'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'example_mice_behavior_decision_value.svg'), format='svg', dpi=300)
plt.close(fig)



# ============================================================================
# ADDITIONAL LEARNING ANALYSES
# ============================================================================

# Load behaviour table with learning trials.
path = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(path)
# Select day 0 performance for whisker trials.
bh_df = table.loc[(table['day'] == 0)]

mice = results_combined['mouse_id'].unique()
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/gradual_learning'


# Analysis: Error-Driven Learning Signals
# ========================================

print("\n" + "="*80)
print("ERROR-DRIVEN LEARNING SIGNALS ANALYSIS")
print("="*80 + "\n")

# Parameters
trial_history_depth = 1  # Look at previous trial only

# Part 1: Decision value adjustments after different trial types
# ----------------------------------------------------------------
print("Part 1: Analyzing decision value adjustments after all trial types...\n")

error_driven_data = []

for mouse in mice:
    group = results_combined.loc[results_combined['mouse_id'] == mouse, 'reward_group'].iloc[0]
    dec_mouse = results_combined[results_combined['mouse_id'] == mouse].copy()
    bh_mouse = bh_df[bh_df['mouse_id'] == mouse].copy()
    
    # Align data
    if align_to_learning:
        learning_trial = bh_mouse['learning_trial'].iloc[0] if 'learning_trial' in bh_mouse.columns else 0
        bh_mouse['trial_w_aligned'] = bh_mouse['trial_w'] - learning_trial
        
        # For decoder: only whisker trials
        dec_mouse = dec_mouse.dropna(subset=['trial_center_aligned'])
        dec_mouse = dec_mouse.sort_values('trial_center_aligned').reset_index(drop=True)
        
        # For behavior: all trials
        bh_mouse = bh_mouse.sort_values('trial_w').reset_index(drop=True)
        trial_col = 'trial_w_aligned'
    else:
        dec_mouse = dec_mouse.sort_values('trial_start').reset_index(drop=True)
        bh_mouse = bh_mouse.sort_values('trial_w').reset_index(drop=True)
        trial_col = 'trial_w'
    
    if len(dec_mouse) < 10 or len(bh_mouse) < 10:
        continue
    
    # Create a lookup for decision values by whisker trial number
    dv_lookup = {}
    for _, row in dec_mouse.iterrows():
        whisker_trial = row['trial_start'] if not align_to_learning else row['trial_center_aligned']
        dv_lookup[whisker_trial] = row['mean_decision_value']
    
    # Go through all behavior trials
    last_whisker_trial = None
    last_dv = None
    
    for i, curr_bh_row in bh_mouse.iterrows():
        curr_trial = curr_bh_row[trial_col]
        
        # Check if current trial is a whisker trial with decision value
        if curr_trial in dv_lookup:
            curr_dv = dv_lookup[curr_trial]
            
            # If we have a previous trial, compute adjustment
            if last_whisker_trial is not None and i > 0:
                prev_bh_row = bh_mouse.iloc[i-1]
                
                # Classify previous trial type and outcome
                if prev_bh_row['whisker_stim'] == 1:
                    if prev_bh_row['lick_flag'] == 1:
                        prev_type = 'Whisker Hit'
                    else:
                        prev_type = 'Whisker Miss'
                elif prev_bh_row['no_stim'] == 1:
                    if prev_bh_row['lick_flag'] == 1:
                        prev_type = 'No-stim FA'
                    else:
                        prev_type = 'No-stim CR'
                else:
                    # Skip auditory trials
                    last_whisker_trial = curr_trial
                    last_dv = curr_dv
                    continue
                
                # Compute adjustment
                adjustment = curr_dv - last_dv
                
                # Classify trial phase
                if align_to_learning:
                    phase = 'post' if curr_trial >= 0 else 'pre'
                else:
                    phase = 'post' if curr_trial >= np.median(bh_mouse[trial_col]) else 'pre'
                
                error_driven_data.append({
                    'mouse_id': mouse,
                    'reward_group': group,
                    'trial': curr_trial,
                    'prev_trial_type': prev_type,
                    'adjustment': adjustment,
                    'absolute_adjustment': abs(adjustment),
                    'phase': phase
                })
            
            # Update last whisker trial
            last_whisker_trial = curr_trial
            last_dv = curr_dv

df_error = pd.DataFrame(error_driven_data)
df_error.to_csv(os.path.join(output_dir, 'error_driven_analysis_data.csv'), index=False)
print("Saved: error_driven_analysis_data.csv")

# Compute per-mouse summary statistics
print("Computing per-mouse adjustments...")
mouse_adjustments = []

trial_types = ['Whisker Hit', 'Whisker Miss', 'No-stim CR', 'No-stim FA']

for mouse in df_error['mouse_id'].unique():
    mouse_data = df_error[df_error['mouse_id'] == mouse]
    group = mouse_data['reward_group'].iloc[0]
    
    perf_dict = {
        'mouse_id': mouse,
        'reward_group': group
    }
    
    # Get adjustments for each trial type
    for trial_type in trial_types:
        type_data = mouse_data[mouse_data['prev_trial_type'] == trial_type]
        if len(type_data) > 0:
            perf_dict[f'mean_adj_after_{trial_type}'] = type_data['absolute_adjustment'].mean()
            perf_dict[f'n_after_{trial_type}'] = len(type_data)
        else:
            perf_dict[f'mean_adj_after_{trial_type}'] = np.nan
            perf_dict[f'n_after_{trial_type}'] = 0
    
    # Also compute phase-specific adjustments for panel 2 (keep old analysis)
    adj_pre = mouse_data[mouse_data['phase'] == 'pre']['absolute_adjustment'].values
    adj_post = mouse_data[mouse_data['phase'] == 'post']['absolute_adjustment'].values
    
    perf_dict['mean_adj_pre'] = np.mean(adj_pre) if len(adj_pre) > 0 else np.nan
    perf_dict['mean_adj_post'] = np.mean(adj_post) if len(adj_post) > 0 else np.nan
    perf_dict['n_trials'] = len(mouse_data)
    
    mouse_adjustments.append(perf_dict)

df_mouse_adj = pd.DataFrame(mouse_adjustments)
df_mouse_adj.to_csv(os.path.join(output_dir, 'error_driven_per_mouse_adjustments.csv'), index=False)
print("Saved: error_driven_per_mouse_adjustments.csv")

# Statistical analysis
print("\nPopulation-level statistics for decision value adjustments:")

error_stats = []

for group in ['R+', 'R-']:
    sub = df_mouse_adj[df_mouse_adj['reward_group'] == group]
    
    if len(sub) == 0:
        continue
    
    print(f"\n{group} Group (N={len(sub)} mice):")
    
    # Test within-group comparisons for trial types
    # 1. Whisker Hit vs. Whisker Miss
    w_hit = sub['mean_adj_after_Whisker Hit'].dropna().values
    w_miss = sub['mean_adj_after_Whisker Miss'].dropna().values
    
    if len(w_hit) >= 3 and len(w_miss) >= 3:
        n_paired = min(len(w_hit), len(w_miss))
        stat_w, p_whisker = wilcoxon(w_hit[:n_paired] - w_miss[:n_paired])
        print(f"  Adjustment after Whisker Hit: {np.mean(w_hit):.4f} ± {np.std(w_hit):.4f}")
        print(f"  Adjustment after Whisker Miss: {np.mean(w_miss):.4f} ± {np.std(w_miss):.4f}")
        print(f"  Wilcoxon test (Hit vs Miss): p={p_whisker:.4f}")
    else:
        p_whisker = np.nan
    
    # 2. No-stim CR vs. No-stim FA
    ns_cr = sub['mean_adj_after_No-stim CR'].dropna().values
    ns_fa = sub['mean_adj_after_No-stim FA'].dropna().values
    
    if len(ns_cr) >= 3 and len(ns_fa) >= 3:
        n_paired = min(len(ns_cr), len(ns_fa))
        stat_w, p_nostim = wilcoxon(ns_cr[:n_paired] - ns_fa[:n_paired])
        print(f"  Adjustment after No-stim CR: {np.mean(ns_cr):.4f} ± {np.std(ns_cr):.4f}")
        print(f"  Adjustment after No-stim FA: {np.mean(ns_fa):.4f} ± {np.std(ns_fa):.4f}")
        print(f"  Wilcoxon test (CR vs FA): p={p_nostim:.4f}")
    else:
        p_nostim = np.nan
    
    # Also keep pre vs post comparison for panel 2
    adj_pre = sub['mean_adj_pre'].dropna().values
    adj_post = sub['mean_adj_post'].dropna().values
    
    if len(adj_pre) > 0 and len(adj_post) > 0:
        n_paired = min(len(adj_pre), len(adj_post))
        if n_paired >= 3:
            stat_w, p_phase = wilcoxon(adj_pre[:n_paired] - adj_post[:n_paired], alternative='greater')
            print(f"  Adjustment pre-learning: mean={np.mean(adj_pre):.4f} ± {np.std(adj_pre):.4f}")
            print(f"  Adjustment post-learning: mean={np.mean(adj_post):.4f} ± {np.std(adj_post):.4f}")
            print(f"  Wilcoxon paired test (pre > post): p={p_phase:.4f}")
        else:
            p_phase = np.nan
    else:
        p_phase = np.nan
    
    error_stats.append({
        'reward_group': group,
        'n_mice': len(sub),
        'p_whisker_hit_vs_miss': p_whisker,
        'p_nostim_cr_vs_fa': p_nostim,
        'mean_adj_pre': np.nanmean(adj_pre) if len(adj_pre) > 0 else np.nan,
        'std_adj_pre': np.nanstd(adj_pre) if len(adj_pre) > 0 else np.nan,
        'mean_adj_post': np.nanmean(adj_post) if len(adj_post) > 0 else np.nan,
        'std_adj_post': np.nanstd(adj_post) if len(adj_post) > 0 else np.nan,
        'p_pre_vs_post': p_phase
    })

df_error_stats = pd.DataFrame(error_stats)
df_error_stats.to_csv(os.path.join(output_dir, 'error_driven_analysis_stats.csv'), index=False)
print("\nSaved: error_driven_analysis_stats.csv")


# Part 2: Behavioral performance after different trial types
# -----------------------------------------------------------
print("\n" + "-"*80)
print("Part 2: Analyzing behavioral performance after different trial types...\n")

behavioral_performance_data = []

for mouse in mice:
    group = results_combined.loc[results_combined['mouse_id'] == mouse, 'reward_group'].iloc[0] if mouse in results_combined['mouse_id'].values else None
    bh_mouse = bh_df[bh_df['mouse_id'] == mouse].copy()
    
    if len(bh_mouse) < 10 or group is None:
        continue
    
    # Sort by trial number
    bh_mouse = bh_mouse.sort_values('id').reset_index(drop=True)
    
    # Identify trial types and outcomes for previous and current trials
    for i in range(1, len(bh_mouse)):
        prev_trial = bh_mouse.iloc[i-1]
        curr_trial = bh_mouse.iloc[i]
        
        # Classify previous trial type and outcome
        if prev_trial['whisker_stim'] == 1:
            if prev_trial['lick_flag'] == 1:
                prev_type = 'Whisker Hit'
            else:
                prev_type = 'Whisker Miss'
        elif prev_trial['no_stim'] == 1:
            if prev_trial['lick_flag'] == 1:
                prev_type = 'No-stim FA'  # False alarm
            else:
                prev_type = 'No-stim CR'  # Correct rejection
        else:
            continue  # Skip auditory trials as previous trials
        
        # Classify current trial type and determine correct performance
        # We want performance on whisker trials only (go trials with reward)
        if curr_trial['whisker_stim'] == 1:
            # Whisker trial: correct = hit (lick)
            curr_correct = curr_trial['lick_flag'] == 1
        else:
            # Skip non-whisker current trials
            continue
        
        behavioral_performance_data.append({
            'mouse_id': mouse,
            'reward_group': group,
            'prev_trial_type': prev_type,
            'current_correct': int(curr_correct)
        })

df_behavior_perf = pd.DataFrame(behavioral_performance_data)
df_behavior_perf.to_csv(os.path.join(output_dir, 'behavioral_performance_after_trial_types_data.csv'), index=False)
print("Saved: behavioral_performance_after_trial_types_data.csv")

# Debug: Check what trial types we have
print(f"\nTrial types found in data: {df_behavior_perf['prev_trial_type'].unique()}")
print(f"Counts per trial type:")
print(df_behavior_perf['prev_trial_type'].value_counts())

# Compute per-mouse performance after each trial type
print("Computing per-mouse performance after each trial type...")
mouse_behavior_perf = []

trial_types = ['Whisker Hit', 'Whisker Miss', 'No-stim CR', 'No-stim FA']

for mouse in df_behavior_perf['mouse_id'].unique():
    mouse_data = df_behavior_perf[df_behavior_perf['mouse_id'] == mouse]
    group = mouse_data['reward_group'].iloc[0]
    
    perf_dict = {
        'mouse_id': mouse,
        'reward_group': group
    }
    
    for trial_type in trial_types:
        type_data = mouse_data[mouse_data['prev_trial_type'] == trial_type]
        if len(type_data) > 0:
            perf = type_data['current_correct'].mean()
            perf_dict[f'perf_after_{trial_type}'] = perf
            perf_dict[f'n_after_{trial_type}'] = len(type_data)
        else:
            perf_dict[f'perf_after_{trial_type}'] = np.nan
            perf_dict[f'n_after_{trial_type}'] = 0
    
    mouse_behavior_perf.append(perf_dict)

df_mouse_behavior_perf = pd.DataFrame(mouse_behavior_perf)
df_mouse_behavior_perf.to_csv(os.path.join(output_dir, 'behavioral_performance_per_mouse.csv'), index=False)
print("Saved: behavioral_performance_per_mouse.csv")

# Debug: Check column names and data availability
print(f"\nColumns in df_mouse_behavior_perf: {df_mouse_behavior_perf.columns.tolist()}")
for col in ['perf_after_Whisker Hit', 'perf_after_Whisker Miss', 'perf_after_No-stim CR', 'perf_after_No-stim FA']:
    if col in df_mouse_behavior_perf.columns:
        n_valid = df_mouse_behavior_perf[col].notna().sum()
        print(f"  {col}: {n_valid} mice with data")

# Statistical analysis
print("\nPopulation-level statistics for behavioral performance:")

behavior_perf_stats = []

for group in ['R+', 'R-']:
    sub = df_mouse_behavior_perf[df_mouse_behavior_perf['reward_group'] == group]
    
    if len(sub) == 0:
        continue
    
    print(f"\n{group} Group (N={len(sub)} mice):")
    
    # Test within-group comparisons
    # 1. Whisker Hit vs. Whisker Miss
    w_hit = sub['perf_after_Whisker Hit'].dropna().values
    w_miss = sub['perf_after_Whisker Miss'].dropna().values
    
    if len(w_hit) >= 3 and len(w_miss) >= 3:
        n_paired = min(len(w_hit), len(w_miss))
        stat_w, p_whisker = wilcoxon(w_hit[:n_paired] - w_miss[:n_paired])
        print(f"  After Whisker Hit: {np.mean(w_hit):.3f} ± {np.std(w_hit):.3f}")
        print(f"  After Whisker Miss: {np.mean(w_miss):.3f} ± {np.std(w_miss):.3f}")
        print(f"  Wilcoxon test (Hit vs Miss): p={p_whisker:.4f}")
        
        behavior_perf_stats.append({
            'reward_group': group,
            'comparison': 'Whisker Hit vs Miss',
            'mean_condition1': np.mean(w_hit),
            'mean_condition2': np.mean(w_miss),
            'p_value': p_whisker,
            'n_mice': n_paired
        })
    
    # 2. No-stim CR vs. No-stim FA
    ns_cr = sub['perf_after_No-stim CR'].dropna().values
    ns_fa = sub['perf_after_No-stim FA'].dropna().values
    
    if len(ns_cr) >= 3 and len(ns_fa) >= 3:
        n_paired = min(len(ns_cr), len(ns_fa))
        stat_w, p_nostim = wilcoxon(ns_cr[:n_paired] - ns_fa[:n_paired])
        print(f"  After No-stim CR: {np.mean(ns_cr):.3f} ± {np.std(ns_cr):.3f}")
        print(f"  After No-stim FA: {np.mean(ns_fa):.3f} ± {np.std(ns_fa):.3f}")
        print(f"  Wilcoxon test (CR vs FA): p={p_nostim:.4f}")
        
        behavior_perf_stats.append({
            'reward_group': group,
            'comparison': 'No-stim CR vs FA',
            'mean_condition1': np.mean(ns_cr),
            'mean_condition2': np.mean(ns_fa),
            'p_value': p_nostim,
            'n_mice': n_paired
        })

df_behavior_perf_stats = pd.DataFrame(behavior_perf_stats)
df_behavior_perf_stats.to_csv(os.path.join(output_dir, 'behavioral_performance_stats.csv'), index=False)
print("\nSaved: behavioral_performance_stats.csv")

# Visualization
# -------------
print("\nGenerating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Row 1: Behavioral performance
# ==============================

# Panel 1 (top-left): Behavioral performance for R+ group
ax = axes[0, 0]

sub = df_mouse_behavior_perf[df_mouse_behavior_perf['reward_group'] == 'R+']
color = reward_palette[1]

# Prepare data for barplot
conditions = ['W Hit', 'W Miss', 'NS CR', 'NS FA']
trial_type_map = {
    'W Hit': 'Whisker Hit',
    'W Miss': 'Whisker Miss',
    'NS CR': 'No-stim CR',
    'NS FA': 'No-stim FA'
}

plot_data_p1 = []
for cond in conditions:
    col_name = f'perf_after_{trial_type_map[cond]}'
    if col_name in sub.columns:
        for idx, row in sub.iterrows():
            if not np.isnan(row[col_name]):
                plot_data_p1.append({
                    'Condition': cond,
                    'Performance': row[col_name],
                    'mouse_id': row['mouse_id']
                })

df_plot_p1 = pd.DataFrame(plot_data_p1)

# Use seaborn barplot with bootstrap CI
if len(df_plot_p1) > 0:
    sns.barplot(data=df_plot_p1, x='Condition', y='Performance', order=conditions,
               color=color, alpha=0.7, errorbar=('ci', 95), capsize=0.1,
               err_kws={'linewidth': 2}, ax=ax)
    
    # Add individual points
    sns.stripplot(data=df_plot_p1, x='Condition', y='Performance', order=conditions,
                 color=color, alpha=0.4, size=6, ax=ax)

# Add p-values for comparisons
stats_r_plus = df_behavior_perf_stats[df_behavior_perf_stats['reward_group'] == 'R+']
if len(stats_r_plus) > 0:
    # Whisker comparison
    p_whisker = stats_r_plus[stats_r_plus['comparison'] == 'Whisker Hit vs Miss']['p_value'].values
    if len(p_whisker) > 0:
        y_max = df_plot_p1[df_plot_p1['Condition'].isin(['W Hit', 'W Miss'])]['Performance'].max()
        ax.plot([0, 1], [y_max + 0.05, y_max + 0.05], 'k-', linewidth=1.5)
        ax.text(0.5, y_max + 0.06, f'p={p_whisker[0]:.4f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # No-stim comparison
    p_nostim = stats_r_plus[stats_r_plus['comparison'] == 'No-stim CR vs FA']['p_value'].values
    if len(p_nostim) > 0:
        y_max = df_plot_p1[df_plot_p1['Condition'].isin(['NS CR', 'NS FA'])]['Performance'].max()
        ax.plot([2, 3], [y_max + 0.05, y_max + 0.05], 'k-', linewidth=1.5)
        ax.text(2.5, y_max + 0.06, f'p={p_nostim[0]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('Previous Trial Type', fontsize=11)
ax.set_ylabel('Performance on Next Whisker Trial', fontsize=11)
ax.set_title('Behavioral: R+ Group', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.2])

# Panel 2 (top-right): Behavioral performance for R- group
ax = axes[0, 1]

sub = df_mouse_behavior_perf[df_mouse_behavior_perf['reward_group'] == 'R-']
color = reward_palette[0]

plot_data_p2 = []
for cond in conditions:
    col_name = f'perf_after_{trial_type_map[cond]}'
    if col_name in sub.columns:
        for idx, row in sub.iterrows():
            if not np.isnan(row[col_name]):
                plot_data_p2.append({
                    'Condition': cond,
                    'Performance': row[col_name],
                    'mouse_id': row['mouse_id']
                })

df_plot_p2 = pd.DataFrame(plot_data_p2)

if len(df_plot_p2) > 0:
    sns.barplot(data=df_plot_p2, x='Condition', y='Performance', order=conditions,
               color=color, alpha=0.7, errorbar=('ci', 95), capsize=0.1,
               err_kws={'linewidth': 2}, ax=ax)
    
    sns.stripplot(data=df_plot_p2, x='Condition', y='Performance', order=conditions,
                 color=color, alpha=0.4, size=6, ax=ax)

# Add p-values for comparisons
stats_r_minus = df_behavior_perf_stats[df_behavior_perf_stats['reward_group'] == 'R-']
if len(stats_r_minus) > 0:
    # Whisker comparison
    p_whisker = stats_r_minus[stats_r_minus['comparison'] == 'Whisker Hit vs Miss']['p_value'].values
    if len(p_whisker) > 0:
        y_max = df_plot_p2[df_plot_p2['Condition'].isin(['W Hit', 'W Miss'])]['Performance'].max()
        ax.plot([0, 1], [y_max + 0.05, y_max + 0.05], 'k-', linewidth=1.5)
        ax.text(0.5, y_max + 0.06, f'p={p_whisker[0]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # No-stim comparison
    p_nostim = stats_r_minus[stats_r_minus['comparison'] == 'No-stim CR vs FA']['p_value'].values
    if len(p_nostim) > 0:
        y_max = df_plot_p2[df_plot_p2['Condition'].isin(['NS CR', 'NS FA'])]['Performance'].max()
        ax.plot([2, 3], [y_max + 0.05, y_max + 0.05], 'k-', linewidth=1.5)
        ax.text(2.5, y_max + 0.06, f'p={p_nostim[0]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('Previous Trial Type', fontsize=11)
ax.set_ylabel('Performance on Next Whisker Trial', fontsize=11)
ax.set_title('Behavioral: R- Group', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.2])


# Row 2: Neural decision value adjustments
# =========================================

# Panel 3 (bottom-left): Decision value adjustment after different trial types for R+
ax = axes[1, 0]

sub = df_mouse_adj[df_mouse_adj['reward_group'] == 'R+']
color = reward_palette[1]

plot_data_p3 = []
for cond in conditions:
    col_name = f'mean_adj_after_{trial_type_map[cond]}'
    if col_name in sub.columns:
        for idx, row in sub.iterrows():
            if not np.isnan(row[col_name]):
                plot_data_p3.append({
                    'Condition': cond,
                    'Adjustment': row[col_name],
                    'mouse_id': row['mouse_id']
                })

df_plot_p3 = pd.DataFrame(plot_data_p3)

if len(df_plot_p3) > 0:
    sns.barplot(data=df_plot_p3, x='Condition', y='Adjustment', order=conditions,
               color=color, alpha=0.7, errorbar=('ci', 95), capsize=0.1,
               err_kws={'linewidth': 2}, ax=ax)
    
    sns.stripplot(data=df_plot_p3, x='Condition', y='Adjustment', order=conditions,
                 color=color, alpha=0.4, size=6, ax=ax)

# Add p-values
error_stats_r_plus = df_error_stats[df_error_stats['reward_group'] == 'R+']
if len(error_stats_r_plus) > 0:
    # Whisker comparison
    p_whisker = error_stats_r_plus['p_whisker_hit_vs_miss'].values
    if len(p_whisker) > 0 and not np.isnan(p_whisker[0]):
        y_max = df_plot_p3[df_plot_p3['Condition'].isin(['W Hit', 'W Miss'])]['Adjustment'].max()
        ax.plot([0, 1], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
        ax.text(0.5, y_max + 0.03, f'p={p_whisker[0]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # No-stim comparison
    p_nostim = error_stats_r_plus['p_nostim_cr_vs_fa'].values
    if len(p_nostim) > 0 and not np.isnan(p_nostim[0]):
        y_max = df_plot_p3[df_plot_p3['Condition'].isin(['NS CR', 'NS FA'])]['Adjustment'].max()
        ax.plot([2, 3], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
        ax.text(2.5, y_max + 0.03, f'p={p_nostim[0]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Previous Trial Type', fontsize=11)
ax.set_ylabel('Mean |Decision Value Adjustment|', fontsize=11)
ax.set_title('Neural: R+ Group', fontsize=12, fontweight='bold')

# Panel 4 (bottom-right): Decision value adjustment after different trial types for R-
ax = axes[1, 1]

sub = df_mouse_adj[df_mouse_adj['reward_group'] == 'R-']
color = reward_palette[0]

plot_data_p4 = []
for cond in conditions:
    col_name = f'mean_adj_after_{trial_type_map[cond]}'
    if col_name in sub.columns:
        for idx, row in sub.iterrows():
            if not np.isnan(row[col_name]):
                plot_data_p4.append({
                    'Condition': cond,
                    'Adjustment': row[col_name],
                    'mouse_id': row['mouse_id']
                })

df_plot_p4 = pd.DataFrame(plot_data_p4)

if len(df_plot_p4) > 0:
    sns.barplot(data=df_plot_p4, x='Condition', y='Adjustment', order=conditions,
               color=color, alpha=0.7, errorbar=('ci', 95), capsize=0.1,
               err_kws={'linewidth': 2}, ax=ax)
    
    sns.stripplot(data=df_plot_p4, x='Condition', y='Adjustment', order=conditions,
                 color=color, alpha=0.4, size=6, ax=ax)

# Add p-values
error_stats_r_minus = df_error_stats[df_error_stats['reward_group'] == 'R-']
if len(error_stats_r_minus) > 0:
    # Whisker comparison
    p_whisker = error_stats_r_minus['p_whisker_hit_vs_miss'].values
    if len(p_whisker) > 0 and not np.isnan(p_whisker[0]):
        y_max = df_plot_p4[df_plot_p4['Condition'].isin(['W Hit', 'W Miss'])]['Adjustment'].max()
        ax.plot([0, 1], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
        ax.text(0.5, y_max + 0.03, f'p={p_whisker[0]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # No-stim comparison
    p_nostim = error_stats_r_minus['p_nostim_cr_vs_fa'].values
    if len(p_nostim) > 0 and not np.isnan(p_nostim[0]):
        y_max = df_plot_p4[df_plot_p4['Condition'].isin(['NS CR', 'NS FA'])]['Adjustment'].max()
        ax.plot([2, 3], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
        ax.text(2.5, y_max + 0.03, f'p={p_nostim[0]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Previous Trial Type', fontsize=11)
ax.set_ylabel('Mean |Decision Value Adjustment|', fontsize=11)
ax.set_title('Neural: R- Group', fontsize=12, fontweight='bold')

plt.tight_layout()
sns.despine()

sns.despine()

plt.savefig(os.path.join(output_dir, 'error_driven_analysis.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'error_driven_analysis.png'), format='png', dpi=300)
print("Saved: error_driven_analysis figure")

print("\n" + "="*80)
print("ERROR-DRIVEN ANALYSIS COMPLETE")
print("="*80 + "\n")
# ================================================

