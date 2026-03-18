"""
Figure 4e: Correlation between decoder decision value and behavioural
performance across Day-0 whisker trials — R+ vs R-.

For each mouse a Pearson correlation is computed between the sliding-window
mean decision value and the behavioural learning curve, then compared between
reward groups (swarm + point plot, Wilcoxon one-sample test against zero).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon, ttest_1samp, pearsonr, linregress
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

sampling_rate = 30
win = (0, 0.300)          # response window from stimulus onset (seconds)
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
window_size = 10
step_size = 1

RESULTS_DIR = os.path.join(io.processed_dir, 'decoding')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_4', 'output')


# ============================================================================
# Load data
# ============================================================================

_, _, mice, db = io.select_sessions_from_db(io.db_path, io.nwb_dir, two_p_imaging='yes')

bh_path = os.path.join(io.processed_dir, 'behavior',
                        'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1)]

vectors_rew_mapping = []
vectors_nonrew_mapping = []
mice_rew = []
mice_nonrew = []
vectors_rew_day0_learning = []
vectors_nonrew_day0_learning = []

for mouse in mice:
    folder = os.path.join(io.processed_dir, 'mice')
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # --- mapping data ---
    xarray = utils_imaging.load_mouse_xarray(
        mouse, folder, 'tensor_xarray_mapping_data.nc', substracted=True)
    xarray = xarray.sel(trial=xarray['day'].isin(days))

    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}, skipping.')
        continue

    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')
    d = d.fillna(0)

    # --- Day 0 learning data ---
    xarray_l = utils_imaging.load_mouse_xarray(
        mouse, folder, 'tensor_xarray_learning_data.nc')
    xarray_l = xarray_l.sel(trial=xarray_l['day'].isin([0]))
    xarray_l = xarray_l.sel(trial=xarray_l['whisker_stim'] == 1)
    xarray_l = xarray_l.sel(time=slice(win[0], win[1])).mean(dim='time')
    xarray_l = xarray_l.fillna(0)

    if rew_gp == 'R+':
        vectors_rew_mapping.append(d)
        vectors_rew_day0_learning.append(xarray_l)
        mice_rew.append(mouse)
    elif rew_gp == 'R-':
        vectors_nonrew_mapping.append(d)
        vectors_nonrew_day0_learning.append(xarray_l)
        mice_nonrew.append(mouse)


# ============================================================================
# Analysis
# ============================================================================

def progressive_learning_analysis(vectors_mapping, vectors_learning, mice_list,
                                   pre_days=[-2, -1], post_days=[1, 2],
                                   window_size=10, step_size=1, seed=42):
    """
    Train a pre/post decoder on mapping trials and apply it with a sliding
    window to Day-0 learning trials.

    Returns a DataFrame with columns: mouse_id, trial_start, trial_center,
    mean_decision_value.
    """
    results = []

    for d_mapping, d_learning, mouse in zip(vectors_mapping, vectors_learning, mice_list):
        day_per_trial = d_mapping['day'].values
        train_mask = np.isin(day_per_trial, pre_days + post_days)
        if np.sum(train_mask) < 4:
            continue

        X_train = d_mapping.values[:, train_mask].T
        y_train = np.array([0 if day in pre_days else 1 for day in day_per_trial[train_mask]])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        clf.fit(X_train_scaled, y_train)

        # Check sign convention
        pre_mask = np.isin(day_per_trial, pre_days)
        post_mask = np.isin(day_per_trial, post_days)
        mean_dec_pre = np.mean(clf.decision_function(scaler.transform(d_mapping.values[:, pre_mask].T)))
        mean_dec_post = np.mean(clf.decision_function(scaler.transform(d_mapping.values[:, post_mask].T)))
        sign_flip = -1 if mean_dec_pre > mean_dec_post else 1

        # Sliding-window application to Day-0 learning trials
        n_trials = d_learning.sizes['trial']
        for start_idx in range(0, max(0, n_trials - window_size + 1), step_size):
            end_idx = start_idx + window_size
            X_win = d_learning.values[:, start_idx:end_idx].T
            if X_win.shape[0] == 0:
                continue
            decision_values = clf.decision_function(scaler.transform(X_win))
            results.append({
                'mouse_id': mouse,
                'trial_start': start_idx,
                'trial_center': start_idx + window_size // 2,
                'mean_decision_value': np.mean(decision_values) * sign_flip,
            })

    return pd.DataFrame(results)


results_rew = progressive_learning_analysis(
    vectors_rew_mapping, vectors_rew_day0_learning, mice_rew,
    window_size=window_size, step_size=step_size)
results_rew['reward_group'] = 'R+'

results_nonrew = progressive_learning_analysis(
    vectors_nonrew_mapping, vectors_nonrew_day0_learning, mice_nonrew,
    window_size=window_size, step_size=step_size)
results_nonrew['reward_group'] = 'R-'

results_combined = pd.concat([results_rew, results_nonrew], ignore_index=True)

# Per-mouse Pearson correlation between decision value and behaviour
corr_real = []
corr_mice = []
corr_groups = []

for mouse in results_combined['mouse_id'].unique():
    group = results_combined.loc[results_combined['mouse_id'] == mouse, 'reward_group'].iloc[0]
    dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
    bh_mouse = bh_df[bh_df['mouse_id'] == mouse]

    common_trials = np.intersect1d(dec_mouse['trial_start'], bh_mouse['trial_w'])
    if len(common_trials) < 10:
        continue

    dec_vals = dec_mouse.set_index('trial_start').loc[common_trials]['mean_decision_value'].values
    perf_vals = bh_mouse.set_index('trial_w').loc[common_trials]['learning_curve_w'].values

    corr = pearsonr(perf_vals, dec_vals)[0]
    corr_real.append(corr)
    corr_mice.append(mouse)
    corr_groups.append(group)
    print(f"{mouse} ({group}): r={corr:.3f}")

df_corr = pd.DataFrame({
    'mouse_id': corr_mice,
    'reward_group': corr_groups,
    'correlation': corr_real,
})

# Population-level statistics (Wilcoxon one-sample, H0: median correlation <= 0)
pop_stats = {}
pop_stats_rows = []
for group in ['R+', 'R-']:
    sub = df_corr[df_corr['reward_group'] == group]
    if len(sub) >= 3:
        _, p_wilcox = wilcoxon(sub['correlation'].values, alternative='greater')
        _, p_ttest = ttest_1samp(sub['correlation'].values, 0, alternative='greater')
        pop_stats[group] = p_wilcox
        pop_stats_rows.append({
            'reward_group': group,
            'n': len(sub),
            'mean_correlation': np.mean(sub['correlation'].values),
            'std_correlation': np.std(sub['correlation'].values),
            'p_wilcoxon': p_wilcox,
            'p_ttest': p_ttest,
        })
        print(f"{group} (N={len(sub)}): mean r={np.mean(sub['correlation'].values):.3f}, "
              f"Wilcoxon p={p_wilcox:.4f}")

df_pop_stats = pd.DataFrame(pop_stats_rows)


# ============================================================================
# Figure
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(4, 5))

sns.swarmplot(data=df_corr, x='reward_group', y='correlation',
              palette=reward_palette[::-1], size=8, alpha=0.6, ax=ax)
sns.pointplot(data=df_corr, x='reward_group', y='correlation',
              palette=reward_palette[::-1], errorbar='ci', markersize=10,
              join=False, ax=ax)
ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)

# Annotate p-values
for i, group in enumerate(['R+', 'R-']):
    if group in pop_stats:
        p = pop_stats[group]
        p_text = f'p={p:.4f}' if p >= 0.001 else 'p<0.001'
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        y_pos = ax.get_ylim()[1] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(i, y_pos, f'{p_text}\n{sig}', ha='center', va='top',
                fontsize=9, fontweight='bold')

ax.set_ylim(-1, 1)
ax.set_xlabel('Reward group')
ax.set_ylabel('Pearson r\n(Decision value vs Performance)')
sns.despine()
plt.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'figure_4e.svg'), format='svg', dpi=300, bbox_inches='tight')
print(f"Saved: figure_4e.svg")

df_corr.to_csv(os.path.join(OUTPUT_DIR, 'figure_4e_correlations.csv'), index=False)
print(f"Saved: figure_4e_correlations.csv")

df_pop_stats.to_csv(os.path.join(OUTPUT_DIR, 'figure_4e_stats.csv'), index=False)
print(f"Saved: figure_4e_stats.csv")

plt.show()
