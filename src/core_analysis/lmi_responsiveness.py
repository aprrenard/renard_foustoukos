"""
Stimulus responsiveness across days for LMI-defined cell populations.

This script computes and plots the proportion of cells significantly encoding
the stimulus (based on ROC analysis) across learning days, separately for:
  - All cells
  - Positive LMI cells (lmi_p >= 0.975)
  - Negative LMI cells (lmi_p <= 0.025)

Output: SVG figures and CSV data files saved to the decoding output directory.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging
from src.utils.utils_plot import *
from src.utils.utils_behavior import *


# =============================================================================
# PARAMETERS
# =============================================================================

LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025
ROC_UPPER_THRESHOLD = 0.975
ROC_LOWER_THRESHOLD = 0.025

OUTPUT_DIR = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'


# =============================================================================
# LOAD DATA
# =============================================================================

output_dir = io.adjust_path_to_host(OUTPUT_DIR)

roc_df = pd.read_csv(os.path.join(io.processed_dir, 'roc_stimvsbaseline_results.csv'))
lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

roc_df = io.add_reward_col_to_df(roc_df)


def mark_significant(df):
    df = df.copy()
    df['significant'] = (df['roc_p'] <= ROC_LOWER_THRESHOLD) | (df['roc_p'] >= ROC_UPPER_THRESHOLD)
    return df


def compute_proportion(df):
    return df.groupby(['mouse_id', 'day', 'reward_group'], as_index=False)['significant'].mean()


def plot_proportion(prop_df, title_suffix, ax_left, ax_right):
    sns.barplot(x='day', y='significant',
                data=prop_df[prop_df['reward_group'] == 'R+'],
                estimator=np.mean, color=reward_palette[1], ax=ax_left)
    ax_left.set_title(f'R+ ({title_suffix})')
    ax_left.set_ylim(0, 1)
    ax_left.set_xlabel('Day')
    ax_left.set_ylabel('Proportion Significant')

    sns.barplot(x='day', y='significant',
                data=prop_df[prop_df['reward_group'] == 'R-'],
                estimator=np.mean, color=reward_palette[0], ax=ax_right)
    ax_right.set_title(f'R- ({title_suffix})')
    ax_right.set_ylim(0, 1)
    ax_right.set_xlabel('Day')
    ax_right.set_ylabel('Proportion Significant')


def save_outputs(prop_df, svg_name, csv_name):
    plt.savefig(os.path.join(output_dir, svg_name), format='svg', dpi=300)
    prop_df.to_csv(os.path.join(output_dir, csv_name), index=False)


# =============================================================================
# ALL CELLS
# =============================================================================

roc_all = mark_significant(roc_df)
prop_all = compute_proportion(roc_all)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plot_proportion(prop_all, 'All Cells', axes[0], axes[1])
sns.despine()
save_outputs(
    prop_all,
    'proportion_significant_cells_encoding_stimulus_all_cells.svg',
    'proportion_significant_cells_encoding_stimulus_all_cells.csv',
)


# =============================================================================
# POSITIVE LMI CELLS
# =============================================================================

pos_cells = lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD, ['mouse_id', 'roi']]
roc_pos = mark_significant(roc_df.merge(pos_cells, on=['mouse_id', 'roi']))
prop_pos = compute_proportion(roc_pos)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plot_proportion(prop_pos, 'Positive LMI', axes[0], axes[1])
sns.despine()
save_outputs(
    prop_pos,
    'proportion_significant_cells_encoding_stimulus_positive_lmi.svg',
    'proportion_significant_cells_encoding_stimulus_positive_lmi.csv',
)


# =============================================================================
# NEGATIVE LMI CELLS
# =============================================================================

neg_cells = lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD, ['mouse_id', 'roi']]
roc_neg = mark_significant(roc_df.merge(neg_cells, on=['mouse_id', 'roi']))
prop_neg = compute_proportion(roc_neg)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plot_proportion(prop_neg, 'Negative LMI', axes[0], axes[1])
sns.despine()
save_outputs(
    prop_neg,
    'proportion_significant_cells_encoding_stimulus_negative_lmi.svg',
    'proportion_significant_cells_encoding_stimulus_negative_lmi.csv',
)


# =============================================================================
# PSTH FOR TOP 5 POSITIVE LMI CELLS NOT RESPONSIVE ON DAY +2
# =============================================================================
# Cells with positive LMI have increased mapping trial responses post-learning,
# yet some show no significant ROC discrimination of baseline vs stimulus on
# day +2. We pick the 5 strongest such cells to inspect their PSTHs.

# Identify positive LMI cells that are NOT stimulus-significant on day +2.
roc_day2 = roc_df[roc_df['day'] == 2].copy()
roc_day2['significant'] = (
    (roc_day2['roc_p'] <= ROC_LOWER_THRESHOLD) |
    (roc_day2['roc_p'] >= ROC_UPPER_THRESHOLD)
)
not_resp_day2 = roc_day2.loc[~roc_day2['significant'], ['mouse_id', 'roi', 'roc', 'roc_p']]

pos_lmi = lmi_df.loc[
    lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD,
    ['mouse_id', 'roi', 'lmi', 'lmi_p']
]

top5 = (
    pos_lmi
    .merge(not_resp_day2, on=['mouse_id', 'roi'])
    .nlargest(20, 'lmi')
    .reset_index(drop=True)
)
print("Top 5 positive LMI cells not responsive on day +2:")
print(top5.to_string())

# Plot mapping PSTH for each cell across days.
# Two rows: top = baseline-subtracted, bottom = raw fluorescence.
WIN_SEC = (-0.5, 1.5)
DAYS = [-2, -1, 0, 1, 2]
folder = os.path.join(io.processed_dir, 'mice')


def _plot_psth_row(axes, xarr, days, color='k'):
    """Plot individual trials + mean PSTH into a row of axes."""
    for j, day in enumerate(days):
        ax = axes[j]
        day_data = xarr.sel(trial=xarr['day'] == day)
        if day_data.sizes['trial'] == 0:
            ax.set_title(f'Day {day:+d}\n(no data)')
            continue
        time = day_data.time.values
        for t in range(day_data.sizes['trial']):
            ax.plot(time, day_data.isel(trial=t).squeeze().values * 100,
                    color='gray', alpha=0.2, linewidth=0.5)
        mean_trace = day_data.mean(dim='trial').squeeze().values * 100
        ax.plot(time, mean_trace, color=color, linewidth=1.5)
        ax.axvline(0, color='#FF9600', linestyle='-', linewidth=1)
        ax.set_title(f'Day {day:+d}')
        ax.set_xlabel('Time (s)')


for _, row in top5.iterrows():
    mouse_id = row['mouse_id']
    roi = int(row['roi'])

    xarr_sub = utils_imaging.load_mouse_xarray(
        mouse_id, folder, 'tensor_xarray_mapping_data.nc', substracted=True
    )
    xarr_raw = utils_imaging.load_mouse_xarray(
        mouse_id, folder, 'tensor_xarray_mapping_data.nc', substracted=False
    )
    xarr_sub = xarr_sub.sel(cell=xarr_sub['roi'].isin([roi])).sel(time=slice(*WIN_SEC))
    xarr_raw = xarr_raw.sel(cell=xarr_raw['roi'].isin([roi])).sel(time=slice(*WIN_SEC))

    fig, axes = plt.subplots(2, len(DAYS), figsize=(15, 6), sharey='row')
    _plot_psth_row(axes[0], xarr_sub, DAYS)
    _plot_psth_row(axes[1], xarr_raw, DAYS)

    axes[0, 0].set_ylabel('ΔF/F₀ (%) — baseline sub.')
    axes[1, 0].set_ylabel('ΔF/F₀ (%) — raw')
    fig.suptitle(
        f'{mouse_id} — ROI {roi} | '
        f'LMI = {row["lmi"]:.2f} (p = {row["lmi_p"]:.3f}) | '
        f'ROC day+2 = {row["roc"]:.2f} (p = {row["roc_p"]:.3f})'
    )
    plt.tight_layout()
    sns.despine()
    plt.show()
