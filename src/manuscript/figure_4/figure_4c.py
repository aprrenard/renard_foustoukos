"""
Figure 4c: Progressive learning during Day 0 — behaviour and decoder value,
per mouse (multi-page PDF).

One page per mouse, two-row layout:
  Row 1: Behavioural learning curve across Day 0 whisker trials.
  Row 2: Decoder decision value applied to Day 0 whisker trials using a
         sliding window.

Decoder weights (trained on Days -2/-1 vs +1/+2 mapping trials) are loaded
from RESULTS_DIR/decoder_weights.pkl, produced by figure_3m_o.py.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

win = (0, 0.300)          # response window from stimulus onset (seconds)
window_size = 10
step_size = 1
cut_n_trials = 100

RESULTS_DIR = os.path.join(io.processed_dir, 'decoding')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_4', 'output')


# ============================================================================
# Load decoder weights
# ============================================================================

weights_path = os.path.join(RESULTS_DIR, 'decoder_weights.pkl')
with open(weights_path, 'rb') as f:
    weights = pickle.load(f)
print(f"Loaded decoder weights for {len(weights)} mice.")


# ============================================================================
# Load behaviour and Day-0 learning data
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                        'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1)]

folder = os.path.join(io.processed_dir, 'mice')
xarrays_learning = {}
for mouse in weights:
    xarr = utils_imaging.load_mouse_xarray(mouse, folder, 'tensor_xarray_learning_data.nc')
    xarr = xarr.sel(trial=xarr['day'].isin([0]))
    xarr = xarr.sel(trial=xarr['whisker_stim'] == 1)
    xarr = xarr.sel(time=slice(win[0], win[1])).mean(dim='time')
    xarr = xarr.fillna(0)
    xarrays_learning[mouse] = xarr


# ============================================================================
# Apply decoder (sliding window)
# ============================================================================

def apply_decoder(weights, xarr, mouse, window_size=10, step_size=1):
    w = weights[mouse]
    scaler, clf, sign_flip = w['scaler'], w['clf'], w['sign_flip']
    n_trials = xarr.sizes['trial']
    rows = []
    for start_idx in range(0, max(0, n_trials - window_size + 1), step_size):
        end_idx = start_idx + window_size
        X_win = xarr.values[:, start_idx:end_idx].T
        if X_win.shape[0] == 0:
            continue
        dec_vals = clf.decision_function(scaler.transform(X_win))
        rows.append({
            'mouse_id': mouse,
            'trial_start': start_idx,
            'trial_center': start_idx + window_size // 2,
            'mean_decision_value': np.mean(dec_vals) * sign_flip,
            'reward_group': w['reward_group'],
        })
    return pd.DataFrame(rows)


results_all = pd.concat(
    [apply_decoder(weights, xarrays_learning[m], m, window_size, step_size)
     for m in weights],
    ignore_index=True,
)


# ============================================================================
# Plotting helpers
# ============================================================================

def plot_behavior(ax, data, color, title):
    if data is None or data.empty:
        ax.set_title(title + ' (no data)')
        ax.set_xlim(0, cut_n_trials)
        ax.set_ylim(0, 1)
        return
    sns.lineplot(data=data, x='trial_w', y='learning_curve_w',
                 color=color, errorbar=None, ax=ax)
    ax.set_xlabel('Trial within Day 0')
    ax.set_ylabel('Learning curve (w)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, cut_n_trials)


def plot_decision(ax, data, color, title, ylim=(-5, 3)):
    if data is None or data.empty:
        ax.set_title(title + ' (no data)')
        ax.set_xlim(0, cut_n_trials)
        return
    sns.lineplot(data=data, x='trial_center', y='mean_decision_value',
                 color=color, errorbar=None, ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trial within Day 0')
    ax.set_ylabel('Decision value')
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim(0, cut_n_trials)


# ============================================================================
# Figure — one page per mouse
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, 'figure_4c.pdf')

with PdfPages(out_path) as pdf:
    for mouse in results_all['mouse_id'].unique():
        mouse_data = results_all[results_all['mouse_id'] == mouse]
        mouse_bh = bh_df[bh_df['mouse_id'] == mouse]
        reward_group = mouse_data['reward_group'].iloc[0]
        color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        plot_behavior(ax1, mouse_bh, color, f'{mouse} ({reward_group}) — Behaviour')
        plot_decision(ax2, mouse_data, color, f'{mouse} ({reward_group}) — Decoder value')

        plt.tight_layout()
        sns.despine()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"\nSaved: {out_path}")
