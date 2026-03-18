"""
Figure 4b: Example mice — behaviour and decoder decision value during Day 0.

Two-column layout (one example R+ mouse, one example R- mouse):
  Row 1: Behavioural learning curve across Day 0 whisker trials.
  Row 2: Decoder decision value applied to Day 0 whisker trials.

Decoder weights (trained on Days -2/-1 vs +1/+2 mapping trials) are loaded
from RESULTS_DIR/decoder_weights.pkl, produced by figure_3m_o.py.

Set EXAMPLE_MOUSE_RPLUS and EXAMPLE_MOUSE_RMINUS to the desired mouse IDs.
Use figure_4c.pdf to browse all mice and pick representative examples.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# Set these to the desired example mouse IDs.
# Browse figure_4c.pdf to pick representative mice.
EXAMPLE_MOUSE_RPLUS = None   # e.g. 'GF314'
EXAMPLE_MOUSE_RMINUS = None  # e.g. 'AR127'

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
# Select example mice
# ============================================================================

if EXAMPLE_MOUSE_RPLUS is None or EXAMPLE_MOUSE_RMINUS is None:
    # Fall back to first available mouse per group from the weights dict.
    rplus_mice = [m for m, w in weights.items() if w['reward_group'] == 'R+']
    rminus_mice = [m for m, w in weights.items() if w['reward_group'] == 'R-']
    if EXAMPLE_MOUSE_RPLUS is None:
        EXAMPLE_MOUSE_RPLUS = rplus_mice[0]
    if EXAMPLE_MOUSE_RMINUS is None:
        EXAMPLE_MOUSE_RMINUS = rminus_mice[0]

example_mice = [EXAMPLE_MOUSE_RPLUS, EXAMPLE_MOUSE_RMINUS]
print(f"Example mice: R+ = {EXAMPLE_MOUSE_RPLUS}, R- = {EXAMPLE_MOUSE_RMINUS}")


# ============================================================================
# Load behaviour and Day-0 learning data for example mice
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                        'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1)]

folder = os.path.join(io.processed_dir, 'mice')
xarrays_learning = {}
for mouse in example_mice:
    xarr = utils_imaging.load_mouse_xarray(mouse, folder, 'tensor_xarray_learning_data.nc')
    xarr = xarr.sel(trial=xarr['day'].isin([0]))
    xarr = xarr.sel(trial=xarr['whisker_stim'] == 1)
    xarr = xarr.sel(time=slice(win[0], win[1])).mean(dim='time')
    xarr = xarr.fillna(0)
    xarrays_learning[mouse] = xarr


# ============================================================================
# Apply decoder
# ============================================================================

def apply_decoder(weights, xarr, mouse, window_size=10, step_size=1):
    if mouse not in weights:
        return pd.DataFrame()
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
            'trial_start': start_idx,
            'trial_center': start_idx + window_size // 2,
            'mean_decision_value': np.mean(dec_vals) * sign_flip,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Figure
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=False)

for col, mouse in enumerate(example_mice):
    reward_group = weights[mouse]['reward_group']
    color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]

    bh_mouse = bh_df[bh_df['mouse_id'] == mouse]
    bh_mouse = bh_mouse[bh_mouse['trial_w'] < cut_n_trials]

    dec_df = apply_decoder(weights, xarrays_learning[mouse], mouse,
                           window_size=window_size, step_size=step_size)

    # Row 1: behaviour
    ax_beh = axes[0, col]
    if not bh_mouse.empty:
        sns.lineplot(data=bh_mouse, x='trial_w', y='learning_curve_w',
                     ax=ax_beh, color=color, linewidth=2.5)
    ax_beh.set_ylabel('Performance (whisker trials)')
    ax_beh.set_ylim(0, 1)
    ax_beh.set_xlim(0, cut_n_trials)
    ax_beh.set_title(f'{mouse} ({reward_group})')

    # Row 2: decision value
    ax_dec = axes[1, col]
    if not dec_df.empty and not bh_mouse.empty:
        common_trials = np.intersect1d(dec_df['trial_start'], bh_mouse['trial_w'])
        dec_plot = dec_df.set_index('trial_start').loc[common_trials]
        sns.lineplot(x=common_trials, y=dec_plot['mean_decision_value'],
                     ax=ax_dec, color=color, linewidth=2.5)
    ax_dec.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax_dec.set_ylabel('Decoder decision value')
    ax_dec.set_xlabel('Trial within Day 0')
    ax_dec.set_xlim(0, cut_n_trials)
    ax_dec.set_ylim(-5, 3)

plt.tight_layout(h_pad=2.5)
sns.despine()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, 'figure_4b.svg')
fig.savefig(out_path, format='svg', dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_path}")

plt.show()
