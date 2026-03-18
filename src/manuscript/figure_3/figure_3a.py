"""
FOV with LMI overlay + calcium transient illustration for the same mouse.

Left panel : mean image overlaid with per-cell LMI values (custom blue→grey→red
             colormap) with ROI numbers annotated for the cells shown in the
             transient panel.
Right panel: concatenated single-trial traces for a fixed set of ROIs (in the
             specified plotting order), with ROI labels on the y-axis.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging
from nwb_wrappers.nwb_reader_functions import get_image_mask


# #############################################################################
# Parameters.
# #############################################################################

MOUSE_ID        = 'GF314'
NWB_FILE        = os.path.join(io.nwb_dir, 'GF314_28112020_171800.nwb')
OPS_PATH        = io.adjust_path_to_host(os.path.join(io.processed_dir, 'GF314_ops.npy'))

sampling_rate   = 30
day_for_trials  = 2
trials_range    = (10, 16)   # (start, stop) indices into the day's trials
SELECTED_ROIS   = [192, 105, 10, 189, 186, 50, 3, 190, 48]  # plotting order
nan_gap         = 60         # NaN frames inserted between trials
offset_step     = 400        # % dF/F vertical offset between cells

file_name = 'tensor_xarray_mapping_data.nc'
folder    = os.path.join(io.processed_dir, 'mice')

OUTPUT_DIR = io.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/manuscript/outputs/figure_3/output'
)


# #############################################################################
# Custom diverging colormap: bright blue → dark → bright red.
# #############################################################################

lmi_cmap = mcolors.LinearSegmentedColormap.from_list(
    'blue_grey_red',
    [
        (0.0,  (0.0,  0.0, 1.0)),   # bright blue
        (0.5,  (0.7, 0.7, 0.7)),  # mid-grey centre
        (1.0,  (1.0,  0.0,  0.0)),   # bright red
    ]
)

# lmi_cmap = plt.cm.bwr

# #############################################################################
# Load FOV data.
# #############################################################################

lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
lmi_df = lmi_df[lmi_df['mouse_id'] == MOUSE_ID].reset_index(drop=True)
print(f"LMI entries for {MOUSE_ID}: {len(lmi_df)}")

image_masks  = get_image_mask(NWB_FILE)
ops          = np.load(OPS_PATH, allow_pickle=True)
mean_img     = ops.item()['meanImg']

lmi_values   = lmi_df['lmi'].values
roi_indices  = lmi_df['roi'].values.astype(int)

lmi_abs_max  = np.nanmax(np.abs(lmi_values))
norm         = mcolors.Normalize(vmin=-lmi_abs_max, vmax=lmi_abs_max)

overlay = np.zeros((*mean_img.shape, 4), dtype=float)
for roi, lmi_val in zip(roi_indices, lmi_values):
    if np.isnan(lmi_val):
        continue
    mask   = image_masks[roi]
    rgba   = lmi_cmap(norm(lmi_val))
    overlay[mask > 0] = rgba


# #############################################################################
# Load calcium data and select cells by ROI.
# #############################################################################

xarr     = utils_imaging.load_mouse_xarray(MOUSE_ID, folder, file_name, substracted=False)
xarr_day = xarr.sel(trial=xarr['day'] == day_for_trials)

if xarr_day.sizes['trial'] < trials_range[1]:
    raise ValueError(
        f"Not enough trials: {xarr_day.sizes['trial']} < {trials_range[1]}"
    )

trial_indices = xarr_day['trial'][trials_range[0]:trials_range[1]].values
n_trials_plot = len(trial_indices)

# Map ROI id → positional index in the xarray cell dimension
roi_to_idx = {int(r): i for i, r in enumerate(xarr_day['cell'].values)}
top_cells   = pd.DataFrame({
    'roi':      SELECTED_ROIS,
    'cell_idx': [roi_to_idx[r] for r in SELECTED_ROIS],
})

print(f"Plotting {len(top_cells)} selected ROIs (day {day_for_trials}):")
for i, (_, cell) in enumerate(top_cells.iterrows()):
    print(f"  {i+1}. ROI {int(cell['roi'])}")


# #############################################################################
# Build concatenated traces.
# #############################################################################

concatenated_traces = []
time_vec            = None

for _, cell in top_cells.iterrows():
    cell_idx   = int(cell['cell_idx'])
    xarr_cell  = xarr_day.isel(cell=cell_idx).sel(trial=trial_indices)
    traces     = xarr_cell.values * 100   # trials × time → % dF/F

    n_trials, n_timepoints_per_trial = traces.shape
    parts = []
    for t in range(n_trials):
        parts.append(traces[t, :])
        if t < n_trials - 1:
            parts.append(np.full(nan_gap, np.nan))
    concatenated_traces.append(np.concatenate(parts))

    if time_vec is None:
        time_vec       = xarr_cell['time'].values
        n_timepoints   = n_timepoints_per_trial

trial_duration = n_timepoints / sampling_rate
gap_duration   = nan_gap / sampling_rate

time_parts = []
for t in range(n_trials_plot):
    time_parts.append(time_vec + t * (trial_duration + gap_duration))
    if t < n_trials_plot - 1:
        time_parts.append(np.full(nan_gap, np.nan))
t_full = np.concatenate(time_parts)


# #############################################################################
# Figure layout.
# #############################################################################

fig = plt.figure(figsize=(12, 7))
gs  = gridspec.GridSpec(
    1, 3,
    figure=fig,
    width_ratios=[10, 0.4, 4],
    wspace=0.08,
)

ax_fov   = fig.add_subplot(gs[0, 0])
ax_cbar  = fig.add_subplot(gs[0, 1])
ax_trans = fig.add_subplot(gs[0, 2])


# ---- FOV panel ----

ax_fov.imshow(mean_img, cmap='gray', interpolation='none')
ax_fov.imshow(overlay,  interpolation='none', alpha=1)
ax_fov.axis('off')

# Colorbar
sm = plt.cm.ScalarMappable(cmap=lmi_cmap, norm=norm)
cb = fig.colorbar(sm, cax=ax_cbar, label='LMI')
cb.set_ticks([-lmi_abs_max, 0, lmi_abs_max])
cb.set_ticklabels([f'{-lmi_abs_max:.2f}', '0', f'{lmi_abs_max:.2f}'])

# Annotate selected cells on FOV image
selected_roi_ids = top_cells['roi'].values.astype(int)
for roi_id in selected_roi_ids:
    # Find this ROI in the lmi_df to get its mask index
    mask = image_masks[roi_id]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        continue
    cx, cy = xs.mean(), ys.mean()
    ax_fov.text(
        cx, cy, str(roi_id),
        color='white', fontsize=7, fontweight='bold',
        ha='center', va='center',
    )


# ---- Calcium transients panel ----

roi_labels = [str(int(r)) for r in top_cells['roi'].values]

for i, trace in enumerate(concatenated_traces):
    ax_trans.plot(t_full, trace + i * offset_step, color='black', linewidth=0.5)

# Stimulus onset lines
for t in range(n_trials_plot):
    stim_time = t * (trial_duration + gap_duration)
    ax_trans.axvline(stim_time, color='#FF9600', linestyle='-', linewidth=0.8, alpha=0.7)

# ROI labels on y-axis
tick_positions = [i * offset_step for i in range(len(top_cells))]
ax_trans.set_yticks(tick_positions)
ax_trans.set_yticklabels(roi_labels, fontsize=7)
ax_trans.set_ylabel('ROI', fontsize=9)
ax_trans.set_xlabel('Time (s)', fontsize=9)
ax_trans.tick_params(axis='x', labelsize=8)
ax_trans.spines['top'].set_visible(False)
ax_trans.spines['right'].set_visible(False)


# #############################################################################
# Save.
# #############################################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f'fov_and_calcium_transients_{MOUSE_ID}.svg')
fig.savefig(out_path, format='svg', dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_path}")

plt.show()
