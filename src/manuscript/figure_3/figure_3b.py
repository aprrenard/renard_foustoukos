"""
Raster plot of single-cell PSTH activity across learning days for GF314.

Displays all cells ordered by LMI (significant only) across learning days.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging


# #############################################################################
# Parameters.
# #############################################################################

sampling_rate = 30
win_sec = (-0.5, 1.5)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, 1, 2]
days_str = ['Day -2', 'Day -1', 'Day 0', 'Day +1', 'Day +2']

OUTPUT_DIR = io.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/manuscript/outputs/figure_3/output'
)


# #############################################################################
# Load LMI data.
# #############################################################################

lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))


# #############################################################################
# Single session raster (all cells ordered by LMI).
# #############################################################################

session_mouse_id = 'GF314'
use_significant_only = True  # If True, only include cells with significant LMI

print(f"\nCreating raster for cells from {session_mouse_id} ordered by LMI...")

# Get LMI values for this mouse
mouse_lmi = lmi_df[lmi_df['mouse_id'] == session_mouse_id][['roi', 'lmi', 'lmi_p']].copy()

# Filter to significant LMI cells if requested
if use_significant_only:
    mouse_lmi = mouse_lmi.loc[(mouse_lmi['lmi_p'] >= 0.975) | (mouse_lmi['lmi_p'] <= 0.025)]  # Keep only lmi_p == 1 or lmi_p == -1

mouse_lmi = mouse_lmi.sort_values('lmi', ascending=False)  # Positive on top
print(f"Found {len(mouse_lmi)} cells with LMI data for {session_mouse_id}")

# Load activity data for this mouse
file_name = 'tensor_xarray_mapping_data.nc'
folder = os.path.join(io.processed_dir, 'mice')
xarr = utils_imaging.load_mouse_xarray(session_mouse_id, folder, file_name)
xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)

xarr = xarr.sel(trial=xarr['day'].isin(days))
xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))
xarr = xarr.groupby('day').mean(dim='trial')

xarr.name = 'psth'
session_df = xarr.to_dataframe().reset_index()
session_df['mouse_id'] = session_mouse_id

# Merge with LMI data to get ordering
session_df = session_df.merge(mouse_lmi, on='roi')

# Create cell identifier for ordering
session_df['cell_id'] = session_df['roi'].astype(str)

# Get ordered cell list (by LMI, descending)
session_cell_order = mouse_lmi['roi'].astype(str).tolist()

# Get time points
session_time_points = np.sort(session_df['time'].unique())

# Create data matrices for each day (cells x time)
session_raster_data = {}
for day in days:
    day_data = session_df[session_df['day'] == day]

    pivot = day_data.pivot_table(
        index='cell_id',
        columns='time',
        values='psth',
        aggfunc='mean'
    )

    # Reorder rows by LMI (positive on top)
    pivot = pivot.reindex(session_cell_order)

    # Convert to percent DF/F0
    session_raster_data[day] = pivot.values * 100

# Determine color scale
session_vmax = np.percentile(np.concatenate([session_raster_data[d].flatten() for d in days]), 98)
session_vmin = 0

# Create figure
fig = plt.figure(figsize=(7.5, 4))
gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.08)

axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
cbar_ax = fig.add_subplot(gs[0, 5])

# Plot rasters for each day
for i, day in enumerate(days):
    ax = axes[i]

    im = ax.imshow(
        session_raster_data[day],
        aspect='auto',
        cmap='Reds',
        vmin=session_vmin,
        vmax=session_vmax,
        extent=[session_time_points[0], session_time_points[-1], len(session_cell_order), 0],
        interpolation='nearest'
    )

    # Add stimulus onset line
    ax.axvline(0, color='#FF9600', linestyle='-', linewidth=1.2)

    # Title
    ax.set_title(days_str[i], fontsize=10)

    # X-axis
    if i == 2:
        ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
    ax.tick_params(axis='x', labelsize=8)

    # Y-axis
    if i == 0:
        ax.set_ylabel('Cells (by LMI)', fontsize=9)
    ax.set_yticks([])

# Add colorbar
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('DF/F0 (%)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()


# #############################################################################
# Save.
# #############################################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)
sig_suffix = '_significant' if use_significant_only else '_all'
out_path = os.path.join(OUTPUT_DIR, f'psth_raster_{session_mouse_id}{sig_suffix}.svg')
fig.savefig(out_path, format='svg', dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_path}")

plt.show()
