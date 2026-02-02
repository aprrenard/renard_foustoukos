"""
Raster plot of single-cell PSTH activity across learning days.

Displays average activity for 30 cells (15 best LMI+ and 15 best LMI-)
to illustrate single-cell responses alongside grand average PSTHs.
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

n_cells_per_group = 10


# #############################################################################
# Load mice list.
# #############################################################################

_, _, mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes',
    experimenters=['AR', 'GF', 'MI']
)


# #############################################################################
# Load LMI data.
# #############################################################################

lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

# Pre-select candidate cells: top LMI+ and bottom LMI- (take more than needed for filtering)
n_candidates = 100  # Take more candidates to filter by peak response
lmi_positive_candidates = lmi_df.nlargest(n_candidates, 'lmi')[['mouse_id', 'roi', 'lmi']]
lmi_negative_candidates = lmi_df.nsmallest(n_candidates, 'lmi')[['mouse_id', 'roi', 'lmi']]
candidate_cells = pd.concat([lmi_positive_candidates, lmi_negative_candidates])


# #############################################################################
# Load activity data for candidate cells and compute peak responses.
# #############################################################################

mice_with_candidates = candidate_cells['mouse_id'].unique()
candidate_data = []

print("Loading data for candidate cells...")
for mouse_id in mice_with_candidates:
    mouse_cells = candidate_cells[candidate_cells['mouse_id'] == mouse_id]
    rois = mouse_cells['roi'].values

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)

    xarr = xarr.sel(trial=xarr['day'].isin(days))
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))
    xarr = xarr.groupby('day').mean(dim='trial')

    xarr.name = 'psth'
    df = xarr.to_dataframe().reset_index()
    df['mouse_id'] = mouse_id
    df = df[df['roi'].isin(rois)]
    candidate_data.append(df)

candidate_data = pd.concat(candidate_data)

# Compute peak response for each cell (max across all days and time points)
# Convert to percent dF/F
candidate_data['psth_pct'] = candidate_data['psth'] * 100

peak_responses = candidate_data.groupby(['mouse_id', 'roi'])['psth_pct'].max().reset_index()
peak_responses.columns = ['mouse_id', 'roi', 'peak_response']

# Merge peak responses with LMI data
candidate_cells = candidate_cells.merge(peak_responses, on=['mouse_id', 'roi'])

# Filter cells with peak response between 100% and 200% dF/F
peak_min, peak_max = 100, 200
filtered_cells = candidate_cells[
    (candidate_cells['peak_response'] >= peak_min) &
    (candidate_cells['peak_response'] <= peak_max)
]

print(f"Cells with peak response in [{peak_min}, {peak_max}]% dF/F: {len(filtered_cells)}")


# #############################################################################
# Select top cells from filtered candidates.
# #############################################################################

# Select top 15 LMI+ cells from filtered
lmi_positive = filtered_cells[filtered_cells['lmi'] > 0].nlargest(n_cells_per_group, 'lmi')
lmi_positive['lmi_group'] = 'LMI+'

# Select top 15 LMI- cells from filtered
lmi_negative = filtered_cells[filtered_cells['lmi'] < 0].nsmallest(n_cells_per_group, 'lmi')
lmi_negative['lmi_group'] = 'LMI-'

# Combine selected cells: LMI+ on top, LMI- on bottom
selected_cells = pd.concat([
    lmi_positive.sort_values('lmi', ascending=False),  # Highest first
    lmi_negative.sort_values('lmi', ascending=True)    # Lowest first
]).reset_index(drop=True)

print(f"Selected {len(selected_cells)} cells:")
print(f"  LMI+ range: {lmi_positive['lmi'].min():.3f} to {lmi_positive['lmi'].max():.3f}")
print(f"  LMI- range: {lmi_negative['lmi'].min():.3f} to {lmi_negative['lmi'].max():.3f}")
print(f"  Peak response range: {selected_cells['peak_response'].min():.1f} to {selected_cells['peak_response'].max():.1f}% dF/F")


# #############################################################################
# Filter activity data for selected cells (reuse candidate_data).
# #############################################################################

# Filter candidate_data to keep only selected cells
cell_data = candidate_data.merge(
    selected_cells[['mouse_id', 'roi', 'lmi', 'lmi_group']],
    on=['mouse_id', 'roi']
)


# #############################################################################
# Prepare data matrix for plotting.
# #############################################################################

# Create cell identifier for ordering
cell_data['cell_id'] = cell_data['mouse_id'] + '_' + cell_data['roi'].astype(str)

# Get ordered cell list (matching selected_cells order)
selected_cells['cell_id'] = selected_cells['mouse_id'] + '_' + selected_cells['roi'].astype(str)
cell_order = selected_cells['cell_id'].tolist()

# Get time points
time_points = np.sort(cell_data['time'].unique())

# Create data matrices for each day (cells x time)
raster_data = {}
for day in days:
    day_data = cell_data[cell_data['day'] == day]

    # Pivot to get cells x time matrix
    pivot = day_data.pivot_table(
        index='cell_id',
        columns='time',
        values='psth',
        aggfunc='mean'
    )

    # Reorder rows to match cell_order
    pivot = pivot.reindex(cell_order)

    # Convert to percent DF/F0
    raster_data[day] = pivot.values * 100

# Determine common color scale
all_values = np.concatenate([raster_data[d].flatten() for d in days])
vmax = 130
vmin = 0


# #############################################################################
# Create figure with raster plots.
# #############################################################################

# A4 width is ~8.27 inches, use slightly less for margins
fig_width = 7.5
fig_height = 2.5

fig = plt.figure(figsize=(fig_width, fig_height))

# Create grid: 5 day panels + 1 narrow colorbar panel
gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.08)

axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
cbar_ax = fig.add_subplot(gs[0, 5])

# Plot rasters for each day
for i, day in enumerate(days):
    ax = axes[i]

    im = ax.imshow(
        raster_data[day],
        aspect='auto',
        cmap='Greys',
        vmin=vmin,
        vmax=vmax,
        extent=[time_points[0], time_points[-1], len(cell_order), 0],
        interpolation='nearest'
    )

    # Add stimulus onset line
    ax.axvline(0, color='#FF9600', linestyle='-', linewidth=1)

    # Add horizontal separator between LMI+ and LMI- cells
    ax.axhline(n_cells_per_group, color='black', linestyle='-', linewidth=0.5)

    # Title
    ax.set_title(days_str[i], fontsize=10)

    # X-axis
    if i == 2:  # Middle panel
        ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
    ax.tick_params(axis='x', labelsize=8)

    # Y-axis: only show on leftmost panel
    if i == 0:
        ax.set_ylabel('Cells', fontsize=9)
        ax.set_yticks([n_cells_per_group / 2, n_cells_per_group + n_cells_per_group / 2])
        ax.set_yticklabels(['LMI+', 'LMI-'], fontsize=8)
    else:
        ax.set_yticks([])

# Add colorbar
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('DF/F0 (%)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()


# #############################################################################
# Save figure.
# #############################################################################

output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)

svg_file = 'psth_raster_lmi_cells.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
print(f"Saved: {os.path.join(output_dir, svg_file)}")


# #############################################################################
# Second figure: Single session raster (all cells ordered by LMI).
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
fig2 = plt.figure(figsize=(7.5, 4))
gs2 = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.08)

axes2 = [fig2.add_subplot(gs2[0, i]) for i in range(5)]
cbar_ax2 = fig2.add_subplot(gs2[0, 5])

# Plot rasters for each day
for i, day in enumerate(days):
    ax = axes2[i]

    im2 = ax.imshow(
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
cbar2 = fig2.colorbar(im2, cax=cbar_ax2)
cbar2.set_label('DF/F0 (%)', fontsize=9)
cbar2.ax.tick_params(labelsize=8)

plt.tight_layout()

# Save
sig_suffix = '_significant' if use_significant_only else '_all'
svg_file2 = f'psth_raster_single_session_{session_mouse_id}{sig_suffix}.svg'
plt.savefig(os.path.join(output_dir, svg_file2), format='svg', dpi=300)
print(f"Saved: {os.path.join(output_dir, svg_file2)}")

plt.show()
