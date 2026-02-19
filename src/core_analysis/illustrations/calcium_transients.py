"""
Calcium transient illustration figure.

Shows concatenated single-trial traces for cells with the best transients,
to illustrate typical calcium dynamics during whisker stimulation.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging


# #############################################################################
# Parameters.
# #############################################################################

sampling_rate = 30
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))

n_cells_to_plot = 60  # Number of cells to display
n_trials_plot = 16  # First 8 whisker trials
day_for_trials = 0


# #############################################################################
# Select single mouse.
# #############################################################################

mouse_id = 'GF314'  # Single mouse to use for illustration


# #############################################################################
# Search for cells with best transients.
# #############################################################################

file_name = 'tensor_xarray_mapping_data.nc'
folder = os.path.join(io.processed_dir, 'mice')

cell_metrics = []

# Load data for single mouse
try:
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)
    # Don't cut time dimension - use full trace

    # Select trials from target day
    xarr_day = xarr.sel(trial=xarr['day'] == day_for_trials)

    if xarr_day.sizes['trial'] < n_trials_plot:
        raise ValueError(f"Not enough trials: {xarr_day.sizes['trial']} < {n_trials_plot}")

    # Take first n_trials_plot trials
    trial_indices = xarr_day['trial'][:n_trials_plot].values
    xarr_trials = xarr_day.sel(trial=trial_indices)

    # Get time indices for response window (0 to 0.5s post-stimulus)
    time_vals = xarr_trials['time'].values
    response_mask = (time_vals >= 0) & (time_vals <= 0.5)

    # Compute simple metric for each cell: mean peak response
    n_cells = xarr_trials.sizes['cell']
    for cell_idx in range(n_cells):
        cell_data = xarr_trials.isel(cell=cell_idx).values  # trials x time
        roi = xarr_trials['cell'].values[cell_idx]

        # Simple criterion: mean peak response across trials
        response_peaks = np.max(cell_data[:, response_mask], axis=1)
        mean_peak = np.mean(response_peaks)

        cell_metrics.append({
            'mouse_id': mouse_id,
            'roi': roi,
            'cell_idx': cell_idx,
            'mean_peak': mean_peak
        })
except Exception as e:
    raise RuntimeError(f"Failed to process {mouse_id}: {e}")

cell_metrics_df = pd.DataFrame(cell_metrics)
print(f"Evaluated {len(cell_metrics_df)} cells from {mouse_id}")


# #############################################################################
# Load LMI values for this mouse.
# #############################################################################

print("\nLoading LMI values...")
lmi_df_path = os.path.join(io.processed_dir, 'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df_path)

# Filter for this mouse and get LMI values
mouse_lmi = lmi_df[lmi_df['mouse_id'] == mouse_id].copy()
print(f"Found {len(mouse_lmi)} cells with LMI values")

# Merge LMI with cell metrics
cell_metrics_df = cell_metrics_df.merge(
    mouse_lmi[['roi', 'lmi']],
    on='roi',
    how='left'
)
cell_metrics_df['abs_lmi'] = cell_metrics_df['lmi'].abs()


# #############################################################################
# Select cells for both plots.
# #############################################################################

# Plot 1: Top cells by mean peak response
n_cells_per_plot = 16
top_cells_transient = cell_metrics_df.nlargest(n_cells_per_plot, 'mean_peak')
print(f"\nPlot 1 - Selected {len(top_cells_transient)} cells with best transients:")
for i, (_, cell) in enumerate(top_cells_transient.iterrows()):
    print(f"  {i+1}. ROI {cell['roi']}, mean_peak={cell['mean_peak']:.2f}")

# Plot 2: Top cells by absolute LMI
top_cells_lmi = cell_metrics_df.dropna(subset=['lmi']).nlargest(n_cells_per_plot, 'abs_lmi')
print(f"\nPlot 2 - Selected {len(top_cells_lmi)} cells with highest |LMI|:")
for i, (_, cell) in enumerate(top_cells_lmi.iterrows()):
    print(f"  {i+1}. ROI {cell['roi']}, LMI={cell['lmi']:.3f}, |LMI|={cell['abs_lmi']:.3f}")

# #############################################################################
# Helper function to create plot.
# #############################################################################

def create_transient_plot(selected_cells, title_suffix):
    """Create a calcium transient plot for selected cells."""
    print(f"\nLoading full traces for {title_suffix}...")

    concatenated_traces = []
    time_vec = None

    for _, cell in selected_cells.iterrows():
        cell_idx = int(cell['cell_idx'])

        # Load xarray (full time dimension - no slicing)
        xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
        xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)

        # Select trials from target day
        xarr_day = xarr.sel(trial=xarr['day'] == day_for_trials)
        trial_indices = xarr_day['trial'][:n_trials_plot].values
        xarr_cell = xarr_day.isel(cell=cell_idx).sel(trial=trial_indices)

        # Get traces and concatenate with NaN padding between trials
        traces = xarr_cell.values * 100  # trials x time, convert to percent dF/F

        # Add NaN padding between trials
        n_trials, n_timepoints_per_trial = traces.shape
        nan_gap = 60  # Number of NaN values between trials

        # Create concatenated trace with gaps
        concatenated_parts = []
        for trial_idx in range(n_trials):
            concatenated_parts.append(traces[trial_idx, :])
            if trial_idx < n_trials - 1:  # Don't add gap after last trial
                concatenated_parts.append(np.full(nan_gap, np.nan))

        concatenated = np.concatenate(concatenated_parts)
        concatenated_traces.append(concatenated)

        if time_vec is None:
            time_vec = xarr_cell['time'].values
            n_timepoints = n_timepoints_per_trial

    # Create figure
    fig, ax = plt.subplots(figsize=(3, 3))

    # Offset between traces for visibility
    offset_step = 400  # Vertical offset between cells in % dF/F

    # Create time vector accounting for NaN gaps
    trial_duration = n_timepoints / sampling_rate
    nan_gap = 60
    gap_duration = nan_gap / sampling_rate

    # Build time vector with gaps
    time_parts = []
    for trial_idx in range(n_trials_plot):
        trial_time = time_vec + trial_idx * (trial_duration + gap_duration)
        time_parts.append(trial_time)
        if trial_idx < n_trials_plot - 1:
            time_parts.append(np.full(nan_gap, np.nan))
    t_full = np.concatenate(time_parts)

    # Plot traces
    for i, trace in enumerate(concatenated_traces):
        offset = i * offset_step
        ax.plot(t_full, trace + offset, color='black', linewidth=0.5)

    # Add vertical lines at stimulus onsets
    for i in range(n_trials_plot):
        stim_time = i * (trial_duration + gap_duration) + abs(time_vec[0])
        ax.axvline(stim_time, color='#FF9600', linestyle='-', linewidth=0.8, alpha=0.7)

    # Styling
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('DF/F0 (%)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig, ax


# #############################################################################
# Create both plots.
# #############################################################################

output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)

# Plot 1: Best transients
fig1, ax1 = create_transient_plot(top_cells_transient, "best transients")
svg_file_1 = f'calcium_transient_traces_{mouse_id}_best_transients.svg'
fig1.savefig(os.path.join(output_dir, svg_file_1), format='svg', dpi=300)
print(f"\nSaved Plot 1: {os.path.join(output_dir, svg_file_1)}")

# Plot 2: Highest LMI
fig2, ax2 = create_transient_plot(top_cells_lmi, "highest |LMI|")
svg_file_2 = f'calcium_transient_traces_{mouse_id}_highest_LMI.svg'
fig2.savefig(os.path.join(output_dir, svg_file_2), format='svg', dpi=300)
print(f"Saved Plot 2: {os.path.join(output_dir, svg_file_2)}")

plt.show()
