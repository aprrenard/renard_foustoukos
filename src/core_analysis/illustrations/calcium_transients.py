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
day_for_trials = 1  # Day +2


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
# Search for cells with best transients.
# #############################################################################

file_name = 'tensor_xarray_mapping_data.nc'
folder = os.path.join(io.processed_dir, 'mice')

cell_metrics = []

for mouse_id in mice[:10]:  # Sample from first 10 mice for speed
    try:
        xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
        xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)
        # Don't cut time dimension - use full trace

        # Select trials from target day
        xarr_day = xarr.sel(trial=xarr['day'] == day_for_trials)

        if xarr_day.sizes['trial'] < n_trials_plot:
            continue

        # Take first n_trials_plot trials
        trial_indices = xarr_day['trial'][:n_trials_plot].values
        xarr_trials = xarr_day.sel(trial=trial_indices)

        # Get time indices for response window (0 to 0.5s post-stimulus)
        time_vals = xarr_trials['time'].values
        response_mask = (time_vals >= 0) & (time_vals <= 0.5)
        baseline_mask = (time_vals >= -0.5) & (time_vals < 0)

        # Compute metrics for each cell
        n_cells = xarr_trials.sizes['cell']
        for cell_idx in range(n_cells):
            cell_data = xarr_trials.isel(cell=cell_idx).values  # trials x time
            roi = xarr_trials['cell'].values[cell_idx]

            # Peak response (mean across trials of max in response window)
            response_peaks = np.max(cell_data[:, response_mask], axis=1)
            mean_peak = np.mean(response_peaks)

            # Baseline std (noise estimate)
            baseline_std = np.std(cell_data[:, baseline_mask])

            # SNR: peak / baseline noise
            snr = mean_peak / (baseline_std + 1e-6)

            # Response reliability: fraction of trials with peak > 2*baseline_std
            threshold = 2 * baseline_std
            reliability = np.mean(response_peaks > threshold)

            # Combined score: prioritize high SNR and reliability
            score = snr * reliability * mean_peak

            cell_metrics.append({
                'mouse_id': mouse_id,
                'roi': roi,
                'cell_idx': cell_idx,
                'mean_peak': mean_peak,
                'snr': snr,
                'reliability': reliability,
                'score': score
            })
    except Exception as e:
        print(f"  Skipping {mouse_id}: {e}")
        continue

cell_metrics_df = pd.DataFrame(cell_metrics)
print(f"Evaluated {len(cell_metrics_df)} cells")

# Select top cells by combined score
top_cells = cell_metrics_df.nlargest(n_cells_to_plot, 'score')
print(f"Selected {len(top_cells)} cells with best transients:")
for i, (_, cell) in enumerate(top_cells.iterrows()):
    print(f"  {i+1}. {cell['mouse_id']} ROI {cell['roi']}, SNR={cell['snr']:.1f}, reliability={cell['reliability']:.2f}")


top_cells = top_cells.iloc[[0,1,2,4,5,6,7,8,11,19,21,26,47,52,58,59]]

# #############################################################################
# Load full traces for selected cells.
# #############################################################################

print("\nLoading full traces for selected cells...")

concatenated_traces = []
time_vec = None

for _, cell in top_cells.iterrows():
    mouse_id = cell['mouse_id']
    cell_idx = int(cell['cell_idx'])

    # Load xarray (full time dimension - no slicing)
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)

    # Select trials from target day
    xarr_day = xarr.sel(trial=xarr['day'] == day_for_trials)
    trial_indices = xarr_day['trial'][:n_trials_plot].values
    xarr_cell = xarr_day.isel(cell=cell_idx).sel(trial=trial_indices)

    # Get traces and concatenate
    traces = xarr_cell.values * 100  # trials x time, convert to percent dF/F
    concatenated = traces.flatten()
    concatenated_traces.append(concatenated)

    if time_vec is None:
        time_vec = xarr_cell['time'].values

n_timepoints = len(time_vec)


# #############################################################################
# Create figure with stacked line plots (10 rows).
# #############################################################################

# Create figure
fig, ax = plt.subplots(figsize=(3, 3))

# Offset between traces for visibility
offset_step = 400  # Vertical offset between cells in % dF/F

for i, trace in enumerate(concatenated_traces):
    offset = i * offset_step
    t = np.arange(len(trace)) / sampling_rate
    ax.plot(t, trace + offset, color='black', linewidth=0.5)

# Add vertical lines at stimulus onsets
trial_duration = n_timepoints / sampling_rate
for i in range(n_trials_plot):
    stim_time = i * trial_duration + abs(time_vec[0])  # Offset by pre-stimulus time
    ax.axvline(stim_time, color='#FF9600', linestyle='-', linewidth=0.8, alpha=0.7)

# Styling
ax.set_xlabel('Time (s)', fontsize=9)
ax.set_ylabel('DF/F0 (%)', fontsize=9)
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()


# #############################################################################
# Save figure.
# #############################################################################

output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)

svg_file = 'calcium_transient_traces.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
print(f"\nSaved: {os.path.join(output_dir, svg_file)}")

plt.show()
