
"""
This script detects reactivation events during no-stimulus trials and analyzes their
relationship with behavioral performance across days and mice.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr, linregress, friedmanchisquare
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *


# ============================================================================
# PARAMETERS
# ============================================================================

sampling_rate = 30  # Hz
win = (0, 0.300)  # Window for template: stimulus onset to 300ms after
days = [-2, -1, 0, 1, 2]
days_str = ['-2', '-1', '0', '+1', '+2']
n_map_trials = 40  # Number of mapping trials to use

# Template and event detection parameters
threshold_type = 'percentile'  # Options: 'percentile' or 'max' (FWER)
threshold_mode = 'mouse'  # Options: 'mouse' (baseline-derived, same for all days) or 'day' (per-day thresholds)
threshold_dff = None  # 5% dff threshold for including cells in template (use None for all cells)
threshold_corr = 0.45  # Default correlation threshold for event detection (if no surrogate thresholds available)
min_event_distance_ms = 200  # Minimum distance between events (ms)
min_event_distance_frames = int(min_event_distance_ms / 1000 * sampling_rate)
prominence = 0.15  # Minimum prominence of peaks for event detection (vertical distance to contour line)

# NOTE: Surrogate-based thresholds
# If reactivation_surrogates.py or reactivation_surrogates_per_day.py has been run,
# the script will automatically load and use thresholds instead of the fixed threshold_corr value.
#
# threshold_type options:
#   - 'percentile': Uses the percentile specified in surrogate scripts (default: 99.9th)
#   - 'max' (FWER/maximum): More conservative, controls family-wise error rate
#
# threshold_mode options:
#   - 'mouse': One threshold per mouse (from baseline days -2 and -1), applied to all days
#              Loads from: reactivation_surrogatesd/surrogate_thresholds.csv
#   - 'day': Separate threshold per mouse-day combination, computed from each day's data
#            Loads from: reactivation_surrogates_per_day/surrogate_thresholds_per_day.csv

# Visualization parameters
time_per_row = 200  # seconds per row in correlation trace plots (rows calculated dynamically)

# Trial type selection (fixed to no_stim trials only)
trial_type = 'no_stim'

# Analysis mode
mode = 'analyze'  # Options: 'compute' (run analysis and save results) or 'analyze' (load results and generate plots)

# Parallel processing parameters
n_jobs = 35  # Number of parallel jobs for processing mice (set to -1 to use all available cores)

# Load database and available mice
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes'
)

# Separate mice by reward group
r_plus_mice = []
r_minus_mice = []

for mouse in all_mice:
    try:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse, db=db)
        if reward_group == 'R+':
            r_plus_mice.append(mouse)
        elif reward_group == 'R-':
            r_minus_mice.append(mouse)
    except:
        continue

print(f"Found {len(r_plus_mice)} R+ mice: {r_plus_mice}")
print(f"Found {len(r_minus_mice)} R- mice: {r_minus_mice}")

# ============================================================================
# THRESHOLD LOADING FUNCTIONS
# ============================================================================

def load_surrogate_thresholds(surrogate_csv_path, threshold_type='max', threshold_mode='day'):
    """
    Load thresholds from surrogate analysis.

    Parameters
    ----------
    surrogate_csv_path : str
        Path to surrogate threshold CSV file
    threshold_type : str
        Type of threshold to use: 'percentile' for percentile-based, 'max' for FWER/maximum
    threshold_mode : str
        'mouse': One threshold per mouse (from baseline days), applied to all days
        'day': Separate threshold per mouse-day combination

    Returns
    -------
    threshold_dict : dict
        If threshold_mode='mouse': {mouse_id: threshold_value}
        If threshold_mode='day': {mouse_id: {day: threshold_value}}
    """
    if not os.path.exists(surrogate_csv_path):
        raise FileNotFoundError(f"Surrogate threshold file not found: {surrogate_csv_path}")

    df = pd.read_csv(surrogate_csv_path)

    # Select the appropriate threshold column
    if threshold_type in ['percentile', '95']:  # Accept both for backwards compatibility
        threshold_col = 'threshold_percentile_mean'
    elif threshold_type == 'max':
        threshold_col = 'threshold_max_mean'
    else:
        raise ValueError(f"Invalid threshold_type '{threshold_type}'. Must be 'percentile' or 'max'.")

    # Build dictionary based on mode
    threshold_dict = {}

    if threshold_mode == 'mouse':
        # One threshold per mouse (original behavior)
        for _, row in df.iterrows():
            mouse = row['mouse_id']
            threshold = row[threshold_col]
            threshold_dict[mouse] = threshold

    elif threshold_mode == 'day':
        # Separate threshold per mouse-day combination
        for _, row in df.iterrows():
            mouse = row['mouse_id']
            day = int(row['day'])
            threshold = row[threshold_col]

            if mouse not in threshold_dict:
                threshold_dict[mouse] = {}
            threshold_dict[mouse][day] = threshold

    else:
        raise ValueError(f"Invalid threshold_mode '{threshold_mode}'. Must be 'mouse' or 'day'.")

    return threshold_dict


def get_threshold_for_mouse_day(threshold_dict, mouse, day, default_threshold=0.45, threshold_mode='day'):
    """
    Get threshold for a specific mouse and day, with fallback to default.

    Parameters
    ----------
    threshold_dict : dict or None
        Dictionary from load_surrogate_thresholds()
    mouse : str
        Mouse ID
    day : int
        Day number
    default_threshold : float
        Default threshold if mouse/day not found
    threshold_mode : str
        'mouse': Use same threshold for all days (threshold_dict[mouse])
        'day': Use day-specific threshold (threshold_dict[mouse][day])

    Returns
    -------
    threshold : float
        Threshold value to use
    """
    if threshold_dict is None:
        return default_threshold

    if threshold_mode == 'mouse':
        # Mouse-wise threshold (same for all days)
        if mouse in threshold_dict:
            return threshold_dict[mouse]
        else:
            return default_threshold

    elif threshold_mode == 'day':
        # Day-specific threshold
        if mouse in threshold_dict and day in threshold_dict[mouse]:
            return threshold_dict[mouse][day]
        else:
            return default_threshold

    else:
        raise ValueError(f"Invalid threshold_mode '{threshold_mode}'. Must be 'mouse' or 'day'.")



def select_trials_by_type(xarray_day, trial_type='no_stim'):
    """
    Select no-stim trials only.

    Parameters
    ----------
    xarray_day : xarray.DataArray
        Day-specific data with trial metadata
    trial_type : str
        Trial type: only 'no_stim' is supported (parameter kept for backward compatibility)

    Returns
    -------
    selected_trials : xarray.DataArray
        Filtered trials (no-stim only)
    n_trials : int
        Number of trials selected
    """
    # Only no-stim trials are analyzed
    selected = xarray_day.sel(trial=xarray_day['no_stim'] == 1)
    return selected, len(selected.trial)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_whisker_template(mouse, day, threshold_dff=0.05, verbose=True):
    """
    Create whisker response template from mapping data for a specific day.

    Parameters
    ----------
    mouse : str
        Mouse ID
    day : int
        Day number (-2, -1, 0, 1, 2)
    threshold_dff : float or None
        Minimum absolute dff response for including cells (default: 0.05 = 5%).
        If None, all cells are included without filtering.
    verbose : bool
        Print information about template creation

    Returns
    -------
    template : np.ndarray
        Template vector (n_cells,)
    cells_mask : np.ndarray
        Boolean mask indicating which cells pass threshold (all True if threshold_dff is None)
    """
    if verbose:
        print(f"\n  Creating template for {mouse}, Day {day}")

    # Load mapping data
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray_map = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)

    # Select the specific day
    xarray_day = xarray_map.sel(trial=xarray_map['day'] == day)

    # Select last n_map_trials for this day
    xarray_day = xarray_day.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))

    # Average over time window
    d = xarray_day.sel(time=slice(win[0], win[1])).mean(dim='time')

    # Handle NaN values
    d = d.fillna(0)

    # Average across trials to get template
    template = d.mean(dim='trial').values

    # Filter cells by threshold
    if threshold_dff is None:
        # Use all cells, no filtering
        cells_mask = np.ones(len(template), dtype=bool)
        template_filtered = template.copy()
    else:
        # Filter cells by threshold
        cells_mask = template >= threshold_dff
        template_filtered = template.copy()
        template_filtered[~cells_mask] = 0

    if verbose:
        print(f"    Total cells: {len(template)}")
        if threshold_dff is None:
            print(f"    Using ALL cells (no threshold applied)")
            print(f"    Template mean: {np.mean(template_filtered):.4f}")
        else:
            print(f"    Cells above {threshold_dff*100}% threshold: {cells_mask.sum()} ({cells_mask.sum()/len(template)*100:.1f}%)")
            print(f"    Template mean: {np.mean(template_filtered[cells_mask]):.4f}")

    return template_filtered, cells_mask


def compute_template_correlation(data, template):
    """
    Compute correlation between neural activity and template at each timepoint.

    Parameters
    ----------
    data : np.ndarray
        Neural activity data (n_cells, n_timepoints)
    template : np.ndarray
        Template vector (n_cells,)

    Returns
    -------
    correlations : np.ndarray
        Correlation values at each timepoint (n_timepoints,)
    """
    n_cells, n_timepoints = data.shape
    correlations = np.zeros(n_timepoints)

    for t in range(n_timepoints):
        activity = data[:, t]
        # Handle potential NaN or constant values
        if np.std(activity) > 0 and np.std(template) > 0:
            correlations[t] = np.corrcoef(template, activity)[0, 1]
        else:
            correlations[t] = 0

    return correlations


def detect_reactivation_events(correlations, threshold=0.45, min_distance=15, prominence=0.1):
    """
    Detect reactivation events as peaks in correlation timeseries.

    Uses scipy.signal.find_peaks to identify local maxima above threshold,
    ensuring peaks are separated by at least min_distance frames and have
    sufficient prominence.

    Parameters
    ----------
    correlations : np.ndarray
        Correlation timeseries
    threshold : float
        Minimum correlation height for peak detection
    min_distance : int
        Minimum distance between peaks (in frames)
    prominence : float
        Minimum prominence of peaks (vertical distance to contour line).
        Higher values = more selective peak detection.

    Returns
    -------
    event_indices : np.ndarray
        Indices of detected peak events
    """
    # Use find_peaks to detect local maxima above threshold
    peaks, properties = find_peaks(correlations, height=threshold, distance=min_distance, prominence=prominence)

    return peaks


def compute_time_above_threshold(correlations, threshold):
    """
    Compute percentage of time spent above correlation threshold.

    This treats reactivation as a continuous state rather than discrete events,
    quantifying how much time the neural population spends in a "reactivated" state.

    Parameters
    ----------
    correlations : np.ndarray
        Correlation timeseries (n_timepoints,)
    threshold : float
        Correlation threshold

    Returns
    -------
    percent_time_above : float
        Percentage of time spent above threshold (0-100%)
    n_frames_above : int
        Number of frames above threshold
    total_frames : int
        Total number of frames
    """
    above_threshold = correlations > threshold
    n_frames_above = np.sum(above_threshold)
    total_frames = len(correlations)

    if total_frames > 0:
        percent_time_above = (n_frames_above / total_frames) * 100
    else:
        percent_time_above = 0.0

    return percent_time_above, n_frames_above, total_frames


def map_events_to_blocks(event_indices, nostim_trials, n_timepoints_per_trial, sampling_rate=30):
    """
    Map event indices to block IDs and count events per block.

    Parameters
    ----------
    event_indices : np.ndarray
        Indices of detected events in concatenated timeseries
    nostim_trials : xarray.DataArray
        No-stim trial data with block_id coordinate
    n_timepoints_per_trial : int
        Number of timepoints per trial
    sampling_rate : float
        Sampling rate in Hz (default: 30)

    Returns
    -------
    events_per_block : dict
        Dictionary mapping block_id to event count
    event_frequency_per_block : dict
        Dictionary mapping block_id to event frequency (events/min)
    event_blocks : list
        List of block IDs for each event (for plotting)
    """
    # Get block IDs for each no_stim trial
    block_ids = nostim_trials['block_id'].values

    # Map event indices to trial numbers
    event_trials = event_indices // n_timepoints_per_trial

    # Map trials to blocks
    event_blocks = []
    for trial_idx in event_trials:
        if trial_idx < len(block_ids):
            event_blocks.append(block_ids[trial_idx])

    # Count events per block and calculate duration per block
    unique_blocks = np.unique(block_ids)
    events_per_block = {int(block): 0 for block in unique_blocks}
    trials_per_block = {int(block): 0 for block in unique_blocks}

    # Count trials per block
    for block_id in block_ids:
        trials_per_block[int(block_id)] += 1

    # Count events per block
    for block in event_blocks:
        if block in events_per_block:
            events_per_block[int(block)] += 1

    # Calculate event frequency per block (events per minute)
    event_frequency_per_block = {}
    for block in unique_blocks:
        block_int = int(block)
        n_trials_in_block = trials_per_block[block_int]
        block_duration_sec = (n_trials_in_block * n_timepoints_per_trial) / sampling_rate
        block_duration_min = block_duration_sec / 60
        event_frequency_per_block[block_int] = events_per_block[block_int] / block_duration_min

    return events_per_block, event_frequency_per_block, event_blocks


def compute_time_above_per_block(correlations, threshold, trial_block_ids, n_timepoints_per_trial):
    """
    Compute percentage of time above threshold per block.

    Parameters
    ----------
    correlations : np.ndarray
        Full correlation timeseries (n_trials * n_timepoints,)
    threshold : float
        Correlation threshold
    trial_block_ids : np.ndarray
        Block ID for each trial (n_trials,)
    n_timepoints_per_trial : int
        Number of timepoints per trial

    Returns
    -------
    percent_time_per_block : dict
        {block_id: percent_time_above}
    """
    n_trials = len(trial_block_ids)
    unique_blocks = np.unique(trial_block_ids)
    percent_time_per_block = {}

    for block in unique_blocks:
        block_int = int(block)
        # Get trials in this block
        trials_in_block = np.where(trial_block_ids == block)[0]

        # Get correlation frames for these trials
        block_correlations = []
        for trial_idx in trials_in_block:
            start_idx = trial_idx * n_timepoints_per_trial
            end_idx = start_idx + n_timepoints_per_trial
            block_correlations.extend(correlations[start_idx:end_idx])

        block_correlations = np.array(block_correlations)

        # Compute percentage
        above_threshold = block_correlations > threshold
        if len(block_correlations) > 0:
            percent_time_per_block[block_int] = (np.sum(above_threshold) / len(block_correlations)) * 100
        else:
            percent_time_per_block[block_int] = 0.0

    return percent_time_per_block


def get_block_boundaries(nostim_trials, n_timepoints_per_trial):
    """
    Find indices where block transitions occur in concatenated no_stim data.

    Parameters
    ----------
    nostim_trials : xarray.DataArray
        No-stim trial data with block_id coordinate
    n_timepoints_per_trial : int
        Number of timepoints per trial

    Returns
    -------
    boundaries : list
        List of indices where blocks change
    """
    block_ids = nostim_trials['block_id'].values
    boundaries = []

    for i in range(1, len(block_ids)):
        if block_ids[i] != block_ids[i-1]:
            boundaries.append(i * n_timepoints_per_trial)

    return boundaries


def extract_performance_per_block(nostim_trials):
    """
    Extract whisker hit rate (hr_w) per block from no_stim trial data.

    Parameters
    ----------
    nostim_trials : xarray.DataArray
        No-stim trial data with hr_w and block_id coordinates

    Returns
    -------
    hr_per_block : dict
        Dictionary mapping block_id to hit rate
    """
    block_ids = nostim_trials['block_id'].values
    hr_w = nostim_trials['hr_w'].values

    unique_blocks = np.unique(block_ids)
    hr_per_block = {}

    for block in unique_blocks:
        # Get hr_w values for this block (should be constant per block)
        block_mask = block_ids == block
        hr_values = hr_w[block_mask]
        # Take the first non-NaN value (they should all be the same per block)
        valid_hr = hr_values[~np.isnan(hr_values)]
        if len(valid_hr) > 0:
            hr_per_block[int(block)] = valid_hr[0]

    return hr_per_block


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_reactivation_frequency_by_time(time_bins, event_rate, event_rate_sem,
                                         n_trials, trial_type, threshold_label):
    """
    Plot average reactivation frequency as function of time in trial.

    Parameters
    ----------
    time_bins : np.ndarray
        Time bin centers in seconds
    event_rate : np.ndarray
        Average event rate per bin (events/second)
    event_rate_sem : np.ndarray
        Standard error of the mean
    n_trials : int
        Number of trials
    trial_type : str
        Trial type label
    threshold_label : str
        Threshold label for title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot with error bars
    ax.plot(time_bins, event_rate, 'o-', linewidth=2, markersize=6,
            color='steelblue', label='Mean ± SEM')
    ax.fill_between(time_bins,
                     event_rate - event_rate_sem,
                     event_rate + event_rate_sem,
                     alpha=0.3, color='steelblue')

    # Add stimulus onset marker (at t=0 for stimulus trials)
    if trial_type != 'no_stim':
        ax.axvline(0, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='Stimulus onset')

    # Formatting
    ax.set_xlabel('Time in Trial (s)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Reactivation Rate (events/s)', fontweight='bold', fontsize=12)
    ax.set_title(f'Temporal Dynamics of Reactivations\n{trial_type.replace("_", " ").title()} Trials (n={n_trials}, {threshold_label})',
                fontweight='bold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add summary statistics
    mean_rate = np.mean(event_rate)
    max_rate = np.max(event_rate)
    max_time = time_bins[np.argmax(event_rate)]

    stats_text = f'Mean rate: {mean_rate:.3f} events/s\n'
    stats_text += f'Peak rate: {max_rate:.3f} events/s @ {max_time:.2f}s'

    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    return fig


def plot_correlation_traces(correlations, events, block_boundaries, mouse, day,
                           save_path=None, time_per_row=200, sampling_rate=30,
                           ylim=None, threshold_used=None):
    """
    Plot correlation traces split across multiple rows with event markers.

    Parameters
    ----------
    correlations : np.ndarray
        Correlation timeseries
    events : np.ndarray
        Event indices
    block_boundaries : list
        Indices where blocks change
    mouse : str
        Mouse ID
    day : int
        Day number
    save_path : str, optional
        Path to save figure
    time_per_row : float
        Time duration per row (seconds)
    sampling_rate : float
        Sampling rate (Hz)
    ylim : tuple, optional
        Y-axis limits (min, max) to use across all days
    threshold_used : float, optional
        Threshold value used for event detection (for display)
    """
    # Calculate time axis
    time_axis = np.arange(len(correlations)) / sampling_rate
    total_time = time_axis[-1] if len(time_axis) > 0 else 0

    # Calculate number of rows needed based on total time
    n_rows = int(np.ceil(total_time / time_per_row))
    if n_rows == 0:
        n_rows = 1

    # Create figure
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Calculate frames per row
    frames_per_row = int(time_per_row * sampling_rate)

    for row_idx in range(n_rows):
        ax = axes[row_idx]

        # Define time window - always show full time_per_row duration
        t_start = row_idx * time_per_row
        t_end = (row_idx + 1) * time_per_row

        # Get data indices for this window
        idx_start = row_idx * frames_per_row
        idx_end = min((row_idx + 1) * frames_per_row, len(correlations))

        if idx_start >= len(correlations):
            # No data for this row
            ax.set_xlim(t_start, t_end)
            ax.set_ylabel('Correlation', fontsize=9)
            ax.grid(True, alpha=0.2)
            if ylim is not None:
                ax.set_ylim(ylim)
            continue

        # Get data for this window
        time_window = time_axis[idx_start:idx_end]
        corr_window = correlations[idx_start:idx_end]

        # Plot correlation
        ax.plot(time_window, corr_window, 'k-', linewidth=0.8, alpha=0.7)
        # Use threshold_used if provided, otherwise fall back to global threshold_corr
        display_threshold = threshold_used if threshold_used is not None else threshold_corr
        ax.axhline(display_threshold, color='gray', linestyle='--', linewidth=0.5, alpha=0.5,
                  label=f'Threshold ({display_threshold:.3f})' if row_idx == 0 else '')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add event markers (red lines)
        for event_idx in events:
            event_time = event_idx / sampling_rate
            if t_start <= event_time <= t_end:
                ax.axvline(event_time, color='red', linewidth=1.0, alpha=0.5)

        # Formatting - always use full time window
        ax.set_xlim(t_start, t_end)
        ax.set_ylabel('Correlation', fontsize=9)
        ax.grid(True, alpha=0.2)

        # Set y-limits (consistent across days if provided)
        if ylim is not None:
            ax.set_ylim(ylim)

        if row_idx == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=9)

        # Add row label
        ax.text(0.01, 0.95, f'Row {row_idx+1}/{n_rows}', transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        if row_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Overall title
    fig.suptitle(f'{mouse} - Day {day} - Correlation Traces\n'
                f'{len(events)} events detected, Mean r = {np.mean(correlations):.3f}',
                fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_events_vs_performance_per_block(event_frequency_per_block, hr_per_block, mouse, day, save_path=None):
    """
    Plot relationship between reactivation event frequency and performance per block.
    Dual-axis plot with performance on top and event frequency on bottom.

    Parameters
    ----------
    event_frequency_per_block : dict
        Event frequency per block (events/min)
    hr_per_block : dict
        Hit rate per block
    mouse : str
        Mouse ID
    day : int
        Day number
    save_path : str, optional
        Path to save figure
    """
    # Align data (only blocks with both event frequency and performance data)
    common_blocks = set(event_frequency_per_block.keys()) & set(hr_per_block.keys())

    if len(common_blocks) == 0:
        print(f"Warning: No common blocks with both event and performance data for {mouse} Day {day}")
        return None

    blocks = sorted(common_blocks)
    frequencies = [event_frequency_per_block[b] for b in blocks]
    hr = [hr_per_block[b] for b in blocks]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: Performance (hr_w)
    ax1.plot(blocks, hr, 'o-', color='#2ca02c', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_ylabel('Whisker Hit Rate (hr_w)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{mouse} - Day {day}\nPerformance and Reactivation Frequency per Block',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])

    # Bottom panel: Reactivation event frequency
    ax2.plot(blocks, frequencies, 'o-', color='#d62728', linewidth=2, markersize=6, alpha=0.7)
    ax2.set_xlabel('Block ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Reactivation Frequency (events/min)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)

    # Add correlation statistics in text box
    if len(frequencies) > 1 and np.std(frequencies) > 0:
        # Only calculate regression if there's variance in frequencies
        slope, intercept, r_value, p_value, std_err = linregress(frequencies, hr)
        stats_text = f'Correlation:\nr = {r_value:.3f}, p = {p_value:.4f}'
        ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    elif len(frequencies) > 1 and np.std(frequencies) == 0:
        # All frequencies are identical (e.g., all zeros)
        stats_text = f'All events = {frequencies[0]:.2f}\n(no variance)'
        ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_events_per_day(frequency_by_day, mouse, save_path=None):
    """
    Plot bar chart of event frequency per day for a single mouse.

    Parameters
    ----------
    frequency_by_day : dict
        Event frequency per day {day: frequency (events/min)}
    mouse : str
        Mouse ID
    save_path : str, optional
        Path to save figure
    """
    days_list = sorted(frequency_by_day.keys())
    frequency_list = [frequency_by_day[d] for d in days_list]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = ['#1f77b4' if d != 0 else '#ff7f0e' for d in days_list]
    ax.bar(days_list, frequency_list, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Formatting
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Reactivation Frequency (events/min)', fontsize=11)
    ax.set_title(f'{mouse} - Reactivation Frequency Across Days', fontsize=12, fontweight='bold')
    ax.set_xticks(days_list)
    ax.set_xticklabels([str(d) for d in days_list])
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight Day 0
    ax.axvline(0, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Day 0')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_percent_time_above_per_day(r_plus_results, r_minus_results, save_path):
    """
    Plot percentage of time above threshold per day (two-panel bar chart).

    Creates a two-panel figure with R+ and R- side by side with shared y-axis.
    Includes statistical comparisons across days within each reward group.

    Parameters
    ----------
    r_plus_results : dict
        {mouse_id: results} from analyze_mouse_reactivation() for R+ mice
    r_minus_results : dict
        {mouse_id: results} from analyze_mouse_reactivation() for R- mice
    save_path : str
        Path to save SVG file
    """
    # Extract data for both groups
    all_data = []
    for reward_group, results_dict in [('R+', r_plus_results), ('R-', r_minus_results)]:
        for mouse, results in results_dict.items():
            for day in days:
                if day in results['days']:
                    day_data = results['days'][day]
                    all_data.append({
                        'mouse': mouse,
                        'reward_group': reward_group,
                        'day': day,
                        'percent_time_above': day_data.get('percent_time_above', 0)
                    })

    df = pd.DataFrame(all_data)

    # Create two-panel plot with shared y-axis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot each reward group
    for idx, (reward_group, ax) in enumerate(zip(['R+', 'R-'], axes)):
        group_df = df[df['reward_group'] == reward_group]

        if len(group_df) == 0:
            ax.text(0.5, 0.5, f'No data for {reward_group}',
                   transform=ax.transAxes, ha='center', va='center')
            continue

        # Compute mean and SEM per day
        day_means = group_df.groupby('day')['percent_time_above'].mean()
        day_sems = group_df.groupby('day')['percent_time_above'].sem()

        # Bar plot
        color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]
        ax.bar(days_str, day_means.values, yerr=day_sems.values,
               color=color, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)

        # Formatting
        ax.set_xlabel('Day', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('% Time Above Threshold', fontsize=12, fontweight='bold')
        ax.set_title(f'{reward_group} Mice', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add n mice annotation
        n_mice = group_df['mouse'].nunique()
        ax.text(0.02, 0.98, f'n = {n_mice} mice',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Statistical test across days (Friedman test for repeated measures)
        # Only if we have data from multiple days and multiple mice
        if len(day_means) > 1 and n_mice >= 3:
            # Prepare data for Friedman test (mice x days matrix)
            mice = group_df['mouse'].unique()
            day_data_per_mouse = []

            # Build matrix: each row is a mouse, each column is a day
            for mouse in mice:
                mouse_data = group_df[group_df['mouse'] == mouse]
                mouse_values = []
                has_all_days = True
                for day in days:
                    day_value = mouse_data[mouse_data['day'] == day]['percent_time_above']
                    if len(day_value) > 0:
                        mouse_values.append(day_value.values[0])
                    else:
                        has_all_days = False
                        break

                # Only include mice with all days present
                if has_all_days:
                    day_data_per_mouse.append(mouse_values)

            # Run Friedman test if we have enough complete cases
            if len(day_data_per_mouse) >= 3:
                day_data_per_mouse = np.array(day_data_per_mouse)
                # Each column is a day, pass as separate arguments
                stat, p_value = friedmanchisquare(*[day_data_per_mouse[:, i] for i in range(len(days))])

                # Add stats text
                stats_text = f'Friedman test:\nχ²={stat:.2f}, p={p_value:.4f}'
                ax.text(0.98, 0.98, stats_text,
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.suptitle('Percentage of Time Above Threshold Per Day', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def compute_reactivation_frequency_per_trial(selected_trials, template, threshold,
                                              time_bin_ms=500, sampling_rate=30,
                                              min_distance=15, prominence=0.1):
    """
    Compute average reactivation frequency as function of time within trials.

    Aligns trials by stimulus onset, bins time, computes event rate per bin,
    then averages across trials.

    Parameters
    ----------
    selected_trials : xarray.DataArray
        (n_cells, n_trials, n_timepoints) trial data
    template : np.ndarray
        (n_cells,) whisker template
    threshold : float
        Correlation threshold for event detection
    time_bin_ms : float
        Time bin size in milliseconds (default: 500ms)
    sampling_rate : float
        Sampling rate in Hz (default: 30)
    min_distance : int
        Minimum distance between peaks in frames (default: 15)
    prominence : float
        Minimum prominence of peaks (default: 0.1)

    Returns
    -------
    time_bins : np.ndarray
        Time bin centers in seconds (relative to trial start)
    event_rate : np.ndarray
        Average event rate per bin (events/second)
    event_rate_sem : np.ndarray
        Standard error of the mean across trials
    n_trials : int
        Number of trials analyzed
    """
    n_cells, n_trials, n_timepoints = selected_trials.shape

    # Convert bin size to frames
    frames_per_bin = int((time_bin_ms / 1000.0) * sampling_rate)
    n_bins = n_timepoints // frames_per_bin

    # Check if trials are long enough for at least one bin
    if n_bins < 1:
        print(f"    Warning: Trials too short for temporal analysis (timepoints={n_timepoints}, need >={frames_per_bin} for one {time_bin_ms}ms bin)")
        # Return empty arrays with shape that won't break downstream code
        return np.array([]), np.array([]), np.array([]), n_trials

    # Truncate to complete bins
    n_frames_to_use = n_bins * frames_per_bin

    # Initialize storage for events per trial per bin
    events_per_trial = np.zeros((n_trials, n_bins))

    # Process each trial independently
    for trial_idx in range(n_trials):
        # Extract trial data
        trial_data = selected_trials[:, trial_idx, :n_frames_to_use].values
        trial_data = np.nan_to_num(trial_data, nan=0.0)

        # Compute correlation timeseries
        corr = compute_template_correlation(trial_data, template)

        # Detect events
        events = detect_reactivation_events(corr, threshold, min_distance, prominence)

        # Bin events
        for bin_idx in range(n_bins):
            bin_start = bin_idx * frames_per_bin
            bin_end = (bin_idx + 1) * frames_per_bin

            # Count events in this bin
            events_in_bin = np.sum((events >= bin_start) & (events < bin_end))
            events_per_trial[trial_idx, bin_idx] = events_in_bin

    # Compute time bin centers (in seconds, relative to trial start)
    time_bins = (np.arange(n_bins) + 0.5) * frames_per_bin / sampling_rate

    # Convert to event rate (events per second)
    bin_duration_sec = frames_per_bin / sampling_rate
    event_rate_per_trial = events_per_trial / bin_duration_sec

    # Average across trials
    event_rate = np.mean(event_rate_per_trial, axis=0)
    event_rate_sem = np.std(event_rate_per_trial, axis=0) / np.sqrt(n_trials)

    return time_bins, event_rate, event_rate_sem, n_trials


def plot_threshold_comparison(save_path):
    """
    Visualize surrogate-based thresholds for both mouse-wise and day-wise modes.

    Creates a 2-page PDF:
    - Page 1: Mouse-wise thresholds (one threshold per mouse)
    - Page 2: Day-wise thresholds (threshold per mouse-day, showing evolution across days)

    Parameters
    ----------
    save_path : str
        Path to save the PDF file
    """
    print("\n" + "="*60)
    print("GENERATING THRESHOLD COMPARISON PLOTS")
    print("="*60)

    # Load mouse-wise thresholds
    mouse_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates', 'surrogate_thresholds.csv')
    day_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates_per_day', 'surrogate_thresholds_per_day.csv')

    has_mouse_data = os.path.exists(mouse_csv_path)
    has_day_data = os.path.exists(day_csv_path)

    if not has_mouse_data and not has_day_data:
        print(f"  Warning: No threshold files found!")
        print(f"    Mouse-wise: {mouse_csv_path}")
        print(f"    Day-wise: {day_csv_path}")
        print(f"  Run surrogate analysis first to generate threshold files.")
        return

    with PdfPages(save_path) as pdf:
        # Page 1: Mouse-wise thresholds
        if has_mouse_data:
            print(f"\n  Loading mouse-wise thresholds from: {mouse_csv_path}")
            df_mouse = pd.read_csv(mouse_csv_path)

            # Get max threshold (FWER-corrected)
            mouse_thresholds = df_mouse.groupby('mouse')['threshold_max'].first().sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(12, 6))

            # Bar plot
            x_pos = np.arange(len(mouse_thresholds))
            bars = ax.bar(x_pos, mouse_thresholds.values, color='steelblue',
                         alpha=0.7, edgecolor='black', linewidth=1.5)

            # Formatting
            ax.set_xlabel('Mouse ID', fontsize=12, fontweight='bold')
            ax.set_ylabel('Correlation Threshold (FWER-corrected)', fontsize=12, fontweight='bold')
            ax.set_title('Mouse-Wise Surrogate Thresholds\n(Single threshold per mouse computed from day 0)',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(mouse_thresholds.index, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

            # Add summary statistics
            mean_threshold = mouse_thresholds.mean()
            std_threshold = mouse_thresholds.std()
            stats_text = f'Mean: {mean_threshold:.4f}\nStd: {std_threshold:.4f}\nn = {len(mouse_thresholds)} mice'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Generated mouse-wise threshold plot (n={len(mouse_thresholds)} mice)")
        else:
            print(f"  Skipping mouse-wise plot: {mouse_csv_path} not found")

        # Page 2: Day-wise thresholds
        if has_day_data:
            print(f"\n  Loading day-wise thresholds from: {day_csv_path}")
            df_day = pd.read_csv(day_csv_path)

            # Get max threshold (FWER-corrected) per mouse-day
            df_day_max = df_day[df_day['threshold_type'] == 'max'].copy()

            # Create line plot showing threshold evolution across days
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Plot 1: Individual mice trajectories
            mice = df_day_max['mouse'].unique()
            for mouse in mice:
                mouse_data = df_day_max[df_day_max['mouse'] == mouse].sort_values('day')
                ax1.plot(mouse_data['day'], mouse_data['threshold'],
                        marker='o', alpha=0.6, linewidth=1.5, label=mouse)

            ax1.set_xlabel('Day', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Correlation Threshold (FWER-corrected)', fontsize=12, fontweight='bold')
            ax1.set_title('Day-Wise Thresholds: Individual Mice', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

            # Plot 2: Mean ± SEM across mice
            day_summary = df_day_max.groupby('day')['threshold'].agg(['mean', 'sem']).reset_index()

            ax2.errorbar(day_summary['day'], day_summary['mean'],
                        yerr=day_summary['sem'], marker='o', markersize=8,
                        linewidth=2, capsize=5, capthick=2, color='steelblue',
                        ecolor='steelblue', alpha=0.8)
            ax2.fill_between(day_summary['day'],
                           day_summary['mean'] - day_summary['sem'],
                           day_summary['mean'] + day_summary['sem'],
                           alpha=0.3, color='steelblue')

            ax2.set_xlabel('Day', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Correlation Threshold (FWER-corrected)', fontsize=12, fontweight='bold')
            ax2.set_title('Day-Wise Thresholds: Mean ± SEM', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Add n mice annotation
            n_mice = len(mice)
            ax2.text(0.02, 0.98, f'n = {n_mice} mice', transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle('Per-Day Surrogate Thresholds\n(Separate threshold for each mouse-day)',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Generated day-wise threshold plot (n={n_mice} mice, {len(day_summary)} days)")
        else:
            print(f"  Skipping day-wise plot: {day_csv_path} not found")

    print(f"\n  ✓ Saved threshold comparison plots to: {save_path}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_mouse_reactivation(mouse, days=[-2, -1, 0, 1, 2], verbose=True, threshold_dict=None):
    """
    Analyze reactivation events for a single mouse across multiple days.

    Parameters
    ----------
    mouse : str
        Mouse ID
    days : list
        List of days to analyze
    verbose : bool
        Print progress information
    threshold_dict : dict, optional
        Per-mouse, per-day thresholds from surrogate analysis.
        If None, uses global threshold_corr parameter.

    Returns
    -------
    results : dict
        Nested dictionary containing all analysis results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING MOUSE: {mouse}")
        print(f"{'='*60}")

    results = {
        'mouse': mouse,
        'days': {}
    }

    for day in days:
        if verbose:
            print(f"\nProcessing Day {day}...")

        try:
            # Step 1: Create template
            template, cells_mask = create_whisker_template(mouse, day, threshold_dff, verbose=verbose)

            # Step 2: Load learning data
            folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
            file_name = 'tensor_xarray_learning_data.nc'
            xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)

            # Select this day
            xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)

            # Select trials by type
            selected_trials, n_selected_trials = select_trials_by_type(xarray_day, trial_type)

            if verbose:
                print(f"  {trial_type.replace('_', ' ').title()} trials: {n_selected_trials}")

            if n_selected_trials == 0:
                if verbose:
                    print(f"  Warning: No {trial_type} trials for Day {day}, skipping...")
                continue

            # Step 3: Prepare data and compute correlations
            n_cells, n_trials, n_timepoints = selected_trials.shape
            data = selected_trials.values.reshape(n_cells, -1)
            data = np.nan_to_num(data, nan=0.0)

            correlations = compute_template_correlation(data, template)

            if verbose:
                print(f"  Correlation: mean={np.mean(correlations):.3f}, max={np.max(correlations):.3f}")

            # Step 4: Detect events - use per-mouse, per-day threshold if available
            current_threshold = get_threshold_for_mouse_day(threshold_dict, mouse, day, threshold_corr, threshold_mode)
            if verbose and threshold_dict is not None:
                print(f"  Using threshold: {current_threshold:.4f} (surrogate-based, {threshold_mode} mode)")
            events = detect_reactivation_events(correlations, current_threshold, min_event_distance_frames, prominence)

            if verbose:
                print(f"  Events detected: {len(events)}")

            # Compute time above threshold metric
            percent_time_above, n_frames_above, total_frames = compute_time_above_threshold(
                correlations, current_threshold
            )

            if verbose:
                print(f"  Time above threshold: {percent_time_above:.2f}% ({n_frames_above}/{total_frames} frames)")

            # Step 5: Map events to blocks
            events_per_block, event_frequency_per_block, event_blocks = map_events_to_blocks(
                events, selected_trials, n_timepoints, sampling_rate
            )

            # Compute per-block time above threshold
            trial_block_ids = selected_trials['block_id'].values
            percent_time_per_block = compute_time_above_per_block(
                correlations, current_threshold, trial_block_ids, n_timepoints
            )

            # Step 6: Get block boundaries
            block_boundaries = get_block_boundaries(selected_trials, n_timepoints)

            # Step 7: Extract performance
            hr_per_block = extract_performance_per_block(selected_trials)

            # Step 8: Calculate session duration and event frequency
            session_duration_sec = (n_trials * n_timepoints) / sampling_rate
            session_duration_min = session_duration_sec / 60
            event_frequency = len(events) / session_duration_min  # events per minute

            if verbose:
                print(f"  Session duration: {session_duration_sec:.1f}s ({session_duration_min:.2f}min)")
                print(f"  Event frequency: {event_frequency:.3f} events/min")

            # Compute temporal reactivation frequency
            if verbose:
                print(f"  Computing temporal reactivation frequency...")

            time_bins, event_rate, event_rate_sem, n_trials_temporal = \
                compute_reactivation_frequency_per_trial(
                    selected_trials, template, current_threshold,
                    time_bin_ms=500, sampling_rate=sampling_rate,
                    min_distance=min_event_distance_frames, prominence=prominence
                )

            if verbose:
                if len(time_bins) > 0:
                    print(f"  Temporal analysis: {len(time_bins)} time bins, mean rate={np.mean(event_rate):.3f} events/s")
                else:
                    print(f"  Temporal analysis skipped: trials too short")

            # Store results
            results['days'][day] = {
                'template': template,
                'cells_mask': cells_mask,
                'correlations': correlations,
                'events': events,
                'events_per_block': events_per_block,
                'event_frequency_per_block': event_frequency_per_block,
                'percent_time_above': percent_time_above,  # NEW: % time above threshold
                'n_frames_above': n_frames_above,  # NEW: frames above threshold
                'total_frames': total_frames,  # NEW: total frames
                'percent_time_per_block': percent_time_per_block,  # NEW: % time per block
                'hr_per_block': hr_per_block,
                'block_boundaries': block_boundaries,
                'n_trials': n_selected_trials,
                'n_timepoints': n_timepoints,
                'total_events': len(events),
                'session_duration_sec': session_duration_sec,
                'session_duration_min': session_duration_min,
                'event_frequency': event_frequency,
                'session_hr_mean': np.mean(list(hr_per_block.values())) if hr_per_block else np.nan,
                'threshold_used': current_threshold,  # Store threshold for reference
                'selected_trials': selected_trials,  # Store for temporal analysis
                'temporal': {
                    'time_bins': time_bins,
                    'event_rate': event_rate,
                    'event_rate_sem': event_rate_sem,
                    'n_trials': n_trials_temporal
                }
            }

        except Exception as e:
            if verbose:
                print(f"  Error processing Day {day}: {str(e)}")
                import traceback
                traceback.print_exc()
            continue

    return results


def generate_mouse_pdf(results, save_dir):
    """
    Generate multi-page PDF report for a single mouse.

    Parameters
    ----------
    results : dict
        Analysis results from analyze_mouse_reactivation
    save_dir : str
        Directory to save PDF
    """
    mouse = results['mouse']
    pdf_path = os.path.join(save_dir, f'{mouse}_reactivation_analysis.pdf')

    # Calculate global y-limits across all days for consistent y-axis
    all_correlations = []
    for day in days:
        if day in results['days']:
            all_correlations.extend(results['days'][day]['correlations'])

    if len(all_correlations) > 0:
        ylim = (np.min(all_correlations), np.max(all_correlations))
    else:
        ylim = None

    with PdfPages(pdf_path) as pdf:
        # Pages 1-5: Correlation traces for all days (in order: -2, -1, 0, +1, +2)
        for day in days:
            if day in results['days']:
                day_data = results['days'][day]
                fig = plot_correlation_traces(
                    day_data['correlations'],
                    day_data['events'],
                    day_data['block_boundaries'],
                    mouse, day,
                    time_per_row=time_per_row,
                    sampling_rate=sampling_rate,
                    ylim=ylim,
                    threshold_used=day_data.get('threshold_used', None)
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        # Page 6: Day 0 event frequency vs performance per block
        if 0 in results['days']:
            day0 = results['days'][0]
            fig = plot_events_vs_performance_per_block(
                day0['event_frequency_per_block'],
                day0['hr_per_block'],
                mouse, 0
            )
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        # Page 7: Event frequency per day
        frequency_by_day = {day: results['days'][day]['event_frequency']
                           for day in results['days'].keys()}
        fig = plot_events_per_day(frequency_by_day, mouse)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Pages 8+: Temporal reactivation frequency for each day
        for day in days:
            if day in results['days'] and 'temporal' in results['days'][day]:
                temporal = results['days'][day]['temporal']

                # Skip if temporal data is empty (trials too short)
                if len(temporal['time_bins']) == 0:
                    continue

                threshold_used = results['days'][day].get('threshold_used', 'default')
                threshold_label = f"threshold={threshold_used:.3f}" if isinstance(threshold_used, float) else str(threshold_used)

                fig = plot_reactivation_frequency_by_time(
                    temporal['time_bins'],
                    temporal['event_rate'],
                    temporal['event_rate_sem'],
                    temporal['n_trials'],
                    trial_type,
                    threshold_label
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

    print(f"  PDF saved: {pdf_path}")


# ============================================================================
# ACROSS-MICE ANALYSIS FUNCTIONS
# ============================================================================

def plot_session_level_across_mice(r_plus_results, r_minus_results, save_path):
    """
    Plot Day 0 session-level reactivation frequency vs performance across mice.
    Two panels: R+ mice (left) and R- mice (right).

    Parameters
    ----------
    r_plus_results : dict
        Dictionary with results for R+ mice
    r_minus_results : dict
        Dictionary with results for R- mice
    save_path : str
        Path to save SVG figure
    """
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Process each group
    for ax, all_results, group_name, group_color in [
        (ax1, r_plus_results, 'R+', reward_palette[1]),
        (ax2, r_minus_results, 'R-', reward_palette[0])
    ]:
        # Extract data
        mice_list = []
        event_frequency_list = []
        session_hr_list = []

        for mouse, results in all_results.items():
            if 0 in results['days']:
                mice_list.append(mouse)
                event_frequency_list.append(results['days'][0]['event_frequency'])
                session_hr_list.append(results['days'][0]['session_hr_mean'])

        if len(mice_list) == 0:
            ax.text(0.5, 0.5, f'No Day 0 data\nfor {group_name} mice',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            continue

        # Use a colormap that can handle any number of mice
        cmap = plt.cm.get_cmap('tab20' if len(mice_list) <= 20 else 'gist_ncar')
        colors = [cmap(i / len(mice_list)) for i in range(len(mice_list))]

        # Scatter plot
        for i, (mouse, freq, hr) in enumerate(zip(mice_list, event_frequency_list, session_hr_list)):
            ax.scatter(freq, hr, s=120, alpha=0.7, c=[colors[i]],
                      edgecolors='black', linewidth=1, label=mouse)

        # Regression line
        if len(event_frequency_list) > 1 and np.std(event_frequency_list) > 0:
            slope, intercept, r_value, p_value, std_err = linregress(event_frequency_list, session_hr_list)
            x_line = np.linspace(min(event_frequency_list), max(event_frequency_list), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=group_color, linestyle='--', linewidth=2.5,
                   alpha=0.7, label='Linear fit')

            # Add statistics text
            stats_text = f'n = {len(mice_list)} mice\n'
            stats_text += f'r = {r_value:.3f}\n'
            stats_text += f'p = {p_value:.4f}\n'
            stats_text += f'R² = {r_value**2:.3f}'

            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        elif len(event_frequency_list) > 1:
            # No variance in event frequencies
            stats_text = f'n = {len(mice_list)} mice\n(no variance in events)'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'))

        # Formatting
        ax.set_xlabel('Reactivation Event Frequency (events/min)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Session Hit Rate (hr_w)', fontsize=11, fontweight='bold')
        ax.set_title(f'{group_name} Mice\nReactivation Frequency vs Performance',
                    fontsize=12, fontweight='bold', color=group_color)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)

    fig.suptitle('Session-Level Reactivation vs Performance (Day 0, Across Mice)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  SVG saved: {save_path}")
        plt.close()

    return fig


def plot_percent_time_above_vs_performance(r_plus_results, r_minus_results, save_path):
    """
    Scatter plot: Day 0 percentage time above threshold vs hit rate across mice.

    Similar to plot_session_level_across_mice() but shows % time instead of events/min.

    Parameters
    ----------
    r_plus_results : dict
        {mouse_id: results} for R+ mice
    r_minus_results : dict
        {mouse_id: results} for R- mice
    save_path : str
        Path to save SVG file
    """
    # Extract Day 0 data
    data_list = []

    for reward_group, results_dict in [('R+', r_plus_results), ('R-', r_minus_results)]:
        for mouse, results in results_dict.items():
            if 0 in results['days']:
                day0_data = results['days'][0]
                data_list.append({
                    'mouse': mouse,
                    'reward_group': reward_group,
                    'percent_time_above': day0_data.get('percent_time_above', 0),
                    'hit_rate': day0_data.get('session_hr_mean', np.nan)
                })

    df = pd.DataFrame(data_list)
    df = df.dropna(subset=['hit_rate', 'percent_time_above'])

    if len(df) == 0:
        print(f"  Warning: No valid data for time-above vs performance plot")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each reward group
    for reward_group in ['R+', 'R-']:
        group_df = df[df['reward_group'] == reward_group]

        if len(group_df) > 0:
            ax.scatter(group_df['percent_time_above'], group_df['hit_rate'],
                      color=reward_palette[1 if reward_group == 'R+' else 0], s=100, alpha=0.7,
                      edgecolor='black', linewidth=1.5,
                      label=f'{reward_group} (n={len(group_df)})')

            # Linear regression
            if len(group_df) >= 3:
                x = group_df['percent_time_above'].values
                y = group_df['hit_rate'].values

                slope, intercept, r_value, p_value, std_err = linregress(x, y)

                # Plot regression line
                x_line = np.array([x.min(), x.max()])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=reward_palette[1 if reward_group == 'R+' else 0],
                       linestyle='--', linewidth=2, alpha=0.8)

                # Add stats text
                stats_text = f'{reward_group}: r={r_value:.3f}, p={p_value:.3f}'
                ax.text(0.02, 0.98 - (0.05 * ['R+', 'R-'].index(reward_group)),
                       stats_text, transform=ax.transAxes, fontsize=10,
                       color=reward_palette[1 if reward_group == 'R+' else 0], fontweight='bold')

    # Formatting
    ax.set_xlabel('% Time Above Threshold (Day 0)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hit Rate (Day 0)', fontsize=12, fontweight='bold')
    ax.set_title('Day 0: Percentage Time Above Threshold vs Performance',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def plot_block_level_across_mice(r_plus_results, r_minus_results, save_path):
    """
    Plot Day 0 block-level reactivation frequency vs performance averaged across mice.
    Two columns: R+ mice (left) and R- mice (right). Each column has performance on top and event frequency on bottom.

    Parameters
    ----------
    r_plus_results : dict
        Dictionary with results for R+ mice
    r_minus_results : dict
        Dictionary with results for R- mice
    save_path : str
        Path to save SVG figure
    """
    # Create figure with 2 rows x 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex='col')

    # Process each group
    for col_idx, (all_results, group_name, group_color) in enumerate([
        (r_plus_results, 'R+', reward_palette[1]),
        (r_minus_results, 'R-', reward_palette[0])
    ]):
        # Build dataframe for this group
        data_rows = []

        for mouse, results in all_results.items():
            if 0 in results['days']:
                day0 = results['days'][0]
                event_frequency_per_block = day0['event_frequency_per_block']
                hr_per_block = day0['hr_per_block']

                # Get common blocks
                common_blocks = set(event_frequency_per_block.keys()) & set(hr_per_block.keys())

                for block in common_blocks:
                    data_rows.append({
                        'mouse': mouse,
                        'block_id': block,
                        'event_frequency': event_frequency_per_block[block],
                        'hr_w': hr_per_block[block]
                    })

        if len(data_rows) == 0:
            for row_idx in range(2):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, f'No Day 0 block data\nfor {group_name} mice',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            continue

        df = pd.DataFrame(data_rows)

        # Top panel: Performance (hr_w) with confidence interval
        ax1 = axes[0, col_idx]
        sns.lineplot(data=df, x='block_id', y='hr_w', ax=ax1,
                    color=group_color, linewidth=2.5, marker='o', markersize=8,
                    errorbar=('ci', 95), err_style='band', alpha=0.8)
        ax1.set_ylabel('Whisker Hit Rate (hr_w)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{group_name} Mice - Performance per Block',
                     fontsize=12, fontweight='bold', color=group_color)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1])

        # Bottom panel: Reactivation event frequency with confidence interval
        ax2 = axes[1, col_idx]
        sns.lineplot(data=df, x='block_id', y='event_frequency', ax=ax2,
                    color=group_color, linewidth=2.5, marker='o', markersize=8,
                    errorbar=('ci', 95), err_style='band', alpha=0.8)
        ax2.set_xlabel('Block ID', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Reactivation Frequency (events/min)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(bottom=0)

        # Add sample size info
        n_mice = df['mouse'].nunique()
        n_blocks_total = len(df)
        info_text = f'n = {n_mice} mice\n{n_blocks_total} blocks'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        # Calculate correlation between average frequency and average hr_w per block
        block_avg = df.groupby('block_id').agg({'event_frequency': 'mean', 'hr_w': 'mean'}).reset_index()
        if len(block_avg) > 1 and block_avg['event_frequency'].std() > 0:
            slope, intercept, r_value, p_value, std_err = linregress(block_avg['event_frequency'], block_avg['hr_w'])
            stats_text = f'r = {r_value:.3f}\np = {p_value:.4f}'
            ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
                    fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        elif len(block_avg) > 1:
            stats_text = 'No variance in events'
            ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
                    fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray'))

    fig.suptitle('Block-Level Performance and Reactivation Frequency (Day 0, Across Mice)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  SVG saved: {save_path}")
        plt.close()

    return fig


def plot_events_per_day_across_mice(r_plus_results, r_minus_results, save_path):
    """
    Plot bar chart of event frequency per day averaged across mice.
    Two panels: R+ mice (left) and R- mice (right).

    Parameters
    ----------
    r_plus_results : dict
        Dictionary with results for R+ mice
    r_minus_results : dict
        Dictionary with results for R- mice
    save_path : str
        Path to save SVG figure
    """
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Process each group
    for ax, all_results, group_name, group_color in [
        (ax1, r_plus_results, 'R+', reward_palette[1]),
        (ax2, r_minus_results, 'R-', reward_palette[0])
    ]:
        # Collect data in format for seaborn
        plot_data = []
        for mouse, results in all_results.items():
            for day in days:
                if day in results['days']:
                    plot_data.append({
                        'Day': day,
                        'Frequency': results['days'][day]['event_frequency'],
                        'Mouse': mouse
                    })

        # Convert to DataFrame
        df = pd.DataFrame(plot_data)

        if len(df) > 0:
            # Use seaborn barplot
            sns.barplot(data=df, x='Day', y='Frequency', ax=ax,
                       color=group_color, alpha=0.7, errorbar='ci',
                       )

            # Add individual data points with stripplot
            sns.swarmplot(data=df, x='Day', y='Frequency', ax=ax,
                         color='black', alpha=0.5, size=5)

        # Formatting
        ax.set_xlabel('Day', fontsize=11, fontweight='bold')
        ax.set_ylabel('Reactivation Frequency (events/min)', fontsize=11, fontweight='bold')
        ax.set_title(f'{group_name} Mice\n(Mean ± SEM, n={len(all_results)} mice)',
                    fontsize=12, fontweight='bold', color=group_color)
        ax.set_xticks(days)
        ax.set_xticklabels([str(d) for d in days_str])
        ax.grid(True, alpha=0.3, axis='y')

        ax.legend(fontsize=9)

    fig.suptitle('Reactivation Frequency Across Days (Across Mice)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  SVG saved: {save_path}")
        plt.close()

    return fig


def plot_group_comparison_per_day(r_plus_results, r_minus_results, save_path):
    """
    Compare R+ vs R- event frequency for each day with statistical tests.
    Grouped bar plot with significance stars using seaborn and pandas.

    Parameters
    ----------
    r_plus_results : dict
        Dictionary with results for R+ mice
    r_minus_results : dict
        Dictionary with results for R- mice
    save_path : str
        Path to save SVG figure
    """

    # Collect data into a DataFrame in long format
    data_list = []

    for mouse, results in r_plus_results.items():
        for day in days:
            if day in results['days']:
                data_list.append({
                    'Day': day,
                    'Frequency': results['days'][day]['event_frequency'],
                    'Group': 'R+',
                    'Mouse': mouse
                })

    for mouse, results in r_minus_results.items():
        for day in days:
            if day in results['days']:
                data_list.append({
                    'Day': day,
                    'Frequency': results['days'][day]['event_frequency'],
                    'Group': 'R-',
                    'Mouse': mouse
                })

    df = pd.DataFrame(data_list)

    # Calculate statistics for each day (for significance testing)
    days_list = sorted(days)
    p_values = []
    r_plus_by_day = {day: [] for day in days}
    r_minus_by_day = {day: [] for day in days}

    for mouse, results in r_plus_results.items():
        for day in days:
            if day in results['days']:
                r_plus_by_day[day].append(results['days'][day]['event_frequency'])

    for mouse, results in r_minus_results.items():
        for day in days:
            if day in results['days']:
                r_minus_by_day[day].append(results['days'][day]['event_frequency'])

    for day in days_list:
        r_plus_data = r_plus_by_day[day]
        r_minus_data = r_minus_by_day[day]

        # Statistical test (Mann-Whitney U test)
        if len(r_plus_data) > 0 and len(r_minus_data) > 0:
            try:
                _, p_val = mannwhitneyu(r_plus_data, r_minus_data, alternative='two-sided')
                p_values.append(p_val)
            except:
                p_values.append(1.0)
        else:
            p_values.append(1.0)

    # Ensure all days are present in the x-axis, even if no data
    all_days = sorted(days)
    # Add dummy rows for missing days (for plotting only)
    for day in all_days:
        if day not in df['Day'].unique():
            df = pd.concat([df, pd.DataFrame({'Day': [day], 'Frequency': [np.nan], 'Group': ['R+'], 'Mouse': ['']})], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'Day': [day], 'Frequency': [np.nan], 'Group': ['R-'], 'Mouse': ['']})], ignore_index=True)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Create barplot with 95% confidence intervals using seaborn
    sns.barplot(data=df, x='Day', y='Frequency', hue='Group',
                errorbar=('ci', 95),
                palette={'R+': reward_palette[1], 'R-': reward_palette[0]},
                hue_order=['R+', 'R-'],
                alpha=0.7, edgecolor='black', ax=ax)

    # Add individual data points with stripplot
    sns.swarmplot(data=df, x='Day', y='Frequency', hue='Group',
                    color='black',
                    hue_order=['R+', 'R-'],
                    dodge=True, size=5, alpha=0.6, linewidth=0.5,
                    edgecolor='black', ax=ax, legend=False)

    # Calculate max y for significance bracket positioning
    y_max = df['Frequency'].max()
    y_range = y_max - 0

    def get_significance_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    # Add significance stars
    width = 0.35  # approximate bar width

    for day_idx, p_val in enumerate(p_values):
        stars = get_significance_stars(p_val)
        if stars != 'ns':
            # Get data for this day
            r_plus_data = df[(df['Day'] == days_list[day_idx]) & (df['Group'] == 'R+')]['Frequency']
            r_minus_data = df[(df['Day'] == days_list[day_idx]) & (df['Group'] == 'R-')]['Frequency']

            # Calculate CI upper bounds (95% CI = mean ± 1.96 * SEM)
            if len(r_plus_data) > 0:
                r_plus_ci_upper = r_plus_data.mean() + 1.96 * r_plus_data.std() / np.sqrt(len(r_plus_data))
            else:
                r_plus_ci_upper = 0

            if len(r_minus_data) > 0:
                r_minus_ci_upper = r_minus_data.mean() + 1.96 * r_minus_data.std() / np.sqrt(len(r_minus_data))
            else:
                r_minus_ci_upper = 0

            y1 = max(r_plus_ci_upper, r_minus_ci_upper)
            y2 = y1 + y_range * 0.05
            x1 = day_idx - width/2
            x2 = day_idx + width/2

            ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], 'k-', linewidth=1)
            ax.text((x1 + x2) / 2, y2, stars, ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

    # Formatting
    ax.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reactivation Frequency (events/min)', fontsize=12, fontweight='bold')
    ax.set_title('R+ vs R- Reactivation Frequency Comparison\n' +
                f'(R+: n={len(r_plus_results)} mice, R-: n={len(r_minus_results)} mice)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  SVG saved: {save_path}")
        plt.close()

    # Print statistics summary
    print("\n" + "="*60)
    print("R+ vs R- STATISTICAL COMPARISON")
    print("="*60)
    for day_idx, day in enumerate(days_list):
        p_val = p_values[day_idx]
        stars = get_significance_stars(p_val)
        r_plus_data = df[(df['Day'] == day) & (df['Group'] == 'R+')]['Frequency']
        r_minus_data = df[(df['Day'] == day) & (df['Group'] == 'R-')]['Frequency']

        r_plus_mean = r_plus_data.mean() if len(r_plus_data) > 0 else 0
        r_plus_ci = 1.96 * r_plus_data.std() / np.sqrt(len(r_plus_data)) if len(r_plus_data) > 0 else 0
        r_minus_mean = r_minus_data.mean() if len(r_minus_data) > 0 else 0
        r_minus_ci = 1.96 * r_minus_data.std() / np.sqrt(len(r_minus_data)) if len(r_minus_data) > 0 else 0

        print(f"Day {day:+d}: p = {p_val:.4f} {stars}")
        print(f"  R+: {r_plus_mean:.3f} ± {r_plus_ci:.3f} (95% CI, n={len(r_plus_data)})")
        print(f"  R-: {r_minus_mean:.3f} ± {r_minus_ci:.3f} (95% CI, n={len(r_minus_data)})")

    return fig


def plot_reactivation_vs_performance_delta(r_plus_results, r_minus_results, save_path):
    """
    Plot reactivation frequency on day 0 vs performance improvement (day +1 - day 0).

    Two panels: R+ (left) and R- (right).
    Shows scatter plot with regression line and Pearson correlation.

    Parameters
    ----------
    r_plus_results : list of dict
        Results for R+ mice
    r_minus_results : list of dict
        Results for R- mice
    save_path : str
        Path to save the figure
    """
    from scipy.stats import pearsonr
    from scipy import stats
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, all_results, group_name, group_color in [
        (ax1, r_plus_results, 'R+', reward_palette[1]),
        (ax2, r_minus_results, 'R-', reward_palette[0])
    ]:
        # Extract day 0 reactivation and delta performance (day +1 - day 0)
        reactivation_freq_day0 = []
        delta_performance = []
        mouse_names = []

        for mouse_id, results in all_results.items():
            # Check if this mouse has both day 0 and day +1
            if 0 in results['days'] and 1 in results['days']:
                # Extract reactivation frequency and performance
                reactivation_day0 = results['days'][0]['event_frequency']
                hr_day0 = results['days'][0]['session_hr_mean']
                hr_day1 = results['days'][1]['session_hr_mean']
                delta_hr = hr_day1 - hr_day0

                reactivation_freq_day0.append(reactivation_day0)
                delta_performance.append(delta_hr)
                mouse_names.append(mouse_id)

        # Convert to numpy arrays
        x = np.array(reactivation_freq_day0)
        y = np.array(delta_performance)

        # Scatter plot
        ax.scatter(x, y, s=120, alpha=0.7, color=group_color, edgecolors='black', linewidths=1.5)

        # Add mouse labels
        for i, mouse in enumerate(mouse_names):
            ax.text(x[i], y[i], f' {mouse}', fontsize=9, alpha=0.7,
                   ha='left', va='center')

        # Calculate Pearson correlation and add regression line
        if len(x) >= 3 and np.std(x) > 0:  # Need at least 3 points and variance for meaningful correlation
            r, p_val = pearsonr(x, y)

            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, '--', color=group_color, linewidth=2.5, alpha=0.8, label='Linear fit')

            # Add correlation text
            significance = ""
            if p_val < 0.001:
                significance = " ***"
            elif p_val < 0.01:
                significance = " **"
            elif p_val < 0.05:
                significance = " *"

            ax.text(0.05, 0.95, f'Pearson r = {r:.3f}{significance}\np = {p_val:.4f}\nn = {len(x)}',
                   transform=ax.transAxes, fontsize=13, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=group_color, linewidth=2))

            print(f"\n{group_name} Group - Reactivation vs Performance Delta:")
            print(f"  Pearson r = {r:.3f}, p = {p_val:.4f}{significance} (n={len(x)})")
        else:
            # Insufficient data or no variance in reactivation frequencies
            reason = 'insufficient data' if len(x) < 3 else 'no variance in events'
            ax.text(0.05, 0.95, f'n = {len(x)}\n({reason})',
                   transform=ax.transAxes, fontsize=13, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor=group_color, linewidth=2))
            print(f"\n{group_name} Group: n = {len(x)} ({reason} for correlation)")

        # Formatting
        ax.set_xlabel('Reactivation Frequency on Day 0 (events/min)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Performance Improvement (HR Day+1 - HR Day0)', fontsize=14, fontweight='bold')
        ax.set_title(f'{group_name} Mice: Reactivation vs Learning Consolidation',
                    fontsize=16, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)

        # Improve tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved reactivation vs performance delta plot: {save_path}")
        plt.close()

    return fig


def calculate_within_day0_performance_delta(mouse, verbose=False):
    """
    Calculate within-session performance delta for day 0.

    Computes hit rate for first 20% and last 20% of whisker trials (hits + misses)
    and returns the delta (last 20% - first 20%).

    Parameters
    ----------
    mouse : str
        Mouse ID
    verbose : bool
        Print progress messages

    Returns
    -------
    delta_hr : float
        Change in hit rate (last 20% - first 20%)
    first_20_hr : float
        Hit rate in first 20% of whisker trials
    last_20_hr : float
        Hit rate in last 20% of whisker trials
    n_whisker_trials : int
        Total number of whisker trials
    """
    # Load learning data
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_learning_data.nc'

    try:
        xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)
    except:
        if verbose:
            print(f"  Warning: Could not load data for {mouse}")
        return np.nan, np.nan, np.nan, 0

    # Select day 0 trials
    xarray_day0 = xarray_learning.sel(trial=xarray_learning['day'] == 0)

    # Get all whisker trials (hits + misses)
    whisker_mask = xarray_day0['whisker_stim'] == 1
    whisker_trials = xarray_day0.sel(trial=whisker_mask)

    n_whisker_trials = len(whisker_trials.trial)

    if n_whisker_trials < 10:  # Need enough trials for meaningful split
        if verbose:
            print(f"  Warning: Only {n_whisker_trials} whisker trials for {mouse}, skipping")
        return np.nan, np.nan, np.nan, n_whisker_trials

    # Calculate indices for first and last 20%
    n_first_20 = max(1, int(n_whisker_trials * 0.2))
    n_last_20 = max(1, int(n_whisker_trials * 0.2))

    # Get first 20% trials
    first_20_trials = whisker_trials.isel(trial=slice(0, n_first_20))
    first_20_licks = first_20_trials['lick_flag'].values
    first_20_hr = np.mean(first_20_licks)

    # Get last 20% trials
    last_20_trials = whisker_trials.isel(trial=slice(-n_last_20, None))
    last_20_licks = last_20_trials['lick_flag'].values
    last_20_hr = np.mean(last_20_licks)

    # Calculate delta
    delta_hr = last_20_hr - first_20_hr

    if verbose:
        print(f"  {mouse}: First 20% HR={first_20_hr:.3f}, Last 20% HR={last_20_hr:.3f}, Delta={delta_hr:.3f} (n={n_whisker_trials})")

    return delta_hr, first_20_hr, last_20_hr, n_whisker_trials


def plot_within_day0_performance_vs_reactivation(r_plus_results, r_minus_results,
                                                   reactivation_trial_type, save_path):
    """
    Plot within-day-0 performance improvement vs reactivation frequency during day 0.

    Performance delta = (last 20% whisker trials hit rate) - (first 20% whisker trials hit rate)
    Reactivation frequency from specified trial type during day 0.

    Two panels: R+ (left) and R- (right).
    Shows scatter plot with regression line and Pearson correlation.

    Parameters
    ----------
    r_plus_results : dict
        {mouse_id: results} for R+ mice
    r_minus_results : dict
        {mouse_id: results} for R- mice
    reactivation_trial_type : str
        Trial type to use for reactivation frequency ('no_stim' or 'whisker_hit')
    save_path : str
        Path to save the figure
    """
    from scipy.stats import pearsonr
    from scipy import stats
    import numpy as np

    print(f"\n  Calculating within-day-0 performance vs {reactivation_trial_type} reactivation correlation...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, all_results, group_name, group_color in [
        (ax1, r_plus_results, 'R+', reward_palette[1]),
        (ax2, r_minus_results, 'R-', reward_palette[0])
    ]:
        # Extract day 0 reactivation frequency and within-day performance delta
        reactivation_freq_day0 = []
        delta_performance = []
        mouse_names = []

        for mouse_id, results in all_results.items():
            # Check if this mouse has day 0 data
            if 0 in results['days']:
                # Get reactivation frequency for day 0
                reactivation_day0 = results['days'][0]['event_frequency']

                # Calculate within-day performance delta
                delta_hr, first_hr, last_hr, n_trials = calculate_within_day0_performance_delta(
                    mouse_id, verbose=False
                )

                # Only include if we have valid data
                if not np.isnan(delta_hr):
                    reactivation_freq_day0.append(reactivation_day0)
                    delta_performance.append(delta_hr)
                    mouse_names.append(mouse_id)

        # Convert to numpy arrays
        x = np.array(reactivation_freq_day0)
        y = np.array(delta_performance)

        # Scatter plot
        ax.scatter(x, y, s=120, alpha=0.7, color=group_color, edgecolors='black', linewidths=1.5)

        # Add mouse labels
        for i, mouse in enumerate(mouse_names):
            ax.text(x[i], y[i], f' {mouse}', fontsize=9, alpha=0.7,
                   ha='left', va='center')

        # Calculate Pearson correlation and add regression line
        if len(x) >= 3 and np.std(x) > 0:  # Need at least 3 points and variance for meaningful correlation
            r, p_val = pearsonr(x, y)

            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, '--', color=group_color, linewidth=2.5, alpha=0.8, label='Linear fit')

            # Add correlation text
            significance = ""
            if p_val < 0.001:
                significance = " ***"
            elif p_val < 0.01:
                significance = " **"
            elif p_val < 0.05:
                significance = " *"

            ax.text(0.05, 0.95, f'Pearson r = {r:.3f}{significance}\np = {p_val:.4f}\nn = {len(x)}',
                   transform=ax.transAxes, fontsize=13, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=group_color, linewidth=2))

            print(f"  {group_name} Group - Pearson r = {r:.3f}, p = {p_val:.4f}{significance} (n={len(x)})")
        else:
            # Insufficient data or no variance
            reason = 'insufficient data' if len(x) < 3 else 'no variance in reactivation'
            ax.text(0.05, 0.95, f'n = {len(x)}\n({reason})',
                   transform=ax.transAxes, fontsize=13, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor=group_color, linewidth=2))
            print(f"  {group_name} Group: n = {len(x)} ({reason})")

        # Formatting
        ax.set_xlabel(f'Reactivation Frequency (events/min)\nDay 0 {reactivation_trial_type.replace("_", " ").title()} Trials',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Within-Session Performance Improvement\n(Last 20% - First 20% Whisker Hit Rate)',
                     fontsize=14, fontweight='bold')
        ax.set_title(f'{group_name} Mice: Day 0 Within-Session Learning\nvs {reactivation_trial_type.replace("_", " ").title()} Reactivation',
                    fontsize=16, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)

        # Improve tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved within-day-0 performance vs reactivation plot: {save_path}")
        plt.close()

    return fig


def calculate_per_trial_reactivation_frequency(trials_xarray, template, threshold,
                                                min_distance, prominence, sampling_rate=30,
                                                blank_stimulus=True, blank_window=(0, 1)):
    """
    Calculate reactivation frequency for individual trials.

    Parameters
    ----------
    trials_xarray : xarray.DataArray
        Trial data with shape (n_cells, n_trials, n_timepoints)
    template : np.ndarray
        Whisker response template (n_cells,)
    threshold : float
        Correlation threshold for event detection
    min_distance : int
        Minimum distance between events (frames)
    prominence : float
        Minimum prominence for peak detection
    sampling_rate : int
        Sampling rate in Hz (default: 30)
    blank_stimulus : bool
        Whether to blank the stimulus period
    blank_window : tuple
        (start, end) time window to blank around stimulus in seconds

    Returns
    -------
    frequencies : np.ndarray
        Array of reactivation frequencies (events/min) for each trial
    """
    n_cells, n_trials, n_timepoints = trials_xarray.shape
    frequencies = np.zeros(n_trials)

    # Calculate blank window indices if needed
    if blank_stimulus:
        blank_start_idx = int((0 - blank_window[0]) * sampling_rate)
        blank_end_idx = int((0 + blank_window[1]) * sampling_rate)
        # Ensure indices are within bounds
        blank_start_idx = max(0, blank_start_idx)
        blank_end_idx = min(n_timepoints, blank_end_idx)

    # Process each trial
    for trial_idx in range(n_trials):
        # Extract trial data
        trial_data = trials_xarray[:, trial_idx, :].values.copy()
        trial_data = np.nan_to_num(trial_data, nan=0.0)

        # Apply stimulus blanking if requested
        if blank_stimulus:
            trial_data[:, blank_start_idx:blank_end_idx] = 0.0

        # Compute correlation with template
        correlations = compute_template_correlation(trial_data, template)

        # Detect events
        events = detect_reactivation_events(correlations, threshold, min_distance, prominence)

        # Calculate frequency for this trial
        trial_duration_sec = n_timepoints / sampling_rate
        trial_duration_min = trial_duration_sec / 60.0

        if trial_duration_min > 0:
            frequencies[trial_idx] = len(events) / trial_duration_min
        else:
            frequencies[trial_idx] = 0.0

    return frequencies


def analyze_reactivation_around_first_hit(mouse, verbose=False):
    """
    Analyze reactivation frequency around the first whisker hit trial on day 0.

    Calculates average reactivation frequency for:
    - All whisker miss trials before the first hit
    - The first whisker hit trial itself
    - The next 5 whisker trials after the first hit

    Parameters
    ----------
    mouse : str
        Mouse ID
    verbose : bool
        Print progress messages

    Returns
    -------
    results : dict
        Dictionary with:
        - 'mouse': mouse ID
        - 'freq_before': average frequency before first hit (NaN if no misses)
        - 'freq_at_hit': frequency at first hit trial
        - 'freq_after': average frequency after first hit
        - 'n_misses_before': number of miss trials before first hit
        - 'n_trials_after': actual number of trials after (may be <5)
    """
    # Load learning data
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_learning_data.nc'

    try:
        xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load data for {mouse}: {e}")
        return None

    # Select day 0 trials
    xarray_day0 = xarray_learning.sel(trial=xarray_learning['day'] == 0)

    # Get all whisker trials (hits + misses)
    whisker_mask = xarray_day0['whisker_stim'] == 1
    whisker_trials = xarray_day0.sel(trial=whisker_mask)

    n_whisker_trials = len(whisker_trials.trial)

    if n_whisker_trials == 0:
        if verbose:
            print(f"  Warning: No whisker trials for {mouse} on day 0")
        return None

    # Get lick sequence to find first hit
    licks = whisker_trials['lick_flag'].values

    # Find first hit
    hit_indices = np.where(licks == 1)[0]
    if len(hit_indices) == 0:
        if verbose:
            print(f"  Warning: No whisker hits for {mouse} on day 0")
        return None

    first_hit_idx = hit_indices[0]

    if verbose:
        print(f"  {mouse}: First hit at whisker trial {first_hit_idx} (of {n_whisker_trials} total)")

    # Create template for day 0
    try:
        template, cells_mask = create_whisker_template(mouse, day=0,
                                                       threshold_dff=threshold_dff,
                                                       verbose=False)
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not create template for {mouse}: {e}")
        return None

    # Load surrogate thresholds if available
    # Choose file based on threshold_mode
    if threshold_mode == 'mouse':
        surrogate_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates', 'surrogate_thresholds.csv')
    else:  # threshold_mode == 'day'
        surrogate_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates_per_day', 'surrogate_thresholds_per_day.csv')

    threshold_dict = None
    if os.path.exists(surrogate_csv_path):
        try:
            threshold_dict = load_surrogate_thresholds(surrogate_csv_path, threshold_type='percentile', threshold_mode=threshold_mode)
        except:
            pass

    threshold = get_threshold_for_mouse_day(threshold_dict, mouse, 0, threshold_corr, threshold_mode)

    # Get trial groups
    # Before: all miss trials before first hit
    if first_hit_idx > 0:
        miss_trials_before = whisker_trials.isel(trial=slice(0, first_hit_idx))
    else:
        miss_trials_before = None

    # At first hit: the first hit trial
    first_hit_trial = whisker_trials.isel(trial=slice(first_hit_idx, first_hit_idx + 1))

    # After: next 5 whisker trials (or fewer if not available)
    n_after = min(5, n_whisker_trials - first_hit_idx - 1)
    if n_after > 0:
        trials_after = whisker_trials.isel(trial=slice(first_hit_idx + 1, first_hit_idx + 1 + n_after))
    else:
        trials_after = None

    # Calculate per-trial frequencies for each group
    freq_before = np.nan
    n_misses_before = 0
    if miss_trials_before is not None and len(miss_trials_before.trial) > 0:
        freqs_before = calculate_per_trial_reactivation_frequency(
            miss_trials_before, template, threshold,
            min_event_distance_frames, prominence,
            sampling_rate=sampling_rate,
            blank_stimulus=True,
            blank_window=(0, 1)
        )
        freq_before = np.mean(freqs_before)
        n_misses_before = len(freqs_before)

    # Frequency at first hit
    freq_at_hit_array = calculate_per_trial_reactivation_frequency(
        first_hit_trial, template, threshold,
        min_event_distance_frames, prominence,
        sampling_rate=sampling_rate,
        blank_stimulus=True,
        blank_window=(0, 1)
    )
    freq_at_hit = freq_at_hit_array[0]

    # Frequency after first hit
    freq_after = np.nan
    n_trials_after = 0
    if trials_after is not None and len(trials_after.trial) > 0:
        freqs_after = calculate_per_trial_reactivation_frequency(
            trials_after, template, threshold,
            min_event_distance_frames, prominence,
            sampling_rate=sampling_rate,
            blank_stimulus=True,
            blank_window=(0, 1)
        )
        freq_after = np.mean(freqs_after)
        n_trials_after = len(freqs_after)

    if verbose:
        print(f"    Freq before: {freq_before:.2f} (n={n_misses_before}), " +
              f"At hit: {freq_at_hit:.2f}, After: {freq_after:.2f} (n={n_trials_after})")

    return {
        'mouse': mouse,
        'freq_before': freq_before,
        'freq_at_hit': freq_at_hit,
        'freq_after': freq_after,
        'n_misses_before': n_misses_before,
        'n_trials_after': n_trials_after
    }


def analyze_reactivation_trial_by_trial(mouse, n_trials_after_hit=60):
    """
    Calculate per-trial reactivation frequency starting from first whisker hit.

    Returns array of frequencies: [first_hit, trial+1, trial+2, ..., trial+n]
    """
    # Load data
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_learning_data.nc'

    try:
        xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)
    except:
        return None

    # Get day 0 whisker trials
    xarray_day0 = xarray_learning.sel(trial=xarray_learning['day'] == 0)
    whisker_mask = xarray_day0['whisker_stim'] == 1
    whisker_trials = xarray_day0.sel(trial=whisker_mask)

    # Find first hit
    licks = whisker_trials['lick_flag'].values
    hit_indices = np.where(licks == 1)[0]
    if len(hit_indices) == 0:
        return None

    first_hit_idx = hit_indices[0]

    # Get trials from first hit onward
    n_available = len(whisker_trials.trial) - first_hit_idx
    n_to_use = min(n_trials_after_hit + 1, n_available)  # +1 to include first hit

    if n_to_use < 1:
        return None

    trials_from_hit = whisker_trials.isel(trial=slice(first_hit_idx, first_hit_idx + n_to_use))

    # Create template
    try:
        template, _ = create_whisker_template(mouse, day=0, threshold_dff=threshold_dff, verbose=False)
    except:
        return None

    # Load threshold
    # Choose file based on threshold_mode
    if threshold_mode == 'mouse':
        surrogate_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates', 'surrogate_thresholds.csv')
    else:  # threshold_mode == 'day'
        surrogate_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates_per_day', 'surrogate_thresholds_per_day.csv')

    threshold_dict = None
    if os.path.exists(surrogate_csv_path):
        try:
            threshold_dict = load_surrogate_thresholds(surrogate_csv_path, threshold_type='percentile', threshold_mode=threshold_mode)
        except:
            pass

    threshold = get_threshold_for_mouse_day(threshold_dict, mouse, 0, threshold_corr, threshold_mode)

    # Calculate per-trial frequencies
    frequencies = calculate_per_trial_reactivation_frequency(
        trials_from_hit, template, threshold,
        min_event_distance_frames, prominence,
        sampling_rate=sampling_rate,
        blank_stimulus=True,
        blank_window=(0, 1)
    )

    return frequencies


def plot_reactivation_trial_by_trial(r_plus_results, r_minus_results, save_path, n_trials_after_hit=60):
    """
    Plot trial-by-trial reactivation frequency aligned to first whisker hit.

    Single panel with two lines (R+ and R-) showing average frequency trajectory
    starting from first hit (x=0) through subsequent whisker trials.
    """
    import pandas as pd

    print(f"\n  Analyzing trial-by-trial reactivation from first hit (+{n_trials_after_hit} trials)...")

    # Collect data
    data_list = []

    # Process R+ mice
    for mouse_id in r_plus_results.keys():
        freqs = analyze_reactivation_trial_by_trial(mouse_id, n_trials_after_hit)
        if freqs is not None:
            for trial_idx, freq in enumerate(freqs):
                data_list.append({
                    'Mouse': mouse_id,
                    'Group': 'R+',
                    'Trial': trial_idx,
                    'Frequency': freq
                })

    # Process R- mice
    for mouse_id in r_minus_results.keys():
        freqs = analyze_reactivation_trial_by_trial(mouse_id, n_trials_after_hit)
        if freqs is not None:
            for trial_idx, freq in enumerate(freqs):
                data_list.append({
                    'Mouse': mouse_id,
                    'Group': 'R-',
                    'Trial': trial_idx,
                    'Frequency': freq
                })

    df = pd.DataFrame(data_list)

    print(f"  Collected data from {len(df['Mouse'].unique())} mice")

    # Create single panel figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot with seaborn lineplot
    sns.lineplot(
        data=df, x='Trial', y='Frequency', hue='Group',
        errorbar=('ci', 95),
        palette={'R+': reward_palette[1], 'R-': reward_palette[0]},
        linewidth=2.5, ax=ax
    )

    # Formatting
    ax.set_xlabel('Whisker Trial (0 = First Hit)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reactivation Frequency (events/min)', fontsize=14, fontweight='bold')
    ax.set_title('Trial-by-Trial Reactivation from First Whisker Hit (Day 0)',
                 fontsize=16, fontweight='bold', pad=10)
    ax.legend(title='Group', fontsize=12, title_fontsize=13, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Add vertical line at first hit
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='First Hit')

    sns.despine(fig=fig)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved trial-by-trial reactivation plot: {save_path}")
        plt.close()

    return fig


def plot_reactivation_around_first_hit(r_plus_results, r_minus_results, save_path):
    """
    Plot reactivation frequency around the first whisker hit trial on day 0.

    Creates a 2-panel figure (R+ and R-) showing average reactivation frequency
    at three timepoints: before first hit, at first hit, and after first hit.

    Parameters
    ----------
    r_plus_results : dict
        {mouse_id: results} for R+ mice
    r_minus_results : dict
        {mouse_id: results} for R- mice
    save_path : str
        Path to save the SVG figure
    """
    print("\n  Analyzing reactivation around first whisker hit on day 0...")

    # Collect data for both groups
    data_list = []

    # Process R+ mice
    print(f"  Processing {len(r_plus_results)} R+ mice...")
    for mouse_id in r_plus_results.keys():
        result = analyze_reactivation_around_first_hit(mouse_id, verbose=False)
        if result is not None:
            # Add three rows for this mouse (one per timepoint)
            data_list.append({
                'Mouse': mouse_id,
                'Group': 'R+',
                'Timepoint': 'Before',
                'Frequency': result['freq_before']
            })
            data_list.append({
                'Mouse': mouse_id,
                'Group': 'R+',
                'Timepoint': 'At First Hit',
                'Frequency': result['freq_at_hit']
            })
            data_list.append({
                'Mouse': mouse_id,
                'Group': 'R+',
                'Timepoint': 'After',
                'Frequency': result['freq_after']
            })

    # Process R- mice
    print(f"  Processing {len(r_minus_results)} R- mice...")
    for mouse_id in r_minus_results.keys():
        result = analyze_reactivation_around_first_hit(mouse_id, verbose=False)
        if result is not None:
            # Add three rows for this mouse (one per timepoint)
            data_list.append({
                'Mouse': mouse_id,
                'Group': 'R-',
                'Timepoint': 'Before',
                'Frequency': result['freq_before']
            })
            data_list.append({
                'Mouse': mouse_id,
                'Group': 'R-',
                'Timepoint': 'At First Hit',
                'Frequency': result['freq_at_hit']
            })
            data_list.append({
                'Mouse': mouse_id,
                'Group': 'R-',
                'Timepoint': 'After',
                'Frequency': result['freq_after']
            })

    # Create DataFrame
    df = pd.DataFrame(data_list)

    print(f"  Collected data from {len(df['Mouse'].unique())} mice total")

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Define timepoint order
    timepoint_order = ['Before', 'At First Hit', 'After']

    # Plot R+ mice
    df_rplus = df[df['Group'] == 'R+']
    if len(df_rplus) > 0:
        sns.pointplot(data=df_rplus, x='Timepoint', y='Frequency',
                     order=timepoint_order,
                     errorbar=('ci', 95),
                     estimator=np.nanmean,
                     markers='o', linestyles='-',
                     color=reward_palette[1], ax=ax1)

        # Add individual mouse lines
        for mouse in df_rplus['Mouse'].unique():
            mouse_data = df_rplus[df_rplus['Mouse'] == mouse]
            # Plot Before → At First Hit if 'Before' exists
            if 'Before' in mouse_data['Timepoint'].values:
                before_val = mouse_data[mouse_data['Timepoint'] == 'Before']['Frequency'].values
                at_hit_val = mouse_data[mouse_data['Timepoint'] == 'At First Hit']['Frequency'].values
                ax1.plot([0, 1], [before_val[0], at_hit_val[0]], 'o-', alpha=0.3, color=reward_palette[1], linewidth=1, markersize=4)
            # Always plot At First Hit → After
            at_hit_val = mouse_data[mouse_data['Timepoint'] == 'At First Hit']['Frequency'].values
            after_val = mouse_data[mouse_data['Timepoint'] == 'After']['Frequency'].values
            ax1.plot([1, 2], [at_hit_val[0], after_val[0]], 'o-', alpha=0.3, color=reward_palette[1], linewidth=1, markersize=4)

    ax1.set_ylabel('Reactivation Frequency (events/min)', fontsize=14, fontweight='bold')
    ax1.set_title(f'R+ Mice: Reactivation Around First Hit\n(Day 0, n={len(df_rplus["Mouse"].unique())} mice)',
                  fontsize=16, fontweight='bold', pad=10)

    # Plot R- mice
    df_rminus = df[df['Group'] == 'R-']
    if len(df_rminus) > 0:
        sns.pointplot(data=df_rminus, x='Timepoint', y='Frequency',
                     order=timepoint_order,
                     errorbar=('ci', 95),
                     estimator=np.nanmean,
                     markers='o', linestyles='-',
                     color=reward_palette[0], ax=ax2)

        # Add individual mouse lines
        for mouse in df_rminus['Mouse'].unique():
            mouse_data = df_rminus[df_rminus['Mouse'] == mouse]
            # Plot Before → At First Hit if 'Before' exists
            if 'Before' in mouse_data['Timepoint'].values:
                before_val = mouse_data[mouse_data['Timepoint'] == 'Before']['Frequency'].values
                at_hit_val = mouse_data[mouse_data['Timepoint'] == 'At First Hit']['Frequency'].values
                ax2.plot([0, 1], [before_val[0], at_hit_val[0]], 'o-', alpha=0.3, color=reward_palette[0], linewidth=1, markersize=4)
            # Always plot At First Hit → After
            at_hit_val = mouse_data[mouse_data['Timepoint'] == 'At First Hit']['Frequency'].values
            after_val = mouse_data[mouse_data['Timepoint'] == 'After']['Frequency'].values
            ax2.plot([1, 2], [at_hit_val[0], after_val[0]], 'o-', alpha=0.3, color=reward_palette[0], linewidth=1, markersize=4)

    ax2.set_ylabel('Reactivation Frequency (events/min)', fontsize=14, fontweight='bold')
    ax2.set_title(f'R- Mice: Reactivation Around First Hit\n(Day 0, n={len(df_rminus["Mouse"].unique())} mice)',
                  fontsize=16, fontweight='bold', pad=10)

    sns.despine(fig=fig)

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved reactivation around first hit plot: {save_path}")
        plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("REACTIVATION AROUND FIRST HIT SUMMARY")
    print("="*60)

    for group_name, df_group in [('R+', df_rplus), ('R-', df_rminus)]:
        print(f"\n{group_name} Mice:")
        for timepoint in timepoint_order:
            tp_data = df_group[df_group['Timepoint'] == timepoint]['Frequency']
            if len(tp_data) > 0:
                mean_freq = tp_data.mean()
                sem_freq = tp_data.sem()
                n_mice = len(tp_data)
                print(f"  {timepoint:15s}: {mean_freq:.2f} ± {sem_freq:.2f} events/min (n={n_mice})")
            else:
                print(f"  {timepoint:15s}: No data")

    return fig


def plot_temporal_dynamics_across_mice(r_plus_results, r_minus_results, save_path):
    """
    Plot temporal reactivation dynamics averaged across mice for each day.

    Creates a 5-page PDF with one page per day, showing R+ and R- traces side by side.

    Parameters
    ----------
    r_plus_results : dict
        {mouse_id: results} for R+ mice
    r_minus_results : dict
        {mouse_id: results} for R- mice
    save_path : str
        Path to save PDF
    """
    print("\n  Generating temporal dynamics across mice PDF...")

    with PdfPages(save_path) as pdf:
        for day_idx, day in enumerate(days):
            # Collect temporal data from all mice for this day
            r_plus_temporal_data = []
            r_minus_temporal_data = []

            # Extract R+ mice temporal data
            for mouse, results in r_plus_results.items():
                if day in results['days'] and 'temporal' in results['days'][day]:
                    temporal = results['days'][day]['temporal']
                    if len(temporal['time_bins']) > 0:  # Check not empty
                        r_plus_temporal_data.append({
                            'time_bins': temporal['time_bins'],
                            'event_rate': temporal['event_rate'],
                            'event_rate_sem': temporal['event_rate_sem'],
                            'n_trials': temporal['n_trials']
                        })

            # Extract R- mice temporal data
            for mouse, results in r_minus_results.items():
                if day in results['days'] and 'temporal' in results['days'][day]:
                    temporal = results['days'][day]['temporal']
                    if len(temporal['time_bins']) > 0:  # Check not empty
                        r_minus_temporal_data.append({
                            'time_bins': temporal['time_bins'],
                            'event_rate': temporal['event_rate'],
                            'event_rate_sem': temporal['event_rate_sem'],
                            'n_trials': temporal['n_trials']
                        })

            # Skip this day if no data available for both groups
            if len(r_plus_temporal_data) == 0 and len(r_minus_temporal_data) == 0:
                print(f"    Warning: No temporal data for day {day}, skipping...")
                continue

            # Create figure with two panels (R+ and R-)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

            # Process each group
            for group_idx, (group_data, group_name, group_color, ax) in enumerate([
                (r_plus_temporal_data, 'R+', reward_palette[1], axes[0]),
                (r_minus_temporal_data, 'R-', reward_palette[0], axes[1])
            ]):
                if len(group_data) == 0:
                    ax.text(0.5, 0.5, f'No data available for {group_name} mice',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, color='gray')
                    ax.set_title(f'{group_name} Mice (n=0)', fontweight='bold', fontsize=14)
                    continue

                # Aggregate across mice
                # First, find common time bins (use the most common binning)
                all_time_bins = [d['time_bins'] for d in group_data]
                all_n_bins = [len(tb) for tb in all_time_bins]

                # Use the most common number of bins
                from collections import Counter

                most_common_n_bins = Counter(all_n_bins).most_common(1)[0][0]

                # Filter to only mice with this number of bins
                filtered_data = [d for d in group_data if len(d['time_bins']) == most_common_n_bins]

                if len(filtered_data) == 0:
                    ax.text(0.5, 0.5, f'Inconsistent binning\nfor {group_name} mice',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, color='gray')
                    ax.set_title(f'{group_name} Mice (data issue)', fontweight='bold', fontsize=14)
                    continue

                # Stack event rates across mice
                time_bins = filtered_data[0]['time_bins'] 
                event_rates_all_mice = np.array([d['event_rate'] for d in filtered_data])

                # Compute mean and SEM across mice
                mean_event_rate = np.mean(event_rates_all_mice, axis=0)
                sem_event_rate = np.std(event_rates_all_mice, axis=0) / np.sqrt(len(filtered_data))

                # Plot mean trace with confidence interval
                ax.plot(time_bins, mean_event_rate, '-', linewidth=3,
                       color=group_color, label='Mean ± SEM', zorder=3)
                ax.fill_between(time_bins,
                               mean_event_rate - sem_event_rate,
                               mean_event_rate + sem_event_rate,
                               alpha=0.3, color=group_color, zorder=2)

                # Add stimulus onset marker (at t=0 for stimulus trials)
                if trial_type != 'no_stim':
                    ax.axvline(0, color='orange', linestyle='-', linewidth=2,
                              alpha=0.7, label='Stimulus onset', zorder=1)

                # Formatting
                ax.set_xlabel('Time in Trial (s)', fontweight='bold', fontsize=13)
                ax.set_ylabel('Reactivation Rate (events/s)', fontweight='bold', fontsize=13)
                ax.set_title(f'{group_name} Mice (n={len(filtered_data)})',
                            fontweight='bold', fontsize=14, color=group_color)
                ax.legend(loc='best', fontsize=11)

                
                # Add summary statistics
                mean_rate = np.mean(mean_event_rate)
                max_rate = np.max(mean_event_rate)
                max_time = time_bins[np.argmax(mean_event_rate)]

                stats_text = f'Mean rate: {mean_rate:.3f} events/s\n'
                stats_text += f'Peak rate: {max_rate:.3f} events/s @ {max_time:.2f}s'

                ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                                edgecolor=group_color, linewidth=2))

            # Overall title
            day_str = days_str[day_idx]
            fig.suptitle(f'Temporal Reactivation Dynamics - Day {day_str}\n{trial_type.replace("_", " ").title()} Trials',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"  ✓ Saved temporal dynamics across mice PDF: {save_path}")


def plot_trial_type_comparison_day0(save_dir):
    """
    Compare reactivation frequencies across trial types for day 0 within each reward group.

    Loads all trial type results and creates a 2-panel figure comparing reactivation
    frequencies for different trial types (no_stim, whisker_hit, whisker_miss, auditory_hit)
    separately for R+ and R- mice.

    Parameters
    ----------
    save_dir : str
        Directory where trial type result files are stored and where figure will be saved
    """
    # Define trial types and their labels
    # trial_types = ['no_stim', 'whisker_hit', 'whisker_miss', 'auditory_hit']
    # trial_labels = ['No Stim', 'Whisker Hit', 'Whisker Miss', 'Auditory Hit']
    trial_types = ['no_stim', 'whisker']
    trial_labels = ['No Stim', 'Whisker']

    # Map trial types to color indices
    # Palette order: auditory misses, auditory hits, whisker misses, whisker hits, correct rejection, false alarm
    color_map = {
        'no_stim': 4,        # Use correct rejection color (grey)
        'whisker': 3    # Whisker
    }

    print("\n" + "="*60)
    print("GENERATING TRIAL TYPE COMPARISON PLOT (DAY 0)")
    print("="*60)

    # Load all trial type results
    all_trial_data = {}
    for tt in trial_types:
        results_file = os.path.join(save_dir, f'reactivation_results_{tt}.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                all_trial_data[tt] = pickle.load(f)
            print(f"  ✓ Loaded {tt} results")
        else:
            print(f"  ✗ Warning: {tt} results not found, skipping")

    if len(all_trial_data) == 0:
        print("  Error: No trial type results found. Run with mode='compute' first.")
        return

    # Create figure with two panels
    fig, (ax_rplus, ax_rminus) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Process each reward group
    for ax, group_name, palette in [
        (ax_rplus, 'R+', trial_type_rew_palette),
        (ax_rminus, 'R-', trial_type_nonrew_palette)
    ]:
        # Collect data for this reward group across trial types
        plot_data = []

        for tt in trial_types:
            if tt not in all_trial_data:
                continue

            # Get results for this reward group
            if group_name == 'R+':
                group_results = all_trial_data[tt]['r_plus_results']
            else:
                group_results = all_trial_data[tt]['r_minus_results']

            # Extract day 0 frequencies
            for mouse, results in group_results.items():
                if 0 in results['days']:
                    frequency = results['days'][0]['event_frequency']
                    plot_data.append({
                        'Trial Type': trial_labels[trial_types.index(tt)],
                        'Frequency': frequency,
                        'Mouse': mouse,
                        'trial_type_key': tt
                    })

        # Convert to DataFrame
        df = pd.DataFrame(plot_data)

        if len(df) > 0:
            # Create color list for this reward group
            colors = [palette[color_map[tt]] for tt in trial_types if tt in all_trial_data]

            # Create barplot
            sns.barplot(data=df, x='Trial Type', y='Frequency', ax=ax,
                       palette=colors, alpha=0.7, errorbar='se',
                       )

            # Add individual data points
            sns.stripplot(data=df, x='Trial Type', y='Frequency', ax=ax,
                         color='black', alpha=0.5, size=5, jitter=0.2)

            # Formatting
            ax.set_xlabel('Trial Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Reactivation Frequency (events/min)', fontsize=12, fontweight='bold')
            ax.set_title(f'{group_name} Mice\n(n={df["Mouse"].nunique()} mice)',
                        fontsize=13, fontweight='bold', color=reward_palette[1 if group_name == 'R+' else 0])
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=15)

            # Add sample sizes to x-tick labels
            trial_counts = df.groupby('Trial Type')['Mouse'].nunique()
            new_labels = []
            for label in ax.get_xticklabels():
                trial_label = label.get_text()
                if trial_label in trial_counts.index:
                    n = trial_counts[trial_label]
                    new_labels.append(f'{trial_label}\n(n={n})')
                else:
                    new_labels.append(trial_label)
            ax.set_xticklabels(new_labels, fontsize=10)
        else:
            ax.text(0.5, 0.5, f'No data for {group_name}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='gray')
            ax.set_title(f'{group_name} Mice', fontsize=13, fontweight='bold')

    # Overall title
    fig.suptitle('Reactivation Frequency by Trial Type (Day 0)\nMean ± SEM',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, 'trial_type_comparison_day0.svg')
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved trial type comparison: {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_single_mouse(mouse, save_dir, verbose=False, threshold_dict=None):
    """
    Process a single mouse (analysis + PDF generation).

    Parameters
    ----------
    mouse : str
        Mouse ID
    save_dir : str
        Directory to save PDF
    verbose : bool
        Print progress information
    threshold_dict : dict, optional
        Per-mouse, per-day thresholds from surrogate analysis

    Returns
    -------
    tuple
        (mouse_id, results_dict)
    """
    if verbose:
        print(f"\nProcessing {mouse}...")

    results = analyze_mouse_reactivation(mouse, days=days, verbose=verbose, threshold_dict=threshold_dict)
    generate_mouse_pdf(results, save_dir)

    if verbose:
        print(f"  Completed {mouse}")

    return (mouse, results)


def process_mouse_group(mice_list, group_name, save_dir, n_jobs=30, threshold_dict=None):
    """
    Process a group of mice and generate their figures in parallel.

    Parameters
    ----------
    mice_list : list
        List of mouse IDs to process
    group_name : str
        Name of the group (e.g., 'R+', 'R-')
    save_dir : str
        Directory to save results
    n_jobs : int
        Number of parallel jobs (default: 30)
    threshold_dict : dict, optional
        Per-mouse, per-day thresholds from surrogate analysis

    Returns
    -------
    all_results : dict
        Dictionary with results for all mice in the group
    """
    print("\n" + "="*60)
    print(f"ANALYZING {group_name} MICE")
    print("="*60)
    print(f"Mice in group: {mice_list}")
    print(f"Processing {len(mice_list)} mice in parallel using {n_jobs} cores...")

    # Process all mice in parallel
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_mouse)(mouse, save_dir, verbose=False, threshold_dict=threshold_dict)
        for mouse in mice_list
    )

    # Convert list of tuples to dictionary
    all_results = dict(results_list)

    print(f"\nCompleted processing {len(all_results)} {group_name} mice")

    return all_results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("REACTIVATION EVENT DETECTION AND ANALYSIS")
    print("="*60)
    print(f"\nAnalysis mode: {mode.upper()}")
    print(f"\nParameters:")
    if mode == 'compute':
        print(f"  Trial types: ALL (no_stim, whisker_hit, whisker_miss, auditory_hit)")
    else:
        print(f"  Trial type: {trial_type.replace('_', ' ').title()}")
    print(f"  Correlation threshold: {threshold_corr} (default)")
    print(f"  Min event distance: {min_event_distance_ms}ms ({min_event_distance_frames} frames)")
    print(f"  Prominence: {prominence}")
    print(f"  DFF threshold: {threshold_dff*100 if threshold_dff is not None else 'None'}%")
    print(f"  Days: {days}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"\nMice groups:")
    print(f"  R+ mice ({len(r_plus_mice)}): {r_plus_mice}")
    print(f"  R- mice ({len(r_minus_mice)}): {r_minus_mice}")

    # Create output directory
    save_dir = os.path.join(io.results_dir, 'reactivation')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nResults will be saved to: {save_dir}")

    # Define results file path (only relevant for analyze mode)
    results_file = os.path.join(save_dir, 'reactivation_results.pkl')

    if mode == 'compute':
        # ===== COMPUTE MODE: Run analysis for no_stim trials only =====
        print("\n" + "="*60)
        print("COMPUTING REACTIVATION ANALYSIS (NO_STIM TRIALS)")
        print("="*60)

        # Load surrogate thresholds if available
        # Choose file based on threshold_mode
        if threshold_mode == 'mouse':
            surrogate_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates', 'surrogate_thresholds.csv')
        elif threshold_mode == 'day':
            surrogate_csv_path = os.path.join(io.results_dir, 'reactivation_surrogates_per_day', 'surrogate_thresholds_per_day.csv')
        else:
            raise ValueError(f"Invalid threshold_mode '{threshold_mode}'. Must be 'mouse' or 'day'.")

        threshold_dict = None
        if os.path.exists(surrogate_csv_path):
            print(f"\n{'='*60}")
            print("LOADING SURROGATE-BASED THRESHOLDS")
            print(f"{'='*60}")
            threshold_dict = load_surrogate_thresholds(surrogate_csv_path, threshold_type=threshold_type, threshold_mode=threshold_mode)
            print(f"Loaded thresholds from: {surrogate_csv_path}")
            print(f"Threshold type: {threshold_type} ({'percentile-based' if threshold_type in ['percentile', '95'] else 'FWER/maximum'})")
            print(f"Threshold mode: {threshold_mode} ({'one per mouse' if threshold_mode == 'mouse' else 'per mouse-day'})")
            if threshold_mode == 'mouse':
                print(f"Loaded thresholds for {len(threshold_dict)} mice")
            else:
                n_mice = len(threshold_dict)
                n_mouse_days = sum(len(days_dict) for days_dict in threshold_dict.values())
                print(f"Loaded thresholds for {n_mice} mice, {n_mouse_days} mouse-day combinations")
        else:
            print(f"\nSurrogate threshold file not found: {surrogate_csv_path}")
            print(f"Using default threshold: {threshold_corr}")

        # Process R+ mice in parallel
        print(f"\nProcessing R+ mice...")
        r_plus_results = process_mouse_group(r_plus_mice, 'R+', save_dir, n_jobs=n_jobs, threshold_dict=threshold_dict)

        # Process R- mice in parallel
        print(f"\nProcessing R- mice...")
        r_minus_results = process_mouse_group(r_minus_mice, 'R-', save_dir, n_jobs=n_jobs, threshold_dict=threshold_dict)

        # Save results to file
        results_file = os.path.join(save_dir, 'reactivation_results.pkl')
        print("\n" + "-"*60)
        print(f"SAVING RESULTS")
        print("-"*60)
        results_data = {
            'r_plus_results': r_plus_results,
            'r_minus_results': r_minus_results,
            'parameters': {
                'trial_type': 'no_stim',
                'threshold_corr': threshold_corr,
                'min_event_distance_ms': min_event_distance_ms,
                'prominence': prominence,
                'days': days
            }
        }
        with open(results_file, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"✓ Saved results to: {results_file}")

        # Exit after computing (don't generate plots in compute mode)
        print("\nTo generate plots, run with mode='analyze'")
        import sys
        sys.exit(0)

    elif mode == 'analyze':
        # ===== ANALYZE MODE: Load results and generate plots =====
        print("\n" + "="*60)
        print("LOADING SAVED RESULTS")
        print("="*60)

        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}\nRun with mode='compute' first.")

        with open(results_file, 'rb') as f:
            results_data = pickle.load(f)

        r_plus_results = results_data['r_plus_results']
        r_minus_results = results_data['r_minus_results']
        params = results_data['parameters']

        print(f"✓ Loaded results from: {results_file}")
        print(f"\nParameters from saved results:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"\nLoaded {len(r_plus_results)} R+ mice and {len(r_minus_results)} R- mice")

    else:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'compute' or 'analyze'")

    # Generate threshold comparison plots
    # threshold_pdf_path = os.path.join(save_dir, 'threshold_comparison.pdf')
    # plot_threshold_comparison(threshold_pdf_path)

    # Generate across-mice comparison figures (combining R+ and R-)
    print("\n" + "="*60)
    print("GENERATING ACROSS-MICE COMPARISON FIGURES")
    print("="*60)

    # Figure 1: Session-level analysis (R+ vs R-)
    svg_path = os.path.join(save_dir, 'across_mice_session_level_comparison.svg')
    plot_session_level_across_mice(r_plus_results, r_minus_results, svg_path)

    # Figure 2: Block-level analysis (R+ vs R-)
    svg_path = os.path.join(save_dir, 'across_mice_block_level_comparison.svg')
    plot_block_level_across_mice(r_plus_results, r_minus_results, svg_path)

    # Figure 3: Events per day (R+ vs R-)
    svg_path = os.path.join(save_dir, 'across_mice_events_per_day_comparison.svg')
    plot_events_per_day_across_mice(r_plus_results, r_minus_results, svg_path)

    # Figure 4: Direct R+ vs R- comparison with statistics
    svg_path = os.path.join(save_dir, 'across_mice_group_comparison_per_day.svg')
    plot_group_comparison_per_day(r_plus_results, r_minus_results, svg_path)

    # Figure 5: Reactivation frequency vs performance improvement (day 0 → day +1)
    svg_path = os.path.join(save_dir, 'reactivation_vs_performance_delta.svg')
    plot_reactivation_vs_performance_delta(r_plus_results, r_minus_results, svg_path)

    # Figure 6: Temporal reactivation dynamics across mice (5-page PDF, one per day)
    pdf_path = os.path.join(save_dir, 'temporal_dynamics_across_mice.pdf')
    plot_temporal_dynamics_across_mice(r_plus_results, r_minus_results, pdf_path)

    # Generate time-above-threshold plots (SVG)
    print("\n" + "="*60)
    print("GENERATING TIME-ABOVE-THRESHOLD VISUALIZATIONS")
    print("="*60)

    # Plot 1: Percent time above per day (two-panel plot with R+ and R-)
    if len(r_plus_results) > 0 or len(r_minus_results) > 0:
        svg_path = os.path.join(save_dir, 'percent_time_above_per_day.svg')
        print(f"\nGenerating two-panel time-above per day plot...")
        plot_percent_time_above_per_day(r_plus_results, r_minus_results, svg_path)

    # Plot 2: Percent time above vs performance (combined R+ and R-)
    if len(r_plus_results) > 0 or len(r_minus_results) > 0:
        svg_path = os.path.join(save_dir, 'percent_time_above_vs_performance.svg')
        print(f"\nGenerating time-above vs performance plot...")
        plot_percent_time_above_vs_performance(r_plus_results, r_minus_results, svg_path)

    print(f"\nTime-above-threshold plots saved to: {save_dir}")

    # # Figure 8 & 9: Within-day-0 performance improvement vs reactivation frequency
    # # These analyses require loading results from specific trial types
    # print("\n" + "="*60)
    # print("WITHIN-DAY-0 PERFORMANCE VS REACTIVATION ANALYSIS")
    # print("="*60)

    # # Load no_stim results
    # nostim_results_file = os.path.join(save_dir, 'reactivation_results_no_stim.pkl')
    # if os.path.exists(nostim_results_file):
    #     with open(nostim_results_file, 'rb') as f:
    #         nostim_data = pickle.load(f)
    #     r_plus_nostim = nostim_data['r_plus_results']
    #     r_minus_nostim = nostim_data['r_minus_results']

    #     # Figure 8: Within-day-0 performance vs no_stim reactivation
    #     svg_path = os.path.join(save_dir, 'within_day0_performance_vs_nostim_reactivation.svg')
    #     plot_within_day0_performance_vs_reactivation(r_plus_nostim, r_minus_nostim, 'no_stim', svg_path)
    # else:
    #     print(f"  Warning: No_stim results not found at {nostim_results_file}")
    #     print(f"  Skipping within-day-0 performance vs no_stim reactivation analysis")

    # # Load whisker_hit results
    # whiskerhit_results_file = os.path.join(save_dir, 'reactivation_results_whisker_hit.pkl')
    # if os.path.exists(whiskerhit_results_file):
    #     with open(whiskerhit_results_file, 'rb') as f:
    #         whiskerhit_data = pickle.load(f)
    #     r_plus_whiskerhit = whiskerhit_data['r_plus_results']
    #     r_minus_whiskerhit = whiskerhit_data['r_minus_results']

    #     # Figure 9: Within-day-0 performance vs whisker_hit reactivation
    #     svg_path = os.path.join(save_dir, 'within_day0_performance_vs_whiskerhit_reactivation.svg')
    #     plot_within_day0_performance_vs_reactivation(r_plus_whiskerhit, r_minus_whiskerhit, 'whisker_hit', svg_path)
    # else:
    #     print(f"  Warning: Whisker_hit results not found at {whiskerhit_results_file}")
    #     print(f"  Skipping within-day-0 performance vs whisker_hit reactivation analysis")

    # # Figure 10: Reactivation frequency around first whisker hit
    # print("\n" + "="*60)
    # print("FIRST WHISKER HIT ANALYSIS")
    # print("="*60)

    # svg_path = os.path.join(save_dir, 'reactivation_around_first_hit_day0.svg')
    # plot_reactivation_around_first_hit(r_plus_results, r_minus_results, svg_path)

    # # Figure 11: Trial-by-trial reactivation trajectory from first hit
    # svg_path = os.path.join(save_dir, 'reactivation_trial_by_trial_from_first_hit_day0.svg')
    # plot_reactivation_trial_by_trial(r_plus_results, r_minus_results, svg_path, n_trials_after_hit=60)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(r_plus_results)} R+ mice")
    print(f"Processed {len(r_minus_results)} R- mice")
    print(f"Total: {len(r_plus_results) + len(r_minus_results)} mice")
    print(f"\nPDFs saved to: {save_dir}")
    print(f"SVG figures saved to: {save_dir}")



# # Correlation trace smoothing and event detection demo

# import scipy.ndimage
# import matplotlib.pyplot as plt

# mouse = 'AR163'
# day = 0
# min_event_distance_frames = 15
# threshold = 0.5
# prominence = 0.15

# # Create template
# template, cells_mask = create_whisker_template(mouse, day, threshold_dff=0.05, verbose=True)

# # Load learning data
# folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
# file_name = 'tensor_xarray_learning_data.nc'
# xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)

# # Select trials for day 0
# xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)
# selected_trials, n_selected_trials = select_trials_by_type(xarray_day, 'no_stim')

# # Prepare data and compute correlations
# n_cells, n_trials, n_timepoints = selected_trials.shape
# data = selected_trials.values.reshape(n_cells, -1)
# data = np.nan_to_num(data, nan=0.0)
# correlations = compute_template_correlation(data, template)

# # Detect events using prominence-based peak detection
# events_nodist_noprominence = detect_reactivation_events(correlations, threshold, 1, prominence=0)
# events_dist_noprominence = detect_reactivation_events(correlations, threshold, min_event_distance_frames, prominence=0)
# events_dist_prominence = detect_reactivation_events(correlations, threshold, min_event_distance_frames, prominence=prominence)

# n_segments = 4
# frames_per_segment = len(correlations) // n_segments

# # Plot segmented traces with event markers
# fig, axes = plt.subplots(n_segments, 2, figsize=(18, 2.5 * n_segments), sharey=True)
# trace_types = [
#     ('With dist, no prominence', correlations, events_dist_noprominence),
#     ('With distance and prominence', correlations, events_dist_prominence),
# ]
# for seg_idx in range(n_segments):
#     start = seg_idx * frames_per_segment
#     end = (seg_idx + 1) * frames_per_segment if seg_idx < n_segments - 1 else len(correlations)
#     for col_idx, (label, trace, events) in enumerate(trace_types):
#         ax = axes[seg_idx, col_idx]
#         ax.plot(np.arange(start, end), trace[start:end], label=f'{label}')
#         ax.axhline(threshold, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
#         for ev in events:
#             if start <= ev < end:
#                 ax.axvline(ev, color='red', linestyle='-', linewidth=1.2, alpha=0.7)
#         ax.set_xlabel('Frame')
#         ax.set_ylabel('Correlation')
#         if seg_idx == 0:
#             ax.set_title(label)
#             ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.3)
# plt.suptitle(f'{mouse} Day {day} Correlation Trace (no_stim trials) - Raw vs Filtered - Events Marked')
# plt.tight_layout()
# plt.show()

# trace_types = [
#     ('No distance no prominence', correlations, events_nodist_noprominence),
#     ('With dist, no prominence', correlations, events_dist_noprominence),
#     ('With distance and prominence', correlations, events_dist_prominence),
# ]
# # Zoomed plot: single segment
# t1, t2 = 24000, 25000
# fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
# for col_idx, (label, trace, events) in enumerate(trace_types):
#     ax = axes[col_idx]
#     ax.plot(np.arange(t1, t2), trace[t1:t2], label=f'{label}')
#     ax.axhline(threshold, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
#     for ev in events:
#         if t1 <= ev < t2:
#             ax.axvline(ev, color='red', linestyle='-', linewidth=1.2, alpha=0.7)
#     ax.set_xlabel('Frame')
#     ax.set_ylabel('Correlation')
#     ax.set_title(f'{label} (Frames {t1}-{t2})')
#     ax.grid(True, alpha=0.3)
#     ax.legend(fontsize=9)
# plt.suptitle(f'{mouse} Day {day} Correlation Trace (no_stim trials) - Zoomed')
# plt.tight_layout()
# plt.show()
