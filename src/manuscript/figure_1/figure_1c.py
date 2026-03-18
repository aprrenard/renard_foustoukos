"""
Figure 1c: Lick raster plot illustrating task structure

This script generates Panel c for Figure 1, showing behavioral responses
(licking) across different trial types aligned to stimulus onset.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import stim_palette


OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_1', 'output')


# ============================================================================
# Helper Functions
# ============================================================================

def detect_piezo_lick_times(
    lick_data,
    ni_session_sr=5000,
    sigma=100,
    height=None,
    distance=None,
    prominence=None,
    width=None
):
    """
    Detect lick times from piezo sensor data using Hilbert transform envelope.

    The envelope is extracted using the Hilbert transform and then smoothed with
    a Gaussian filter. Peaks in the envelope correspond to individual licks.

    Args:
        lick_data: Raw piezo lick trace (1D numpy array)
        ni_session_sr: Sampling rate in Hz
        sigma: Standard deviation of Gaussian filter for envelope smoothing
        height: Minimum peak height for find_peaks
        distance: Minimum distance between peaks (in samples)
        prominence: Minimum prominence for find_peaks
        width: Minimum width for find_peaks

    Returns:
        lick_times: Array of detected lick times in seconds
    """
    # Extract envelope using Hilbert transform
    analytic_signal = hilbert(lick_data)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope
    envelope = gaussian_filter1d(envelope, sigma=sigma)

    # Detect peaks in the smoothed envelope
    peaks, _ = find_peaks(
        envelope,
        height=height,
        distance=distance,
        prominence=prominence,
        width=width
    )

    lick_times = peaks / ni_session_sr

    return lick_times


# ============================================================================
# Main Panel Generation
# ============================================================================

def generate_panel(
    results_file=os.path.join(io.processed_dir, 'behavior', 'GF305_29112020_103331_results.txt'),
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 1 Panel c: Lick raster plot.

    Shows licking behavior aligned to stimulus onset for whisker, auditory,
    and no-stimulus trials to illustrate the task structure.

    Args:
        results_file: Path to the preprocessed Results.txt file
        save_path: Directory to save output figure
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    results_file = io.adjust_path_to_host(results_file)

    # Load behavioral results
    df_results = pd.read_csv(results_file, sep=r'\s+', engine='python')

    # Piezo sensor parameters
    piezo_sr = 100000  # Hz

    # Detect licks for each trial
    df_lick_raster = pd.DataFrame(columns=['trialnumber', 'trial_type', 'lick_times'])
    trial_counter = 1

    for _, trial in df_results.iterrows():
        # Skip early lick trials
        if trial['EarlyLick'] == 1:
            continue

        # Determine trial type
        if trial['Whisker/NoWhisker'] == 1:
            trial_type = 'whisker'
        elif trial['Auditory/NoAuditory'] == 1:
            trial_type = 'auditory'
        elif trial['Stim/NoStim'] == 0:
            trial_type = 'no_stim'
        else:
            continue

        # Load and process lick trace
        lick_traces_dir = io.adjust_path_to_host(os.path.join(io.processed_dir, 'behavior', 'GF305_lick_traces'))
        lick_file = os.path.join(lick_traces_dir, f"LickTrace{int(trial['trialnumber'])}.bin")
        lick_trace = np.fromfile(lick_file)[1::2]

        lick_times = detect_piezo_lick_times(
            lick_trace,
            ni_session_sr=piezo_sr,
            sigma=200,
            height=0.04,
            distance=piezo_sr * 0.05,
            width=None
        )

        df_lick_raster = pd.concat([
            df_lick_raster,
            pd.DataFrame({
                'trialnumber': [trial_counter],
                'trial_type': [trial_type],
                'lick_times': [lick_times.tolist()]
            })
        ], ignore_index=True)
        trial_counter += 1

    # Remove mapping trials (keep only first 320 trials)
    df_lick_raster = df_lick_raster[df_lick_raster.trialnumber <= 320]

    # Plot licking raster
    # Figure dimensions optimized for manuscript panel (in mm, converted to inches)
    fig_width_mm = 45
    fig_height_mm = 90
    fig_width_in = fig_width_mm / 25.4
    fig_height_in = fig_height_mm / 25.4

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    # Color mapping for trial types
    colors = {
        'whisker': stim_palette[1],
        'auditory': stim_palette[0],
        'no_stim': stim_palette[2]
    }

    # Plot licks for each trial
    for i, row in df_lick_raster.iterrows():
        trial_type = row['trial_type']
        # Adjust for 2 sec baseline in GF305 data
        lick_times = np.array(row['lick_times']) - 2

        ax.scatter(
            lick_times,
            np.full_like(lick_times, i + 0.5),
            color=colors.get(trial_type, 'grey'),
            s=1,
            alpha=1,
            marker='o',
            linewidths=0
        )

    # Highlight stimulus period (0 to 1 second)
    ax.axvspan(0, 1, color='lightgrey', alpha=0.5, zorder=0)

    # Formatting
    ax.set_xlabel('Lick time from stim onset (secs)')
    ax.set_ylabel('Trial')
    ax.set_xlim([-1, 4])
    sns.despine()

    # Save figure
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'figure_1c.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 1c saved to: {output_file}")

    # Save data: one row per lick event, time aligned to stimulus onset
    lick_events = []
    for _, row in df_lick_raster.iterrows():
        for t in row['lick_times']:
            lick_events.append({
                'trialnumber': row['trialnumber'],
                'trial_type': row['trial_type'],
                'lick_time_s': t - 2,  # aligned to stimulus onset
            })
    pd.DataFrame(lick_events).to_csv(os.path.join(save_path, 'figure_1c_data.csv'), index=False)
    print(f"Figure 1c data saved to: {os.path.join(save_path, 'figure_1c_data.csv')}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    generate_panel()
