"""
Figure 4i: Example correlation traces across days (mouse AR127)

Generates a 5-row panel showing reactivation correlation traces across
days for an example mouse. Each row is one day; reactivation events are
marked as red vertical lines.
"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io


# ============================================================================
# Parameters
# ============================================================================

MOUSE = 'AR127'
DAYS = [-2, -1, 0, 1, 2]
SAMPLING_RATE = 30
TIME_WINDOW_PER_DAY = 180  # seconds; None = full trace

RESULTS_FILE = os.path.join(io.processed_dir, 'reactivation', 'reactivation_results_p99.pkl')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_4', 'output')


# ============================================================================
# Panel i
# ============================================================================

def panel_i_correlation_traces(
    r_plus_results,
    r_minus_results,
    mouse=MOUSE,
    days=DAYS,
    sampling_rate=SAMPLING_RATE,
    time_window=TIME_WINDOW_PER_DAY,
    nan_gap=0,
    filename='figure_4i',
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 4 Panel i: correlation traces across days for an example mouse.

    Args:
        nan_gap: Number of frames inserted as NaN between trials (0 = seamless).
        filename: Output filename stem (without extension).
    """
    if mouse in r_plus_results:
        results = r_plus_results[mouse]
    elif mouse in r_minus_results:
        results = r_minus_results[mouse]
    else:
        available = list(r_plus_results.keys()) + list(r_minus_results.keys())
        raise ValueError(f"Mouse {mouse} not found in results. Available: {available}")

    # Global y-limits across all days
    all_correlations = []
    for day in days:
        if day in results['days']:
            all_correlations.extend(results['days'][day]['correlations'])
    if not all_correlations:
        raise ValueError(f"No correlation data found for {mouse}")
    ylim = (np.min(all_correlations), np.max(all_correlations))

    fig, axes = plt.subplots(len(days), 1, figsize=(12, 6.0), sharex=False)

    for i, day in enumerate(days):
        ax = axes[i]

        if day not in results['days']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_ylabel(f'Day {day}', fontsize=9, fontweight='bold')
            ax.set_ylim(ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue

        day_data = results['days'][day]
        correlations = np.array(day_data['correlations'])
        events = np.array(day_data['events'])

        # Use stored frames_per_trial to avoid wrong splits after truncation
        frames_per_trial = day_data.get('n_timepoints', None)

        if time_window is not None:
            max_frames = int(time_window * sampling_rate)
            correlations = correlations[:max_frames]
            events = events[events < max_frames]

        total_frames = len(correlations)

        if frames_per_trial is None or frames_per_trial <= 0:
            frames_per_trial = int(12 * sampling_rate)
        n_trials = total_frames // frames_per_trial

        # Build concatenated trace with optional NaN gaps between trials
        trial_parts = []
        event_parts = []
        cumulative_frames = 0

        for trial_idx in range(n_trials):
            start_idx = trial_idx * frames_per_trial
            end_idx = min((trial_idx + 1) * frames_per_trial, total_frames)
            if start_idx >= total_frames:
                break

            trial_data = correlations[start_idx:end_idx]
            trial_parts.append(trial_data)

            trial_events = events[(events >= start_idx) & (events < end_idx)]
            shifted_events = trial_events - start_idx + cumulative_frames
            event_parts.append(shifted_events)
            cumulative_frames += len(trial_data)

            if nan_gap > 0 and trial_idx < n_trials - 1 and end_idx < total_frames:
                trial_parts.append(np.full(nan_gap, np.nan))
                cumulative_frames += nan_gap

        corr_with_gaps = np.concatenate(trial_parts)
        events_shifted = np.concatenate(event_parts) if event_parts else np.array([])

        # Build time vector
        time_parts = []
        current_time = 0
        gap_duration = nan_gap / sampling_rate

        for part in trial_parts:
            n_points = len(part)
            if np.all(np.isnan(part)):
                time_parts.append(np.full(n_points, np.nan))
            else:
                trial_time = np.arange(n_points) / sampling_rate + current_time
                time_parts.append(trial_time)
                current_time = trial_time[-1] + 1 / sampling_rate + gap_duration

        time_vec = np.concatenate(time_parts)

        # Plot
        ax.plot(time_vec, corr_with_gaps, 'k-', linewidth=0.7, alpha=0.9)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

        total_events = len(events_shifted)
        for event_idx in events_shifted:
            event_time = time_vec[int(event_idx)]
            if not np.isnan(event_time):
                ax.axvline(event_time, color='red', linewidth=0.8, alpha=0.7,
                           ymin=0.1, ymax=0.9)

        ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(f'Day {day}', fontsize=9, fontweight='bold')
        ax.text(0.98, 0.98, f'n={total_events}',
                transform=ax.transAxes, fontsize=7, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))
        ax.tick_params(axis='both', labelsize=7)
        if i == len(days) - 1:
            ax.set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {os.path.join(output_dir, filename + '.' + save_format)}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Loading reactivation results from: {RESULTS_FILE}")
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Please run reactivation.py with mode='compute' first."
        )

    with open(RESULTS_FILE, 'rb') as f:
        results_data = pickle.load(f)

    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']
    print(f"Loaded results for {len(r_plus_results)} R+ mice and {len(r_minus_results)} R- mice")

    panel_i_correlation_traces(r_plus_results, r_minus_results,
                               nan_gap=0, filename='figure_4i_no_gaps')
    panel_i_correlation_traces(r_plus_results, r_minus_results,
                               nan_gap=45, filename='figure_4i_with_gaps')
