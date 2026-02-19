"""
Standalone script to generate a 5-column panel showing correlation traces
across days for an example mouse (AR127).

This creates a publication-ready figure panel showing reactivation events
detected across all 5 days of the experiment.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io

# ============================================================================
# PARAMETERS
# ============================================================================

# Mouse to plot
MOUSE = 'AR127'

# Days to include
DAYS = [-2, -1, 0, 1, 2]

# Sampling rate (Hz)
SAMPLING_RATE = 30

# Time duration to display per day (in seconds)
# Set to None to show entire trace, or specify duration (e.g., 180 for 3 minutes)
TIME_WINDOW_PER_DAY = 120

# Path to saved reactivation results
RESULTS_DIR = os.path.join(io.results_dir, 'reactivation')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'reactivation_results.pkl')

# Output path
OUTPUT_DIR = os.path.join(io.results_dir, 'reactivation', 'illustrations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_example_mouse_correlation_traces_panel(r_plus_results, r_minus_results,
                                                 mouse='AR127',
                                                 days=[-2, -1, 0, 1, 2],
                                                 sampling_rate=30,
                                                 time_window=None,
                                                 n_lines=None,  # Deprecated parameter
                                                 save_path=None):
    """
    Create a 5-row panel showing correlation traces across days for an example mouse.
    Each row shows one day as a continuous horizontal trace with gaps between trials.
    Sized to fit as a panel in an A4 figure.

    Parameters
    ----------
    r_plus_results : dict
        Results dictionary for R+ mice
    r_minus_results : dict
        Results dictionary for R- mice
    mouse : str, optional
        Mouse ID (default: 'AR127')
    days : list, optional
        List of days to plot (default: [-2, -1, 0, 1, 2])
    sampling_rate : float, optional
        Sampling rate (Hz, default: 30)
    time_window : float, optional
        Time duration to display per day in seconds (default: None shows entire trace)
    n_lines : int, optional
        Deprecated parameter (kept for backward compatibility)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Find mouse in results
    if mouse in r_plus_results:
        results = r_plus_results[mouse]
        reward_group = 'R+'
    elif mouse in r_minus_results:
        results = r_minus_results[mouse]
        reward_group = 'R-'
    else:
        print(f"Warning: Mouse {mouse} not found in results")
        return None

    # Calculate global y-limits across all days
    all_correlations = []
    for day in days:
        if day in results['days']:
            all_correlations.extend(results['days'][day]['correlations'])

    if len(all_correlations) == 0:
        print(f"Warning: No correlation data found for {mouse}")
        return None

    ylim = (np.min(all_correlations), np.max(all_correlations))

    # Create figure with 5 rows (one per day)
    # Size: ~8.27 inches wide (A4 width), ~6.0 inches tall for 5 stacked traces
    fig, axes = plt.subplots(5, 1, figsize=(8.27, 6.0), sharex=False)

    # Number of NaN values to insert between trials as gaps (small = subtle separator)
    nan_gap = 45

    for i, day in enumerate(days):
        ax = axes[i]

        if day not in results['days']:
            # No data for this day
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_ylabel(f'Day {day}', fontsize=9, fontweight='bold')
            ax.set_ylim(ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue

        # Get data for this day
        day_data = results['days'][day]
        correlations = np.array(day_data['correlations'])
        events = np.array(day_data['events'])
        threshold_used = day_data.get('threshold_used', 0.45)

        # Use stored n_timepoints as the authoritative frames-per-trial value.
        # Do NOT derive it from total_frames/n_trials: if time_window truncates
        # total_frames while n_trials stays at the full-session count, the
        # division gives a wrongly small frames_per_trial and gaps appear
        # far too often.
        frames_per_trial = day_data.get('n_timepoints', None)

        # Apply time window if specified
        if time_window is not None:
            max_frames = int(time_window * sampling_rate)
            correlations = correlations[:max_frames]
            events = events[events < max_frames]

        total_frames = len(correlations)

        if frames_per_trial is None or frames_per_trial <= 0:
            frames_per_trial = int(12 * sampling_rate)   # last-resort fallback
        n_trials = total_frames // frames_per_trial       # recompute after any truncation

        # Insert NaN gaps between trials to create visual separation
        # Split correlations into trials and add gaps
        trial_parts = []
        event_parts = []
        cumulative_frames = 0

        for trial_idx in range(n_trials):
            start_idx = trial_idx * frames_per_trial
            end_idx = min((trial_idx + 1) * frames_per_trial, total_frames)

            if start_idx >= total_frames:
                break

            # Add trial data
            trial_data = correlations[start_idx:end_idx]
            trial_parts.append(trial_data)

            # Update event indices for this trial (shift by cumulative frames + gaps)
            trial_events = events[(events >= start_idx) & (events < end_idx)]
            shifted_events = trial_events - start_idx + cumulative_frames
            event_parts.append(shifted_events)

            cumulative_frames += len(trial_data)

            # Add NaN gap (except after last trial)
            if trial_idx < n_trials - 1 and end_idx < total_frames:
                trial_parts.append(np.full(nan_gap, np.nan))
                cumulative_frames += nan_gap

        # Concatenate all parts
        corr_with_gaps = np.concatenate(trial_parts)
        events_shifted = np.concatenate(event_parts) if event_parts else np.array([])

        # Create time vector with gaps
        # Build time accounting for NaN gaps
        time_parts = []
        current_time = 0
        gap_duration = nan_gap / sampling_rate

        for trial_idx in range(len(trial_parts)):
            part = trial_parts[trial_idx]
            n_points = len(part)

            if np.all(np.isnan(part)):
                # This is a gap - add NaN times
                time_parts.append(np.full(n_points, np.nan))
            else:
                # This is actual data - add sequential times
                trial_time = np.arange(n_points) / sampling_rate + current_time
                time_parts.append(trial_time)
                current_time = trial_time[-1] + 1/sampling_rate + gap_duration

        time_vec = np.concatenate(time_parts)

        # Plot correlation trace with gaps
        ax.plot(time_vec, corr_with_gaps, 'k-', linewidth=0.7, alpha=0.9)

        # Plot reference lines
        ax.axhline(threshold_used, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

        # Add vertical lines for reactivation events
        total_events = len(events_shifted)
        for event_idx in events_shifted:
            event_time = time_vec[int(event_idx)]
            if not np.isnan(event_time):
                ax.axvline(event_time, color='red', linewidth=0.8, alpha=0.7,
                          ymin=0.1, ymax=0.9)

        # Formatting
        ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Y-label shows day
        ax.set_ylabel(f'Day {day}', fontsize=9, fontweight='bold')

        # Add event count annotation
        ax.text(0.98, 0.98, f'n={total_events}',
               transform=ax.transAxes, fontsize=7,
               ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='gray', alpha=0.8))

        # X-axis formatting
        ax.tick_params(axis='both', labelsize=7)

        # Only show x-label on bottom subplot
        if i == len(days) - 1:
            ax.set_xlabel('Time (s)', fontsize=9, fontweight='bold')

    # Overall title
    fig.suptitle(f'{mouse} ({reward_group}) - Correlation Traces Across Days',
                fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved example mouse panel: {save_path}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING EXAMPLE MOUSE CORRELATION TRACES PANEL")
    print("="*70)
    print(f"\nMouse: {MOUSE}")
    print(f"Days: {DAYS}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    if TIME_WINDOW_PER_DAY is not None:
        print(f"Time window per day: {TIME_WINDOW_PER_DAY}s ({TIME_WINDOW_PER_DAY/60:.1f} min)")
    else:
        print(f"Time window per day: Full trace")

    # Load saved reactivation results
    print(f"\nLoading results from: {RESULTS_FILE}")
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            f"Please run reactivation.py with mode='compute' first."
        )

    with open(RESULTS_FILE, 'rb') as f:
        results_data = pickle.load(f)

    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']

    print(f"✓ Loaded results for {len(r_plus_results)} R+ mice and {len(r_minus_results)} R- mice")

    # Check if mouse exists
    if MOUSE in r_plus_results:
        print(f"✓ Found {MOUSE} in R+ group")
    elif MOUSE in r_minus_results:
        print(f"✓ Found {MOUSE} in R- group")
    else:
        available_mice = list(r_plus_results.keys()) + list(r_minus_results.keys())
        print(f"\n✗ Error: Mouse {MOUSE} not found in results")
        print(f"Available mice: {available_mice}")
        sys.exit(1)

    # Generate panel figure
    print(f"\nGenerating correlation traces panel...")

    # Save as both SVG and PNG
    svg_path = os.path.join(OUTPUT_DIR, f'{MOUSE}_correlation_traces_panel.svg')
    png_path = os.path.join(OUTPUT_DIR, f'{MOUSE}_correlation_traces_panel.png')

    fig = plot_example_mouse_correlation_traces_panel(
        r_plus_results,
        r_minus_results,
        mouse=MOUSE,
        days=DAYS,
        sampling_rate=SAMPLING_RATE,
        time_window=TIME_WINDOW_PER_DAY,
        save_path=svg_path
    )

    # Also save as PNG
    if fig is not None:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved PNG version: {png_path}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  • {svg_path}")
    print(f"  • {png_path}")
    print()
