"""
Spontaneous activity controls to investigate why surrogate thresholds increase on Day 0.

Two analyses:
1. No-stim PSTH across days: Mean population dF/F during no-stim trials, R+ vs R-.
   Scatter: mean 0-300ms response vs reactivation frequency (Day 0).
2. Calcium transient frequency per cell across days, R+ vs R-.
   Scatter: transient frequency vs reactivation frequency (Day 0).
   QC figure for example mouse showing mean dF/F trace with transient markers.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr, linregress

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *


# ============================================================================
# PARAMETERS
# ============================================================================

days = [-2, -1, 0, 1, 2]
sampling_rate = 30                # Hz
win_psth = (-0.5, 1.5)           # PSTH time window in seconds
win_response = (0, 0.300)         # Response window for scatter (seconds)

threshold_transient = 0.40        # dF/F threshold for transient detection (10%)
min_distance_ms = 200             # Minimum distance between transient peaks (ms)
min_distance_frames = int(min_distance_ms / 1000 * sampling_rate)
prominence_transient = 0.2      # Prominence for transient detection (10% dF/F)
savgol_window = 10                 # Savitzky-Golay smoothing window (frames)
savgol_order = 2                  # Savitzky-Golay polynomial order
nan_gap_frames = 60               # NaN gap between trials in QC figure

example_mouse = 'AR127'           # Example mouse for transient QC figure

# Paths
save_dir = os.path.join(io.results_dir, 'reactivation')
results_pkl = os.path.join(save_dir, 'reactivation_results_p99.pkl')
folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')

os.makedirs(save_dir, exist_ok=True)


# ============================================================================
# MOUSE LOADING
# ============================================================================

_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes'
)

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

print(f"Found {len(r_plus_mice)} R+ mice and {len(r_minus_mice)} R- mice")


# ============================================================================
# ANALYSIS 1: NO-STIM PSTH
# ============================================================================

def compute_nostim_psth(mice_list):
    """
    Compute mean population PSTH from no-stim trials for each mouse across days.

    Returns a DataFrame with columns: mouse_id, reward_group, day, time, psth, mean_response.
    mean_response is the mean dF/F in win_response (0-300ms window).
    """
    psth_rows = []

    for mouse_id in mice_list:
        try:
            reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id, db=db)
        except:
            continue

        try:
            xarr = utils_imaging.load_mouse_xarray(
                mouse_id, folder, 'tensor_xarray_learning_data.nc', substracted=True
            )
        except Exception as e:
            print(f"  Warning: Could not load data for {mouse_id}: {e}")
            continue

        # Filter to no-stim trials across all days of interest
        xarr = xarr.sel(trial=xarr['no_stim'] == 1)
        xarr = xarr.sel(trial=xarr['day'].isin(days))

        if len(xarr.trial) == 0:
            continue

        # Slice PSTH time window
        xarr = xarr.sel(time=slice(win_psth[0], win_psth[1]))

        # Average across trials per day → (cell, day, time)
        xarr_avg = xarr.groupby('day').mean(dim='trial')

        # Convert to DataFrame and average across cells
        xarr_avg.name = 'psth'
        df = xarr_avg.to_dataframe().reset_index()
        df['mouse_id'] = mouse_id
        df['reward_group'] = reward_group
        df = df.groupby(['mouse_id', 'reward_group', 'day', 'time'])['psth'].mean().reset_index()

        # Compute mean response in win_response window per day
        resp_mask = (df['time'] >= win_response[0]) & (df['time'] <= win_response[1])
        for day in df['day'].unique():
            day_resp_mask = (df['day'] == day) & resp_mask
            mean_resp = df.loc[day_resp_mask, 'psth'].mean()
            df.loc[df['day'] == day, 'mean_response'] = mean_resp

        psth_rows.append(df)
        print(f"  Processed {mouse_id}")

    if not psth_rows:
        return pd.DataFrame()
    return pd.concat(psth_rows, ignore_index=True)


def plot_nostim_psth_per_day(df, save_path):
    """
    Plot mean population PSTH for no-stim trials per day, R+ vs R-.
    1 row × 5 columns layout, sharey=True.
    """
    df = df.copy()
    df['psth'] = df['psth'] * 100  # convert to percent dF/F

    fig, axes = plt.subplots(1, len(days), figsize=(18, 5), sharey=True)

    for j, day in enumerate(days):
        d = df.loc[df['day'] == day]
        sns.lineplot(
            data=d, x='time', y='psth', errorbar='ci',
            hue='reward_group', hue_order=['R-', 'R+'],
            palette=reward_palette, estimator='mean',
            ax=axes[j], legend=(j == 0)
        )
        axes[j].axvline(0, color='#FF9600', linestyle='-', linewidth=1)
        axes[j].set_title(f'Day {day:+d}')
        axes[j].set_xlabel('Time (s)')
        axes[j].set_ylabel('dF/F (%)' if j == 0 else '')

    fig.suptitle('No-Stim Trial PSTH Across Days', fontsize=13, fontweight='bold')
    plt.tight_layout()
    sns.despine()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    return fig


def plot_nostim_response_vs_reactivation(psth_df, r_plus_results, r_minus_results, save_path):
    """
    Scatter: mean no-stim dF/F (0-300ms window) vs reactivation frequency, Day 0.
    """
    scatter_data = []

    for mouse_id in psth_df['mouse_id'].unique():
        mouse_psth = psth_df[(psth_df['mouse_id'] == mouse_id) & (psth_df['day'] == 0)]
        if len(mouse_psth) == 0:
            continue

        mean_response = mouse_psth['mean_response'].iloc[0]
        reward_group = mouse_psth['reward_group'].iloc[0]

        reac_freq = None
        if mouse_id in r_plus_results and 0 in r_plus_results[mouse_id]['days']:
            reac_freq = r_plus_results[mouse_id]['days'][0]['event_frequency']
        elif mouse_id in r_minus_results and 0 in r_minus_results[mouse_id]['days']:
            reac_freq = r_minus_results[mouse_id]['days'][0]['event_frequency']

        if reac_freq is not None:
            scatter_data.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'mean_response': mean_response * 100,  # percent dF/F
                'reac_freq': reac_freq
            })

    df_scatter = pd.DataFrame(scatter_data)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for group, color in [('R+', reward_palette[1]), ('R-', reward_palette[0])]:
        g = df_scatter[df_scatter['reward_group'] == group]
        if len(g) < 2:
            continue
        ax.scatter(g['mean_response'], g['reac_freq'], color=color, label=group, alpha=0.8, s=50)
        slope, intercept, _, _, _ = linregress(g['mean_response'], g['reac_freq'])
        x_range = np.linspace(g['mean_response'].min(), g['mean_response'].max(), 50)
        ax.plot(x_range, slope * x_range + intercept, color=color, linewidth=1.5, alpha=0.7)
        r, p = pearsonr(g['mean_response'], g['reac_freq'])
        print(f"  {group}: r={r:.3f}, p={p:.4f}, n={len(g)}")

    ax.set_xlabel('Mean no-stim dF/F, 0-300ms (%) - Day 0')
    ax.set_ylabel('Reactivation frequency (events/min) - Day 0')
    ax.set_title('No-Stim Response vs Reactivation Frequency (Day 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    return fig


# ============================================================================
# ANALYSIS 2: CALCIUM TRANSIENT FREQUENCY
# ============================================================================

def detect_transients(cell_trace):
    """Detect calcium transients in a single cell trace. Returns peak frame indices."""
    smoothed = savgol_filter(cell_trace, savgol_window, savgol_order)
    peaks, _ = find_peaks(
        smoothed,
        height=threshold_transient,
        distance=min_distance_frames,
        prominence=prominence_transient
    )
    return peaks


def compute_transient_freq_per_mouse(mouse_id):
    """
    Detect transients per cell on no-stim trials for each day.
    Returns list of dicts: mouse_id, reward_group, day, transient_freq, n_cells, n_trials.
    """
    try:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id, db=db)
    except:
        return []

    try:
        xarr = utils_imaging.load_mouse_xarray(
            mouse_id, folder, 'tensor_xarray_learning_data.nc', substracted=True
        )
    except Exception as e:
        print(f"  Warning: Could not load data for {mouse_id}: {e}")
        return []

    rows = []
    for day in days:
        xarr_day = xarr.sel(trial=(xarr['day'] == day) & (xarr['no_stim'] == 1))

        n_trials = len(xarr_day.trial)
        if n_trials == 0:
            continue

        # Reshape to (n_cells, n_total_frames)
        n_cells = len(xarr_day.cell)
        data = xarr_day.values.reshape(n_cells, -1)
        data = np.nan_to_num(data, nan=0.0)

        session_duration_min = data.shape[1] / sampling_rate / 60

        cell_freqs = [
            len(detect_transients(data[c])) / session_duration_min
            for c in range(n_cells)
        ]

        rows.append({
            'mouse_id': mouse_id,
            'reward_group': reward_group,
            'day': day,
            'transient_freq': np.mean(cell_freqs),
            'n_cells': n_cells,
            'n_trials': n_trials,
        })

    return rows


def compute_transient_frequencies(mice_list):
    """Compute transient frequencies for all mice. Returns long-format DataFrame."""
    all_rows = []
    for mouse_id in mice_list:
        print(f"  Processing {mouse_id}...")
        all_rows.extend(compute_transient_freq_per_mouse(mouse_id))
    return pd.DataFrame(all_rows)


def plot_transient_freq_per_day(df, save_path):
    """
    Grouped barplot of mean per-cell transient frequency per day, R+ vs R-.
    Same style as plot_group_comparison_per_day in reactivation.py.
    """
    all_days = sorted(days)

    # Ensure all days are present for plotting
    df_plot = df.copy()
    for day in all_days:
        if day not in df_plot['day'].unique():
            df_plot = pd.concat([df_plot, pd.DataFrame({
                'day': [day], 'transient_freq': [np.nan],
                'reward_group': ['R+'], 'mouse_id': ['']
            })], ignore_index=True)
            df_plot = pd.concat([df_plot, pd.DataFrame({
                'day': [day], 'transient_freq': [np.nan],
                'reward_group': ['R-'], 'mouse_id': ['']
            })], ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    sns.barplot(
        data=df_plot, x='day', y='transient_freq', hue='reward_group',
        errorbar=('ci', 95),
        palette={'R+': reward_palette[1], 'R-': reward_palette[0]},
        hue_order=['R+', 'R-'],
        alpha=0.7, edgecolor='black', ax=ax
    )

    # Individual mouse trajectories
    x_positions = {day: idx for idx, day in enumerate(all_days)}
    bar_width = 0.35
    group_offsets = {'R+': -bar_width / 2, 'R-': bar_width / 2}

    for mouse_id in df_plot['mouse_id'].unique():
        if mouse_id == '':
            continue
        mouse_data = df_plot[df_plot['mouse_id'] == mouse_id].sort_values('day')
        if len(mouse_data) == 0:
            continue
        group = mouse_data['reward_group'].iloc[0]
        mouse_x = [x_positions[d] + group_offsets[group] for d in mouse_data['day']]
        mouse_y = mouse_data['transient_freq'].values
        group_color = reward_palette[1] if group == 'R+' else reward_palette[0]
        ax.plot(mouse_x, mouse_y, '-', color=group_color, linewidth=0.8, alpha=0.5, zorder=5)

    ax.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transient frequency (events/min/cell)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Calcium Transient Frequency Across Days\n'
        f'(threshold={threshold_transient*100:.0f}% dF/F, '
        f'prominence={prominence_transient*100:.0f}% dF/F)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    return fig


def plot_transient_vs_reactivation(transient_df, r_plus_results, r_minus_results, save_path):
    """
    Scatter: mean per-cell transient frequency vs reactivation frequency, Day 0.
    """
    scatter_data = []

    for _, row in transient_df[transient_df['day'] == 0].iterrows():
        mouse_id = row['mouse_id']
        reward_group = row['reward_group']
        transient_freq = row['transient_freq']

        reac_freq = None
        if mouse_id in r_plus_results and 0 in r_plus_results[mouse_id]['days']:
            reac_freq = r_plus_results[mouse_id]['days'][0]['event_frequency']
        elif mouse_id in r_minus_results and 0 in r_minus_results[mouse_id]['days']:
            reac_freq = r_minus_results[mouse_id]['days'][0]['event_frequency']

        if reac_freq is not None:
            scatter_data.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'transient_freq': transient_freq,
                'reac_freq': reac_freq
            })

    df_scatter = pd.DataFrame(scatter_data)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for group, color in [('R+', reward_palette[1]), ('R-', reward_palette[0])]:
        g = df_scatter[df_scatter['reward_group'] == group]
        if len(g) < 2:
            continue
        ax.scatter(g['transient_freq'], g['reac_freq'], color=color, label=group, alpha=0.8, s=50)
        slope, intercept, _, _, _ = linregress(g['transient_freq'], g['reac_freq'])
        x_range = np.linspace(g['transient_freq'].min(), g['transient_freq'].max(), 50)
        ax.plot(x_range, slope * x_range + intercept, color=color, linewidth=1.5, alpha=0.7)
        r, p = pearsonr(g['transient_freq'], g['reac_freq'])
        print(f"  {group}: r={r:.3f}, p={p:.4f}, n={len(g)}")

    ax.set_xlabel('Transient frequency (events/min/cell) - Day 0')
    ax.set_ylabel('Reactivation frequency (events/min) - Day 0')
    ax.set_title('Transient Frequency vs Reactivation Frequency (Day 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    return fig


def plot_transient_qc(mouse_id, save_path, n_cells_to_show=5, max_trials=100):
    """
    QC figure for one example mouse: 5 cells (rows) × 5 days (columns).
    Each panel shows one cell's dF/F trace concatenated across up to max_trials
    no-stim trials (with NaN gaps between trials), with vertical red bars for
    detected transients in that cell on that day.

    Cells are selected as the 5 with the highest mean transient frequency
    across days where data is available.
    """
    try:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id, db=db)
    except:
        reward_group = '?'

    try:
        xarr = utils_imaging.load_mouse_xarray(
            mouse_id, folder, 'tensor_xarray_learning_data.nc', substracted=True
        )
    except Exception as e:
        print(f"  Warning: Could not load data for {mouse_id}: {e}")
        return None

    # Load and cache data for each day (capped at max_trials)
    day_cache = {}
    for day in days:
        xarr_day = xarr.sel(trial=(xarr['day'] == day) & (xarr['no_stim'] == 1))
        if len(xarr_day.trial) == 0:
            day_cache[day] = None
            continue
        # Cap trials
        n_trials_avail = len(xarr_day.trial)
        n_trials = min(n_trials_avail, max_trials)
        xarr_day = xarr_day.isel(trial=slice(0, n_trials))

        n_cells, _, n_timepoints = xarr_day.shape
        data = xarr_day.values.reshape(n_cells, -1)
        data = np.nan_to_num(data, nan=0.0)
        day_cache[day] = (data, n_trials, n_timepoints, n_cells)

    # Select 5 cells with highest mean transient frequency across available days
    first_day_data = next((v for v in day_cache.values() if v is not None), None)
    if first_day_data is None:
        print(f"  Warning: No data found for {mouse_id}")
        return None

    n_cells_total = first_day_data[3]
    cell_freqs = np.zeros(n_cells_total)
    n_days_with_data = 0

    for day, cached in day_cache.items():
        if cached is None:
            continue
        data, n_trials, n_timepoints, _ = cached
        duration_min = (n_trials * n_timepoints) / sampling_rate / 60
        for c in range(n_cells_total):
            cell_freqs[c] += len(detect_transients(data[c])) / duration_min
        n_days_with_data += 1

    if n_days_with_data > 0:
        cell_freqs /= n_days_with_data

    selected_cells = np.argsort(cell_freqs)[-n_cells_to_show:][::-1]

    # Global y-limits from selected cells across all days
    all_values = []
    for cached in day_cache.values():
        if cached is None:
            continue
        data, _, _, _ = cached
        for c in selected_cells:
            all_values.extend((data[c] * 100).tolist())

    ylim = (np.percentile(all_values, 1), np.percentile(all_values, 99))

    # Figure: n_cells_to_show rows × len(days) columns
    fig, axes = plt.subplots(
        n_cells_to_show, len(days),
        figsize=(3.5 * len(days), 2.0 * n_cells_to_show),
        sharex=False, sharey=True
    )

    gap_duration = nan_gap_frames / sampling_rate

    for row, cell_idx in enumerate(selected_cells):
        for col, day in enumerate(days):
            ax = axes[row, col]

            if day_cache[day] is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                continue

            data, n_trials, n_timepoints, _ = day_cache[day]
            cell_trace = data[cell_idx] * 100  # percent dF/F

            # Build concatenated trace with NaN gaps between trials
            trial_parts = []
            for trial_idx in range(n_trials):
                start = trial_idx * n_timepoints
                end = (trial_idx + 1) * n_timepoints
                trial_parts.append(cell_trace[start:end])
                if trial_idx < n_trials - 1:
                    trial_parts.append(np.full(nan_gap_frames, np.nan))

            trace_with_gaps = np.concatenate(trial_parts)

            # Build time vector
            time_parts = []
            current_time = 0
            for part in trial_parts:
                if np.all(np.isnan(part)):
                    time_parts.append(np.full(len(part), np.nan))
                else:
                    t = np.arange(len(part)) / sampling_rate + current_time
                    time_parts.append(t)
                    current_time = t[-1] + 1 / sampling_rate + gap_duration

            time_vec = np.concatenate(time_parts)

            # Detect transients on this cell's full concatenated trace
            peaks = detect_transients(data[cell_idx])

            # Map peak frame → position in gapped trace
            def orig_to_gapped_pos(f):
                t_idx = f // n_timepoints
                f_in_t = f % n_timepoints
                return t_idx * (n_timepoints + nan_gap_frames) + f_in_t

            ax.plot(time_vec, trace_with_gaps, 'k-', linewidth=0.5, alpha=0.9)
            ax.axhline(threshold_transient * 100, color='gray', linestyle=':',
                       linewidth=0.5, alpha=0.5)
            ax.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

            for orig_frame in peaks:
                pos = orig_to_gapped_pos(orig_frame)
                if pos < len(time_vec) and not np.isnan(time_vec[pos]):
                    ax.axvline(time_vec[pos], color='red', linewidth=0.6, alpha=0.7,
                               ymin=0.1, ymax=0.9)

            ax.set_ylim(ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=6)
            ax.text(0.98, 0.98, f'n={len(peaks)}',
                    transform=ax.transAxes, fontsize=6,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='gray', alpha=0.8))

            # Row labels (cell index) on left column only
            if col == 0:
                ax.set_ylabel(f'Cell {cell_idx}', fontsize=7, fontweight='bold')

            # Day labels on top row only
            if row == 0:
                ax.set_title(f'Day {day:+d}', fontsize=8, fontweight='bold')

            # x-label on bottom row only
            if row == n_cells_to_show - 1:
                ax.set_xlabel('Time (s)', fontsize=7)

    fig.suptitle(
        f'{mouse_id} ({reward_group}) - Individual cell dF/F with transients '
        f'(top {n_cells_to_show} by freq, ≤{max_trials} trials)\n'
        f'threshold={threshold_transient*100:.0f}% dF/F, '
        f'prominence={prominence_transient*100:.0f}% dF/F',
        fontsize=9, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

all_mice = r_plus_mice + r_minus_mice

print("\n" + "="*60)
print("ANALYSIS 1: NO-STIM PSTH")
print("="*60)

print(f"\nComputing no-stim PSTH for {len(all_mice)} mice...")
psth_df = compute_nostim_psth(all_mice)

if len(psth_df) > 0:
    svg_path = os.path.join(save_dir, 'nostim_psth_per_day.svg')
    plot_nostim_psth_per_day(psth_df, svg_path)

    print("\nLoading reactivation results...")
    with open(results_pkl, 'rb') as f:
        results_data = pickle.load(f)
    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']

    svg_path = os.path.join(save_dir, 'nostim_response_vs_reactivation_day0.svg')
    print("\nNo-stim response vs reactivation frequency (Day 0):")
    plot_nostim_response_vs_reactivation(psth_df, r_plus_results, r_minus_results, svg_path)
else:
    print("  Warning: No PSTH data computed.")

print("\n" + "="*60)
print("ANALYSIS 2: CALCIUM TRANSIENT FREQUENCY")
print("="*60)

print(f"\nDetecting transients for {len(all_mice)} mice...")
transient_df = compute_transient_frequencies(all_mice)

svg_path = os.path.join(save_dir, 'transient_freq_per_day.svg')
plot_transient_freq_per_day(transient_df, svg_path)

if 'r_plus_results' not in dir():
    with open(results_pkl, 'rb') as f:
        results_data = pickle.load(f)
    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']

svg_path = os.path.join(save_dir, 'transient_freq_vs_reactivation_day0.svg')
print("\nTransient vs reactivation frequency (Day 0):")
plot_transient_vs_reactivation(transient_df, r_plus_results, r_minus_results, svg_path)

svg_path = os.path.join(save_dir, f'transient_qc_{example_mouse}.svg')
print(f"\nGenerating QC figure for {example_mouse}...")
plot_transient_qc(example_mouse, svg_path)


print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"\nResults saved to: {save_dir}")




# ### Debug — average no-stim PSTH per mouse, plotted sequentially
# _day = -1

# for _mouse in r_plus_mice:
#     _xarr = utils_imaging.load_mouse_xarray(_mouse, folder, 'tensor_xarray_learning_data.nc', substracted=True)
#     _sel  = _xarr.sel(trial=(_xarr['day'] == _day) & (_xarr['no_stim'] == 1))
#     if len(_sel.trial) == 0:
#         print(f'{_mouse}: no data'); continue
#     _psth = _sel.mean(dim=['trial', 'cell']).values * 100
#     _t    = _xarr.time.values

#     plt.figure(figsize=(6, 2))
#     plt.plot(_t, _psth, 'k-', lw=0.8)
#     plt.axvline(0, color='gray', linestyle=':', lw=0.7)
#     plt.axhline(0, color='gray', linestyle=':', lw=0.5)
#     plt.title(f'{_mouse}  day={_day:+d}'); plt.ylabel('dF/F (%)'); plt.xlabel('Time (s)')
#     plt.tight_layout(); plt.show()
