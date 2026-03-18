"""
Supplementary Figure 4a-b: Spontaneous activity controls for the
LMI-participation relationship.

Panel 4a: Scatter plot of cell participation rate vs spontaneous transient
frequency (Day 0). One dot per cell, color-coded by LMI using the coolwarm
colormap (blue = negative LMI, red = positive LMI, centered at 0).
One panel per reward group.

Panel 4b: Partial correlation — LMI vs participation rate, raw and after
controlling for spontaneous transient frequency. Layout: 2 rows (R+, R-) x
2 columns (raw, partial). Tests whether spontaneous activity explains the
LMI-participation relationship.

Execution modes:
    MODE = 'compute' : run full pipeline, save intermediate data, then plot
    MODE = 'plot'    : load previously saved data from RESULTS_DIR and plot only

Intermediate data (participation rates, merged day-0 dataset) are saved to
data_processed/reactivation/.
Figures and data/stats CSVs are saved to io.manuscript_output_dir/supp_4/output/.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr, linregress
from joblib import Parallel, delayed

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
SAMPLING_RATE = 30

# Participation parameters (must match reactivation_lmi_prediction.py)
EVENT_WINDOW_MS = 150
EVENT_WINDOW_FRAMES = int(EVENT_WINDOW_MS / 1000 * SAMPLING_RATE)
PARTICIPATION_THRESHOLD = 0.10
MIN_EVENTS_FOR_RELIABILITY = 5

# Spontaneous transient detection parameters
MIN_DISTANCE_MS = 200
MIN_DISTANCE_FRAMES = int(MIN_DISTANCE_MS / 1000 * SAMPLING_RATE)
PROMINENCE_TRANSIENT = 0.2
N_STD_THRESHOLD = 3               # Per-cell threshold: N_STD_THRESHOLD * std(trace)
SAVGOL_WINDOW = 10
SAVGOL_ORDER = 2

N_JOBS = 10

RESULTS_DIR = os.path.join(io.processed_dir, 'reactivation')
REACTIVATION_RESULTS_FILE = os.path.join(RESULTS_DIR, 'reactivation_results_p99.pkl')
PARTICIPATION_CSV = os.path.join(RESULTS_DIR, 'cell_participation_rates_per_day.csv')
LMI_RESULTS_CSV = os.path.join(io.processed_dir, 'lmi_results.csv')
LMI_DATA_CSV = os.path.join(RESULTS_DIR, 'supp4ab_lmi_data_day0.csv')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_4', 'output')
FOLDER = os.path.join(io.solve_common_paths('processed_data'), 'mice')

# Execution mode
MODE = 'plot'


# ============================================================================
# Mouse loading
# ============================================================================

_, _, _all_mice, _db = io.select_sessions_from_db(
    io.db_path, io.nwb_dir, two_p_imaging='yes'
)

r_plus_mice, r_minus_mice = [], []
for _mouse in _all_mice:
    try:
        _rg = io.get_mouse_reward_group_from_db(io.db_path, _mouse, db=_db)
        if _rg == 'R+':
            r_plus_mice.append(_mouse)
        elif _rg == 'R-':
            r_minus_mice.append(_mouse)
    except:
        continue

print(f"Found {len(r_plus_mice)} R+ mice and {len(r_minus_mice)} R- mice")


# ============================================================================
# Participation computation
# ============================================================================

def _extract_event_responses(mouse, day, preloaded_events):
    """
    Extract per-cell dF/F responses around pre-computed reactivation events.

    Uses no-stim trials only. Responses are averaged over ±EVENT_WINDOW_FRAMES
    around each event. Cells are flagged as participating if their mean response
    exceeds PARTICIPATION_THRESHOLD.

    Returns DataFrame (mouse_id, day, roi, event_idx, avg_response, participates)
    or None if insufficient data (<10 no-stim trials or no valid events).
    """
    xarr = utils_imaging.load_mouse_xarray(
        mouse, FOLDER, 'tensor_xarray_learning_data.nc', substracted=False
    )
    xarr_day = xarr.sel(trial=xarr['day'] == day)
    nostim = xarr_day.sel(trial=xarr_day['no_stim'] == 1)

    if len(nostim.trial) < 10:
        return None

    n_cells, n_trials, n_timepoints = nostim.shape
    data_3d = nostim.values
    roi_list = nostim['roi'].values
    win = EVENT_WINDOW_FRAMES

    rows = []
    for event_idx in preloaded_events:
        trial_idx = event_idx // n_timepoints
        time_idx  = event_idx % n_timepoints
        if time_idx < win or time_idx >= n_timepoints - win or trial_idx >= n_trials:
            continue
        window_data  = data_3d[:, trial_idx, time_idx - win:time_idx + win + 1]
        avg_response = np.mean(window_data, axis=1)
        participates = avg_response >= PARTICIPATION_THRESHOLD
        for icell in range(n_cells):
            rows.append({
                'mouse_id': mouse, 'day': day, 'roi': roi_list[icell],
                'event_idx': event_idx, 'avg_response': float(avg_response[icell]),
                'participates': bool(participates[icell]),
            })

    return pd.DataFrame(rows) if rows else None


def _compute_participation_rate(responses_df):
    """Aggregate cell-event responses to per-cell participation rates."""
    grouped = responses_df.groupby(['mouse_id', 'day', 'roi']).agg(
        n_participations=('participates', 'sum'),
        n_events=('participates', 'count'),
    ).reset_index()
    grouped['participation_rate'] = grouped['n_participations'] / grouped['n_events']
    grouped['reliable'] = grouped['n_events'] >= MIN_EVENTS_FOR_RELIABILITY
    return grouped


def _process_mouse_participation(mouse, preloaded_results):
    """Compute participation rates across all days for one mouse."""
    all_responses = []
    for day in DAYS:
        events = preloaded_results.get('days', {}).get(day, {}).get('events', None)
        if events is None or len(events) == 0:
            continue
        try:
            resp_df = _extract_event_responses(mouse, day, events)
            if resp_df is not None and len(resp_df) > 0:
                all_responses.append(resp_df)
        except Exception as e:
            print(f"  Warning: {mouse} day {day}: {e}")
    if not all_responses:
        return mouse, None
    all_resp_df = pd.concat(all_responses, ignore_index=True)
    return mouse, _compute_participation_rate(all_resp_df)


def compute_participation_csv(save_path):
    """
    Compute per-cell participation rates across days from pre-computed
    reactivation events (reactivation_results_p99.pkl) and save to save_path.

    Runs in parallel across mice.
    """
    if not os.path.exists(REACTIVATION_RESULTS_FILE):
        raise FileNotFoundError(
            f"Reactivation results not found: {REACTIVATION_RESULTS_FILE}\n"
            "Run reactivation.py first."
        )

    with open(REACTIVATION_RESULTS_FILE, 'rb') as f:
        data = pickle.load(f)
    all_results = {**data['r_plus_results'], **data['r_minus_results']}

    all_mice = r_plus_mice + r_minus_mice
    print(f"\nComputing participation rates for {len(all_mice)} mice...")
    results_list = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_process_mouse_participation)(mouse, all_results.get(mouse, {}))
        for mouse in all_mice
    )

    all_data = [df for _, df in results_list if df is not None]
    if not all_data:
        raise RuntimeError("No participation data computed.")

    participation_df_all = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    participation_df_all.to_csv(save_path, index=False)
    print(f"Saved: {save_path}  ({len(participation_df_all)} cell-day records)")
    return participation_df_all


# ============================================================================
# Transient detection
# ============================================================================

def _detect_transients(cell_trace):
    """Detect calcium transients in a single cell trace. Returns peak indices.

    The height threshold is set per cell as N_STD_THRESHOLD * std(cell_trace),
    preventing noisy cells from generating spuriously high transient counts.
    """
    smoothed = savgol_filter(cell_trace, SAVGOL_WINDOW, SAVGOL_ORDER)
    cell_threshold = N_STD_THRESHOLD * np.std(cell_trace)
    peaks, _ = find_peaks(
        smoothed,
        height=cell_threshold,
        distance=MIN_DISTANCE_FRAMES,
        prominence=PROMINENCE_TRANSIENT,
    )
    return peaks


def _compute_transient_freq_per_cell(mouse_id, day=0):
    """
    Compute spontaneous transient frequency (events/min) per cell for a given
    mouse and day, using no-stim trials only.

    Returns DataFrame: mouse_id, roi, transient_freq.
    """
    try:
        xarr = utils_imaging.load_mouse_xarray(
            mouse_id, FOLDER, 'tensor_xarray_learning_data.nc', substracted=False
        )
    except Exception as e:
        print(f"  Warning: Could not load data for {mouse_id}: {e}")
        return pd.DataFrame()

    xarr_day = xarr.sel(trial=(xarr['day'] == day) & (xarr['no_stim'] == 1))
    if len(xarr_day.trial) == 0:
        return pd.DataFrame()

    n_cells = len(xarr_day.cell)
    roi_ids = xarr_day['roi'].values
    data = xarr_day.values.reshape(n_cells, -1)
    data = np.nan_to_num(data, nan=0.0)
    session_duration_min = data.shape[1] / SAMPLING_RATE / 60

    rows = []
    for c in range(n_cells):
        n_peaks = len(_detect_transients(data[c]))
        rows.append({
            'mouse_id': mouse_id,
            'roi': roi_ids[c],
            'transient_freq': n_peaks / session_duration_min,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Build merged day-0 dataset
# ============================================================================

def compute_lmi_data_csv(save_path):
    """
    Build the per-cell day-0 dataset used by both panels by merging:
      - PARTICIPATION_CSV  (mouse_id, roi, participation_rate — day 0 only)
      - transient freq     (computed from raw imaging, no-stim trials, day 0)
      - LMI_RESULTS_CSV    (mouse_id, roi, lmi, lmi_p)

    Computes PARTICIPATION_CSV first if it does not exist.
    Saves the merged DataFrame to save_path.
    """
    # Participation rates
    if not os.path.exists(PARTICIPATION_CSV):
        print("Participation CSV not found, computing it...")
        compute_participation_csv(PARTICIPATION_CSV)

    part_df = pd.read_csv(PARTICIPATION_CSV)
    part_df = part_df[part_df['day'] == 0][['mouse_id', 'roi', 'participation_rate']].copy()
    if len(part_df) == 0:
        raise RuntimeError("No day-0 participation data found.")

    # Transient frequencies
    mice = part_df['mouse_id'].unique()
    transient_parts = []
    for mouse_id in mice:
        print(f"  Computing transient freq for {mouse_id}...")
        transient_parts.append(_compute_transient_freq_per_cell(mouse_id, day=0))
    transient_df = pd.concat(
        [d for d in transient_parts if len(d) > 0], ignore_index=True
    )
    if len(transient_df) == 0:
        raise RuntimeError("No transient data computed.")

    # LMI
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)[['mouse_id', 'roi', 'lmi', 'lmi_p']]

    # Merge
    merged = part_df.merge(transient_df, on=['mouse_id', 'roi'], how='inner')
    merged = merged.merge(lmi_df, on=['mouse_id', 'roi'], how='inner')

    group_map = {m: 'R+' for m in r_plus_mice}
    group_map.update({m: 'R-' for m in r_minus_mice})
    merged['reward_group'] = merged['mouse_id'].map(group_map)
    merged = merged.dropna(
        subset=['reward_group', 'transient_freq', 'participation_rate', 'lmi']
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False)
    print(f"Saved: {save_path}  ({len(merged)} cells, {merged['mouse_id'].nunique()} mice)")
    return merged


# ============================================================================
# Panel 4a: participation rate vs transient frequency scatter (LMI color)
# ============================================================================

def panel_supp4a_scatter(
    data_csv_path,
    output_dir=OUTPUT_DIR,
    filename='supp_4a',
    save_format='svg',
    dpi=300,
):
    """
    Supp Figure 4a: scatter of participation rate vs transient frequency (Day 0).
    One dot per cell, colored by LMI (coolwarm colormap, centered at 0).
    One panel per reward group. Regression line and Pearson r annotated.

    Saves:
        <filename>.svg       -- figure
        <filename>_data.csv  -- data used for the plot
        <filename>_stats.csv -- Pearson r and p per reward group
    """

    lmi_cmap = mcolors.LinearSegmentedColormap.from_list(
    'blue_grey_red',
    [
        (0.0,  (0.0,  0.0, 1.0)),   # bright blue
        (0.5,  (0.7, 0.7, 0.7)),  # mid-grey centre
        (1.0,  (1.0,  0.0,  0.0)),   # bright red
    ]
    )

    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(
        subset=['lmi', 'transient_freq', 'participation_rate', 'reward_group']
    )

    lmi_abs_max = np.abs(merged['lmi']).max()
    norm = mcolors.TwoSlopeNorm(vmin=-lmi_abs_max, vcenter=0, vmax=lmi_abs_max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats_rows = []

    for i, reward_group in enumerate(['R+', 'R-']):
        ax = axes[i]
        gdata = merged[merged['reward_group'] == reward_group]

        if len(gdata) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(reward_group, fontweight='bold')
            continue

        sc = ax.scatter(
            gdata['transient_freq'], gdata['participation_rate'],
            c=gdata['lmi'], cmap=lmi_cmap, norm=norm,
            alpha=0.6, s=15, linewidths=0,
        )

        slope, intercept, _, _, _ = linregress(
            gdata['transient_freq'], gdata['participation_rate']
        )
        x_range = np.linspace(
            gdata['transient_freq'].min(), gdata['transient_freq'].max(), 100
        )
        ax.plot(x_range, slope * x_range + intercept, 'k-', linewidth=1.5)

        r, p = pearsonr(gdata['transient_freq'], gdata['participation_rate'])
        p_str = ('p < 0.001 ***' if p < 0.001 else
                 f'p = {p:.3f} **' if p < 0.01 else
                 f'p = {p:.3f} *' if p < 0.05 else
                 f'p = {p:.3f} ns')
        n_mice = gdata['mouse_id'].nunique()
        ax.text(
            0.05, 0.95,
            f'r = {r:.3f}\n{p_str}\nn = {len(gdata)} cells, {n_mice} mice',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
        )

        plt.colorbar(sc, ax=ax, label='LMI')
        ax.set_xlabel('Transient frequency (events/min)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Participation rate' if i == 0 else '', fontweight='bold', fontsize=12)
        ax.set_title(f'{reward_group}  (n={n_mice} mice, {len(gdata)} cells)',
                     fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

        stats_rows.append({
            'reward_group': reward_group,
            'pearson_r': r, 'p_value': p,
            'n_cells': len(gdata), 'n_mice': n_mice,
            'test': 'Pearson r (transient_freq vs participation_rate)',
        })
        print(f"  {reward_group}: r={r:.3f}, p={p:.4f}, "
              f"n={len(gdata)} cells, {n_mice} mice")

    fig.suptitle(
        'Participation Rate vs Transient Frequency (Day 0, colored by LMI)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    merged.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(output_dir, f'{filename}_stats.csv'), index=False
    )
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Panel 4b: partial correlation — LMI vs participation | transient freq
# ============================================================================

def panel_supp4b_partial_corr(
    data_csv_path,
    output_dir=OUTPUT_DIR,
    filename='supp_4b',
    save_format='svg',
    dpi=300,
):
    """
    Supp Figure 4b: added-variable scatter plots comparing raw and partial
    correlation between LMI and participation rate.

    Layout: 2 rows (R+, R-) x 2 columns (raw, partial).
      Left column : raw scatter of LMI vs participation_rate.
      Right column: residuals of LMI and participation_rate after regressing
                    each on transient_freq (partial regression / added-variable
                    plot). Slope and r equal the partial regression coefficient
                    and partial correlation.

    If the partial r (right) remains significant and close in magnitude to the
    raw r (left), spontaneous activity does not explain the LMI-participation
    relationship.

    Saves:
        <filename>.svg       -- figure
        <filename>_data.csv  -- data with residuals per reward group
        <filename>_stats.csv -- raw and partial Pearson r / p per reward group
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(
        subset=['lmi', 'participation_rate', 'transient_freq', 'reward_group']
    )

    group_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}

    def residuals(a, b):
        slope, intercept, _, _, _ = linregress(b, a)
        return a - (slope * b + intercept)

    def annotate(ax, r, p):
        p_str = ('p < 0.001 ***' if p < 0.001 else
                 f'p = {p:.3f} **' if p < 0.01 else
                 f'p = {p:.3f} *' if p < 0.05 else
                 f'p = {p:.3f} ns')
        ax.text(0.05, 0.95, f'r = {r:.3f}\n{p_str}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.9, edgecolor='gray'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    stats_rows = []
    data_rows = []

    for row, reward_group in enumerate(['R+', 'R-']):
        color = group_colors[reward_group]
        gdata = merged[merged['reward_group'] == reward_group].copy()

        if len(gdata) < 5:
            for col in range(2):
                axes[row, col].text(0.5, 0.5, 'Insufficient data',
                                    ha='center', va='center',
                                    transform=axes[row, col].transAxes)
            continue

        lmi  = gdata['lmi'].values
        part = gdata['participation_rate'].values
        freq = gdata['transient_freq'].values

        lmi_resid  = residuals(lmi, freq)
        part_resid = residuals(part, freq)

        r_raw,     p_raw     = pearsonr(lmi, part)
        r_partial, p_partial = pearsonr(lmi_resid, part_resid)

        n_mice = gdata['mouse_id'].nunique()
        print(f"  {reward_group}  raw r={r_raw:.3f} p={p_raw:.4f} | "
              f"partial r={r_partial:.3f} p={p_partial:.4f}  "
              f"(n={len(lmi)} cells, {n_mice} mice)")

        stats_rows.append({
            'reward_group': reward_group,
            'r_raw': r_raw, 'p_raw': p_raw,
            'r_partial': r_partial, 'p_partial': p_partial,
            'n_cells': len(lmi), 'n_mice': n_mice,
            'test': 'Pearson r (LMI vs participation_rate)',
        })

        gdata = gdata.copy()
        gdata['lmi_resid']  = lmi_resid
        gdata['part_resid'] = part_resid
        data_rows.append(gdata)

        for col, (x, y, r, p, xlabel, ylabel) in enumerate([
            (lmi,       part,       r_raw,     p_raw,
             'LMI', 'Participation rate'),
            (lmi_resid, part_resid, r_partial, p_partial,
             'LMI  (residual | transient freq)',
             'Participation rate  (residual | transient freq)'),
        ]):
            ax = axes[row, col]
            ax.scatter(x, y, color=color, alpha=0.3, s=10, linewidths=0)

            slope, intercept, _, _, _ = linregress(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color='black', linewidth=1.5)

            ax.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            annotate(ax, r, p)

            ax.set_xlim(-1, 1)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel if col == 0 else '', fontsize=11)
            title = 'Raw' if col == 0 else 'Partial  (ctrl transient freq)'
            ax.set_title(
                f'{reward_group} — {title}  (n={len(lmi)} cells, {n_mice} mice)',
                fontweight='bold', fontsize=12,
            )
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

    fig.suptitle(
        'LMI vs Participation Rate: Raw and Partial Correlation (Day 0)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False
    )
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(output_dir, f'{filename}_stats.csv'), index=False
    )
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Mode:             {MODE}")
    print(f"Results dir:      {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    if MODE == 'compute':
        compute_lmi_data_csv(LMI_DATA_CSV)
    elif MODE == 'plot':
        if not os.path.exists(LMI_DATA_CSV):
            raise FileNotFoundError(
                f"Data CSV not found: {LMI_DATA_CSV}\n"
                "Run with MODE='compute' first."
            )
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'compute' or 'plot'.")

    print("\nPlotting panel supp_4a...")
    panel_supp4a_scatter(LMI_DATA_CSV, filename='supp_4a')

    print("\nPlotting panel supp_4b...")
    panel_supp4b_partial_corr(LMI_DATA_CSV, filename='supp_4b')

    print(f"\nDone. Figures saved to: {OUTPUT_DIR}")
