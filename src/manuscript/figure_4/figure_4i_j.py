"""
Figure 4k-l: Reactivation participation rate vs LMI.

Panel k: Scatter plot of day-0 participation rate vs LMI (one dot per cell),
         separately for R+ and R- mice, with a linear regression line and
         Pearson r coefficient.

Panel l: Participation rate across days (-2 to +2) for LMI+ vs LMI- cells,
         showing per-mouse averages with individual trajectories. Stats:
         2-way repeated-measures ANOVA (day × LMI category) followed by
         Mann-Whitney U posthoc tests (positive vs negative per day).

Execution modes:
    MODE = 'compute' : run participation-rate pipeline, save CSVs, then plot
    MODE = 'plot'    : load previously saved CSVs and plot only

Processed data files are saved/loaded from data_processed/reactivation/.
Figures and CSVs are saved to io.manuscript_output_dir/figure_4/output/.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress, kruskal
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
EVENT_WINDOW_MS = 150
EVENT_WINDOW_FRAMES = int(EVENT_WINDOW_MS / 1000 * SAMPLING_RATE)
PARTICIPATION_THRESHOLD = 0.1
MIN_EVENTS_FOR_RELIABILITY = 5
LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025
N_JOBS = 35

RESULTS_DIR = os.path.join(io.processed_dir, 'reactivation')
REACTIVATION_RESULTS_FILE = os.path.join(RESULTS_DIR, 'reactivation_results_p99.pkl')
PARTICIPATION_CSV = os.path.join(RESULTS_DIR, 'cell_participation_rates_per_day.csv')
MERGED_CSV = os.path.join(RESULTS_DIR, 'participation_lmi_merged.csv')
LMI_RESULTS_CSV = os.path.join(io.processed_dir, 'lmi_results.csv')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_4', 'output')
FOLDER = os.path.join(io.processed_dir, 'mice')

# Execution mode
#   'compute' : run participation-rate pipeline, save CSVs, then plot
#   'plot'    : load previously saved CSVs and plot only
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
    except Exception:
        continue

print(f"Found {len(r_plus_mice)} R+ mice and {len(r_minus_mice)} R- mice")


# ============================================================================
# Participation pipeline (self-contained, no external reactivation imports)
# ============================================================================

def _load_reactivation_results(results_file):
    """Load pre-computed reactivation results from pickle."""
    import pickle
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Reactivation results file not found: {results_file}\n"
            "Please run reactivation.py with mode='compute' first.")
    print(f"\nLoading reactivation events from: {results_file}")
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    r_plus = data['r_plus_results']
    r_minus = data['r_minus_results']
    print(f"Loaded {len(r_plus)} R+ mice and {len(r_minus)} R- mice")
    return r_plus, r_minus


def _extract_event_responses(mouse, day, preloaded_events):
    """Extract per-cell dF/F responses around pre-computed reactivation events."""
    xarr = utils_imaging.load_mouse_xarray(
        mouse, FOLDER, 'tensor_xarray_learning_data.nc', substracted=True)
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


def _aggregate_across_days(participation_df_all):
    """Aggregate per-day participation rates to baseline/learning/post periods."""
    if participation_df_all is None or len(participation_df_all) == 0:
        return None

    baseline_days = [-2, -1]
    post_days = [1, 2]
    results = []

    for (mouse_id, roi), group in participation_df_all.groupby(['mouse_id', 'roi']):
        baseline_data = group[group['day'].isin(baseline_days)]
        baseline_rate = baseline_data['participation_rate'].mean() if len(baseline_data) > 0 else np.nan
        reliable_baseline = baseline_data['reliable'].all() if len(baseline_data) > 0 else False

        learning_data = group[group['day'] == 0]
        learning_rate = learning_data['participation_rate'].iloc[0] if len(learning_data) > 0 else np.nan
        reliable_learning = learning_data['reliable'].iloc[0] if len(learning_data) > 0 else False

        post_data = group[group['day'].isin(post_days)]
        post_rate = post_data['participation_rate'].mean() if len(post_data) > 0 else np.nan
        reliable_post = post_data['reliable'].all() if len(post_data) > 0 else False

        results.append({
            'mouse_id': mouse_id, 'roi': roi,
            'baseline_rate': baseline_rate,
            'learning_rate': learning_rate,
            'post_rate': post_rate,
            'delta_learning': learning_rate - baseline_rate if not np.isnan(baseline_rate) else np.nan,
            'delta_post': post_rate - baseline_rate if not np.isnan(baseline_rate) else np.nan,
            'reliable_baseline': reliable_baseline,
            'reliable_learning': reliable_learning,
            'reliable_post': reliable_post,
        })

    return pd.DataFrame(results)


def _load_and_match_lmi_data(participation_df):
    """Load LMI data and merge with participation data."""
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)

    if 'reward_group' not in lmi_df.columns:
        group_map = {m: 'R+' for m in r_plus_mice}
        group_map.update({m: 'R-' for m in r_minus_mice})
        lmi_df['reward_group'] = lmi_df['mouse_id'].map(group_map)

    lmi_df['lmi_category'] = 'neutral'
    lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD, 'lmi_category'] = 'positive'
    lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD, 'lmi_category'] = 'negative'

    cols = ['mouse_id', 'roi', 'lmi', 'lmi_p', 'lmi_category', 'reward_group']
    if 'cell_type' in lmi_df.columns:
        cols.append('cell_type')

    merged_df = pd.merge(participation_df, lmi_df[cols], on=['mouse_id', 'roi'], how='inner')

    print(f"\n  Merged: {len(merged_df)} cells total "
          f"({(merged_df['lmi_category']=='positive').sum()} LMI+, "
          f"{(merged_df['lmi_category']=='negative').sum()} LMI-, "
          f"{(merged_df['lmi_category']=='neutral').sum()} neutral)")
    return merged_df


# ============================================================================
# Helpers
# ============================================================================

def _significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


# ============================================================================
# Data loading / computation
# ============================================================================

def _compute_participation_data():
    """Run the full participation-rate computation pipeline.

    1. Load pre-computed reactivation events.
    2. Compute per-cell participation rates per day (parallel across mice).
    3. Aggregate across days and merge with LMI data.

    Saves PARTICIPATION_CSV and MERGED_CSV to RESULTS_DIR.

    Returns
    -------
    merged_df  : pd.DataFrame  – per-cell aggregated data with LMI info
    per_day_df : pd.DataFrame  – per-cell, per-day participation rates
    """
    r_plus, r_minus = _load_reactivation_results(REACTIVATION_RESULTS_FILE)
    all_results = {**r_plus, **r_minus}
    all_mice = list(all_results.keys())

    print(f"\nComputing participation rates for {len(all_mice)} mice...")
    results_list = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_process_mouse_participation)(mouse, all_results.get(mouse))
        for mouse in all_mice
    )

    all_data = [df for _, df in results_list if df is not None]
    if not all_data:
        raise RuntimeError("No participation data computed.")
    per_day_df = pd.concat(all_data, ignore_index=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_day_df.to_csv(PARTICIPATION_CSV, index=False)

    aggregated_df = _aggregate_across_days(per_day_df)
    merged_df = _load_and_match_lmi_data(aggregated_df)
    merged_df.to_csv(MERGED_CSV, index=False)

    return merged_df, per_day_df


def _load_participation_data():
    """Load pre-computed participation data from CSVs."""
    for path in [PARTICIPATION_CSV, MERGED_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pre-computed data not found: {path}\n"
                "Run with MODE='compute' first.")
    merged_df = pd.read_csv(MERGED_CSV)
    per_day_df = pd.read_csv(PARTICIPATION_CSV)
    print(f"Loaded {len(merged_df)} cells and {len(per_day_df)} cell-day records.")
    return merged_df, per_day_df


def _prepare_data(merged_df):
    """Add lmi_category column to merged_df if not present."""
    if 'lmi_category' not in merged_df.columns:
        merged_df = merged_df.copy()
        merged_df['lmi_category'] = 'neutral'
        merged_df.loc[merged_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD,
                      'lmi_category'] = 'positive'
        merged_df.loc[merged_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD,
                      'lmi_category'] = 'negative'
    return merged_df


# ============================================================================
# Panel k: scatter of day-0 participation rate vs LMI
# ============================================================================

def panel_k_participation_vs_lmi(
    merged_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4k',
    save_format='svg',
    dpi=300,
):
    """Figure 4 Panel k: scatter of day-0 participation rate vs LMI.

    One dot per cell. Separate subplots for R+ and R-.
    Linear regression line with Pearson r and p-value displayed.

    Saves:
        <filename>.svg        – figure
        <filename>_stats.csv  – per-reward-group Pearson r, p-value, regression params
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    df = merged_df.dropna(subset=['lmi', 'learning_rate']).copy()

    reward_groups = ['R+', 'R-']
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    stats_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = df[df['reward_group'] == rg]
        x = grp['lmi'].values
        y = grp['learning_rate'].values

        # Scatter
        ax.scatter(x, y, color=rg_colors[rg], s=4, alpha=0.4, linewidths=0,
                   rasterized=True)

        # Linear regression line
        if len(x) >= 3:
            slope, intercept, r_value, p_value, se = linregress(x, y)
            pearson_r, pearson_p = pearsonr(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, slope * x_line + intercept,
                    color='black', linewidth=1.2, zorder=5)
            stars = _significance_stars(pearson_p)
            ax.text(0.05, 0.95,
                    f'r = {pearson_r:.3f}\np = {pearson_p:.3g} {stars}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8)
            stats_rows.append({
                'reward_group': rg,
                'n_cells': len(x),
                'n_mice': grp['mouse_id'].nunique(),
                'pearson_r': pearson_r,
                'p_value': pearson_p,
                'significance': stars,
                'slope': slope,
                'intercept': intercept,
                'stderr': se,
            })
        else:
            pearson_r, pearson_p = np.nan, np.nan

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        n_cells = len(grp)
        n_mice = grp['mouse_id'].nunique()
        ax.set_title(f'{rg}  (n = {n_cells} cells, {n_mice} mice)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('LMI', fontsize=9)
        ax.set_ylabel('Participation rate (day 0)' if i == 0 else '', fontsize=9)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Panel k saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.DataFrame(stats_rows).to_csv(
        os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel k stats saved: {output_dir}")


# ============================================================================
# Panel l: participation rate across days (LMI+ vs LMI-)
# ============================================================================

def panel_l_participation_across_days(
    merged_df,
    per_day_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4l',
    save_format='svg',
    dpi=300,
):
    """Figure 4 Panel l: participation rate across days for LMI+ vs LMI- cells.

    Per-mouse averages with individual trajectories. Stats: Kruskal-Wallis
    test (effect of day) run independently for each of the four groups
    (R+ positive LMI, R+ negative LMI, R- positive LMI, R- negative LMI).

    Saves:
        <filename>.svg         – figure
        <filename>_data.csv    – per-mouse × day × LMI-category averages
        <filename>_stats.csv   – Kruskal-Wallis results per group
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    days_sorted = sorted(DAYS)
    lmi_categories = ['positive', 'negative']
    cat_colors = {'positive': '#d62728', 'negative': '#1f77b4'}
    reward_groups = ['R+', 'R-']

    lmi_cells = merged_df.loc[
        merged_df['lmi_category'].isin(lmi_categories),
        ['mouse_id', 'roi', 'lmi_category', 'reward_group'],
    ]
    day_data = pd.merge(per_day_df, lmi_cells, on=['mouse_id', 'roi'], how='inner')

    mouse_day_avg = (
        day_data
        .groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'],
                 observed=True)['participation_rate']
        .mean()
        .reset_index()
    )

    cell_counts = (
        lmi_cells.groupby(['reward_group', 'lmi_category'], observed=True)
        .size()
        .to_dict()
    )

    # Kruskal-Wallis: effect of day within each (reward_group, lmi_category) group
    all_stats_rows = []
    kw_results = {}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp_data = mouse_day_avg[
                (mouse_day_avg['reward_group'] == rg) &
                (mouse_day_avg['lmi_category'] == cat)
            ]
            day_groups = [
                grp_data[grp_data['day'] == day]['participation_rate'].values
                for day in days_sorted
            ]
            day_groups = [g for g in day_groups if len(g) > 0]
            if len(day_groups) >= 2:
                try:
                    H, p = kruskal(*day_groups)
                except Exception:
                    H, p = np.nan, np.nan
            else:
                H, p = np.nan, np.nan
            kw_results[(rg, cat)] = (H, p)
            all_stats_rows.append({
                'reward_group': rg,
                'lmi_category': cat,
                'test': 'Kruskal-Wallis',
                'effect': 'day',
                'H_statistic': H,
                'p_value': p,
                'significance': _significance_stars(p) if not np.isnan(p) else 'n.a.',
                'n_days': len(day_groups),
            })
            print(f"  KW {rg} {cat} LMI: H={H:.3f}, p={p:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    plot_data_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = mouse_day_avg[mouse_day_avg['reward_group'] == rg]

        sns.barplot(
            data=grp, x='day', y='participation_rate', hue='lmi_category',
            hue_order=lmi_categories, palette=cat_colors, order=days_sorted,
            estimator=np.mean, errorbar=('ci', 95), capsize=0,
            err_kws={'linewidth': 1.5}, alpha=0.7, ax=ax,
        )
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(0.6)

        # # Individual mouse trajectories
        # for j, cat in enumerate(lmi_categories):
        #     if j >= len(ax.containers):
        #         continue
        #     cat_grp = grp[grp['lmi_category'] == cat]
        #     x_centers = {
        #         days_sorted[k]: bar.get_x() + bar.get_width() / 2
        #         for k, bar in enumerate(ax.containers[j])
        #         if k < len(days_sorted)
        #     }
        #     for mouse_id in cat_grp['mouse_id'].unique():
        #         mdata = cat_grp[cat_grp['mouse_id'] == mouse_id].sort_values('day')
        #         mx = [x_centers[d] for d in mdata['day'] if d in x_centers]
        #         my = mdata['participation_rate'].values
        #         ax.plot(mx, my, '-', color=cat_colors[cat],
        #                 linewidth=0.8, alpha=0.4, zorder=5)

        # Annotate Kruskal-Wallis results for each LMI group
        for j, cat in enumerate(lmi_categories):
            H, p = kw_results.get((rg, cat), (np.nan, np.nan))
            stars = _significance_stars(p) if not np.isnan(p) else 'n.a.'
            ax.text(0.02, 0.97 - j * 0.12,
                    f'{cat.capitalize()} LMI: KW p={p:.3g} {stars}',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=7, color=cat_colors[cat])

        n_pos = cell_counts.get((rg, 'positive'), 0)
        n_neg = cell_counts.get((rg, 'negative'), 0)
        ax.set_title(f'{rg}  (LMI+: {n_pos} cells | LMI−: {n_neg} cells)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Day', fontsize=9)
        ax.set_ylabel('Participation rate' if i == 0 else '', fontsize=9)
        ax.set_ylim(0, .4)
        ax.tick_params(labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f'{l.capitalize()} LMI' for l in labels], fontsize=8)
        sns.despine(ax=ax)

        plot_data_rows.append(grp)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Panel l saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(plot_data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(all_stats_rows).to_csv(
        os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel l data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Mode:             {MODE}")
    print(f"Output directory: {OUTPUT_DIR}")

    if MODE == 'compute':
        merged_df, per_day_df = _compute_participation_data()
    elif MODE == 'plot':
        merged_df, per_day_df = _load_participation_data()
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'compute' or 'plot'.")

    merged_df = _prepare_data(merged_df)
    print(f"\nDataset: {len(merged_df)} cells, {len(per_day_df)} cell-day records, "
          f"{merged_df['mouse_id'].nunique()} mice")

    panel_k_participation_vs_lmi(merged_df, filename='figure_4k')
    panel_l_participation_across_days(merged_df, per_day_df, filename='figure_4l')
