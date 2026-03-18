"""
Figure 4h: Reactivation heatmap illustration (mouse AR127, single day)

Layout:
  - Left strips: LMI, participation rate, and whisker template (one bar per cell)
  - Right: neural activity heatmap (cells × time)
  - Top of heatmap: tick marks at detected reactivation events

Result files (reactivation_results.pkl, lmi_results.csv,
cell_participation_rates_per_day.csv, circular_shift_significant_participation.csv)
are loaded from data_processed/reactivation/. Figures are saved to output/.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoLocator

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io


# ============================================================================
# Parameters
# ============================================================================

MOUSE = 'AR127'
DAY = 0
SAMPLING_RATE = 30
TIME_WINDOW = 180  # seconds

HEATMAP_CMAP = 'RdBu_r'

RESULTS_DIR = os.path.join(io.processed_dir, 'reactivation')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_4', 'output')


# ============================================================================
# Panel h
# ============================================================================

def panel_h_reactivation_heatmap(
    r_plus_results,
    r_minus_results,
    mouse=MOUSE,
    day=DAY,
    sampling_rate=SAMPLING_RATE,
    time_window=TIME_WINDOW,
    sort_by='participation',
    top_n=20,
    results_dir=RESULTS_DIR,
    filename='figure_4h',
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 4 Panel h: reactivation heatmap for a single mouse/day.

    Args:
        sort_by: 'template' or 'participation' — cell ordering.
        top_n: If set, restrict to the top N cells by participation rate.
        results_dir: Directory containing all result CSV/pkl files.
        filename: Output filename stem (without extension).
    """
    if mouse in r_plus_results:
        results = r_plus_results[mouse]
        reward_group = 'R+'
    elif mouse in r_minus_results:
        results = r_minus_results[mouse]
        reward_group = 'R-'
    else:
        raise ValueError(f"Mouse {mouse} not found in results")

    if day not in results['days']:
        raise ValueError(f"Day {day} not found for mouse {mouse}")

    day_data = results['days'][day]

    # Extract data
    template = np.array(day_data['template'])
    correlations = np.array(day_data['correlations'])
    events = np.array(day_data['events'])
    threshold = day_data.get('threshold_used', 0.45)

    selected_trials = day_data['selected_trials']
    n_cells, n_trials, n_tp = selected_trials.shape
    neural_data = selected_trials.values.reshape(n_cells, -1)
    neural_data = np.nan_to_num(neural_data, nan=0.0)

    # Truncate to requested time window
    max_frames = int(time_window * sampling_rate)
    neural_data = neural_data[:, :max_frames]
    correlations = correlations[:max_frames]
    events = events[events < max_frames]
    n_frames = neural_data.shape[1]

    # Load per-cell metrics
    roi_ids_orig = selected_trials.coords['roi'].values

    lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
    mouse_lmi = lmi_df[lmi_df['mouse_id'] == mouse].set_index('roi')
    lmi_orig = np.array([
        mouse_lmi.loc[r, 'lmi'] if r in mouse_lmi.index else np.nan
        for r in roi_ids_orig
    ])

    part_df = pd.read_csv(os.path.join(results_dir, 'cell_participation_rates_per_day.csv'))
    mouse_part = (part_df[(part_df['mouse_id'] == mouse) & (part_df['day'] == day)]
                  .set_index('roi'))
    part_orig = np.array([
        mouse_part.loc[r, 'participation_rate'] if r in mouse_part.index else np.nan
        for r in roi_ids_orig
    ])

    # Sort cells
    if sort_by == 'participation':
        key = np.where(np.isnan(part_orig), -np.inf, part_orig)
        sort_idx = np.argsort(key)[::-1]
    else:
        sort_idx = np.argsort(template)[::-1]

    neural_data_sorted = neural_data[sort_idx, :]
    template_sorted = template[sort_idx]
    lmi_values = lmi_orig[sort_idx]
    part_values = part_orig[sort_idx]

    # Restrict to top-N cells by participation
    if top_n is not None:
        neural_data_sorted = neural_data_sorted[:top_n, :]
        template_sorted = template_sorted[:top_n]
        lmi_values = lmi_values[:top_n]
        part_values = part_values[:top_n]
    n_cells = neural_data_sorted.shape[0]

    # Colour ranges — symmetric so white = 0
    act_data_min = float(np.percentile(neural_data_sorted, 2))
    act_data_max = float(np.percentile(neural_data_sorted, 99))
    vmax_act = max(abs(act_data_min), abs(act_data_max))
    vmin_act = -vmax_act
    neural_data_display = np.clip(neural_data_sorted, act_data_min, act_data_max)

    vmin_tmpl = 0.0
    vmax_tmpl = max(float(np.nanmax(template_sorted)), 1e-6)

    # Build figure
    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(
        3, 5,
        figure=fig,
        width_ratios=[1, 1, 1, 14, 0.4],
        height_ratios=[3, 14, 0.8],
        wspace=0.03,
        hspace=0.04,
    )

    ax_lmi = fig.add_subplot(gs[1, 0])
    ax_template = fig.add_subplot(gs[1, 1])
    ax_part = fig.add_subplot(gs[1, 2])
    ax_heatmap = fig.add_subplot(gs[1, 3])
    ax_cbar = fig.add_subplot(gs[1, 4])
    ax_events = fig.add_subplot(gs[0, 3], sharex=ax_heatmap)
    ax_cbar_lmi = fig.add_subplot(gs[2, 0])
    ax_cbar_tmpl = fig.add_subplot(gs[2, 1])
    ax_cbar_part = fig.add_subplot(gs[2, 2])

    for _ax in [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4]),
                fig.add_subplot(gs[2, 3]), fig.add_subplot(gs[2, 4])]:
        _ax.set_visible(False)

    # Heatmap
    ax_heatmap.imshow(
        neural_data_display,
        aspect='auto', cmap=HEATMAP_CMAP,
        vmin=vmin_act, vmax=vmax_act,
        interpolation='none', origin='upper',
        extent=[0, n_frames / sampling_rate, n_cells - 0.5, -0.5],
    )
    ax_heatmap.set_xlabel('Time (s)', fontsize=9)
    ax_heatmap.set_yticks([])
    ax_heatmap.tick_params(axis='x', labelsize=7)
    ax_heatmap.spines['top'].set_visible(False)
    ax_heatmap.spines['right'].set_visible(False)
    ax_heatmap.spines['left'].set_visible(False)

    # Events panel — correlation trace with threshold and event markers
    t_corr = np.arange(n_frames) / sampling_rate
    ax_events.plot(t_corr, correlations, 'k-', linewidth=0.7, alpha=0.9)
    ax_events.axhline(threshold, color='#e84040', linestyle='--', linewidth=0.8, alpha=0.7)
    for ev in events:
        ax_events.axvline(ev / sampling_rate, color='#e84040', linewidth=0.6, alpha=0.5)
    corr_min, corr_max = float(np.nanmin(correlations)), float(np.nanmax(correlations))
    corr_pad = (corr_max - corr_min) * 0.05
    ax_events.set_ylim(corr_min - corr_pad, corr_max + corr_pad)
    ax_events.set_yticks([])
    ax_events.xaxis.set_visible(False)
    ax_events.spines['top'].set_visible(False)
    ax_events.spines['right'].set_visible(False)
    ax_events.spines['bottom'].set_visible(False)
    ax_events.spines['left'].set_visible(False)
    ax_events.set_title(
        f'{mouse} ({reward_group})  |  Day {day}  |  {len(events)} reactivations',
        fontsize=10, fontweight='bold', pad=4,
    )
    ax_heatmap.xaxis.set_major_locator(AutoLocator())

    def _strip(ax, values, cmap, xlabel, vmin=None, vmax=None):
        img = np.ma.masked_invalid(values.reshape(-1, 1))
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='lightgrey')
        im = ax.imshow(img, aspect='auto', cmap=cmap_obj,
                       vmin=vmin, vmax=vmax,
                       interpolation='none', origin='upper')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        return im

    im_tmpl = _strip(ax_template, template_sorted, 'Reds',
                     'Template', vmin=vmin_tmpl, vmax=vmax_tmpl)
    im_lmi = _strip(ax_lmi, lmi_values, HEATMAP_CMAP, 'LMI', vmin=-1, vmax=1)
    im_part = _strip(ax_part, part_values, 'Reds', 'Particip.\nrate', vmin=0, vmax=1)

    # Colourbars
    tmpl_data_min = 0.0
    tmpl_data_max = float(np.nanmax(template_sorted))

    cb_act = fig.colorbar(ax_heatmap.images[0], cax=ax_cbar)
    cb_act.ax.set_ylim(act_data_min, act_data_max)
    cb_act.set_ticks([act_data_min, 0, act_data_max])
    cb_act.set_ticklabels([f'{act_data_min:.2f}', '0', f'{act_data_max:.2f}'])
    ax_cbar.set_ylabel('dF/F', fontsize=8)
    ax_cbar.tick_params(labelsize=7)

    for im, cax, label, dmin, dmax in [
        (im_lmi,  ax_cbar_lmi,  'LMI',  -1,            1),
        (im_tmpl, ax_cbar_tmpl, 'dF/F', tmpl_data_min, tmpl_data_max),
        (im_part, ax_cbar_part, 'Rate',  0,             1),
    ]:
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.set_xlim(dmin, dmax)
        cb.set_ticks([dmin, dmax])
        cb.set_ticklabels([f'{dmin:.2f}', f'{dmax:.2f}'])
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {os.path.join(output_dir, filename + '.' + save_format)}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    results_file = os.path.join(RESULTS_DIR, 'reactivation_results_p99.pkl')
    print(f"Loading reactivation results from: {results_file}")
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            "Please run reactivation.py with mode='compute' first."
        )

    with open(results_file, 'rb') as f:
        results_data = pickle.load(f)

    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']
    print(f"Loaded results for {len(r_plus_results)} R+ mice and {len(r_minus_results)} R- mice")

    panel_h_reactivation_heatmap(
        r_plus_results, r_minus_results,
        sort_by='participation',
        top_n=20,
        filename='figure_4h',
    )
