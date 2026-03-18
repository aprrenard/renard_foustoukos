"""
Figure 4j: Reactivation rate across days, R+ vs R-

Grouped bar plot (one bar per group per day) showing reactivation event
frequency (events/min) across the 5 experimental days. Individual mouse
trajectories are overlaid. Mann-Whitney U test per day (R+ vs R-).

Result files are loaded from data_processed/reactivation/.
Figures and CSVs are saved to output/.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]

RESULTS_FILE = os.path.join(io.processed_dir, 'reactivation', 'reactivation_results_p99.pkl')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_4', 'output')


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
# Panel j
# ============================================================================

def panel_j_reactivation_rate(
    r_plus_results,
    r_minus_results,
    days=DAYS,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 4 Panel j: reactivation frequency per day, R+ vs R-.

    Saves:
        figure_4j_data.csv: mouse_id, reward_group, day, event_frequency
        figure_4j_stats.csv: Mann-Whitney U results per day
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    days_sorted = sorted(days)

    # Build long-format DataFrame
    data_list = []
    for mouse, results in r_plus_results.items():
        for day in days_sorted:
            if day in results['days']:
                data_list.append({'mouse_id': mouse, 'reward_group': 'R+',
                                  'Day': day,
                                  'Frequency': results['days'][day]['event_frequency']})
    for mouse, results in r_minus_results.items():
        for day in days_sorted:
            if day in results['days']:
                data_list.append({'mouse_id': mouse, 'reward_group': 'R-',
                                  'Day': day,
                                  'Frequency': results['days'][day]['event_frequency']})

    df = pd.DataFrame(data_list)

    # Statistics: Mann-Whitney U per day
    stats_rows = []
    p_values = []
    for day in days_sorted:
        r_plus_vals = df[(df['Day'] == day) & (df['reward_group'] == 'R+')]['Frequency'].values
        r_minus_vals = df[(df['Day'] == day) & (df['reward_group'] == 'R-')]['Frequency'].values
        if len(r_plus_vals) > 0 and len(r_minus_vals) > 0:
            stat, p = mannwhitneyu(r_plus_vals, r_minus_vals, alternative='two-sided')
        else:
            stat, p = np.nan, 1.0
        p_values.append(p)
        stats_rows.append({
            'test': 'Mann-Whitney U',
            'day': day,
            'R+_n': len(r_plus_vals),
            'R-_n': len(r_minus_vals),
            'R+_mean': np.nanmean(r_plus_vals),
            'R-_mean': np.nanmean(r_minus_vals),
            'statistic': stat,
            'p_value': p,
            'significance': _significance_stars(p),
        })

    # Add dummy rows so all days appear even if data is missing
    for day in days_sorted:
        for group in ['R+', 'R-']:
            if day not in df[df['reward_group'] == group]['Day'].values:
                df = pd.concat([df, pd.DataFrame(
                    {'mouse_id': [''], 'reward_group': [group],
                     'Day': [day], 'Frequency': [np.nan]}
                )], ignore_index=True)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    sns.barplot(data=df, x='Day', y='Frequency', hue='reward_group',
                errorbar=('ci', 95),
                palette={'R+': reward_palette[1], 'R-': reward_palette[0]},
                hue_order=['R+', 'R-'],
                alpha=0.7, edgecolor='black', ax=ax)

    # # Individual mouse trajectories
    # bar_width = 0.35
    # group_offsets = {'R+': -bar_width / 2, 'R-': bar_width / 2}
    # x_positions = {day: idx for idx, day in enumerate(days_sorted)}

    # for mouse in df['mouse_id'].unique():
    #     if mouse == '':
    #         continue
    #     mouse_data = df[df['mouse_id'] == mouse].sort_values('Day')
    #     group = mouse_data['reward_group'].iloc[0]
    #     mouse_x = [x_positions[d] + group_offsets[group] for d in mouse_data['Day']]
    #     mouse_y = mouse_data['Frequency'].values
    #     color = reward_palette[1] if group == 'R+' else reward_palette[0]
    #     ax.plot(mouse_x, mouse_y, '-', color=color, linewidth=0.8, alpha=0.5, zorder=5)

    # Significance stars
    y_max = df['Frequency'].max()
    y_range = y_max * 0.05
    width = 0.35

    for day_idx, (day, p) in enumerate(zip(days_sorted, p_values)):
        stars = _significance_stars(p)
        if stars != 'n.s.':
            r_plus_vals = df[(df['Day'] == day) & (df['reward_group'] == 'R+') & (df['mouse_id'] != '')]['Frequency']
            r_minus_vals = df[(df['Day'] == day) & (df['reward_group'] == 'R-') & (df['mouse_id'] != '')]['Frequency']
            ci_plus = r_plus_vals.mean() + 1.96 * r_plus_vals.std() / np.sqrt(len(r_plus_vals)) if len(r_plus_vals) > 0 else 0
            ci_minus = r_minus_vals.mean() + 1.96 * r_minus_vals.std() / np.sqrt(len(r_minus_vals)) if len(r_minus_vals) > 0 else 0
            y1 = max(ci_plus, ci_minus)
            y2 = y1 + y_range
            x1 = day_idx - width / 2
            x2 = day_idx + width / 2
            ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], 'k-', linewidth=1)
            ax.text((x1 + x2) / 2, y2, stars, ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Day', fontsize=10)
    ax.set_ylabel('Reactivation rate (events/min)', fontsize=10)
    ax.legend(title='', fontsize=9)
    ax.set_ylim(0, 20)
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'figure_4j.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 4j saved to: {os.path.join(output_dir, 'figure_4j.' + save_format)}")

    # Save CSVs
    df[df['mouse_id'] != ''].to_csv(
        os.path.join(output_dir, 'figure_4j_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(output_dir, 'figure_4j_stats.csv'), index=False)
    print(f"Figure 4j data/stats saved to: {output_dir}")


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

    panel_j_reactivation_rate(r_plus_results, r_minus_results)
