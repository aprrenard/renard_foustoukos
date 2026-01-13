"""
Figure 3e-f: Learning-Modulated Index (LMI) analysis

This script generates Panels e and f for Figure 3:
- Panel e: Proportion of LMI positive and negative cells
- Panel f: Distribution of LMI values across reward groups
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_and_process_lmi_data(
    lmi_file='lmi_results.csv',
    lmi_threshold_pos=0.975,
    lmi_threshold_neg=0.025
):
    """
    Load and process LMI data.

    Args:
        lmi_file: Name of CSV file containing LMI results
        lmi_threshold_pos: Threshold for LMI positive cells (p-value)
        lmi_threshold_neg: Threshold for LMI negative cells (p-value)

    Returns:
        lmi_df: DataFrame with LMI values and classifications
        lmi_prop: Proportions by mouse (all cells)
        lmi_prop_ct: Proportions by mouse and cell type
    """

    # Load LMI data
    processed_folder = io.solve_common_paths('processed_data')
    lmi_df = pd.read_csv(os.path.join(processed_folder, lmi_file))

    # Get list of imaging mice
    _, _, mice, _ = io.select_sessions_from_db(
        io.db_path,
        io.nwb_dir,
        two_p_imaging='yes',
    )

    # Add reward group information
    for mouse in lmi_df.mouse_id.unique():
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse)
        lmi_df.loc[lmi_df.mouse_id == mouse, 'reward_group'] = reward_group

    # Filter for imaging mice
    lmi_df = lmi_df.loc[lmi_df.mouse_id.isin(mice)]

    # Classify cells as LMI positive or negative
    lmi_df['lmi_pos'] = lmi_df['lmi_p'] >= lmi_threshold_pos
    lmi_df['lmi_neg'] = lmi_df['lmi_p'] <= lmi_threshold_neg

    # Calculate proportions by mouse (all cells)
    lmi_prop = lmi_df.groupby(['mouse_id', 'reward_group'])[
        ['lmi_pos', 'lmi_neg']
    ].apply(lambda x: x.sum() / x.count()).reset_index()

    # Calculate proportions by mouse and cell type
    lmi_prop_ct = lmi_df.groupby(['mouse_id', 'reward_group', 'cell_type'])[
        ['lmi_pos', 'lmi_neg']
    ].apply(lambda x: x.sum() / x.count()).reset_index()

    return lmi_df, lmi_prop, lmi_prop_ct


# ============================================================================
# Panel e: LMI proportions
# ============================================================================

def panel_e_lmi_proportions(
    lmi_prop=None,
    lmi_prop_ct=None,
    save_path='/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/cell_proportions',
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 3 Panel e: LMI proportions across reward groups.

    Shows bar plots comparing proportions of LMI positive and negative cells
    between R+ and R- groups for all cells, wS2, and wM1 neurons.

    Args:
        lmi_prop: Pre-loaded proportion data for all cells (if None, will load)
        lmi_prop_ct: Pre-loaded proportion data by cell type (if None, will load)
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load data if not provided
    if lmi_prop is None or lmi_prop_ct is None:
        _, lmi_prop, lmi_prop_ct = load_and_process_lmi_data()

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    # Create figure: 1x6 grid (all pos, all neg, wS2 pos, wS2 neg, wM1 pos, wM1 neg)
    fig, axes = plt.subplots(1, 6, figsize=(15, 3), sharey=True)

    # ========================================================================
    # Statistical testing
    # ========================================================================
    results = []
    groups = ['lmi_pos', 'lmi_neg']
    cell_types = [None, 'wS2', 'wM1']

    for group in groups:
        for cell_type in cell_types:
            if cell_type:
                data = lmi_prop_ct[lmi_prop_ct.cell_type == cell_type]
            else:
                data = lmi_prop

            r_plus = data[data.reward_group == 'R+'][group]
            r_minus = data[data.reward_group == 'R-'][group]

            stat, p_value = mannwhitneyu(
                r_plus, r_minus,
                alternative='two-sided'
            )
            results.append({
                'group': group,
                'cell_type': cell_type if cell_type else 'all',
                'stat': stat,
                'p_value': p_value
            })

    results_df = pd.DataFrame(results)

    # ========================================================================
    # Helper function for significance stars
    # ========================================================================
    def get_star(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    # ========================================================================
    # Plot each panel
    # ========================================================================
    plot_configs = [
        (lmi_prop, None, 'lmi_pos', 'LMI Positive'),
        (lmi_prop, None, 'lmi_neg', 'LMI Negative'),
        (lmi_prop_ct, 'wS2', 'lmi_pos', 'LMI Positive wS2'),
        (lmi_prop_ct, 'wS2', 'lmi_neg', 'LMI Negative wS2'),
        (lmi_prop_ct, 'wM1', 'lmi_pos', 'LMI Positive wM1'),
        (lmi_prop_ct, 'wM1', 'lmi_neg', 'LMI Negative wM1')
    ]

    for i, (data, cell_type, metric, title) in enumerate(plot_configs):
        # Filter data by cell type if specified
        if cell_type:
            plot_data = data.loc[data.cell_type == cell_type]
        else:
            plot_data = data

        # Create bar plot
        sns.barplot(
            data=plot_data,
            x='reward_group',
            order=['R+', 'R-'],
            hue='reward_group',
            y=metric,
            ax=axes[i],
            palette=reward_palette,
            hue_order=['R-', 'R+'],
            legend=False
        )
        axes[i].set_title(title)
        axes[i].set_xlabel('Reward Group')
        axes[i].set_ylabel('Proportion')

        # Add significance stars
        stat_row = results_df[
            (results_df['group'] == metric) &
            (results_df['cell_type'] == (cell_type if cell_type else 'all'))
        ]
        if not stat_row.empty:
            p = stat_row.iloc[0]['p_value']
            star = get_star(p)
            # Add star annotation between bars
            y_max = axes[i].get_ylim()[1]
            axes[i].annotate(
                star,
                xy=(0.5, y_max * 0.95),
                xycoords='axes fraction',
                ha='center',
                va='bottom',
                fontsize=18,
                color='black'
            )

    sns.despine(trim=True)
    plt.tight_layout()

    # Save figure and data
    save_path = io.adjust_path_to_host(save_path)
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_3e.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Save data and statistics
    data_file_all = os.path.join(save_path, 'figure_3e_proportions_all.csv')
    data_file_ct = os.path.join(save_path, 'figure_3e_proportions_by_celltype.csv')
    stats_file = os.path.join(save_path, 'figure_3e_stats.csv')

    lmi_prop.to_csv(data_file_all, index=False)
    lmi_prop_ct.to_csv(data_file_ct, index=False)
    results_df.to_csv(stats_file, index=False)

    print(f"Figure 3e saved to: {output_file}")
    print(f"Figure 3e data saved to: {data_file_all} and {data_file_ct}")
    print(f"Figure 3e statistics saved to: {stats_file}")

    return results_df


# ============================================================================
# Panel f: LMI distributions
# ============================================================================

def panel_f_lmi_distributions(
    lmi_df=None,
    n_bins=30,
    save_path='/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/cell_proportions',
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 3 Panel f: LMI distributions across reward groups.

    Shows histogram distributions of LMI values comparing R+ vs R- groups
    for all cells, wS2, and wM1 neurons.

    Args:
        lmi_df: Pre-loaded LMI dataframe (if None, will load)
        n_bins: Number of bins for histograms
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load data if not provided
    if lmi_df is None:
        lmi_df, _, _ = load_and_process_lmi_data()

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    # Use common bin edges for all three plots
    bin_edges = np.histogram_bin_edges(lmi_df['lmi'], bins=n_bins)

    # Create figure: 1x3 grid (all cells, wS2, wM1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    cell_types = [None, 'wS2', 'wM1']
    titles = ['All cells', 'wS2p', 'wM1p']

    # Plot each cell type
    for i, (cell_type, title) in enumerate(zip(cell_types, titles)):
        # Filter data by cell type
        if cell_type:
            data = lmi_df[lmi_df.cell_type == cell_type]
        else:
            data = lmi_df

        # Plot histograms for both reward groups
        for rg, color in zip(['R-', 'R+'], reward_palette):
            sns.histplot(
                data[data.reward_group == rg]['lmi'],
                bins=bin_edges,
                kde=True,
                ax=axes[i],
                color=color,
                label=rg,
                stat='density',
                alpha=0.5,
            )

        axes[i].set_title(title)
        axes[i].set_xlabel('LMI')
        axes[i].set_ylabel('Density')
        axes[i].legend()

    sns.despine()
    plt.tight_layout()

    # Save figure
    save_path = io.adjust_path_to_host(save_path)
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_3f.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Statistical testing: Kolmogorov-Smirnov test
    # ========================================================================
    ks_results = []
    for cell_type in cell_types:
        if cell_type:
            data = lmi_df[lmi_df.cell_type == cell_type]
        else:
            data = lmi_df

        lmi_r_minus = data[data.reward_group == 'R-']['lmi']
        lmi_r_plus = data[data.reward_group == 'R+']['lmi']

        stat, p = ks_2samp(lmi_r_minus, lmi_r_plus)
        ks_results.append({
            'cell_type': cell_type if cell_type else 'all',
            'stat': stat,
            'p_value': p
        })

    ks_results_df = pd.DataFrame(ks_results)

    # Save statistics
    stats_file = os.path.join(save_path, 'figure_3f_stats.csv')
    ks_results_df.to_csv(stats_file, index=False)

    print(f"Figure 3f saved to: {output_file}")
    print(f"Figure 3f statistics saved to: {stats_file}")

    return ks_results_df


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Load data once
    print("Loading LMI data...")
    lmi_df, lmi_prop, lmi_prop_ct = load_and_process_lmi_data()

    # Generate panel e
    print("\nGenerating panel e (LMI proportions)...")
    panel_e_lmi_proportions(lmi_prop=lmi_prop, lmi_prop_ct=lmi_prop_ct)

    # Generate panel f
    print("\nGenerating panel f (LMI distributions)...")
    panel_f_lmi_distributions(lmi_df=lmi_df)

    print("\nDone!")
