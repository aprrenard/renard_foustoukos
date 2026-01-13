"""
Figure 3c-d: Pre vs post learning response comparison

This script generates Panels c and d for Figure 3:
- Panel c: PSTH and response amplitude before/after learning for all cells
- Panel d: PSTH and response amplitude before/after learning for projection types (wS2, wM1)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io


# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_and_process_response_data(
    mice=None,
    days=[-2, -1, 0, 1, 2],
    win_sec_amp=(0, 0.300),
    win_sec_psth=(-0.5, 1.5),
    baseline_win=(0, 1),
    sampling_rate=30,
    file_name='tensor_xarray_mapping_data.nc'
):
    """
    Load and process average response and PSTH data for pre/post learning comparison.

    Args:
        mice: List of mouse IDs (if None, will query from database)
        days: List of day indices relative to learning day 0
        win_sec_amp: Time window for average response calculation (seconds)
        win_sec_psth: Time window for PSTH (seconds)
        baseline_win: Baseline window for subtraction (seconds)
        sampling_rate: Imaging sampling rate (Hz)
        file_name: Name of xarray file containing imaging data

    Returns:
        avg_resp: DataFrame with average response per trial
        psth: DataFrame with PSTH per day
    """

    # Get mouse list from database if not provided
    if mice is None:
        _, _, mice, db = io.select_sessions_from_db(
            io.db_path,
            io.nwb_dir,
            two_p_imaging='yes',
            experimenters=['AR', 'GF', 'MI']
        )

    # Convert baseline window to samples
    baseline_win_samples = (
        int(baseline_win[0] * sampling_rate),
        int(baseline_win[1] * sampling_rate)
    )

    # Load data for each mouse
    avg_resp_list = []
    psth_list = []

    for mouse_id in mice:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

        # Load xarray data
        folder = os.path.join(io.processed_dir, 'mice')
        xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)

        # Subtract baseline
        xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win_samples)

        # ====================================================================
        # Average response data
        # ====================================================================
        avg = xarr.sel(trial=xarr['day'].isin(days))
        # Average over time window
        avg = avg.sel(time=slice(win_sec_amp[0], win_sec_amp[1])).mean(dim='time')

        # Convert to dataframe
        avg.name = 'average_response'
        avg_df = avg.to_dataframe().reset_index()
        avg_df['mouse_id'] = mouse_id
        avg_df['reward_group'] = reward_group
        avg_resp_list.append(avg_df)

        # ====================================================================
        # PSTH data
        # ====================================================================
        p = xarr.sel(trial=xarr['day'].isin(days))
        # Select time window
        p = p.sel(time=slice(win_sec_psth[0], win_sec_psth[1]))
        # Average across trials for each day
        p = p.groupby('day').mean(dim='trial')

        # Convert to dataframe
        p.name = 'psth'
        p_df = p.to_dataframe().reset_index()
        p_df['mouse_id'] = mouse_id
        p_df['reward_group'] = reward_group
        psth_list.append(p_df)

    # Concatenate all mice
    avg_resp = pd.concat(avg_resp_list)
    psth = pd.concat(psth_list)

    # Convert to percent dF/F0
    avg_resp['average_response'] = avg_resp['average_response'] * 100
    psth['psth'] = psth['psth'] * 100

    # Add learning period column
    avg_resp['learning_period'] = avg_resp['day'].apply(
        lambda x: 'pre' if x in [-2, -1] else 'post'
    )
    psth['learning_period'] = psth['day'].apply(
        lambda x: 'pre' if x in [-2, -1] else 'post'
    )

    return avg_resp, psth


# ============================================================================
# Panel c: All cells pre/post comparison
# ============================================================================

def panel_c_pre_post_all_cells(
    avg_resp=None,
    psth=None,
    days_selected=[-2, -1, 1, 2],
    min_cells=3,
    variance='mice',
    save_path='/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth',
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 3 Panel c: Pre vs post learning for all cells.

    Shows PSTH (left) and response amplitude (right) before and after learning
    for R+ (top) and R- (bottom) reward groups, averaging across all cell types.

    Args:
        avg_resp: Pre-loaded average response dataframe (if None, will load)
        psth: Pre-loaded PSTH dataframe (if None, will load)
        days_selected: Days to include in pre/post comparison
        min_cells: Minimum number of cells per mouse to include
        variance: 'mice' to average by mouse first, 'cells' to average cells directly
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load data if not provided
    if avg_resp is None or psth is None:
        avg_resp, psth = load_and_process_response_data()

    # Select days of interest
    data_plot_avg = avg_resp[avg_resp['day'].isin(days_selected)]
    data_plot_psth = psth[psth['day'].isin(days_selected)]

    # Process based on variance method
    if variance == "mice":
        # Filter by minimum cell count
        data_plot_avg = utils_imaging.filter_data_by_cell_count(data_plot_avg, min_cells)
        data_plot_psth = utils_imaging.filter_data_by_cell_count(data_plot_psth, min_cells)

        # Average across all cells (ignore cell_type)
        data_plot_avg_all = data_plot_avg.groupby(
            ['mouse_id', 'learning_period', 'reward_group']
        )['average_response'].agg('mean').reset_index()

        data_plot_psth_all = data_plot_psth.groupby(
            ['mouse_id', 'learning_period', 'reward_group', 'time']
        )['psth'].agg('mean').reset_index()
    else:
        # Average by cell first
        data_plot_avg_all = data_plot_avg.groupby(
            ['mouse_id', 'learning_period', 'reward_group', 'cell_type', 'roi']
        )['average_response'].agg('mean').reset_index()

        data_plot_psth_all = data_plot_psth.groupby(
            ['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type', 'roi']
        )['psth'].agg('mean').reset_index()

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    # Create figure: 2x2 grid (R+ top, R- bottom; PSTH left, amplitude right)
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=False, sharey=True)

    # ========================================================================
    # Top-left: PSTH for R+ mice
    # ========================================================================
    rewarded_data = data_plot_psth_all[data_plot_psth_all['reward_group'] == 'R+']
    sns.lineplot(
        data=rewarded_data,
        x='time',
        y='psth',
        hue='learning_period',
        hue_order=['pre', 'post'],
        palette=sns.color_palette(['#a3a3a3', '#1b9e77']),
        ax=axes[0, 0],
        legend=False
    )
    axes[0, 0].set_title('PSTH (Rewarded Mice)')
    axes[0, 0].set_ylabel('DF/F0 (%)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].axvline(0, color='orange', linestyle='-')

    # ========================================================================
    # Bottom-left: PSTH for R- mice
    # ========================================================================
    nonrewarded_data = data_plot_psth_all[data_plot_psth_all['reward_group'] == 'R-']
    sns.lineplot(
        data=nonrewarded_data,
        x='time',
        y='psth',
        hue='learning_period',
        hue_order=['pre', 'post'],
        palette=sns.color_palette(['#a3a3a3', '#c959affe']),
        ax=axes[1, 0],
        legend=False
    )
    axes[1, 0].set_title('PSTH (Non-Rewarded Mice)')
    axes[1, 0].set_ylabel('DF/F0 (%)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].axvline(0, color='orange', linestyle='-')

    # ========================================================================
    # Top-right: Response amplitude for R+ mice
    # ========================================================================
    rewarded_avg = data_plot_avg_all[data_plot_avg_all['reward_group'] == 'R+']
    sns.barplot(
        data=rewarded_avg,
        x='learning_period',
        y='average_response',
        order=['pre', 'post'],
        color='#1b9e77',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Response Amplitude (Rewarded Mice)')
    axes[0, 1].set_ylabel('Average Response (dF/F0 %)')
    axes[0, 1].set_xlabel('Learning Period')

    # ========================================================================
    # Bottom-right: Response amplitude for R- mice
    # ========================================================================
    nonrewarded_avg = data_plot_avg_all[data_plot_avg_all['reward_group'] == 'R-']
    sns.barplot(
        data=nonrewarded_avg,
        x='learning_period',
        y='average_response',
        order=['pre', 'post'],
        color='#c959affe',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Response Amplitude (Non-Rewarded Mice)')
    axes[1, 1].set_ylabel('Average Response (dF/F0 %)')
    axes[1, 1].set_xlabel('Learning Period')

    sns.despine()
    plt.tight_layout()

    # Save figure and data
    save_path = io.adjust_path_to_host(save_path)
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(
        save_path,
        f'figure_3c_{variance}.{save_format}'
    )
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Save data
    data_file = os.path.join(save_path, f'figure_3c_{variance}_data.csv')
    data_plot_avg_all.to_csv(data_file, index=False)

    print(f"Figure 3c saved to: {output_file}")
    print(f"Figure 3c data saved to: {data_file}")

    # Statistical testing
    perform_statistics(data_plot_avg_all, 'all', variance, save_path)

    return data_plot_avg_all


# ============================================================================
# Panel d: Projection-specific pre/post comparison
# ============================================================================

def panel_d_pre_post_projection_types(
    avg_resp=None,
    psth=None,
    days_selected=[-2, -1, 1, 2],
    min_cells=3,
    variance='mice',
    projection_types=['wS2', 'wM1'],
    save_path='/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth',
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 3 Panel d: Pre vs post learning for projection types.

    Shows PSTH and response amplitude before and after learning for wS2 and wM1
    projection neurons, with R+ (top) and R- (bottom) reward groups.

    Args:
        avg_resp: Pre-loaded average response dataframe (if None, will load)
        psth: Pre-loaded PSTH dataframe (if None, will load)
        days_selected: Days to include in pre/post comparison
        min_cells: Minimum number of cells per mouse to include
        variance: 'mice' to average by mouse first, 'cells' to average cells directly
        projection_types: List of cell types to plot
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load data if not provided
    if avg_resp is None or psth is None:
        avg_resp, psth = load_and_process_response_data()

    # Select days of interest
    data_plot_avg = avg_resp[avg_resp['day'].isin(days_selected)]
    data_plot_psth = psth[psth['day'].isin(days_selected)]

    # Process based on variance method
    if variance == "mice":
        # Filter by minimum cell count
        data_plot_avg = utils_imaging.filter_data_by_cell_count(data_plot_avg, min_cells)
        data_plot_psth = utils_imaging.filter_data_by_cell_count(data_plot_psth, min_cells)

        # Average for projection types
        data_plot_avg_proj = data_plot_avg.groupby(
            ['mouse_id', 'learning_period', 'reward_group', 'cell_type']
        )['average_response'].agg('mean').reset_index()

        data_plot_psth_proj = data_plot_psth.groupby(
            ['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type']
        )['psth'].agg('mean').reset_index()
    else:
        # Average by cell first
        data_plot_avg_proj = data_plot_avg.groupby(
            ['mouse_id', 'learning_period', 'reward_group', 'cell_type', 'roi']
        )['average_response'].agg('mean').reset_index()

        data_plot_psth_proj = data_plot_psth.groupby(
            ['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type', 'roi']
        )['psth'].agg('mean').reset_index()

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    # Create figure: 2x4 grid (R+ top, R- bottom; wS2 PSTH, wS2 amp, wM1 PSTH, wM1 amp)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=False, sharey=True)

    # ========================================================================
    # wS2 columns (0-1)
    # ========================================================================
    cell_type = 'wS2'

    # Top-left: wS2 PSTH for R+
    rewarded_data = data_plot_psth_proj[
        (data_plot_psth_proj['reward_group'] == 'R+') &
        (data_plot_psth_proj['cell_type'] == cell_type)
    ]
    sns.lineplot(
        data=rewarded_data,
        x='time',
        y='psth',
        hue='learning_period',
        hue_order=['pre', 'post'],
        palette=sns.color_palette(['#a3a3a3', '#1b9e77']),
        ax=axes[0, 0],
        legend=False
    )
    axes[0, 0].set_title(f'{cell_type} PSTH (R+)')
    axes[0, 0].set_ylabel('DF/F0 (%)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].axvline(0, color='orange', linestyle='-')

    # Bottom-left: wS2 PSTH for R-
    nonrewarded_data = data_plot_psth_proj[
        (data_plot_psth_proj['reward_group'] == 'R-') &
        (data_plot_psth_proj['cell_type'] == cell_type)
    ]
    sns.lineplot(
        data=nonrewarded_data,
        x='time',
        y='psth',
        hue='learning_period',
        hue_order=['pre', 'post'],
        palette=sns.color_palette(['#a3a3a3', '#c959affe']),
        ax=axes[1, 0],
        legend=False
    )
    axes[1, 0].set_title(f'{cell_type} PSTH (R-)')
    axes[1, 0].set_ylabel('DF/F0 (%)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].axvline(0, color='orange', linestyle='-')

    # Top-middle: wS2 amplitude for R+
    rewarded_avg = data_plot_avg_proj[
        (data_plot_avg_proj['reward_group'] == 'R+') &
        (data_plot_avg_proj['cell_type'] == cell_type)
    ]
    sns.barplot(
        data=rewarded_avg,
        x='learning_period',
        y='average_response',
        order=['pre', 'post'],
        color='#1b9e77',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title(f'{cell_type} Amplitude (R+)')
    axes[0, 1].set_ylabel('Avg Response (dF/F0 %)')
    axes[0, 1].set_xlabel('Learning Period')

    # Bottom-middle: wS2 amplitude for R-
    nonrewarded_avg = data_plot_avg_proj[
        (data_plot_avg_proj['reward_group'] == 'R-') &
        (data_plot_avg_proj['cell_type'] == cell_type)
    ]
    sns.barplot(
        data=nonrewarded_avg,
        x='learning_period',
        y='average_response',
        order=['pre', 'post'],
        color='#c959affe',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title(f'{cell_type} Amplitude (R-)')
    axes[1, 1].set_ylabel('Avg Response (dF/F0 %)')
    axes[1, 1].set_xlabel('Learning Period')

    # ========================================================================
    # wM1 columns (2-3)
    # ========================================================================
    cell_type = 'wM1'

    # Top-middle-right: wM1 PSTH for R+
    rewarded_data = data_plot_psth_proj[
        (data_plot_psth_proj['reward_group'] == 'R+') &
        (data_plot_psth_proj['cell_type'] == cell_type)
    ]
    sns.lineplot(
        data=rewarded_data,
        x='time',
        y='psth',
        hue='learning_period',
        hue_order=['pre', 'post'],
        palette=sns.color_palette(['#a3a3a3', '#1b9e77']),
        ax=axes[0, 2],
        legend=False
    )
    axes[0, 2].set_title(f'{cell_type} PSTH (R+)')
    axes[0, 2].set_ylabel('DF/F0 (%)')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].axvline(0, color='orange', linestyle='-')

    # Bottom-middle-right: wM1 PSTH for R-
    nonrewarded_data = data_plot_psth_proj[
        (data_plot_psth_proj['reward_group'] == 'R-') &
        (data_plot_psth_proj['cell_type'] == cell_type)
    ]
    sns.lineplot(
        data=nonrewarded_data,
        x='time',
        y='psth',
        hue='learning_period',
        hue_order=['pre', 'post'],
        palette=sns.color_palette(['#a3a3a3', '#c959affe']),
        ax=axes[1, 2],
        legend=False
    )
    axes[1, 2].set_title(f'{cell_type} PSTH (R-)')
    axes[1, 2].set_ylabel('DF/F0 (%)')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].axvline(0, color='orange', linestyle='-')

    # Top-right: wM1 amplitude for R+
    rewarded_avg = data_plot_avg_proj[
        (data_plot_avg_proj['reward_group'] == 'R+') &
        (data_plot_avg_proj['cell_type'] == cell_type)
    ]
    sns.barplot(
        data=rewarded_avg,
        x='learning_period',
        y='average_response',
        order=['pre', 'post'],
        color='#1b9e77',
        ax=axes[0, 3]
    )
    axes[0, 3].set_title(f'{cell_type} Amplitude (R+)')
    axes[0, 3].set_ylabel('Avg Response (dF/F0 %)')
    axes[0, 3].set_xlabel('Learning Period')

    # Bottom-right: wM1 amplitude for R-
    nonrewarded_avg = data_plot_avg_proj[
        (data_plot_avg_proj['reward_group'] == 'R-') &
        (data_plot_avg_proj['cell_type'] == cell_type)
    ]
    sns.barplot(
        data=nonrewarded_avg,
        x='learning_period',
        y='average_response',
        order=['pre', 'post'],
        color='#c959affe',
        ax=axes[1, 3]
    )
    axes[1, 3].set_title(f'{cell_type} Amplitude (R-)')
    axes[1, 3].set_ylabel('Avg Response (dF/F0 %)')
    axes[1, 3].set_xlabel('Learning Period')

    sns.despine()
    plt.tight_layout()

    # Save figure and data
    save_path = io.adjust_path_to_host(save_path)
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(
        save_path,
        f'figure_3d_{variance}.{save_format}'
    )
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Save data
    data_file = os.path.join(save_path, f'figure_3d_{variance}_data.csv')
    data_plot_avg_proj.to_csv(data_file, index=False)

    print(f"Figure 3d saved to: {output_file}")
    print(f"Figure 3d data saved to: {data_file}")

    # Statistical testing for projection types
    for cell_type in projection_types:
        data_ctype = data_plot_avg_proj[data_plot_avg_proj['cell_type'] == cell_type]
        perform_statistics(data_ctype, cell_type, variance, save_path)

    return data_plot_avg_proj


# ============================================================================
# Statistical Testing
# ============================================================================

def perform_statistics(data, cell_type_label, variance, save_path):
    """
    Perform Wilcoxon signed-rank test comparing pre vs post learning periods.

    Args:
        data: DataFrame with columns [reward_group, learning_period, average_response]
        cell_type_label: Label for cell type ('all', 'wS2', 'wM1')
        variance: Variance method used ('mice' or 'cells')
        save_path: Directory to save statistics
    """

    results = []
    for reward_group in ['R+', 'R-']:
        data_group = data[data['reward_group'] == reward_group]
        pre = data_group[data_group['learning_period'] == 'pre']['average_response']
        post = data_group[data_group['learning_period'] == 'post']['average_response']

        stat, p_value = wilcoxon(pre, post)
        results.append({
            'reward_group': reward_group,
            'cell_type': cell_type_label,
            'stat': stat,
            'p_value': p_value
        })

    stats_df = pd.DataFrame(results)
    print(f"\nStatistics for {cell_type_label}:")
    print(stats_df)

    # Save statistics
    stats_file = os.path.join(
        save_path,
        f'figure_3{"c" if cell_type_label == "all" else "d"}_{cell_type_label}_{variance}_stats.csv'
    )
    stats_df.to_csv(stats_file, index=False)
    print(f"Statistics saved to: {stats_file}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Load data once
    print("Loading response and PSTH data...")
    avg_resp, psth = load_and_process_response_data()

    # Generate panel c
    print("\nGenerating panel c (all cells)...")
    panel_c_pre_post_all_cells(avg_resp=avg_resp, psth=psth)

    # Generate panel d
    print("\nGenerating panel d (projection types)...")
    panel_d_pre_post_projection_types(avg_resp=avg_resp, psth=psth)

    print("\nDone!")
