"""
Figure 3b: Grand average PSTHs across learning days

This script generates Panel b for Figure 3, showing peri-stimulus time histograms
(PSTHs) averaged across mice for different learning days. Includes plots for:
- All cells combined
- Projection-specific neurons (wS2, wM1)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_3', 'output')


# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_and_process_psth_data(
    mice=None,
    days=[-2, -1, 0, 1, 2],
    win_sec=(-0.5, 1.5),
    baseline_win=(0, 1),
    sampling_rate=30,
    file_name='tensor_xarray_mapping_data.nc'
):
    """
    Load and process PSTH data from xarray files for all mice.

    Args:
        mice: List of mouse IDs (if None, will query from database)
        days: List of day indices relative to learning day 0
        win_sec: Time window for PSTH (seconds)
        baseline_win: Baseline window for subtraction (seconds)
        sampling_rate: Imaging sampling rate (Hz)
        file_name: Name of xarray file containing imaging data

    Returns:
        psth: DataFrame with columns [mouse_id, day, time, roi, cell_type, psth, reward_group]
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
    psth_list = []

    for mouse_id in mice:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

        # Load xarray data
        folder = os.path.join(io.processed_dir, 'mice')
        xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)

        # Subtract baseline
        xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win_samples)

        # Filter for days of interest
        xarr = xarr.sel(trial=xarr['day'].isin(days))

        # Select time window
        xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))

        # Average across trials for each day
        xarr = xarr.groupby('day').mean(dim='trial')

        # Convert to dataframe
        xarr.name = 'psth'
        xarr_df = xarr.to_dataframe().reset_index()
        xarr_df['mouse_id'] = mouse_id
        xarr_df['reward_group'] = reward_group

        psth_list.append(xarr_df)

    psth = pd.concat(psth_list)

    return psth


# ============================================================================
# Panel b1: All cells PSTH
# ============================================================================

def panel_b1_psth_all_cells(
    psth=None,
    days=[-2, -1, 0, 1, 2],
    min_cells=3,
    variance='mice',
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 3 Panel b1: PSTH for all cells across learning days.

    Shows average PSTH across all cell types for R+ and R- reward groups
    across 5 days of training.

    Args:
        psth: Pre-loaded PSTH dataframe (if None, will load)
        days: List of day indices to plot
        min_cells: Minimum number of cells per mouse to include
        variance: 'mice' to average by mouse first, 'cells' to average cells directly
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load data if not provided
    if psth is None:
        psth = load_and_process_psth_data(days=days)

    # Filter mice with minimum cell count
    if variance == "mice":
        psth_filtered = utils_imaging.filter_data_by_cell_count(psth, min_cells)
        # Average across all cells (ignore cell_type)
        data_allcells = psth_filtered.groupby(
            ['mouse_id', 'day', 'reward_group', 'time']
        )['psth'].mean().reset_index()
    else:
        # Average by cell first, then by mouse
        data_allcells = psth.groupby(
            ['mouse_id', 'day', 'reward_group', 'time', 'roi']
        )['psth'].mean().reset_index()
        data_allcells = data_allcells.groupby(
            ['mouse_id', 'day', 'reward_group', 'time']
        )['psth'].mean().reset_index()

    # Convert to percent dF/F0
    data_allcells['psth'] = data_allcells['psth'] * 100

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    # Create figure
    fig, axes = plt.subplots(1, len(days), figsize=(18, 5), sharey=True)

    # Plot each day
    for j, day in enumerate(days):
        d = data_allcells.loc[data_allcells['day'] == day]
        sns.lineplot(
            data=d,
            x='time',
            y='psth',
            errorbar='ci',
            hue='reward_group',
            hue_order=['R-', 'R+'],
            palette=reward_palette,
            estimator='mean',
            ax=axes[j],
            legend=False
        )
        axes[j].axvline(0, color='#FF9600', linestyle='-')
        axes[j].set_title(f'Day {day} - All Cells')
        axes[j].set_ylabel('DF/F0 (%)')
        axes[j].set_xlabel('Time (s)')

    plt.ylim(-1, 12)
    plt.tight_layout()
    sns.despine()

    # Save figure and data
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(
        save_path,
        f'figure_3b_all_cells_{variance}.{save_format}'
    )
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Save data
    data_file = os.path.join(save_path, f'figure_3b_all_cells_{variance}_data.csv')
    data_allcells.to_csv(data_file, index=False)

    print(f"Figure 3b (all cells) saved to: {output_file}")
    print(f"Figure 3b (all cells) data saved to: {data_file}")


# ============================================================================
# Panel b2: Projection-specific PSTH
# ============================================================================

def panel_b2_psth_projection_types(
    psth=None,
    days=[-2, -1, 0, 1, 2],
    min_cells=3,
    variance='mice',
    projection_types=['wS2', 'wM1'],
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 3 Panel b2: PSTH for projection-specific neurons.

    Shows average PSTH for wS2 and wM1 projection neurons separately
    for R+ and R- reward groups across 5 days of training.

    Args:
        psth: Pre-loaded PSTH dataframe (if None, will load)
        days: List of day indices to plot
        min_cells: Minimum number of cells per mouse to include
        variance: 'mice' to average by mouse first, 'cells' to average cells directly
        projection_types: List of cell types to plot
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load data if not provided
    if psth is None:
        psth = load_and_process_psth_data(days=days)

    # Filter for projection types and minimum cell count
    if variance == "mice":
        psth_filtered = utils_imaging.filter_data_by_cell_count(psth, min_cells)
        data_ctype = psth_filtered[psth_filtered['cell_type'].isin(projection_types)]
        data_ctype = data_ctype.groupby(
            ['mouse_id', 'day', 'reward_group', 'time', 'cell_type']
        )['psth'].mean().reset_index()
    else:
        data_ctype = psth[psth['cell_type'].isin(projection_types)]
        data_ctype = data_ctype.groupby(
            ['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi']
        )['psth'].mean().reset_index()
        data_ctype = data_ctype.groupby(
            ['mouse_id', 'day', 'reward_group', 'time', 'cell_type']
        )['psth'].mean().reset_index()

    # Convert to percent dF/F0
    data_ctype['psth'] = data_ctype['psth'] * 100

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    # Create figure
    fig, axes = plt.subplots(
        len(projection_types), len(days),
        figsize=(18, 10),
        sharey=True
    )

    # Plot each cell type and day
    for i, cell_type in enumerate(projection_types):
        for j, day in enumerate(days):
            d = data_ctype[
                (data_ctype['cell_type'] == cell_type) &
                (data_ctype['day'] == day)
            ]
            sns.lineplot(
                data=d,
                x='time',
                y='psth',
                errorbar='ci',
                hue='reward_group',
                hue_order=['R-', 'R+'],
                palette=reward_palette,
                estimator='mean',
                ax=axes[i, j],
                legend=False
            )
            axes[i, j].axvline(0, color='#FF9600', linestyle='-')
            axes[i, j].set_title(f'{cell_type} - Day {day}')
            axes[i, j].set_ylabel('DF/F0 (%)')
            axes[i, j].set_xlabel('Time (s)')

    plt.ylim(-1, 16)
    plt.tight_layout()
    sns.despine()

    # Save figure and data
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(
        save_path,
        f'figure_3b_projection_types_{variance}.{save_format}'
    )
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Save data
    data_file = os.path.join(
        save_path,
        f'figure_3b_projection_types_{variance}_data.csv'
    )
    data_ctype.to_csv(data_file, index=False)

    print(f"Figure 3b (projection types) saved to: {output_file}")
    print(f"Figure 3b (projection types) data saved to: {data_file}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Load data once
    print("Loading PSTH data...")
    psth = load_and_process_psth_data()

    # Generate both panels
    print("\nGenerating panel b1 (all cells)...")
    panel_b1_psth_all_cells(psth=psth)

    print("\nGenerating panel b2 (projection types)...")
    panel_b2_psth_projection_types(psth=psth)

    print("\nDone!")
