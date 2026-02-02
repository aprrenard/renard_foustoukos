import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu
from multiprocessing import Pool

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *


# #############################################################################
# Pairwise correlations between cells during 2 sec quiet windows
# Pre vs Post comparison
# #############################################################################
#
# This script computes pairwise correlations between neurons and compares
# pre vs post learning periods for wS2-wS2 and wM1-wM1 cell pairs.
#
# Key features:
# - Per-trial correlation computation (avoids drift artifacts)
# - Parallelized computation across mice
# - Two-level analysis: cell-pair level and mouse-average level
# - Pre-post comparison (pre: days -2/-1 combined, post: days +1/+2 combined)
#
# Outputs (per reward group R+/R-):
# - pairwise_correlations_prepost.csv: Raw correlation data
# - pairwise_correlations_mouse_averages_prepost.csv: Mouse-level averages
# - statistical_tests_prepost_pairlevel.csv: Pair-level statistics
# - statistical_tests_prepost_mouselevel.csv: Mouse-level statistics
# - pairwise_correlations_prepost_pairlevel.svg: Pair-level plot
# - pairwise_correlations_prepost_mouselevel.svg: Mouse-level plot
#
# #############################################################################

# Parameters
# ----------
sampling_rate = 30
win_sec = (-2, 0)  # Use quiet window: 2s before stim onset
pre_days = [-2, -1]  # Pre period: pool trials from these days
post_days = [1, 2]   # Post period: pool trials from these days
N_CORES = 35  # Number of cores for parallel processing

# Analysis mode: 'compute' to compute correlations, 'analyze' to load and analyze existing data
ANALYSIS_MODE = 'analyze'  # Options: 'compute' or 'analyze'

# Data type: 'mapping' for mapping trials, 'learning' for learning trials
DATA_TYPE = 'mapping'  # Options: 'mapping' or 'learning'

# Select sessions from database
_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])

# Separate mice by reward group
mice_by_group = {}
for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    if reward_group not in mice_by_group:
        mice_by_group[reward_group] = []
    mice_by_group[reward_group].append(mouse_id)


def process_mouse(mouse_id):
    """
    Process correlations for a single mouse for pre and post periods.

    For each cell pair:
    - Pools all trials from days -2 and -1 for the pre period
    - Pools all trials from days +1 and +2 for the post period
    - Computes correlation for each trial
    - Averages across trials to get one correlation value per period

    Parameters
    ----------
    mouse_id : str
        Mouse identifier

    Returns
    -------
    list of dict
        List of correlation results for this mouse (one per pair per period)
    """
    print(f"\nProcessing mouse: {mouse_id}")

    mouse_results = []

    # Get reward group for this mouse
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    # Load xarray data for this mouse
    file_name = f'tensor_xarray_{DATA_TYPE}_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
    xarr.name = 'dff'

    # Filter to only include pre and post days
    all_days = pre_days + post_days
    xarr = xarr.sel(trial=xarr['day'].isin(all_days))
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))

    # Process pre and post periods separately
    for period, period_days in [('pre', pre_days), ('post', post_days)]:
        print(f"  Mouse {mouse_id}: Processing {period} period (days {period_days})...")

        # Select trials for this period (pooling days together)
        xarr_period = xarr.sel(trial=xarr['day'].isin(period_days))

        # Pre-extract all cell data at once to avoid repeated xarray indexing
        # Shape: (n_cells, n_trials, n_timepoints)
        all_cells_data = xarr_period.values

        n_cells = all_cells_data.shape[0]
        n_trials = all_cells_data.shape[1]

        if n_trials == 0:
            print(f"    Mouse {mouse_id}: No trials for {period} period, skipping")
            continue

        print(f"    Mouse {mouse_id}: Found {n_cells} cells, {n_trials} trials")

        # Get cell type information for this period's data
        cell_types = xarr_period.coords['cell_type'].values
        rois = xarr_period.coords['roi'].values

        # For each pair of cells, compute correlation per trial and average
        for i, j in combinations(range(n_cells), 2):
            # Only process wS2-wS2 and wM1-wM1 pairs
            if cell_types[i] != cell_types[j]:
                continue
            if cell_types[i] not in ['wS2', 'wM1']:
                continue

            # Get activity traces for both cells
            # Shape: (n_trials, n_timepoints)
            cell_i_data = all_cells_data[i, :, :]
            cell_j_data = all_cells_data[j, :, :]

            # Compute correlation for each trial separately
            trial_correlations = []
            for trial_idx in range(n_trials):
                cell_i_trial = cell_i_data[trial_idx, :]
                cell_j_trial = cell_j_data[trial_idx, :]

                # Remove any NaN values
                valid_idx = ~(np.isnan(cell_i_trial) | np.isnan(cell_j_trial))
                cell_i_clean = cell_i_trial[valid_idx]
                cell_j_clean = cell_j_trial[valid_idx]

                # Compute Pearson correlation for this trial
                if len(cell_i_clean) > 1:
                    if np.std(cell_i_clean) > 0 and np.std(cell_j_clean) > 0:
                        corr, _ = pearsonr(cell_i_clean, cell_j_clean)
                        trial_correlations.append(corr)

            # Average correlation across trials for this period
            if len(trial_correlations) > 0:
                final_corr = np.mean(trial_correlations)

                # Determine pair type
                pair_type = f'{cell_types[i]}-{cell_types[i]}'  # Will be wS2-wS2 or wM1-wM1

                # Store result
                mouse_results.append({
                    'mouse_id': mouse_id,
                    'reward_group': reward_group,
                    'period': period,
                    'pair_type': pair_type,
                    'cell_i': i,
                    'cell_j': j,
                    'roi_i': rois[i],
                    'roi_j': rois[j],
                    'correlation': final_corr,
                    'n_trials': len(trial_correlations)
                })

    print(f"  Mouse {mouse_id}: Completed! Computed {len(mouse_results)} correlations")
    return mouse_results


def add_significance_stars(ax, x1, x2, y, p_value):
    """Add significance stars and p-value between two bars"""
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        return  # Don't add anything if not significant

    # Draw line
    h = y
    line_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [h, h + line_height, h + line_height, h], lw=1.5, c='black')

    # Add stars and p-value
    if p_value < 0.001:
        pval_text = f'{stars}\np<0.001'
    else:
        pval_text = f'{stars}\np={p_value:.3f}'
    ax.text((x1 + x2) / 2, h + line_height, pval_text, ha='center', va='bottom',
            fontsize=10, fontweight='bold')


# Main execution: Process mice in parallel
if __name__ == '__main__':
    # Pair types of interest
    pair_types_of_interest = ['wS2-wS2', 'wM1-wM1']

    # Loop through each reward group
    for reward_group in ['R-', 'R+']:
        if reward_group not in mice_by_group:
            print(f"\nNo mice found for reward group {reward_group}, skipping...")
            continue

        group_mice = mice_by_group[reward_group]
        print(f"\n{'='*80}")
        print(f"PROCESSING REWARD GROUP: {reward_group}")
        print(f"{'='*80}")

        # Set up output directory
        output_dir = f'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/pairwise_correlations/{reward_group}/{DATA_TYPE}'
        output_dir = io.adjust_path_to_host(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Check if we should compute or load existing data
        corr_csv_path = os.path.join(output_dir, 'pairwise_correlations_prepost.csv')

        if ANALYSIS_MODE == 'compute' or not os.path.exists(corr_csv_path):
            print(f"\nMode: COMPUTE - Computing pairwise correlations for {len(group_mice)} mice using {N_CORES} cores...")
            print(f"Mice in {reward_group}: {group_mice}")

            # Use multiprocessing to parallelize across mice
            with Pool(processes=N_CORES) as pool:
                results_list = pool.map(process_mouse, group_mice)

            # Flatten the list of lists into a single list
            correlation_results = [item for sublist in results_list for item in sublist]

            print(f"\nCorrelation computation complete for {reward_group}!")

            # Convert results to DataFrame
            corr_df = pd.DataFrame(correlation_results)

            if len(corr_df) == 0:
                print("WARNING: No correlation results computed!")
                continue

            print(f"Successfully computed correlations for {corr_df['mouse_id'].nunique()} mice")
            print(f"Breakdown by pair type:")
            print(corr_df['pair_type'].value_counts())

            # Save correlation data
            corr_df.to_csv(corr_csv_path, index=False)
            print(f"\nSaved correlation data to: {corr_csv_path}")

        else:
            print(f"\nMode: ANALYZE - Loading existing correlation data from {corr_csv_path}")
            corr_df = pd.read_csv(corr_csv_path)
            print(f"Loaded {len(corr_df)} correlation pairs")
            print(f"Breakdown by pair type:")
            print(corr_df['pair_type'].value_counts())

        # #############################################################################
        # Statistical tests - Pair level
        # #############################################################################

        print("\n" + "="*80)
        print("STATISTICAL TESTS (PAIR-LEVEL)")
        print("="*80)

        # Compare pre vs post for each pair type (Mann-Whitney U test at pair level)
        print("\nComparing pre vs post for each pair type (Mann-Whitney U test, n=cell pairs)")
        print("-" * 80)

        prepost_stat_results_pair = []

        for pair_type in pair_types_of_interest:
            print(f"\n{pair_type}:")
            pair_data = corr_df[corr_df['pair_type'] == pair_type]

            data_pre = pair_data[pair_data['period'] == 'pre']['correlation'].values
            data_post = pair_data[pair_data['period'] == 'post']['correlation'].values

            if len(data_pre) > 0 and len(data_post) > 0:
                stat, p_value = mannwhitneyu(data_pre, data_post, alternative='two-sided')

                mean_pre = np.mean(data_pre)
                mean_post = np.mean(data_post)
                sem_pre = np.std(data_pre) / np.sqrt(len(data_pre))
                sem_post = np.std(data_post) / np.sqrt(len(data_post))

                print(f"  Pre: μ={mean_pre:.3f}, SEM={sem_pre:.3f}, n={len(data_pre)}")
                print(f"  Post: μ={mean_post:.3f}, SEM={sem_post:.3f}, n={len(data_post)}")
                print(f"  Mann-Whitney U: U={stat:.2f}, p={p_value:.4f} "
                      f"{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                prepost_stat_results_pair.append({
                    'pair_type': pair_type,
                    'mean_pre': mean_pre,
                    'sem_pre': sem_pre,
                    'mean_post': mean_post,
                    'sem_post': sem_post,
                    'n_pre': len(data_pre),
                    'n_post': len(data_post),
                    'U_statistic': stat,
                    'p_value': p_value
                })
            else:
                print(f"  Insufficient data (n_pre={len(data_pre)}, n_post={len(data_post)})")

        # Save pair-level pre-post statistical results
        prepost_stat_df_pair = pd.DataFrame(prepost_stat_results_pair)
        prepost_stat_df_pair.to_csv(os.path.join(output_dir, 'statistical_tests_prepost_pairlevel.csv'), index=False)
        print(f"\nSaved pair-level statistical test results to: "
              f"{os.path.join(output_dir, 'statistical_tests_prepost_pairlevel.csv')}")

        # #############################################################################
        # Pair-level plot
        # #############################################################################

        print("\n" + "="*80)
        print("GENERATING PAIR-LEVEL PLOT")
        print("="*80)

        # Plot: Pre-Post comparison (pair-level) - 2 panels, one per pair type
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]

            # Get data for this pair type
            pair_data = corr_df[corr_df['pair_type'] == pair_type]

            # Plot bar plot
            sns.barplot(
                data=pair_data,
                x='period',
                y='correlation',
                order=['pre', 'post'],
                ax=ax,
                errorbar='se',  # Standard error
                capsize=0.1
            )

            ax.set_title(f'{pair_type}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Period', fontsize=12)
            ax.set_ylim(0, 0.02)
            if idx == 0:
                ax.set_ylabel('Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations
        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]
            pair_data = corr_df[corr_df['pair_type'] == pair_type]

            # Get pre-post comparison stats
            pair_prepost_stats = prepost_stat_df_pair[prepost_stat_df_pair['pair_type'] == pair_type]

            if not pair_prepost_stats.empty:
                p_val = pair_prepost_stats.iloc[0]['p_value']

                if p_val < 0.05:
                    # Get max y values for pre and post
                    data_pre = pair_data[pair_data['period'] == 'pre']['correlation']
                    data_post = pair_data[pair_data['period'] == 'post']['correlation']

                    # Calculate the top of error bars
                    y_pre_top = data_pre.mean() + data_pre.sem()
                    y_post_top = data_post.mean() + data_post.sem()
                    y_max = max(y_pre_top, y_post_top)

                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_offset = y_range * 0.06
                    add_significance_stars(ax, 0, 1, y_max + y_offset, p_val)
                else:
                    # Add p-value text in corner if not significant
                    if p_val < 0.001:
                        pval_text = 'p<0.001'
                    else:
                        pval_text = f'p={p_val:.3f}'
                    ax.text(0.95, 0.95, pval_text, transform=ax.transAxes,
                            fontsize=10, va='top', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(f'Pre vs Post Pairwise Correlation - Pair Level ({reward_group}, {DATA_TYPE})',
                     fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_prepost_pairlevel.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        plt.close()

        # #############################################################################
        # Mouse-level analysis
        # #############################################################################

        print("\n" + "="*80)
        print("MOUSE-LEVEL ANALYSIS")
        print("="*80)

        # Compute average correlation per mouse for each period and pair type
        mouse_avg_corr = corr_df.groupby(['mouse_id', 'reward_group', 'period', 'pair_type'])['correlation'].mean().reset_index()
        mouse_avg_corr.rename(columns={'correlation': 'mean_correlation'}, inplace=True)

        print(f"\nComputed average correlations for {len(mouse_avg_corr['mouse_id'].unique())} mice")

        # Save mouse-level averages
        mouse_avg_corr.to_csv(os.path.join(output_dir, 'pairwise_correlations_mouse_averages_prepost.csv'), index=False)
        print(f"Saved mouse-level averages to: "
              f"{os.path.join(output_dir, 'pairwise_correlations_mouse_averages_prepost.csv')}")

        # #############################################################################
        # Statistical tests - Mouse level
        # #############################################################################

        print("\n" + "="*80)
        print("STATISTICAL TESTS (MOUSE-LEVEL)")
        print("="*80)

        # Compare pre vs post for each pair type (Wilcoxon paired test at mouse level)
        print("\nComparing pre vs post for each pair type (Wilcoxon paired test, n=mice)")
        print("-" * 80)

        prepost_stat_results_mouse = []

        for pair_type in pair_types_of_interest:
            print(f"\n{pair_type}:")

            # Get data for this pair type - pivot to have pre and post as columns for pairing
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Pivot to get pre and post values for each mouse
            pivot_data = pair_mouse_data.pivot(index='mouse_id', columns='period', values='mean_correlation')

            # Only keep mice that have both pre and post data
            paired_data = pivot_data.dropna()

            if len(paired_data) > 0:
                data_pre = paired_data['pre'].values
                data_post = paired_data['post'].values

                # Wilcoxon signed-rank test (paired)
                stat, p_value = wilcoxon(data_pre, data_post, alternative='two-sided')

                mean_pre = np.mean(data_pre)
                mean_post = np.mean(data_post)
                sem_pre = np.std(data_pre) / np.sqrt(len(data_pre))
                sem_post = np.std(data_post) / np.sqrt(len(data_post))

                print(f"  Pre: μ={mean_pre:.3f}, SEM={sem_pre:.3f}")
                print(f"  Post: μ={mean_post:.3f}, SEM={sem_post:.3f}")
                print(f"  n={len(data_pre)} mice")
                print(f"  Wilcoxon signed-rank: W={stat:.2f}, p={p_value:.4f} "
                      f"{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                prepost_stat_results_mouse.append({
                    'pair_type': pair_type,
                    'mean_pre': mean_pre,
                    'sem_pre': sem_pre,
                    'mean_post': mean_post,
                    'sem_post': sem_post,
                    'n_mice': len(data_pre),
                    'W_statistic': stat,
                    'p_value': p_value
                })
            else:
                print(f"  Insufficient paired data")

        # Save mouse-level pre-post statistical results
        prepost_stat_df_mouse = pd.DataFrame(prepost_stat_results_mouse)
        prepost_stat_df_mouse.to_csv(os.path.join(output_dir, 'statistical_tests_prepost_mouselevel.csv'), index=False)
        print(f"\nSaved mouse-level statistical test results to: "
              f"{os.path.join(output_dir, 'statistical_tests_prepost_mouselevel.csv')}")

        # #############################################################################
        # Mouse-level plot
        # #############################################################################

        print("\n" + "="*80)
        print("GENERATING MOUSE-LEVEL PLOT")
        print("="*80)

        # Plot: Pre-Post comparison (mouse-level) - 2 panels, one per pair type
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]

            # Get data for this pair type
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Plot bar plot
            sns.barplot(
                data=pair_mouse_data,
                x='period',
                y='mean_correlation',
                order=['pre', 'post'],
                ax=ax,
                errorbar='se',  # Standard error
                capsize=0.1
            )

            ax.set_title(f'{pair_type}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Period', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Mean Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations
        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Get pre-post comparison stats
            pair_prepost_mouse_stats = prepost_stat_df_mouse[prepost_stat_df_mouse['pair_type'] == pair_type]

            if not pair_prepost_mouse_stats.empty:
                p_val = pair_prepost_mouse_stats.iloc[0]['p_value']

                if p_val < 0.05:
                    # Get max y values for pre and post
                    data_pre = pair_mouse_data[pair_mouse_data['period'] == 'pre']['mean_correlation']
                    data_post = pair_mouse_data[pair_mouse_data['period'] == 'post']['mean_correlation']

                    # Calculate the top of error bars
                    y_pre_top = data_pre.mean() + data_pre.sem()
                    y_post_top = data_post.mean() + data_post.sem()
                    y_max = max(y_pre_top, y_post_top)

                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_offset = y_range * 0.06
                    add_significance_stars(ax, 0, 1, y_max + y_offset, p_val)
                else:
                    # Add p-value text in corner if not significant
                    if p_val < 0.001:
                        pval_text = 'p<0.001'
                    else:
                        pval_text = f'p={p_val:.3f}'
                    ax.text(0.95, 0.95, pval_text, transform=ax.transAxes,
                            fontsize=10, va='top', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(f'Pre vs Post Pairwise Correlation - Mouse Level ({reward_group}, {DATA_TYPE})',
                     fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_prepost_mouselevel.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        plt.close()

        # #############################################################################
        # Summary
        # #############################################################################

        print(f"\n{'='*80}")
        print(f"Analysis complete for {reward_group}!")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}")
        print("Files generated:")
        print("  Data:")
        print("    - pairwise_correlations_prepost.csv")
        print("    - pairwise_correlations_mouse_averages_prepost.csv")
        print("  Statistics:")
        print("    - statistical_tests_prepost_pairlevel.csv")
        print("    - statistical_tests_prepost_mouselevel.csv")
        print("  Plots:")
        print("    - pairwise_correlations_prepost_pairlevel.svg")
        print("    - pairwise_correlations_prepost_mouselevel.svg")

    print(f"\n{'='*80}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*80}")
