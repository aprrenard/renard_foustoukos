"""
Reactivation Surrogate Analysis for Data-Driven Threshold Determination


This script computes statistically principled correlation thresholds for reactivation
event detection via circular time-shift surrogate analysis. It generates per-mouse
thresholds based on baseline days (-2 and -1), ensuring consistent event detection
across all days.

Approach:
1. For each mouse, use baseline days (-2 and -1) to establish threshold
2. Create whisker response template from each baseline day's mapping trials
3. Load no-stim trial data from baseline days and pool them
4. Generate N surrogate datasets by circular time-shifting each cell independently
5. Compute template correlation for each surrogate (using pooled baseline template)
6. Extract both percentile (pointwise threshold) and maximum (FWER) from each
7. Average across surrogates to get final thresholds with confidence intervals
8. This single threshold is then applied to all 5 days (−2, −1, 0, +1, +2)

Output:
- CSV file with one threshold per mouse (no day column)
- Single-page PDFs showing baseline surrogate distributions for each mouse
- Summary plots comparing thresholds across mice and reward groups

Note: Each day still uses its own template for correlation computation, but event
detection uses the common baseline-derived threshold for consistency.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import percentileofscore
from joblib import Parallel, delayed
import warnings

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.core_analysis.reactivations.reactivation import (
    create_whisker_template,
    compute_template_correlation
)

# =============================================================================
# PARAMETERS
# =============================================================================

sampling_rate = 30  # Hz
days = [-2, -1, 0, 1, 2]
days_str = ['-2', '-1', '0', '+1', '+2']
n_map_trials = 40  # Number of mapping trials for template

# Template parameters
threshold_dff = 0.05  # 5% dF/F threshold for template cells (parametrized)

# Surrogate parameters
n_surrogates = 1000  # Number of surrogate iterations
min_shift_frames = 0  # Minimum shift: 2 seconds at 30Hz
percentile_threshold = 99.9  # Percentile for pointwise threshold (e.g., 90, 95, 99)
np.random.seed(42)  # For reproducibility

# Parallel processing
n_jobs = 35

# Visualization
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

# Load database
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes'
)

# Separate mice by reward group
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

print(f"Found {len(r_plus_mice)} R+ mice: {r_plus_mice}")
print(f"Found {len(r_minus_mice)} R- mice: {r_minus_mice}")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def create_surrogate_by_circular_shift(data, min_shift_frames=30):
    """
    Create one surrogate by independently shifting each cell in time.

    Uses circular (roll) shifts to preserve autocorrelation structure while
    breaking cross-neuron correlations.

    Parameters
    ----------
    data : np.ndarray
        (n_cells, n_frames) neural activity
    min_shift_frames : int
        Minimum shift amount (default: 30 = 1 sec at 30Hz)

    Returns
    -------
    surrogate : np.ndarray
        (n_cells, n_frames) time-shifted data
    """
    n_cells, n_frames = data.shape
    surrogate = np.zeros_like(data)

    for icell in range(n_cells):
        # Random shift in range [min_shift_frames, n_frames)
        shift_frames = np.random.randint(min_shift_frames, n_frames)
        surrogate[icell, :] = np.roll(data[icell, :], shift_frames)

    return surrogate


def compute_surrogate_thresholds(data, template, n_surrogates=1000, min_shift=0,
                                 percentile=99, verbose=True):
    """
    Compute surrogate-based thresholds via circular time shifts.

    Generates n_surrogates time-shifted versions of the data, computes template
    correlation for each, and extracts statistics to define thresholds.

    Parameters
    ----------
    data : np.ndarray
        (n_cells, n_frames) neural activity
    template : np.ndarray
        (n_cells,) response template
    n_surrogates : int
        Number of surrogate iterations (default: 1000)
    min_shift : int
        Minimum shift in frames (default: 30)
    percentile : float
        Percentile for pointwise threshold (default: 95)
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        {
            'threshold_percentile_mean': float,
            'threshold_percentile_ci': (lower, upper),
            'threshold_max_mean': float,
            'threshold_max_ci': (lower, upper),
            'surrogate_percentiles': np.ndarray of percentiles,
            'surrogate_maxs': np.ndarray of maxima,
            'observed_percentile': float,
            'observed_max': float,
            'p_value_percentile': float (percentile of observed in null),
            'p_value_max': float,
            'percentile_value': float (the percentile used)
        }
    """
    n_cells, n_frames = data.shape

    if verbose:
        print(f"    Computing {n_surrogates} surrogates (percentile={percentile})...")
        print(f"    Data shape: {n_cells} cells × {n_frames} frames")

    # First compute observed statistics
    observed_corr = compute_template_correlation(data, template)
    observed_percentile = np.percentile(observed_corr, percentile)
    observed_max = np.max(observed_corr)

    if verbose:
        print(f"    Observed: {percentile}th percentile = {observed_percentile:.4f}, max = {observed_max:.4f}")

    # Generate surrogates and compute statistics
    surrogate_percentiles = np.zeros(n_surrogates)
    surrogate_maxs = np.zeros(n_surrogates)

    for i in range(n_surrogates):
        if verbose and (i+1) % 100 == 0:
            print(f"      Surrogate {i+1}/{n_surrogates}")

        # Generate surrogate
        surrogate_data = create_surrogate_by_circular_shift(data, min_shift)

        # Compute correlation with template
        surrogate_corr = compute_template_correlation(surrogate_data, template)

        # Extract statistics
        surrogate_percentiles[i] = np.percentile(surrogate_corr, percentile)
        surrogate_maxs[i] = np.max(surrogate_corr)

    # Compute threshold means and confidence intervals
    threshold_percentile_mean = np.mean(surrogate_percentiles)
    threshold_percentile_ci = (np.percentile(surrogate_percentiles, 2.5),
                                np.percentile(surrogate_percentiles, 97.5))

    threshold_max_mean = np.mean(surrogate_maxs)
    threshold_max_ci = (np.percentile(surrogate_maxs, 2.5), np.percentile(surrogate_maxs, 97.5))

    # Compute p-values: where does observed fall in surrogate distribution?
    p_value_percentile = percentileofscore(surrogate_percentiles, observed_percentile) / 100.0
    p_value_max = percentileofscore(surrogate_maxs, observed_max) / 100.0

    if verbose:
        print(f"    Threshold ({percentile}th): {threshold_percentile_mean:.4f} [{threshold_percentile_ci[0]:.4f}, {threshold_percentile_ci[1]:.4f}]")
        print(f"    Threshold (max):  {threshold_max_mean:.4f} [{threshold_max_ci[0]:.4f}, {threshold_max_ci[1]:.4f}]")
        print(f"    P-values: {percentile}th={p_value_percentile:.3f}, max={p_value_max:.3f}")

    return {
        'threshold_percentile_mean': threshold_percentile_mean,
        'threshold_percentile_ci': threshold_percentile_ci,
        'threshold_max_mean': threshold_max_mean,
        'threshold_max_ci': threshold_max_ci,
        'surrogate_percentiles': surrogate_percentiles,
        'surrogate_maxs': surrogate_maxs,
        'observed_percentile': observed_percentile,
        'observed_max': observed_max,
        'p_value_percentile': p_value_percentile,
        'p_value_max': p_value_max,
        'percentile_value': percentile
    }


def analyze_mouse_surrogates(mouse, threshold_dff=0.05,
                             n_surrogates=1000, percentile=95, verbose=True):
    """
    Compute surrogate thresholds for one mouse using baseline days (-2 and -1).

    Data from baseline days is pooled to create a null distribution for threshold
    determination. This threshold will be applied to all days to ensure consistent
    event detection.

    Parameters
    ----------
    mouse : str
        Mouse ID
    threshold_dff : float
        Responsiveness threshold for template cells (default: 0.05 = 5%)
    n_surrogates : int
        Number of surrogate iterations (default: 1000)
    percentile : float
        Percentile for pointwise threshold (default: 95)
    verbose : bool
        Print progress

    Returns
    -------
    results_df : pd.DataFrame
        Threshold results for this mouse (single row)
    all_surrogate_data : dict
        Detailed surrogate distributions for baseline (for plotting)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING MOUSE: {mouse}")
        print(f"{'='*60}")

    results_list = []
    all_surrogate_data = {}

    reference_days = [-2, -1]  # Use baseline days as reference

    try:
        if verbose:
            print(f"\n  Processing baseline days {reference_days}...")

        # Step 1: Create templates from baseline days and average them
        templates = []
        cells_masks = []
        for ref_day in reference_days:
            try:
                template, cells_mask = create_whisker_template(mouse, ref_day, threshold_dff, verbose=False)
                templates.append(template)
                cells_masks.append(cells_mask)
                if verbose:
                    print(f"    Day {ref_day}: {cells_mask.sum()} responsive cells")
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not create template for day {ref_day}: {str(e)}")
                continue

        if len(templates) == 0:
            if verbose:
                print(f"    Error: Could not create template for any baseline day")
            return None, None

        # Average templates from baseline days
        template = np.mean(templates, axis=0)
        # Union of responsive cells across baseline days
        combined_mask = np.any(cells_masks, axis=0)
        n_cells_responsive = combined_mask.sum()

        if n_cells_responsive < 3:
            if verbose:
                print(f"    Warning: Only {n_cells_responsive} responsive cells, skipping...")
            return None, None

        if verbose:
            if threshold_dff is not None:
                print(f"    Combined template: {n_cells_responsive} cells above {threshold_dff*100}% threshold")
            else:
                print(f"    Combined template: using all {n_cells_responsive} cells")

        # Step 2: Load no-stim trial data from baseline days and pool them
        folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        file_name = 'tensor_xarray_learning_data.nc'
        xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)

        # Collect no-stim trials from both baseline days
        all_nostim_data = []
        total_trials = 0
        n_timepoints = None

        for ref_day in reference_days:
            xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == ref_day)
            nostim_trials = xarray_day.sel(trial=xarray_day['no_stim'] == 1)

            n_nostim_trials = len(nostim_trials.trial)
            if n_nostim_trials < 1:
                if verbose:
                    print(f"    Warning: No no-stim trials on day {ref_day}")
                continue

            # Get data shape
            n_cells, n_trials, n_tp = nostim_trials.shape
            if n_timepoints is None:
                n_timepoints = n_tp

            # Reshape to 2D (concatenate trials)
            day_data = nostim_trials.values.reshape(n_cells, -1)
            day_data = np.nan_to_num(day_data, nan=0.0)
            all_nostim_data.append(day_data)
            total_trials += n_trials

            if verbose:
                print(f"    Day {ref_day}: {n_trials} no-stim trials × {n_tp} frames")

        if len(all_nostim_data) == 0:
            if verbose:
                print(f"    Error: No valid no-stim data from baseline days")
            return None, None

        # Pool data from baseline days
        data = np.concatenate(all_nostim_data, axis=1)
        n_frames = data.shape[1]

        if n_frames < 60:  # Less than 2 seconds
            if verbose:
                print(f"    Warning: Pooled data too short ({n_frames} frames), skipping...")
            return None, None

        if verbose:
            print(f"    Pooled baseline data: {total_trials} trials = {n_frames} total frames")

        # Step 3: Compute surrogate thresholds
        surrogate_results = compute_surrogate_thresholds(
            data, template, n_surrogates, min_shift_frames, percentile=percentile, verbose=verbose
        )

        # Store results (no day column - single threshold per mouse)
        results_list.append({
            'mouse_id': mouse,
            'n_cells_responsive': n_cells_responsive,
            'n_trials': total_trials,
            'n_timepoints': n_timepoints,
            'n_frames': n_frames,
            'n_surrogates': n_surrogates,
            'percentile_value': percentile,
            'threshold_percentile_mean': surrogate_results['threshold_percentile_mean'],
            'threshold_percentile_ci_lower': surrogate_results['threshold_percentile_ci'][0],
            'threshold_percentile_ci_upper': surrogate_results['threshold_percentile_ci'][1],
            'threshold_max_mean': surrogate_results['threshold_max_mean'],
            'threshold_max_ci_lower': surrogate_results['threshold_max_ci'][0],
            'threshold_max_ci_upper': surrogate_results['threshold_max_ci'][1],
            'observed_percentile': surrogate_results['observed_percentile'],
            'observed_max': surrogate_results['observed_max'],
            'p_value_percentile': surrogate_results['p_value_percentile'],
            'p_value_max': surrogate_results['p_value_max']
        })

        # Store detailed data for plotting
        all_surrogate_data['baseline'] = surrogate_results

    except Exception as e:
        if verbose:
            print(f"    Error processing baseline days: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

    if len(results_list) == 0:
        if verbose:
            print(f"\n  No valid data for mouse {mouse}")
        return None, None

    results_df = pd.DataFrame(results_list)

    if verbose:
        print(f"\n  Completed mouse {mouse}: Baseline threshold computed")

    return results_df, all_surrogate_data


def process_single_mouse(mouse, threshold_dff, n_surrogates, percentile=95, verbose=False):
    """
    Wrapper for parallel processing.
    """
    results_df, surrogate_data = analyze_mouse_surrogates(
        mouse, threshold_dff=threshold_dff,
        n_surrogates=n_surrogates, percentile=percentile, verbose=verbose
    )
    return (mouse, results_df, surrogate_data)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_surrogate_distributions(mouse, surrogate_data, save_path):
    """
    Generate single-page PDF showing surrogate distributions for baseline days.

    Parameters
    ----------
    mouse : str
        Mouse ID
    surrogate_data : dict
        {key: surrogate_results} from analyze_mouse_surrogates (baseline)
    save_path : str
        Path to save PDF
    """
    if surrogate_data is None or len(surrogate_data) == 0:
        print(f"    Warning: No surrogate data for {mouse}, skipping plot")
        return

    # Get baseline results
    if 'baseline' not in surrogate_data:
        print(f"    Warning: No baseline data for {mouse}, skipping plot")
        return

    results = surrogate_data['baseline']

    with PdfPages(save_path) as pdf:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Get percentile value from results
        percentile_val = int(results.get('percentile_value', 95))

        # Left panel: Percentile distribution
        ax1.hist(results['surrogate_percentiles'], bins=50, alpha=0.7, color='steelblue',
                edgecolor='black', linewidth=0.5)

        # Add mean threshold line
        ax1.axvline(results['threshold_percentile_mean'], color='blue', linestyle='-',
                   linewidth=2, label=f"Mean: {results['threshold_percentile_mean']:.4f}")

        # Add confidence interval
        ax1.axvspan(results['threshold_percentile_ci'][0], results['threshold_percentile_ci'][1],
                   alpha=0.2, color='blue', label='95% CI')

        # Add observed value
        ax1.axvline(results['observed_percentile'], color='red', linestyle='--',
                   linewidth=2, label=f"Observed: {results['observed_percentile']:.4f}")

        # Statistics text
        stats_text = f"n_surrogates = {len(results['surrogate_percentiles'])}\n"
        stats_text += f"p-value = {results['p_value_percentile']:.3f}"
        ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax1.set_xlabel(f'{percentile_val}th Percentile Correlation', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title(f'{percentile_val}th Percentile Threshold (Pointwise)', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')

        # Right panel: Max correlation distribution
        ax2.hist(results['surrogate_maxs'], bins=50, alpha=0.7, color='darkorange',
                edgecolor='black', linewidth=0.5)

        # Add mean threshold line
        ax2.axvline(results['threshold_max_mean'], color='orange', linestyle='-',
                   linewidth=2, label=f"Mean: {results['threshold_max_mean']:.4f}")

        # Add confidence interval
        ax2.axvspan(results['threshold_max_ci'][0], results['threshold_max_ci'][1],
                   alpha=0.2, color='orange', label='95% CI')

        # Add observed value
        ax2.axvline(results['observed_max'], color='red', linestyle='--',
                   linewidth=2, label=f"Observed: {results['observed_max']:.4f}")

        # Statistics text
        stats_text = f"n_surrogates = {len(results['surrogate_maxs'])}\n"
        stats_text += f"p-value = {results['p_value_max']:.3f}"
        ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax2.set_xlabel('Maximum Correlation', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(f'Maximum Threshold (FWER)', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')

        # Overall title - emphasize this is baseline threshold applied to all days
        fig.suptitle(f'{mouse} - Baseline (Days -2, -1) Surrogate Analysis (Applied to All Days)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"    Saved PDF: {save_path}")


def plot_threshold_summary_across_mice(all_results_df, save_path):
    """
    Summary plots comparing baseline (days -2, -1) thresholds across mice.

    Parameters
    ----------
    all_results_df : pd.DataFrame
        Combined results from all mice (one row per mouse)
    save_path : str
        Path to save PDF
    """
    # Add reward group information
    all_results_df['reward_group'] = all_results_df['mouse_id'].apply(
        lambda m: io.get_mouse_reward_group_from_db(io.db_path, m, db=db)
    )

    # Get percentile value from data
    percentile_val = int(all_results_df['percentile_value'].iloc[0]) if len(all_results_df) > 0 else 95

    with PdfPages(save_path) as pdf:
        # Page 1: Threshold distributions by reward group
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Percentile thresholds - histogram
        ax = axes[0, 0]
        for reward_group, color in zip(['R+', 'R-'], ['steelblue', 'coral']):
            group_data = all_results_df[all_results_df['reward_group'] == reward_group]
            ax.hist(group_data['threshold_percentile_mean'], bins=15, alpha=0.6,
                   color=color, edgecolor='black', linewidth=0.5,
                   label=f'{reward_group} (n={len(group_data)})')
        ax.set_xlabel(f'{percentile_val}th Percentile Threshold', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f'Baseline {percentile_val}th Percentile Threshold Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Percentile thresholds - box plot
        ax = axes[0, 1]
        data_to_plot = []
        labels = []
        colors_list = []
        for reward_group, color in zip(['R+', 'R-'], ['steelblue', 'coral']):
            group_data = all_results_df[all_results_df['reward_group'] == reward_group]
            data_to_plot.append(group_data['threshold_percentile_mean'].values)
            labels.append(f'{reward_group}\n(n={len(group_data)})')
            colors_list.append(color)

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(f'{percentile_val}th Percentile Threshold', fontweight='bold')
        ax.set_title(f'Baseline {percentile_val}th Percentile by Reward Group', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Max thresholds - histogram
        ax = axes[1, 0]
        for reward_group, color in zip(['R+', 'R-'], ['darkorange', 'orchid']):
            group_data = all_results_df[all_results_df['reward_group'] == reward_group]
            ax.hist(group_data['threshold_max_mean'], bins=15, alpha=0.6,
                   color=color, edgecolor='black', linewidth=0.5,
                   label=f'{reward_group} (n={len(group_data)})')
        ax.set_xlabel('Maximum Threshold (FWER)', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Baseline Maximum (FWER) Threshold Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Max thresholds - box plot
        ax = axes[1, 1]
        data_to_plot = []
        labels = []
        colors_list = []
        for reward_group, color in zip(['R+', 'R-'], ['darkorange', 'orchid']):
            group_data = all_results_df[all_results_df['reward_group'] == reward_group]
            data_to_plot.append(group_data['threshold_max_mean'].values)
            labels.append(f'{reward_group}\n(n={len(group_data)})')
            colors_list.append(color)

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Maximum Threshold (FWER)', fontweight='bold')
        ax.set_title('Baseline Maximum (FWER) by Reward Group', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Baseline (Days -2, -1) Threshold Summary (Applied to All Days)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Scatter plot of percentile vs max thresholds
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, reward_group in enumerate(['R+', 'R-']):
            ax = axes[i]
            group_data = all_results_df[all_results_df['reward_group'] == reward_group]

            ax.scatter(group_data['threshold_percentile_mean'],
                      group_data['threshold_max_mean'],
                      alpha=0.6, s=100, edgecolor='black', linewidth=0.5)

            ax.set_xlabel(f'{percentile_val}th Percentile Threshold', fontweight='bold')
            ax.set_ylabel('Maximum Threshold (FWER)', fontweight='bold')
            ax.set_title(f'{reward_group} Mice: Percentile vs Max\n(n={len(group_data)})',
                        fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add diagonal reference line
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            max_val = max(xlim[1], ylim[1])
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"  Saved summary PDF: {save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("REACTIVATION SURROGATE ANALYSIS - BASELINE THRESHOLD")
    print("="*60)
    print(f"\nParameters:")
    if threshold_dff is not None:
        print(f"  Responsiveness threshold: {threshold_dff*100}% dF/F")
    else:
        print(f"  Responsiveness threshold: None (all cells)")
    print(f"  Number of surrogates: {n_surrogates}")
    print(f"  Percentile threshold: {percentile_threshold}th")
    print(f"  Minimum time shift: {min_shift_frames} frames ({min_shift_frames/sampling_rate:.1f} sec)")
    print(f"  Reference days: Days -2 and -1 (applied to all days: {days})")
    print(f"  Parallel jobs: {n_jobs}")

    # Create output directory
    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/reactivation_surrogates'
    output_dir = io.adjust_path_to_host(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Process all mice in parallel
    print("\n" + "="*60)
    print("PROCESSING ALL MICE")
    print("="*60)

    all_mice_to_process = r_plus_mice + r_minus_mice
    print(f"Processing {len(all_mice_to_process)} mice in parallel (baseline days -2, -1)...")

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_mouse)(mouse, threshold_dff, n_surrogates,
                                     percentile=percentile_threshold, verbose=False)
        for mouse in all_mice_to_process
    )

    # Collect results
    all_results = []
    all_surrogate_data = {}

    for mouse, results_df, surrogate_data in results_list:
        if results_df is not None:
            all_results.append(results_df)
            all_surrogate_data[mouse] = surrogate_data

    if len(all_results) == 0:
        print("\nERROR: No valid results collected!")
        sys.exit(1)

    all_results_df = pd.concat(all_results, ignore_index=True)
    print(f"\nCollected results: {len(all_results_df)} mice with baseline thresholds")

    # Save main results CSV
    csv_path = os.path.join(output_dir, 'surrogate_thresholds.csv')
    all_results_df.to_csv(csv_path, index=False)
    print(f"\nSaved thresholds to: {csv_path}")

    # Generate per-mouse PDFs
    print("\n" + "="*60)
    print("GENERATING PER-MOUSE VISUALIZATIONS")
    print("="*60)

    pdf_dir = os.path.join(output_dir, 'per_mouse_pdfs')
    os.makedirs(pdf_dir, exist_ok=True)

    for mouse, surrogate_data in all_surrogate_data.items():
        if surrogate_data is not None:
            pdf_path = os.path.join(pdf_dir, f'{mouse}_surrogate_analysis.pdf')
            print(f"  Generating PDF for {mouse}...")
            plot_surrogate_distributions(mouse, surrogate_data, pdf_path)

    # Generate summary plots
    print("\n" + "="*60)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("="*60)

    summary_pdf_path = os.path.join(output_dir, 'surrogate_threshold_summary.pdf')
    plot_threshold_summary_across_mice(all_results_df, summary_pdf_path)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Get percentile value for display
    percentile_val = int(all_results_df['percentile_value'].iloc[0]) if len(all_results_df) > 0 else 95

    for reward_group in ['R+', 'R-']:
        group_data = all_results_df[all_results_df['mouse_id'].isin(
            r_plus_mice if reward_group == 'R+' else r_minus_mice
        )]

        print(f"\n{reward_group} Group (n={group_data['mouse_id'].nunique()} mice):")
        print(f"  {percentile_val}th percentile threshold: {group_data['threshold_percentile_mean'].mean():.4f} ± {group_data['threshold_percentile_mean'].std():.4f}")
        print(f"  Max threshold (FWER):         {group_data['threshold_max_mean'].mean():.4f} ± {group_data['threshold_max_mean'].std():.4f}")
        print(f"  Range ({percentile_val}th): [{group_data['threshold_percentile_mean'].min():.4f}, {group_data['threshold_percentile_mean'].max():.4f}]")
        print(f"  Range (max):     [{group_data['threshold_max_mean'].min():.4f}, {group_data['threshold_max_mean'].max():.4f}]")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(all_mice_to_process)} mice")
    print(f"Total mice with thresholds: {len(all_results_df)}")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey points:")
    print(f"  - Each mouse has ONE threshold computed from baseline days (-2, -1)")
    print(f"  - Template and data pooled from days -2 and -1")
    print(f"  - This threshold will be applied to ALL days (−2, −1, 0, +1, +2)")
    print(f"  - Ensures consistent event detection across days")
    print(f"\nNext steps:")
    print(f"1. Review per-mouse PDFs in: {pdf_dir}")
    print(f"2. Examine summary plots: {summary_pdf_path}")
    print(f"3. Choose threshold type ({percentile_val}th percentile or max/FWER)")
    print(f"4. Use reactivation.py and reactivation_lmi_prediction.py with new thresholds")
