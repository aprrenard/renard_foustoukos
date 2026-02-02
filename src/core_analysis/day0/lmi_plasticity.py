"""
Single-cell plasticity analysis during day 0 whisker learning.

This script fits sigmoid models to trial-by-trial responses of LMI-significant cells
and identifies cells showing online plasticity. Two statistical tests are performed:
  1. Sigmoid vs Flat: Tests if there is ANY change over trials
  2. Sigmoid vs Linear: Tests if the change is specifically sigmoid-shaped (more stringent)

Cells are ranked by response amplitude and filtered by statistical significance.

Output: CSV with amplitude metrics, distribution plots including inflection timing,
        and PDF reports showing significant cells with detailed plots and test results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2, mannwhitneyu, kruskal
from scipy.optimize import curve_fit
from scipy.special import expit
from joblib import Parallel, delayed
import warnings

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *

# =============================================================================
# PARAMETERS
# =============================================================================

# Analysis parameters
RUN_FITTING = False
GENERATE_PDFS = False
SAMPLING_RATE = 30  # Hz
RESPONSE_WIN = (0, 0.300)  # 0-300ms response window
PSTH_WIN = (-0.5, 1.5)  # PSTH time window for visualization
RESPONSE_TYPE = 'mean'  # 'mean' or 'peak' within response window
AMPLITUDE_TYPE = 'absolute'  # 'absolute' or 'relative' - how to compute amplitude
MIN_TRIALS = 20  # Minimum whisker trials required for fitting
ALPHA = 0.05  # Significance threshold
N_CORES = 35  # Number of cores for parallel processing (one per mouse)
DAYS_LEARNING = [-2, -1, 0, 1, 2]  # Days for mapping PSTH visualization
USE_SIGMOID_FILTER = False  # Use sigmoid vs linear filter (True) or all LMI cells (False)

# LMI thresholds for cell selection
LMI_POSITIVE_THRESHOLD = 0.975  # Top 2.5% LMI cells
LMI_NEGATIVE_THRESHOLD = 0.025  # Bottom 2.5% LMI cells

# Output directory
OUTPUT_DIR = io.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/plasticity'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADINGa
# =============================================================================

def load_day0_data(mouse_id, response_type='mean', response_win=(0, 0.300)):
    """
    Load and prepare day 0 whisker trial data for a single mouse.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    response_type : str
        'mean' or 'peak' - how to compute response from window
    response_win : tuple
        (start, end) time window in seconds

    Returns
    -------
    responses : np.ndarray
        Shape (n_cells, n_trials) - response values
    trial_indices : np.ndarray
        Shape (n_trials,) - trial_w values (whisker trial numbers)
    roi_ids : np.ndarray
        Shape (n_cells,) - ROI identifiers
    """
    # Load xarray
    folder = os.path.join(io.processed_dir, 'mice')
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)

    # Select day 0 whisker trials
    xarray = xarray.sel(trial=xarray['day'] == 0)
    xarray = xarray.sel(trial=xarray['whisker_stim'] == 1)

    # Extract trial indices
    trial_indices = xarray['trial_w'].values

    # Compute response per trial
    xarray_win = xarray.sel(time=slice(*response_win))

    if response_type == 'mean':
        responses_xarr = xarray_win.mean(dim='time')
    elif response_type == 'peak':
        responses_xarr = xarray_win.max(dim='time')
    else:
        raise ValueError(f"response_type must be 'mean' or 'peak', got {response_type}")

    # Extract ROI identifiers
    roi_ids = xarray['roi'].values

    # Convert to numpy array: (n_cells, n_trials)
    responses = responses_xarr.values

    return responses, trial_indices, roi_ids


# =============================================================================
# MODEL FITTING FUNCTIONS
# =============================================================================

def sigmoid_4pl(x, baseline, max_val, inflection, slope_param):
    """
    4-parameter logistic (sigmoid) function using scipy's expit.

    Parameters
    ----------
    x : array-like
        Independent variable (trial numbers)
    baseline : float
        Lower asymptote (response level at early trials, x → -∞)
    max_val : float
        Upper asymptote (response level at late trials, x → +∞)
    inflection : float
        Inflection point (trial number where change is steepest)
    slope_param : float
        Slope parameter (controls steepness)

    Returns
    -------
    y : array-like
        Sigmoid function output
    """
    # Use scipy's expit (logistic sigmoid) for numerical stability
    # expit(x) = 1 / (1 + exp(-x))
    return baseline + (max_val - baseline) * expit((x - inflection) / slope_param)


def fit_sigmoid_model(x, y):
    """
    Fit 4-parameter logistic sigmoid model to data.

    Parameters
    ----------
    x : np.ndarray
        Trial indices (1D)
    y : np.ndarray
        Response values (1D)

    Returns
    -------
    results : dict or None
        Returns None if fitting fails, otherwise dict with:
        {
            'baseline': float,
            'max_val': float,
            'inflection': float,
            'slope_param': float,
            'predictions': np.ndarray,
            'residuals': np.ndarray,
            'n_params': int,  # Always 4
            'fit_success': bool
        }
    """
    # Remove NaN values
    mask = ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 5:  # Need more points for 4 parameters
        return None

    # Compute initial parameter estimates
    y_min, y_max = np.nanmin(y_clean), np.nanmax(y_clean)
    y_range = y_max - y_min

    # Initial guesses
    p0 = [
        y_min,  # baseline
        y_max,  # max_val
        np.median(x_clean),  # inflection (middle trial)
        (x_clean[-1] - x_clean[0]) / 4  # slope_param (quarter of trial range)
    ]

    # Bounds to ensure numerical stability
    bounds = (
        [y_min - y_range, y_min - y_range, x_clean[0], 0.1],  # Lower bounds
        [y_max + y_range, y_max + y_range, x_clean[-1], (x_clean[-1] - x_clean[0]) * 2]  # Upper bounds
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                sigmoid_4pl, x_clean, y_clean,
                p0=p0, bounds=bounds, maxfev=5000
            )

        predictions = sigmoid_4pl(x_clean, *popt)
        residuals = y_clean - predictions

        # Compute robust amplitude by evaluating fitted curve over trial range
        # This is more robust than |max_val - baseline| which can be affected by outlier parameters
        trial_range = np.linspace(x_clean[0], x_clean[-1], 100)
        predictions_range = sigmoid_4pl(trial_range, *popt)

        # Signed amplitude: max_val - baseline (can be positive or negative)
        # Positive: cell increases response during learning
        # Negative: cell decreases response during learning
        baseline_val = popt[0]
        max_val = popt[1]
        amplitude_absolute = max_val - baseline_val

        # Relative amplitude: normalized by baseline (avoid division by zero)
        if abs(baseline_val) > .1:  # Avoid division by very small values
            amplitude_relative = amplitude_absolute / abs(baseline_val)
        else:
            amplitude_relative = amplitude_absolute  # Fallback to absolute if baseline ~0

        # Compute parameter standard errors from covariance matrix
        try:
            perr = np.sqrt(np.diag(pcov))
        except:
            perr = None

        return {
            'baseline': popt[0],
            'max_val': popt[1],
            'inflection': popt[2],
            'slope_param': popt[3],
            'predictions': predictions,
            'residuals': residuals,
            'n_params': 4,
            'fit_success': True,
            'x_clean': x_clean,
            'y_clean': y_clean,
            'amplitude_absolute': amplitude_absolute,
            'amplitude_relative': amplitude_relative,
            'pcov': pcov,
            'perr': perr
        }

    except (RuntimeError, ValueError):
        # Fitting failed
        return None


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def compute_pseudo_r_squared(residuals, y):
    """
    Compute pseudo-R² for sigmoid model.

    Pseudo-R² = 1 - (RSS / TSS)
    where TSS = total sum of squares

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    y : np.ndarray
        Observed values

    Returns
    -------
    pseudo_r2 : float
    """
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum(residuals ** 2)

    if tss == 0:
        return 0.0

    return 1 - (rss / tss)


def fit_flat_model(y):
    """
    Fit flat (constant mean) model for null hypothesis comparison.

    Parameters
    ----------
    y : np.ndarray
        Response values (1D)

    Returns
    -------
    results : dict
    """
    mask = ~np.isnan(y)
    y_clean = y[mask]

    mean_val = np.nanmean(y_clean)
    predictions = np.full_like(y_clean, mean_val)
    residuals = y_clean - predictions

    return {
        'mean': mean_val,
        'predictions': predictions,
        'residuals': residuals,
        'n_params': 1,
        'y_clean': y_clean
    }


def fit_linear_model(x, y):
    """
    Fit linear regression model for comparison with sigmoid.

    Parameters
    ----------
    x : np.ndarray
        Trial indices (1D)
    y : np.ndarray
        Response values (1D)

    Returns
    -------
    results : dict or None
        Returns None if fitting fails, otherwise dict with:
        {
            'slope': float,
            'intercept': float,
            'predictions': np.ndarray,
            'residuals': np.ndarray,
            'n_params': int  # Always 2
        }
    """
    from scipy.stats import linregress

    # Remove NaN values
    mask = ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 2:
        return None

    try:
        # Fit linear regression: y = slope * x + intercept
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)

        predictions = slope * x_clean + intercept
        residuals = y_clean - predictions

        return {
            'slope': slope,
            'intercept': intercept,
            'predictions': predictions,
            'residuals': residuals,
            'n_params': 2,
            'x_clean': x_clean,
            'y_clean': y_clean
        }

    except (ValueError, RuntimeError):
        return None


def likelihood_ratio_test(residuals_null, residuals_alt, df_diff):
    """
    Perform likelihood ratio test between nested models.

    Parameters
    ----------
    residuals_null : np.ndarray
        Residuals from null (flat) model
    residuals_alt : np.ndarray
        Residuals from alternative (sigmoid) model
    df_diff : int
        Difference in degrees of freedom

    Returns
    -------
    p_value : float
    """
    n = len(residuals_null)
    rss_null = np.sum(residuals_null ** 2)
    rss_alt = np.sum(residuals_alt ** 2)

    if rss_alt <= 0:
        return 0.0

    lr_stat = n * np.log(rss_null / rss_alt)
    p_value = 1 - chi2.cdf(lr_stat, df_diff)

    return p_value


# =============================================================================
# SINGLE-CELL ANALYSIS
# =============================================================================

def analyze_single_cell(x, y, min_trials=20, amplitude_type='absolute'):
    """
    Fit sigmoid model to single cell's trial-by-trial responses.

    Performs two significance tests:
    1. Sigmoid vs Flat: Tests if there is ANY change over trials
    2. Sigmoid vs Linear: Tests if the change is specifically sigmoid-shaped

    Parameters
    ----------
    x : np.ndarray
        Trial indices (1D)
    y : np.ndarray
        Response values (1D)
    min_trials : int
        Minimum number of trials required
    amplitude_type : str
        'absolute' or 'relative' - how to compute amplitude

    Returns
    -------
    results : dict or None
        Returns None if insufficient data or fitting failed
        Otherwise returns dict with sigmoid fit results and both p-values
    """
    # Check data quality
    mask = ~np.isnan(y)
    n_valid = np.sum(mask)

    if n_valid < min_trials:
        return None

    # Fit sigmoid model
    sigmoid_fit = fit_sigmoid_model(x, y)
    if sigmoid_fit is None or not sigmoid_fit.get('fit_success', False):
        return None

    # Fit flat model for significance test #1: Sigmoid vs Flat
    flat_fit = fit_flat_model(y)

    # Fit linear model for significance test #2: Sigmoid vs Linear
    linear_fit = fit_linear_model(x, y)

    # Compute pseudo-R²
    pseudo_r2 = compute_pseudo_r_squared(sigmoid_fit['residuals'], sigmoid_fit['y_clean'])

    # Test #1: Sigmoid vs Flat (tests if there is any change)
    p_value_vs_flat = likelihood_ratio_test(
        flat_fit['residuals'],
        sigmoid_fit['residuals'],
        df_diff=sigmoid_fit['n_params'] - flat_fit['n_params']  # 4 - 1 = 3
    )

    # Test #2: Sigmoid vs Linear (tests if change is sigmoid-shaped)
    if linear_fit is not None:
        p_value_vs_linear = likelihood_ratio_test(
            linear_fit['residuals'],
            sigmoid_fit['residuals'],
            df_diff=sigmoid_fit['n_params'] - linear_fit['n_params']  # 4 - 2 = 2
        )
    else:
        # Linear fit failed, use NaN
        p_value_vs_linear = np.nan

    # Select amplitude type (absolute or relative)
    if amplitude_type == 'relative':
        amplitude = sigmoid_fit['amplitude_relative']
    else:  # Default to 'absolute'
        amplitude = sigmoid_fit['amplitude_absolute']

    return {
        'p_value': p_value_vs_flat,  # Keep original name for backwards compatibility
        'p_value_vs_flat': p_value_vs_flat,
        'p_value_vs_linear': p_value_vs_linear,
        'pseudo_r2': pseudo_r2,
        'amplitude': amplitude,
        'amplitude_absolute': sigmoid_fit['amplitude_absolute'],
        'amplitude_relative': sigmoid_fit['amplitude_relative'],
        'baseline': sigmoid_fit['baseline'],
        'max_val': sigmoid_fit['max_val'],
        'inflection': sigmoid_fit['inflection'],
        'slope_param': sigmoid_fit['slope_param'],
        'predictions': sigmoid_fit['predictions'],
        'residuals': sigmoid_fit['residuals'],
        'x_clean': sigmoid_fit['x_clean'],
        'y_clean': sigmoid_fit['y_clean']
    }


# =============================================================================
# MOUSE-LEVEL PROCESSING
# =============================================================================

def process_mouse(mouse_id, response_type='mean', response_win=(0, 0.300),
                  min_trials=20, amplitude_type='absolute'):
    """
    Process all cells for a single mouse using sigmoid model.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    response_type : str
        'mean' or 'peak'
    response_win : tuple
        (start, end) time window
    min_trials : int
        Minimum trials required for fitting
    amplitude_type : str
        'absolute' or 'relative' - how to compute amplitude

    Returns
    -------
    results_df : pd.DataFrame
        Columns: mouse_id, roi, reward_group, p_value, pseudo_r2, amplitude,
                 amplitude_absolute, amplitude_relative, baseline, max_val,
                 inflection, slope_param, n_trials
    """
    print(f"Processing {mouse_id}...")

    # Load data
    responses, trial_indices, roi_ids = load_day0_data(mouse_id, response_type, response_win)

    # Get reward group
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    # Process each cell
    results = []
    n_cells = responses.shape[0]

    for i, roi in enumerate(roi_ids):
        y = responses[i, :]
        x = trial_indices

        cell_results = analyze_single_cell(x, y, min_trials, amplitude_type)

        if cell_results is not None:
            results.append({
                'mouse_id': mouse_id,
                'roi': roi,
                'reward_group': reward_group,
                'p_value': cell_results['p_value'],  # Sigmoid vs Flat
                'p_value_vs_flat': cell_results['p_value_vs_flat'],
                'p_value_vs_linear': cell_results['p_value_vs_linear'],
                'pseudo_r2': cell_results['pseudo_r2'],
                'amplitude': cell_results['amplitude'],
                'amplitude_absolute': cell_results['amplitude_absolute'],
                'amplitude_relative': cell_results['amplitude_relative'],
                'baseline': cell_results['baseline'],
                'max_val': cell_results['max_val'],
                'inflection': cell_results['inflection'],
                'slope_param': cell_results['slope_param'],
                'n_trials': len(x)
            })

    results_df = pd.DataFrame(results)
    print(f"  Processed {len(results_df)} cells for {mouse_id}")

    return results_df


# =============================================================================
# PRE/POST INFLECTION ANALYSIS
# =============================================================================

def process_mouse_pre_post_inflection(mouse_id, mouse_cells_df, response_win, min_trials_per_period, psth_win=(-0.5, 1.5)):
    """
    Process all cells from a single mouse for pre/post inflection analysis.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    mouse_cells_df : pd.DataFrame
        Dataframe with cells from this mouse (columns: roi, lmi_sign, reward_group, inflection)
    response_win : tuple
        Time window for quantifying responses (start, end) in seconds
    min_trials_per_period : int
        Minimum number of trials required in each period
    psth_win : tuple
        Time window for PSTH traces (start, end) in seconds

    Returns
    -------
    psth_list : list
        List of PSTH dataframes for this mouse (averaged traces only)
    response_list : list
        List of response dicts for this mouse
    n_processed : int
        Number of cells successfully processed
    n_skipped : int
        Number of cells skipped
    """
    folder = os.path.join(io.processed_dir, 'mice')

    psth_list = []
    response_list = []
    n_processed = 0
    n_skipped = 0

    try:
        # Load baseline-subtracted xarray data ONCE for this mouse
        xarr = utils_imaging.load_mouse_xarray(
            mouse_id, folder, 'tensor_xarray_learning_data.nc', substracted=True
        )

        # Filter for day 0 and whisker trials once
        xarr = xarr.sel(trial=xarr['day'] == 0)
        xarr = xarr.sel(trial=xarr['whisker_stim'] == 1)

        # Select PSTH time window EARLY to reduce data size
        xarr = xarr.sel(time=slice(*psth_win))

        # Process each cell from this mouse
        for _, row in mouse_cells_df.iterrows():
            roi = row['roi']
            lmi_sign = row['lmi_sign']
            reward_group = row['reward_group']
            inflection_trial = row['inflection']

            try:
                # Filter for this cell
                xarr_cell = xarr.sel(cell=xarr['roi'] == roi)

                # Get trial indices
                trial_w_indices = xarr_cell['trial_w'].values

                # Split by inflection point
                before_mask = trial_w_indices <= inflection_trial
                after_mask = trial_w_indices > inflection_trial

                # Check minimum trials requirement
                n_before = np.sum(before_mask)
                n_after = np.sum(after_mask)

                if n_before < min_trials_per_period or n_after < min_trials_per_period:
                    n_skipped += 1
                    continue

                trials_before = xarr_cell.isel(trial=before_mask)
                trials_after = xarr_cell.isel(trial=after_mask)

                # PERFORMANCE: Average trials BEFORE converting to dataframe
                # This reduces data from (n_trials × n_timepoints) to just (n_timepoints)
                mean_before = trials_before.mean(dim='trial') * 100  # Convert to %ΔF/F
                mean_after = trials_after.mean(dim='trial') * 100

                # Convert averaged traces to dataframe (much smaller!)
                df_before = mean_before.to_dataframe(name='psth').reset_index()
                df_before['period'] = 'pre'
                df_before['mouse_id'] = mouse_id
                df_before['roi'] = roi
                df_before['lmi_sign'] = lmi_sign
                df_before['reward_group'] = reward_group

                df_after = mean_after.to_dataframe(name='psth').reset_index()
                df_after['period'] = 'post'
                df_after['mouse_id'] = mouse_id
                df_after['roi'] = roi
                df_after['lmi_sign'] = lmi_sign
                df_after['reward_group'] = reward_group

                # Combine pre and post
                df_combined = pd.concat([df_before, df_after], ignore_index=True)
                psth_list.append(df_combined)

                # Compute quantified responses (0-300ms mean)
                before_response_win = trials_before.sel(time=slice(*response_win)).mean(dim='time').values
                after_response_win = trials_after.sel(time=slice(*response_win)).mean(dim='time').values

                # Store responses (average across trials for each period)
                response_list.append({
                    'mouse_id': mouse_id,
                    'roi': roi,
                    'lmi_sign': lmi_sign,
                    'reward_group': reward_group,
                    'response': np.mean(before_response_win) * 100,  # Convert to %ΔF/F
                    'period': 'pre',
                    'n_trials': n_before
                })

                response_list.append({
                    'mouse_id': mouse_id,
                    'roi': roi,
                    'lmi_sign': lmi_sign,
                    'reward_group': reward_group,
                    'response': np.mean(after_response_win) * 100,
                    'period': 'post',
                    'n_trials': n_after
                })

                n_processed += 1

            except Exception as e:
                print(f"    WARNING: Failed to process cell {mouse_id}_{roi}: {e}")
                n_skipped += 1
                continue

    except Exception as e:
        print(f"  ERROR: Failed to load data for {mouse_id}: {e}")
        n_skipped = len(mouse_cells_df)

    return psth_list, response_list, n_processed, n_skipped


def compute_pre_post_inflection_psth(results_df, response_win=(0, 0.300), min_trials_per_period=3, psth_win=(-0.5, 1.5), n_jobs=1):
    """
    Compute average PSTH and response before and after inflection point for each cell.

    Efficiently loads each mouse's data only once and processes all cells from that mouse.
    Can optionally parallelize across mice.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns: mouse_id, roi, lmi_sign, reward_group, inflection
    response_win : tuple
        Time window for quantifying responses (start, end) in seconds
    min_trials_per_period : int
        Minimum number of trials required in each period (before/after inflection)
    psth_win : tuple
        Time window for PSTH traces (start, end) in seconds
    n_jobs : int
        Number of parallel jobs (default: 1 = no parallelization)
        Set to -1 to use all available cores

    Returns
    -------
    psth_df : pd.DataFrame
        PSTH data with columns: mouse_id, roi, lmi_sign, reward_group, time, psth, period
    response_df : pd.DataFrame
        Quantified responses with columns: mouse_id, roi, lmi_sign, reward_group, response, period
    """
    print("\n" + "="*70)
    print("COMPUTING PRE/POST INFLECTION PSTH")
    print("="*70)

    # Group cells by mouse
    grouped = results_df.groupby('mouse_id')
    n_mice = len(grouped)
    n_cells_total = len(results_df)

    print(f"\nProcessing {n_cells_total} cells from {n_mice} mice")
    print(f"PSTH window: {psth_win[0]} to {psth_win[1]} sec")
    print(f"Parallelization: {'Enabled' if n_jobs != 1 else 'Disabled'} (n_jobs={n_jobs})")

    if n_jobs == 1:
        # Sequential processing
        all_results = []
        for idx, (mouse_id, mouse_cells) in enumerate(grouped):
            print(f"  Processing mouse {idx+1}/{n_mice}: {mouse_id} ({len(mouse_cells)} cells)...")
            result = process_mouse_pre_post_inflection(
                mouse_id, mouse_cells, response_win, min_trials_per_period, psth_win
            )
            all_results.append(result)
    else:
        # Parallel processing across mice
        print(f"  Running parallel processing across {n_mice} mice...")
        all_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(process_mouse_pre_post_inflection)(
                mouse_id, mouse_cells, response_win, min_trials_per_period, psth_win
            )
            for mouse_id, mouse_cells in grouped
        )

    # Combine results from all mice
    psth_list = []
    response_list = []
    n_cells_processed = 0
    n_cells_skipped = 0

    for psth_mouse, response_mouse, n_proc, n_skip in all_results:
        psth_list.extend(psth_mouse)
        response_list.extend(response_mouse)
        n_cells_processed += n_proc
        n_cells_skipped += n_skip

    print(f"\nProcessed {n_cells_processed} cells successfully")
    print(f"Skipped {n_cells_skipped} cells (insufficient trials or errors)")

    # Combine all data
    psth_df = pd.concat(psth_list, ignore_index=True) if psth_list else pd.DataFrame()
    response_df = pd.DataFrame(response_list) if response_list else pd.DataFrame()

    if len(psth_df) > 0:
        print(f"\nPSTH data shape: {psth_df.shape}")
        print(f"Response data shape: {response_df.shape}")

        # Print summary statistics
        print("\nCell counts by group:")
        summary = response_df.groupby(['reward_group', 'lmi_sign', 'period'])['roi'].nunique().reset_index()
        summary.columns = ['reward_group', 'lmi_sign', 'period', 'n_cells']
        print(summary.pivot_table(index=['reward_group', 'lmi_sign'], columns='period', values='n_cells'))

    return psth_df, response_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_pre_post_inflection_psth_and_responses(psth_df, response_df, output_dir, stat_level='cells'):
    """
    Plot PSTHs and response quantification for pre vs post inflection periods.

    Creates a 4x2 grid figure:
    - 4 rows: R+/LMI+, R+/LMI-, R-/LMI+, R-/LMI-
    - 2 columns: PSTH (left), response barplot (right)

    Statistical testing: Wilcoxon signed-rank test comparing pre vs post within each group.

    Parameters
    ----------
    psth_df : pd.DataFrame
        PSTH data with columns: mouse_id, roi, lmi_sign, reward_group, time, psth, period
    response_df : pd.DataFrame
        Response data with columns: mouse_id, roi, lmi_sign, reward_group, response, period
    output_dir : str
        Output directory for saving figure
    stat_level : str
        'cells' = stats over individual cells (default)
        'mice' = aggregate cells per mouse first, then stats over mice
    """
    print("\n" + "="*70)
    print(f"PLOTTING PRE/POST INFLECTION PSTH AND RESPONSES (stat_level={stat_level})")
    print("="*70)

    if len(psth_df) == 0 or len(response_df) == 0:
        print("  WARNING: No data to plot!")
        return

    # Aggregate data based on stat_level
    if stat_level == 'mice':
        print("  Aggregating cells per mouse before plotting/statistics...")

        # For PSTH: average across cells within each mouse
        psth_df = psth_df.groupby(['mouse_id', 'lmi_sign', 'reward_group', 'time', 'period'])['psth'].mean().reset_index()

        # For responses: average across cells within each mouse
        response_df = response_df.groupby(['mouse_id', 'lmi_sign', 'reward_group', 'period'])['response'].mean().reset_index()

        # Use 'mouse_id' as the identifier for pairing
        id_col = 'mouse_id'
    else:
        # Use 'roi' as the identifier for pairing
        id_col = 'roi'

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.3)

    # Create 4x2 subplot grid
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), dpi=150)

    # Define colors
    colors_pre_post = {
        'pre': '#a3a3a3',  # Gray
        'R+': '#1b9e77',   # Green
        'R-': '#c959affe'  # Magenta
    }

    # Define row order: R+/LMI+, R+/LMI-, R-/LMI+, R-/LMI-
    row_configs = [
        ('R+', 'Positive', 'R+/LMI+'),
        ('R+', 'Negative', 'R+/LMI-'),
        ('R-', 'Positive', 'R-/LMI+'),
        ('R-', 'Negative', 'R-/LMI-')
    ]

    # Process each row
    for row_idx, (reward_group, lmi_sign, label) in enumerate(row_configs):
        # Filter data for this group
        psth_group = psth_df[
            (psth_df['reward_group'] == reward_group) &
            (psth_df['lmi_sign'] == lmi_sign)
        ]

        response_group = response_df[
            (response_df['reward_group'] == reward_group) &
            (response_df['lmi_sign'] == lmi_sign)
        ]

        # Count data points (cells or mice depending on stat_level)
        if stat_level == 'mice':
            n_data_points = response_group['mouse_id'].nunique()
            data_label = f"n={n_data_points} mice"
        else:
            n_data_points = response_group['roi'].nunique()
            data_label = f"n={n_data_points} cells"

        # Left column: PSTH
        ax_psth = axes[row_idx, 0]

        if len(psth_group) > 0:
            # Create custom palette: pre=gray, post=reward color
            palette_psth = {'pre': colors_pre_post['pre'], 'post': colors_pre_post[reward_group]}

            sns.lineplot(
                data=psth_group, x='time', y='psth', hue='period',
                hue_order=['pre', 'post'], palette=palette_psth,
                errorbar='ci', ax=ax_psth, legend=True
            )

            ax_psth.axvline(0, color='orange', linestyle='-', linewidth=1, alpha=0.7)
            ax_psth.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_psth.set_xlabel('Time from stimulus (s)', fontsize=11)
            ax_psth.set_ylabel('Activity (%ΔF/F)', fontsize=11)
            ax_psth.set_title(f'{label} - PSTH ({data_label})', fontsize=12, fontweight='bold')
            ax_psth.legend(title='Period', loc='best', frameon=False, fontsize=9)
        else:
            ax_psth.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_psth.transAxes)
            ax_psth.set_title(f'{label} - PSTH', fontsize=12, fontweight='bold')
            ax_psth.axis('off')

        # Right column: Barplot with statistics
        ax_bar = axes[row_idx, 1]

        if len(response_group) > 0:
            # Create barplot
            sns.barplot(
                data=response_group, x='period', y='response',
                order=['pre', 'post'], color=colors_pre_post[reward_group],
                errorbar='ci', ax=ax_bar, alpha=0.7, edgecolor='black', linewidth=1.5
            )

            ax_bar.set_xlabel('Period', fontsize=11)
            ax_bar.set_ylabel('Response (% ΔF/F, 0-300ms)', fontsize=11)
            ax_bar.set_title(f'{label} - Response', fontsize=12, fontweight='bold')
            ax_bar.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_bar.set_ylim(0, 20)  # Set consistent y-axis limits

            # Statistical testing: Wilcoxon signed-rank test (paired)
            # Need to match pre and post responses for the same cells/mice
            pre_data = response_group[response_group['period'] == 'pre'].set_index(id_col)['response']
            post_data = response_group[response_group['period'] == 'post'].set_index(id_col)['response']

            # Find common cells/mice (paired test)
            common_ids = pre_data.index.intersection(post_data.index)

            if len(common_ids) >= 3:  # Need at least 3 paired samples
                pre_values = pre_data.loc[common_ids].values
                post_values = post_data.loc[common_ids].values

                try:
                    from scipy.stats import wilcoxon
                    stat, p_value = wilcoxon(pre_values, post_values)

                    # Determine significance stars
                    if p_value < 0.001:
                        stars = '***'
                    elif p_value < 0.01:
                        stars = '**'
                    elif p_value < 0.05:
                        stars = '*'
                    else:
                        stars = 'ns'

                    # Add significance annotation at fixed position (~80% of y-axis)
                    # Y-axis is 0-20, so place bracket at ~16
                    y_bracket = 16
                    ax_bar.plot([0, 0, 1, 1],
                               [y_bracket, y_bracket + 0.3, y_bracket + 0.3, y_bracket],
                               'k-', linewidth=1.5)
                    ax_bar.text(0.5, y_bracket + 0.5, stars,
                               ha='center', va='bottom', fontsize=14, fontweight='bold')

                    # Print result
                    unit_label = 'mice' if stat_level == 'mice' else 'cells'
                    print(f"  {label}: Wilcoxon test p={p_value:.4f} ({stars}), n={len(common_ids)} {unit_label}")

                except Exception as e:
                    print(f"  WARNING: Statistical test failed for {label}: {e}")
            else:
                print(f"  WARNING: Not enough paired samples for {label} (n={len(common_ids)})")
        else:
            ax_bar.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_bar.transAxes)
            ax_bar.set_title(f'{label} - Response', fontsize=12, fontweight='bold')
            ax_bar.axis('off')

        sns.despine(ax=ax_psth)
        sns.despine(ax=ax_bar)

    plt.tight_layout()

    # Save figure with stat_level in filename
    filename = f'pre_post_inflection_psth_and_responses_{stat_level}.svg'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, format='svg', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ Figure saved to: {save_path}")


def plot_inflection_timing_distributions(results_df, output_dir, alpha=0.05):
    """
    Plot distributions of inflection point timing for cells with significant sigmoid fits.

    Creates a 2-panel figure:
    - Panel 1: Distribution of absolute inflection points (whisker trial index)
    - Panel 2: Distribution of inflection points relative to behavioral learning trial

    Uses all cells in input dataframe (assumed to be pre-filtered).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe (should be pre-filtered for significance)
    output_dir : str
        Output directory for saving figure
    alpha : float
        Not used (kept for backwards compatibility)
    """
    print("\n  Computing inflection timing distributions...")

    # Use all cells in input dataframe (already filtered)
    sig_cells = results_df.copy()

    if len(sig_cells) == 0:
        print("  WARNING: No cells found!")
        return

    print(f"    Using {len(sig_cells)} doubly-significant cells")

    # Create figure
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150, sharex='col')

    # Panel 1: Absolute inflection timing (whisker trial index)
    for idx, reward_group in enumerate(['R+', 'R-']):
        ax = axes[0, idx]
        group_data = sig_cells[sig_cells['reward_group'] == reward_group]
        color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]

        if len(group_data) > 0:
            sns.histplot(
                data=group_data, x='inflection', ax=ax,
                color=color, alpha=0.6, binwidth=10,
                stat='percent', kde=False
            )

        ax.set_xlabel('Inflection Point (whisker trial index)', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title(f'{reward_group} (n={len(group_data)})', fontsize=14, fontweight='bold')
        ax.set_xlim(-50, 200)
        ax.set_ylim(0, 25)

    # Panel 2: Inflection timing relative to behavioral learning trial
    for idx, reward_group in enumerate(['R+', 'R-']):
        ax = axes[1, idx]
        group_data = sig_cells[sig_cells['reward_group'] == reward_group].dropna(subset=['learning_trial'])
        color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]

        if len(group_data) > 0:
            sns.histplot(
                data=group_data, x='inflection_relative', ax=ax,
                color=color, alpha=0.6, binwidth=10,
                stat='percent', kde=False
            )

            # Add learning trial reference line
            ax.axvline(0, color='red', linestyle='-', linewidth=2, alpha=0.8,
                      label='Behavioral learning trial')

        ax.set_xlabel('Inflection - Learning Trial (trials)', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title(f'{reward_group} (n={len(group_data)})', fontsize=14, fontweight='bold')
        ax.set_xlim(-50, 200)
        ax.set_ylim(0, 20)
        if len(group_data) > 0:
            ax.legend(frameon=False, fontsize=10)

    # Overall title
    fig.suptitle('Inflection Point Timing - Doubly-Significant Cells (LMI + Sigmoid vs Linear)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    sns.despine()

    # Save figure
    save_path = os.path.join(output_dir, 'inflection_timing_distributions.svg')
    plt.savefig(save_path, format='svg', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Inflection timing distributions saved to: {save_path}")

    # Print summary statistics
    print("\n  Inflection timing summary (doubly-significant cells):")
    for reward in ['R+', 'R-']:
        group_data = sig_cells[sig_cells['reward_group'] == reward]
        if len(group_data) > 0:
            mean_abs = group_data['inflection'].mean()
            std_abs = group_data['inflection'].std()
            median_abs = group_data['inflection'].median()
            print(f"    {reward} absolute: mean={mean_abs:.1f} ± {std_abs:.1f}, median={median_abs:.1f} (n={len(group_data)})")

            group_data_rel = group_data.dropna(subset=['learning_trial'])
            if len(group_data_rel) > 0:
                mean_rel = group_data_rel['inflection_relative'].mean()
                std_rel = group_data_rel['inflection_relative'].std()
                median_rel = group_data_rel['inflection_relative'].median()
                before = (group_data_rel['inflection_relative'] < 0).sum()
                after = (group_data_rel['inflection_relative'] >= 0).sum()
                print(f"    {reward} relative: mean={mean_rel:.1f} ± {std_rel:.1f}, median={median_rel:.1f}")
                print(f"               before learning: {before} ({100*before/len(group_data_rel):.1f}%)")
                print(f"               after learning: {after} ({100*after/len(group_data_rel):.1f}%)")


def plot_inflection_timing_distributions_late_learners(results_df, output_dir, alpha=0.05, learning_trial_threshold=20):
    """
    Plot distributions of inflection point timing excluding mice with early learning trials.

    Creates a 2-panel figure identical to plot_inflection_timing_distributions() but
    excludes mice where the behavioral learning trial occurred in the first N trials.
    This helps determine if results are dominated by early learners.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe (should be pre-filtered for significance)
    output_dir : str
        Output directory for saving figure
    alpha : float
        Not used (kept for backwards compatibility)
    learning_trial_threshold : int
        Exclude mice with learning_trial < this value (default: 20)
    """
    print(f"\n  Computing inflection timing distributions (excluding mice with learning_trial < {learning_trial_threshold})...")

    # Filter out mice with early learning trials
    sig_cells = results_df.dropna(subset=['learning_trial']).copy()
    early_learning_mice = sig_cells[sig_cells['learning_trial'] < learning_trial_threshold]['mouse_id'].unique()

    print(f"    Excluding {len(early_learning_mice)} mice with learning_trial < {learning_trial_threshold}:")
    for mouse_id in early_learning_mice:
        learning_trial = sig_cells[sig_cells['mouse_id'] == mouse_id]['learning_trial'].iloc[0]
        print(f"      - {mouse_id} (learning_trial = {learning_trial:.0f})")

    # Exclude these mice from analysis
    sig_cells = sig_cells[~sig_cells['mouse_id'].isin(early_learning_mice)]

    if len(sig_cells) == 0:
        print("  WARNING: No cells found after excluding early learning mice!")
        return

    remaining_mice = sig_cells['mouse_id'].nunique()
    print(f"    Using {len(sig_cells)} doubly-significant cells from {remaining_mice} mice")

    # Create figure
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150, sharex='col')

    # Panel 1: Absolute inflection timing (whisker trial index)
    for idx, reward_group in enumerate(['R+', 'R-']):
        ax = axes[0, idx]
        group_data = sig_cells[sig_cells['reward_group'] == reward_group]
        color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]

        if len(group_data) > 0:
            sns.histplot(
                data=group_data, x='inflection', ax=ax,
                color=color, alpha=0.6, binwidth=10,
                stat='percent', kde=False
            )

        ax.set_xlabel('Inflection Point (whisker trial index)', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title(f'{reward_group} (n={len(group_data)})', fontsize=14, fontweight='bold')
        ax.set_xlim(-50, 200)
        ax.set_ylim(0, 25)

    # Panel 2: Inflection timing relative to behavioral learning trial
    for idx, reward_group in enumerate(['R+', 'R-']):
        ax = axes[1, idx]
        group_data = sig_cells[sig_cells['reward_group'] == reward_group]
        color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]

        if len(group_data) > 0:
            sns.histplot(
                data=group_data, x='inflection_relative', ax=ax,
                color=color, alpha=0.6, binwidth=10,
                stat='percent', kde=False
            )

            # Add learning trial reference line
            ax.axvline(0, color='red', linestyle='-', linewidth=2, alpha=0.8,
                      label='Behavioral learning trial')

        ax.set_xlabel('Inflection - Learning Trial (trials)', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title(f'{reward_group} (n={len(group_data)})', fontsize=14, fontweight='bold')
        ax.set_xlim(-50, 200)
        ax.set_ylim(0, 20)
        if len(group_data) > 0:
            ax.legend(frameon=False, fontsize=10)

    # Overall title
    fig.suptitle(f'Inflection Point Timing - Excluding Early Learners (learning_trial < {learning_trial_threshold})',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    sns.despine()

    # Save figure
    save_path = os.path.join(output_dir, 'inflection_timing_distributions_late_learners.svg')
    plt.savefig(save_path, format='svg', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Inflection timing distributions (late learners) saved to: {save_path}")

    # Print summary statistics
    print("\n  Inflection timing summary (late learners only):")
    for reward in ['R+', 'R-']:
        group_data = sig_cells[sig_cells['reward_group'] == reward]
        if len(group_data) > 0:
            mean_abs = group_data['inflection'].mean()
            std_abs = group_data['inflection'].std()
            median_abs = group_data['inflection'].median()
            print(f"    {reward} absolute: mean={mean_abs:.1f} ± {std_abs:.1f}, median={median_abs:.1f} (n={len(group_data)})")

            if len(group_data) > 0:
                mean_rel = group_data['inflection_relative'].mean()
                std_rel = group_data['inflection_relative'].std()
                median_rel = group_data['inflection_relative'].median()
                before = (group_data['inflection_relative'] < 0).sum()
                after = (group_data['inflection_relative'] >= 0).sum()
                print(f"    {reward} relative: mean={mean_rel:.1f} ± {std_rel:.1f}, median={median_rel:.1f}")
                print(f"               before learning: {before} ({100*before/len(group_data):.1f}%)")
                print(f"               after learning: {after} ({100*after/len(group_data):.1f}%)")


def plot_cell_psth_split_by_inflection(ax, mouse_id, roi, inflection_trial, reward_group='R+'):
    """
    Plot PSTH of whisker stimulus responses split by inflection point.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    mouse_id : str
        Mouse identifier§
    roi : int
        Cell ROI number
    inflection_trial : int
        Trial index of sigmoid inflection
    reward_group : str, optional
        Reward group ('R+' or 'R-') for color coding (default: 'R+')
    """

    # Load xarray data (baseline-subtracted)¨
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(
        mouse_id, folder, 'tensor_xarray_learning_data.nc', substracted=True)

    # Filter for this cell
    xarr_cell = xarr.sel(cell=xarr['roi'] == roi)
    xarr_cell = xarr_cell.sel(trial=xarr_cell['day'] == 0)

    # Filter for whisker stimulus trials (whisker_stim == 1)
    whisker_trials = xarr_cell.sel(trial=xarr_cell['whisker_stim'] == 1)

    # Get trial_w indices
    trial_w_indices = whisker_trials['trial_w'].values

    # Split trials by inflection point
    before_mask = trial_w_indices <= inflection_trial
    after_mask = trial_w_indices > inflection_trial

    trials_before = whisker_trials.isel(trial=before_mask)
    trials_after = whisker_trials.isel(trial=after_mask)
    
    # Convert to DataFrame without computing mean 
    df_before = trials_before.to_dataframe(name='activity').reset_index()
    df_before['activity'] = df_before['activity'] * 100  # Convert to %ΔF/F
    df_before['period'] = f'Before inflection (n={before_mask.sum()})'

    df_after = trials_after.to_dataframe(name='activity').reset_index()
    df_after['activity'] = df_after['activity'] * 100
    df_after['period'] = f'After inflection (n={after_mask.sum()})'

    # Combine
    df_combined = pd.concat([df_before, df_after], ignore_index=True)

    # Use reward-based colors for after-inflection trace
    if reward_group == 'R+':
        colors = ['gray', reward_palette[1]]  # Gray before, green after (R+)
    else:  # R-
        colors = ['gray', reward_palette[0]]  # Light gray before, magenta after (R-)

    # Plot with seaborn - let it compute mean and CI across trials
    sns.lineplot(
        data=df_combined, x='time', y='activity', hue='period',
        errorbar='ci', ax=ax, palette=colors
    )

    # Styling
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='Stimulus onset')
    ax.set_xlabel('Time from stimulus (s)', fontsize=9)
    ax.set_ylabel('Activity (%ΔF/F)', fontsize=9)
    ax.set_title('Whisker Stimulus Response', fontsize=10)
    ax.legend(loc='best', frameon=False, fontsize=8)
    sns.despine(ax=ax)


def plot_behavior_learning_curves(ax, mouse_id, behavior_table, reward_group):
    """
    Plot behavioral learning curves for a specific mouse.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    mouse_id : str
        Mouse identifier
    behavior_table : pd.DataFrame
        Behavior table with learning curves
    reward_group : str
        Reward group ('R+' or 'R-')
    """
    # Filter for this mouse, day 0, whisker trials
    mouse_data = behavior_table[
        (behavior_table['mouse_id'] == mouse_id) &
        (behavior_table['day'] == 0) &
        (behavior_table['whisker_stim'] == 1)
    ].reset_index(drop=True)

    if len(mouse_data) == 0:
        ax.text(0.5, 0.5, 'No behavior data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    # Extract learning curves
    trial_w = mouse_data['trial_w'].values
    learning_curve_w = mouse_data['learning_curve_w'].values.astype(float)
    learning_curve_w_ci_low = mouse_data['learning_curve_w_ci_low'].values.astype(float)
    learning_curve_w_ci_high = mouse_data['learning_curve_w_ci_high'].values.astype(float)
    learning_curve_chance = mouse_data['learning_curve_chance'].values.astype(float)

    # Colors based on reward group
    w_color = behavior_palette[3] if reward_group == 'R+' else behavior_palette[2]
    ns_color = behavior_palette[5]

    # Plot learning curves
    ax.plot(trial_w, learning_curve_w, color=w_color, linewidth=2, label='Whisker')
    ax.fill_between(trial_w, learning_curve_w_ci_low, learning_curve_w_ci_high,
                     color=w_color, alpha=0.2)
    ax.plot(trial_w, learning_curve_chance, color=ns_color, linewidth=2, label='No stim')

    # Add learning trial vertical line
    learning_trial = mouse_data['learning_trial'].values[0]
    if not pd.isna(learning_trial):
        ax.axvline(learning_trial, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    # Styling
    ax.set_ylim([0, 1])
    ax.set_xlabel('Whisker Trial', fontsize=9)
    ax.set_ylabel('Lick Probability', fontsize=9)
    ax.set_title('Behavioral Learning Curve', fontsize=10)
    ax.legend(loc='best', frameon=False, fontsize=8)
    sns.despine(ax=ax)


def plot_5day_mapping_psth(axes, mouse_id, roi):
    """
    Plot mapping trial PSTH across 5 days for a single cell.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        List of 5 axes (one per day)
    mouse_id : str
        Mouse identifier
    roi : int
        Cell ROI number
    """
    # Load mapping data
    folder = os.path.join(io.processed_dir, 'mice')
    file_name_mapping = 'tensor_xarray_mapping_data.nc'
 
    xarr_mapping = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name_mapping, substracted=True)
    xarr_mapping = xarr_mapping.sel(cell=xarr_mapping['roi'] == roi)
    xarr_mapping.load()

    # Plot each day
    for idx, day in enumerate(DAYS_LEARNING):
        ax = axes[idx]

        # Select day data
        day_data = xarr_mapping.sel(trial=xarr_mapping['day'] == day)

        if len(day_data.trial) > 0:
            # Convert to DataFrame for seaborn
            df = day_data.to_dataframe(name='activity').reset_index()
            df['activity'] = df['activity'] * 100  # Convert to %

            # Plot with CI
            sns.lineplot(data=df, x='time', y='activity', errorbar='ci',
                        ax=ax, color='darkorange', linewidth=1.5)

            # Styling
            ax.axvline(0, color='darkorange', linestyle='-', linewidth=1, alpha=0.5)
            ax.set_title(f'Day {day}\n(n={len(day_data.trial)})', fontsize=9)
            ax.set_ylabel('ΔF/F (%)', fontsize=8)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Day {day}', fontsize=9)

        sns.despine(ax=ax)



def create_cell_pdf_report(results_df, output_dir, pdf_name, n_cells=50, sort_by='lmi'):
    """
    Create PDF report with individual cell plots showing raw data and sigmoid fits.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe (should be pre-filtered for significance)
    output_dir : str
        Output directory
    pdf_name : str
        Name of PDF file to create
    n_cells : int
        Number of top cells to include in report (default: 50)
    sort_by : str
        Column name to sort by (default: 'lmi'). Use 'combined_score' for
        quality-based sorting.
    """
    print(f"\n  Generating {pdf_name} for top {n_cells} cells (sorted by {sort_by})...")

    # Use all cells and sort by specified column (already filtered)
    results_sorted = results_df.sort_values(sort_by, ascending=False).head(n_cells)

    # Load behavior table for learning curves
    behavior_path = io.adjust_path_to_host(
        r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
        r'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    )
    behavior_table = pd.read_csv(behavior_path)

    # Create PDF
    pdf_path = os.path.join(output_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        for idx, (_, row) in enumerate(results_sorted.iterrows()):
            print(f"  Plotting cell {idx+1}/{len(results_sorted)}: {row['mouse_id']}_{row['roi']}")

            # Load raw data for this cell
            responses, trial_indices, roi_ids = load_day0_data(
                row['mouse_id'], RESPONSE_TYPE, RESPONSE_WIN
            )

            # Find this cell's index
            cell_idx = np.where(roi_ids == row['roi'])[0][0]
            y = responses[cell_idx, :]
            x = trial_indices

            # Create figure with expanded layout
            fig = plt.figure(figsize=(16, 12), dpi=150)
            gs = fig.add_gridspec(
                4, 5,
                hspace=0.45,
                wspace=0.35,
                height_ratios=[1, 0.8, 1, 0.9],
                top=0.96,   # shift the grid up on the page
                bottom=0.06
            )

            # Row 0: Sigmoid fit (full width)
            ax_main = fig.add_subplot(gs[1, :])

            # Plot raw data
            color = reward_palette[1] if row['reward_group'] == 'R+' else reward_palette[0]
            ax_main.scatter(x, y * 100, alpha=0.5, s=30, color=color, label='Raw data')

            # Fit sigmoid model
            sigmoid_fit = fit_sigmoid_model(x, y)

            if sigmoid_fit is not None and sigmoid_fit.get('fit_success', False):
                x_fit = sigmoid_fit['x_clean']
                y_fit = sigmoid_fit['predictions']
                ax_main.plot(x_fit, y_fit * 100, 'darkorange', linewidth=3, label='Sigmoid fit')

                inflexion = row['inflection']
                ax_main.axvline(inflexion, color='darkorange', linestyle='-',
                                linewidth=2, alpha=0.8, label='Inflexion point')

            ax_main.set_xlabel('Whisker Trial Number (trial_w)', fontsize=12)
            ax_main.set_ylabel('Response (ΔF/F0 %)', fontsize=12)
            ax_main.legend(loc='best', fontsize=10)
            ax_main.grid(True, alpha=0.3)
            sns.despine(ax=ax_main)

            # Row 1: Behavior learning curves (full width)
            ax_behavior = fig.add_subplot(gs[0, :])
            plot_behavior_learning_curves(ax_behavior, row['mouse_id'], behavior_table,
                                         row['reward_group'])

            # Row 2: Info panel (left) and Day 0 PSTH (right)
            ax_info = fig.add_subplot(gs[2, 0:2])
            ax_info.axis('off')

            # Format learning trial info
            learning_trial_text = ""
            if 'learning_trial' in row and not pd.isna(row['learning_trial']):
                learning_trial_text = f"""
Learning trial: {row['learning_trial']:.0f}
Inflection rel. to learning: {row['inflection_relative']:.1f} trials
"""

            # Format significance test results
            p_vs_flat = row.get('p_value_vs_flat', row['p_value'])
            p_vs_linear = row.get('p_value_vs_linear', np.nan)

            sig_vs_flat = 'YES' if p_vs_flat < ALPHA else 'NO'
            sig_vs_linear = 'YES' if (not pd.isna(p_vs_linear) and p_vs_linear < ALPHA) else 'NO'

            info_text = f"""
LMI: {row['lmi']:.3f}, p: {row['lmi_p']:.3f}

SIGNIFICANCE TESTS:
Sigmoid vs Flat: {p_vs_flat:.4e} ({sig_vs_flat})
Sigmoid vs Linear: {p_vs_linear:.4e} ({sig_vs_linear})

Pseudo-R²: {row['pseudo_r2']:.4f}

SIGMOID PARAMETERS:
Amplitude: {row['amplitude']*100:.2f}% ΔF/F
Inflection: {row['inflection']:.1f} trials
Baseline: {row['baseline']*100:.2f}% ΔF/F
Max: {row['max_val']*100:.2f}% ΔF/F{learning_trial_text}
            """

            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

            # Day 0 PSTH (before/after inflection)
            ax_psth = fig.add_subplot(gs[2, 2:])
            inflection_trial = row['inflection']
            plot_cell_psth_split_by_inflection(ax_psth, row['mouse_id'], row['roi'],
                                                inflection_trial,
                                                reward_group=row['reward_group'])

            # Row 3: 5-day mapping PSTH (one panel per day)
            # Create 5 axes that share the same y-axis
            ax0 = fig.add_subplot(gs[3, 0])
            axes_5day = [ax0] + [fig.add_subplot(gs[3, i], sharey=ax0) for i in range(1, 5)]
            plot_5day_mapping_psth(axes_5day, row['mouse_id'], row['roi'])

            # Overall title
            fig.suptitle(f'Cell Plasticity Report #{idx+1} - {row["mouse_id"]}_{row["roi"]} ({row["reward_group"]})',
                        fontsize=16, fontweight='bold', y=0.995)

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  ✓ PDF report saved to: {pdf_path}")


def plot_example_cell_plasticity(mouse_id, roi, results_df, output_dir):
    """
    Generate example cell plasticity plot showing behavior and sigmoid fit.

    Reproduces the first two plots from PDF report (behavior on top, sigmoid fit below)
    in a format suitable for a figure panel.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    roi : int
        Cell ROI number
    results_df : pd.DataFrame
        Results dataframe with sigmoid fit parameters
    output_dir : str
        Output directory for saving figure
    """
    print(f"\n  Generating example cell plasticity plot for {mouse_id}_{roi}...")

    # Find this cell in results
    cell_data = results_df[(results_df['mouse_id'] == mouse_id) & (results_df['roi'] == roi)]

    if len(cell_data) == 0:
        print(f"  ERROR: Cell {mouse_id}_{roi} not found in results!")
        return

    row = cell_data.iloc[0]

    # Load behavior table for learning curves
    behavior_path = io.adjust_path_to_host(
        r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
        r'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    )
    behavior_table = pd.read_csv(behavior_path)

    # Load raw data for this cell
    responses, trial_indices, roi_ids = load_day0_data(
        mouse_id, RESPONSE_TYPE, RESPONSE_WIN
    )

    # Find this cell's index
    cell_idx = np.where(roi_ids == roi)[0][0]
    y = responses[cell_idx, :]
    x = trial_indices

    # Create figure with 2 rows (behavior on top, sigmoid fit below)
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.0)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), dpi=150, height_ratios=[1, 1.2])

    # Top panel: Behavior learning curve
    ax_behavior = axes[0]
    plot_behavior_learning_curves(ax_behavior, mouse_id, behavior_table, row['reward_group'])

    # Bottom panel: Sigmoid fit with raw data
    ax_sigmoid = axes[1]

    # Plot raw data
    color = reward_palette[1] if row['reward_group'] == 'R+' else reward_palette[0]
    ax_sigmoid.scatter(x, y * 100, alpha=0.5, s=30, color=color, label='Raw data')

    # Fit sigmoid model
    sigmoid_fit = fit_sigmoid_model(x, y)

    if sigmoid_fit is not None and sigmoid_fit.get('fit_success', False):
        x_fit = sigmoid_fit['x_clean']
        y_fit = sigmoid_fit['predictions']
        ax_sigmoid.plot(x_fit, y_fit * 100, 'darkorange', linewidth=3, label='Sigmoid fit')

        inflexion = row['inflection']
        ax_sigmoid.axvline(inflexion, color='darkorange', linestyle='-',
                          linewidth=2, alpha=0.8, label='Inflection point')

    ax_sigmoid.set_xlabel('Whisker Trial Number', fontsize=10)
    ax_sigmoid.set_ylabel('Response (% ΔF/F)', fontsize=10)
    ax_sigmoid.set_title(f'Trial-by-trial plasticity', fontsize=10)
    ax_sigmoid.legend(loc='best', fontsize=8)
    ax_sigmoid.grid(True, alpha=0.3)
    sns.despine(ax=ax_sigmoid)

    # Overall title
    fig.suptitle(f'{mouse_id}_{roi} ({row["reward_group"]})',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    filename = f'example_cell_plasticity_{mouse_id}_{roi}.svg'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, format='svg', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Example cell plot saved to: {save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(run_fitting=RUN_FITTING, generate_pdfs=GENERATE_PDFS):
    """
    Main execution function.

    Parameters
    ----------
    run_fitting : bool, optional
        If True, run sigmoid fitting and amplitude computation (default: True).
        If False, load existing results from CSV files.
    generate_pdfs : bool, optional
        If True, generate single-cell PDF reports (default: True).
        If False, skip PDF generation.
    """
    print("="*70)
    print("SINGLE-CELL PLASTICITY ANALYSIS - DAY 0")
    print("="*70)

    if run_fitting:
        print("\nMode: Running sigmoid fitting and amplitude computation")

        # Load mice list
        _, _, mice, db = io.select_sessions_from_db(
            io.db_path, io.nwb_dir, two_p_imaging='yes'
        )

        print(f"\nProcessing {len(mice)} mice in parallel using {N_CORES} cores...")

        # Process all mice in parallel
        all_results = Parallel(n_jobs=N_CORES, verbose=10)(
            delayed(process_mouse)(
                mouse_id,
                response_type=RESPONSE_TYPE,
                response_win=RESPONSE_WIN,
                min_trials=MIN_TRIALS,
                amplitude_type=AMPLITUDE_TYPE
            )
            for mouse_id in mice
        )

        # Filter out None results and empty dataframes
        all_results = [r for r in all_results if r is not None and len(r) > 0]

        # Combine results
        results_df = pd.concat(all_results, ignore_index=True)

        # Add LMI information
        lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
        results_df = results_df.merge(
            lmi_df[['mouse_id', 'roi', 'lmi', 'lmi_p']],
            on=['mouse_id', 'roi'],
            how='inner'
        )

        print(f"\nTotal cells before LMI filtering: {len(results_df)}")

        # Filter for LMI-significant cells only
        lmi_positive = results_df[results_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD].copy()
        lmi_negative = results_df[results_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD].copy()

        print(f"LMI+ cells (p >= {LMI_POSITIVE_THRESHOLD}): {len(lmi_positive)}")
        print(f"LMI- cells (p <= {LMI_NEGATIVE_THRESHOLD}): {len(lmi_negative)}")

        # Add LMI sign column
        lmi_positive['lmi_sign'] = 'Positive'
        lmi_negative['lmi_sign'] = 'Negative'

        # Combine for saving
        results_lmi = pd.concat([lmi_positive, lmi_negative], ignore_index=True)

        # Load and merge behavioral learning trial data
        learning_path = io.adjust_path_to_host(
            r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
            r'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
        )
        learning_df = pd.read_csv(learning_path)
        learning_df = learning_df[['mouse_id', 'learning_trial']].dropna(subset=['learning_trial']).drop_duplicates()

        # Merge learning_trial into ALL cells (for distribution plots)
        results_df = results_df.merge(learning_df, on='mouse_id', how='left')
        results_df['inflection_relative'] = results_df['inflection'] - results_df['learning_trial']

        # Merge learning_trial into LMI-filtered results
        results_lmi = results_lmi.merge(learning_df, on='mouse_id', how='left')

        # Compute inflection relative to learning trial
        results_lmi['inflection_relative'] = results_lmi['inflection'] - results_lmi['learning_trial']

        # Save results
        csv_path_all = os.path.join(OUTPUT_DIR, 'plasticity_results_all_cells.csv')
        results_df.to_csv(csv_path_all, index=False)
        print(f"\nSaved all cells results to {csv_path_all}")

        csv_path_lmi = os.path.join(OUTPUT_DIR, 'plasticity_results_lmi_cells.csv')
        results_lmi.to_csv(csv_path_lmi, index=False)
        print(f"Saved LMI-filtered results to {csv_path_lmi}")

    else:
        print("\nMode: Loading existing results from CSV files")

        # Load existing results
        csv_path_all = os.path.join(OUTPUT_DIR, 'plasticity_results_all_cells.csv')
        csv_path_lmi = os.path.join(OUTPUT_DIR, 'plasticity_results_lmi_cells.csv')

        if not os.path.exists(csv_path_all) or not os.path.exists(csv_path_lmi):
            raise FileNotFoundError(
                f"Results files not found. Please run with run_fitting=True first.\n"
                f"Expected files:\n  {csv_path_all}\n  {csv_path_lmi}"
            )

        results_df = pd.read_csv(csv_path_all)
        results_lmi = pd.read_csv(csv_path_lmi)

        # Backward compatibility: Add new columns if they don't exist
        if 'p_value_vs_flat' not in results_df.columns:
            print("\n  NOTE: Old CSV format detected. Adding new p-value columns for compatibility.")
            results_df['p_value_vs_flat'] = results_df['p_value']  # Old p_value was vs flat
            results_df['p_value_vs_linear'] = np.nan  # Not computed in old version

        if 'p_value_vs_flat' not in results_lmi.columns:
            results_lmi['p_value_vs_flat'] = results_lmi['p_value']
            results_lmi['p_value_vs_linear'] = np.nan

        print(f"\nLoaded all cells results from {csv_path_all}")
        print(f"Loaded LMI-filtered results from {csv_path_lmi}")
        print(f"Total cells: {len(results_df)}")
        print(f"LMI-filtered cells: {len(results_lmi)}")

    # Filter for cells based on USE_SIGMOID_FILTER parameter
    print("\n" + "="*70)
    print("APPLYING CELL FILTERS")
    print("="*70)
    print(f"\nBefore filtering: {len(results_lmi)} LMI-significant cells")

    n_significant_vs_flat = np.sum(results_lmi['p_value_vs_flat'] < ALPHA)
    n_significant_vs_linear = np.sum(results_lmi['p_value_vs_linear'] < ALPHA)
    print(f"  - Sigmoid vs Flat significant: {n_significant_vs_flat}")
    print(f"  - Sigmoid vs Linear significant: {n_significant_vs_linear}")

    # Apply sigmoid filter if requested
    if USE_SIGMOID_FILTER:
        print(f"\nUSE_SIGMOID_FILTER=True: Applying double significance filter")
        results_lmi_filtered = results_lmi[results_lmi['p_value_vs_linear'] < ALPHA].copy()
        print(f"After filtering: {len(results_lmi_filtered)} cells with BOTH LMI and Sigmoid vs Linear significance")
        print(f"  This represents {100*len(results_lmi_filtered)/len(results_lmi):.1f}% of LMI-significant cells")
        results_lmi = results_lmi_filtered
    else:
        print(f"\nUSE_SIGMOID_FILTER=False: Using all LMI-significant cells")
        print(f"Using {len(results_lmi)} LMI-significant cells for analysis")

    # Extract LMI+ and LMI- subsets from results_lmi for summary statistics
    lmi_positive = results_lmi[results_lmi['lmi_sign'] == 'Positive']
    lmi_negative = results_lmi[results_lmi['lmi_sign'] == 'Negative']

    # Quantify proportions
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (LMI + SIGMOID VS LINEAR SIGNIFICANT CELLS)")
    print("="*70)

    n_total = len(results_lmi)

    print(f"\nTotal doubly-significant cells: {n_total}")
    print(f"  - LMI+ cells: {len(lmi_positive)}")
    print(f"  - LMI- cells: {len(lmi_negative)}")

    # By reward group and LMI sign
    print("\nBy reward group and LMI sign:")
    for lmi_sign in ['Positive', 'Negative']:
        for group in ['R+', 'R-']:
            df_subset = results_lmi[(results_lmi['lmi_sign'] == lmi_sign) &
                                    (results_lmi['reward_group'] == group)]
            n_subset = len(df_subset)
            if n_subset > 0:
                print(f"  LMI{lmi_sign[0]}, {group}: {n_subset} cells")

    # Mean amplitude
    if len(results_lmi) > 0:
        print(f"\nMean amplitude (doubly-significant cells, n={len(results_lmi)}):")
        for lmi_sign in ['Positive', 'Negative']:
            for group in ['R+', 'R-']:
                df_subset = results_lmi[(results_lmi['lmi_sign'] == lmi_sign) &
                                        (results_lmi['reward_group'] == group)]
                if len(df_subset) > 0:
                    mean_amp = df_subset['amplitude'].mean()
                    std_amp = df_subset['amplitude'].std()
                    print(f"  LMI{lmi_sign[0]}, {group}: {mean_amp:.4f} ± {std_amp:.4f}")

    # Inflection timing analysis
    print("\n" + "="*70)
    print("INFLECTION TIMING RELATIVE TO BEHAVIORAL LEARNING")
    print("="*70)
    sig_with_learning = results_lmi.dropna(subset=['learning_trial'])
    if len(sig_with_learning) > 0:
        for reward in ['R+', 'R-']:
            subset = sig_with_learning[sig_with_learning['reward_group'] == reward]
            if len(subset) > 0:
                mean_rel = subset['inflection_relative'].mean()
                std_rel = subset['inflection_relative'].std()
                median_rel = subset['inflection_relative'].median()
                print(f"\n{reward}: mean={mean_rel:.2f} ± {std_rel:.2f} trials, "
                      f"median={median_rel:.2f} trials (n={len(subset)})")

                # Report how many cells have inflection before/after learning
                before = (subset['inflection_relative'] < 0).sum()
                after = (subset['inflection_relative'] >= 0).sum()
                print(f"  Before learning: {before} ({100*before/len(subset):.1f}%)")
                print(f"  After learning: {after} ({100*after/len(subset):.1f}%)")
    else:
        print("\nNo cells with learning trial data available.")

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    filter_description = "sigmoid-filtered" if USE_SIGMOID_FILTER else "LMI-significant"
    print(f"Using {filter_description} cells for analysis\n")

    # Plot inflection timing distributions
    plot_inflection_timing_distributions(results_lmi, OUTPUT_DIR, alpha=ALPHA)

    # Plot inflection timing distributions excluding early learners
    plot_inflection_timing_distributions_late_learners(results_lmi, OUTPUT_DIR, alpha=ALPHA, learning_trial_threshold=20)

    # Compute and plot pre/post inflection analysis
    if len(results_lmi) > 0:
        print("\n" + "="*70)
        print("PRE/POST INFLECTION ANALYSIS")
        print("="*70)

        psth_df, response_df = compute_pre_post_inflection_psth(
            results_lmi,
            response_win=RESPONSE_WIN,
            min_trials_per_period=3,
            psth_win=PSTH_WIN,
            n_jobs=N_CORES  # Use parallel processing across mice
        )

        if len(psth_df) > 0 and len(response_df) > 0:
            # Generate both versions: cells and mice level
            plot_pre_post_inflection_psth_and_responses(psth_df, response_df, OUTPUT_DIR, stat_level='cells')
            plot_pre_post_inflection_psth_and_responses(psth_df, response_df, OUTPUT_DIR, stat_level='mice')
        else:
            print("  WARNING: No data available for pre/post inflection analysis")
    else:
        print("\n  WARNING: No cells available for analysis")

    # Generate example cell plasticity plot
    if len(results_lmi) > 0:
        print("\n" + "="*70)
        print("EXAMPLE CELL PLASTICITY")
        print("="*70)

        # Generate plot for AR175 roi 139
        plot_example_cell_plasticity('AR175', 139, results_lmi, OUTPUT_DIR)

    # Generate PDF report
    if generate_pdfs and len(results_lmi) > 0:
        print("\n" + "="*70)
        print("GENERATING PDF REPORT")
        print("="*70)

        print(f"\n  Found {len(results_lmi)} filtered cells")
        print(f"  Generating PDF for top 50 cells sorted by LMI (descending)...")

        # Sort by LMI in descending order
        results_lmi_sorted = results_lmi.sort_values('lmi', ascending=False)

        create_cell_pdf_report(
            results_lmi_sorted, OUTPUT_DIR,
            'plasticity_cells_top100_by_lmi.pdf',
            n_cells=100,
            sort_by='lmi'
        )
    elif generate_pdfs:
        print("\n  WARNING: No cells available for PDF report")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
