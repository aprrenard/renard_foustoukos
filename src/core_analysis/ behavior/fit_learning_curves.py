"""
Learning Curve Fitting Pipeline
================================

This script fits learning curves to behavioral data using a Bayesian Gaussian Random Walk model.
The fitted curves provide a smoothed estimate of trial-by-trial learning probability and identify
the trial at which learning occurs (defined as sustained performance above chance level).

Main Functions:
--------------
1. fit_learning_curve(): Fits a Gaussian Random Walk model to binary trial outcomes using PyMC
2. compute_learning_curves(): Applies curve fitting to all sessions in the dataset
3. compute_learning_trial(): Identifies the trial where learning occurs (performance > chance)
4. plot_learning_curves_pdf(): Visualizes fitted curves for all sessions

Fitting Procedure:
-----------------
The fitting uses a Bayesian approach with the following model structure:

1. **Latent States**: Trial-by-trial lick probability is modeled as a Gaussian Random Walk (GRW):
   - x[t] ~ GaussianRandomWalk(mu=0, sigma)
   - This allows smooth evolution of probability across trials
   - The random walk captures gradual learning dynamics

2. **Observation Model**: Binary outcomes are related to latent states via logistic transformation:
   - p[t] = logit^-1(x[t])  # Convert unbounded states to [0,1] probabilities
   - outcome[t] ~ Bernoulli(p[t])  # Binary lick/no-lick outcome

3. **Hyperpriors**:
   - tau ~ Gamma(alpha, beta)  # Precision of random walk
   - sigma = 1/sqrt(tau)  # Standard deviation of transitions

4. **Inference**: MCMC sampling (1000 samples, 4 chains) provides posterior distributions
   - Posterior mean gives the fitted learning curve
   - Credible intervals quantify uncertainty

Learning Trial Detection:
------------------------
The "learning trial" is defined as the first trial where:
1. The lower confidence bound exceeds the interpolated false alarm rate (chance level)
2. This criterion is maintained for n consecutive trials (default: 10)

This ensures robust detection that accounts for uncertainty and avoids spurious hits.

Author: Anthony Renard
Date: 2026-02-16
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pymc as pm
import scipy as sp

# Add project paths
sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette, stim_palette


def fit_learning_curve(outcomes, alpha=1, beta=1, n_samples=1000, n_tune=10,
                       n_chains=4, n_cores=8, conf_int=80):
    """
    Fit a Gaussian Random Walk model to binary trial outcomes using Bayesian inference.

    This function models trial-by-trial learning as a smooth probabilistic process where
    the lick probability evolves gradually over trials according to a Gaussian Random Walk.

    Parameters:
    ----------
    outcomes : array-like
        Binary outcomes for each trial (1 = lick/hit, 0 = no-lick/miss)
        Shape: (n_trials,)

    alpha : float, default=1
        Shape parameter for Gamma prior on precision (tau)
        Higher values = stronger prior belief about transition smoothness

    beta : float, default=1
        Rate parameter for Gamma prior on precision (tau)
        Controls the scale of the precision prior

    n_samples : int, default=1000
        Number of MCMC samples to draw from posterior
        More samples = better posterior approximation but slower

    n_tune : int, default=10
        Number of tuning steps for MCMC sampler
        Used to adapt step sizes before sampling

    n_chains : int, default=4
        Number of independent MCMC chains to run
        Multiple chains help diagnose convergence

    n_cores : int, default=8
        Number of CPU cores to use for parallel chain sampling

    conf_int : int, default=80
        Confidence interval percentage for credible intervals (e.g., 80 = 10th to 90th percentile)

    Returns:
    -------
    p_samples : ndarray or None
        Posterior samples of lick probability for each trial
        Shape: (n_samples * n_chains, n_trials)
        Returns None if no trials

    p_mean : ndarray or None
        Posterior mean lick probability for each trial (fitted learning curve)
        Shape: (n_trials,)
        Returns None if no trials

    p_low : ndarray or None
        Lower credible interval bound for each trial
        Shape: (n_trials,)
        Returns None if no trials

    p_high : ndarray or None
        Upper credible interval bound for each trial
        Shape: (n_trials,)
        Returns None if no trials

    Model Structure:
    ---------------
    The generative model is:

    1. Precision prior:
       tau ~ Gamma(alpha, beta)
       sigma = 1 / sqrt(tau)

    2. Latent random walk (in logit space):
       x[0] ~ Normal(0, sigma)
       x[t] ~ Normal(x[t-1], sigma) for t > 0

    3. Transform to probability space:
       p[t] = 1 / (1 + exp(-x[t]))  # Inverse logit

    4. Observation likelihood:
       outcome[t] ~ Bernoulli(p[t])

    Notes:
    -----
    - The random walk in logit space ensures p[t] stays in [0, 1] without constraints
    - Smoothness is controlled by sigma: small sigma = smooth curves, large sigma = noisy
    - MCMC sampling may produce warnings about divergences if the model is poorly specified
    """
    n_trials = len(outcomes)

    # Handle edge case: no trials to fit
    if n_trials == 0:
        return None, None, None, None

    # Build Bayesian model
    with pm.Model() as model:
        # Prior on precision (inverse variance) of random walk transitions
        # Gamma distribution is conjugate prior for precision parameters
        tau = pm.Gamma("tau", alpha=alpha, beta=beta)

        # Convert precision to standard deviation for interpretability
        sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))

        # Define latent states as a Gaussian Random Walk
        # Each state x[t] represents the logit-transformed lick probability
        # mu=0 means the random walk has no drift (unbiased evolution)
        x = pm.GaussianRandomWalk("x", mu=0, sigma=sigma, shape=n_trials)

        # Transform latent states to probabilities via inverse logit
        # This ensures probabilities stay in (0, 1) without explicit constraints
        p = pm.Deterministic("p", pm.math.invlogit(x))

        # Likelihood: observed binary outcomes follow Bernoulli distribution
        # This connects the latent probability p to the observed data
        obs = pm.Bernoulli("obs", p=p, observed=outcomes)

        # Perform MCMC sampling to approximate posterior distribution
        # Multiple chains help diagnose convergence issues
        trace = pm.sample(n_samples, tune=n_tune, cores=n_cores, chains=n_chains,
                         progressbar=False, return_inferencedata=True)

    # Extract posterior samples for lick probabilities
    # Shape: (n_chains, n_samples, n_trials) -> flatten to (n_chains * n_samples, n_trials)
    p_samples = trace.posterior["p"].values.reshape(-1, n_trials)

    # Compute summary statistics across posterior samples
    p_mean = np.mean(p_samples, axis=0)  # Posterior mean (fitted curve)

    # Compute credible intervals (Bayesian analog of confidence intervals)
    lower_percentile = (100 - conf_int) / 2
    upper_percentile = 100 - (100 - conf_int) / 2
    p_low, p_high = np.percentile(p_samples, [lower_percentile, upper_percentile], axis=0)

    return p_samples, p_mean, p_low, p_high


def compute_learning_curves(table, days_to_fit=[-2, -1, 0, 1, 2]):
    """
    Fit learning curves to all sessions in the behavioral data table.

    This function applies the Gaussian Random Walk fitting procedure to each session
    and each stimulus type (whisker, auditory, no-stim), adding the fitted curves
    and confidence intervals as new columns to the table.

    Parameters:
    ----------
    table : pd.DataFrame
        Behavioral data table with columns:
        - session_id: unique session identifier
        - day: training day relative to learning day (e.g., -2, -1, 0, 1, 2)
        - whisker_stim: binary flag for whisker stimulation trials
        - auditory_stim: binary flag for auditory stimulation trials
        - no_stim: binary flag for no-stimulation trials
        - outcome_w: binary outcome for whisker trials (1=lick, 0=no-lick)
        - outcome_a: binary outcome for auditory trials
        - outcome_c: binary outcome for no-stim trials (catch trials)

    days_to_fit : list, default=[-2, -1, 0, 1, 2]
        Which training days to include in fitting
        Typically includes pre-training and post-training days

    Returns:
    -------
    table : pd.DataFrame
        Input table with added columns for each stimulus type:
        - learning_curve_w: fitted whisker learning curve (posterior mean)
        - learning_curve_w_ci_low: lower 80% credible interval
        - learning_curve_w_ci_high: upper 80% credible interval
        - learning_curve_a: fitted auditory learning curve
        - learning_curve_a_ci_low: lower 80% credible interval
        - learning_curve_a_ci_high: upper 80% credible interval
        - learning_curve_ns: fitted no-stim learning curve
        - learning_curve_ns_ci_low: lower 80% credible interval
        - learning_curve_ns_ci_high: upper 80% credible interval

    Processing Steps:
    ----------------
    For each session and each stimulus type:
    1. Extract binary outcomes for that stimulus type
    2. Fit Gaussian Random Walk model to outcomes
    3. Store fitted curve (posterior mean) and credible intervals
    4. Add fitted values to corresponding rows in the table

    Notes:
    -----
    - Sessions are processed independently
    - Each stimulus type is fitted separately (no cross-stimulus information sharing)
    - Progress is printed to console for monitoring long runs
    - If fitting fails for a session/stimulus, those entries remain NaN
    """
    session_list = table.session_id.unique()

    for session in session_list:
        print(f'Processing session {session}...')

        # Extract data for this session (restrict to specified days)
        data = table.loc[table.day.isin(days_to_fit)].reset_index(drop=True).copy()

        # Fit whisker trials
        # ------------------
        data_w = data[(data.session_id == session) & (data.whisker_stim == 1)].reset_index(drop=True)
        outcomes = data_w.outcome_w.values
        p_samples_w, p_mean_w, p_low_w, p_high_w = fit_learning_curve(outcomes)

        # Fit auditory trials
        # -------------------
        data_a = data[(data.session_id == session) & (data.auditory_stim == 1)].reset_index(drop=True)
        outcomes = data_a.outcome_a.values
        p_samples_a, p_mean_a, p_low_a, p_high_a = fit_learning_curve(outcomes)

        # Fit no-stim trials (catch trials)
        # ---------------------------------
        data_ns = data[(data.session_id == session) & (data.no_stim == 1)].reset_index(drop=True)
        outcomes = data_ns.outcome_c.values
        p_samples_ns, p_mean_ns, p_low_ns, p_high_ns = fit_learning_curve(outcomes)

        # Store fitted curves in table
        # ----------------------------
        # Only update if fitting succeeded (returned non-None values)

        if p_mean_w is not None:
            table.loc[(table.session_id == session) & (table.whisker_stim == 1), 'learning_curve_w'] = p_mean_w.astype(float)
            table.loc[(table.session_id == session) & (table.whisker_stim == 1), 'learning_curve_w_ci_low'] = p_low_w.astype(float)
            table.loc[(table.session_id == session) & (table.whisker_stim == 1), 'learning_curve_w_ci_high'] = p_high_w.astype(float)

        if p_mean_a is not None:
            table.loc[(table.session_id == session) & (table.auditory_stim == 1), 'learning_curve_a'] = p_mean_a.astype(float)
            table.loc[(table.session_id == session) & (table.auditory_stim == 1), 'learning_curve_a_ci_low'] = p_low_a.astype(float)
            table.loc[(table.session_id == session) & (table.auditory_stim == 1), 'learning_curve_a_ci_high'] = p_high_a.astype(float)

        if p_mean_ns is not None:
            table.loc[(table.session_id == session) & (table.no_stim == 1), 'learning_curve_ns'] = p_mean_ns.astype(float)
            table.loc[(table.session_id == session) & (table.no_stim == 1), 'learning_curve_ns_ci_low'] = p_low_ns.astype(float)
            table.loc[(table.session_id == session) & (table.no_stim == 1), 'learning_curve_ns_ci_high'] = p_high_ns.astype(float)

    return table


def compute_learning_trial(table, n_consecutive_trials=10):
    """
    Identify the trial at which learning occurs for each session.

    Learning is defined as the first trial where the fitted whisker learning curve
    exceeds the chance level (interpolated from no-stim trials) and remains above
    chance for n consecutive trials. This ensures robust detection while accounting
    for uncertainty in the fitted curves.

    Parameters:
    ----------
    table : pd.DataFrame
        Behavioral data table with fitted learning curves (from compute_learning_curves)
        Required columns:
        - session_id: unique session identifier
        - day: training day (learning trial only computed for day 0)
        - whisker_stim: binary flag for whisker trials
        - no_stim: binary flag for no-stim trials
        - start_time: trial start timestamp
        - trial_w: trial number within whisker sequence
        - learning_curve_w: fitted whisker learning curve
        - learning_curve_w_ci_low: lower credible interval for whisker curve
        - learning_curve_ns: fitted no-stim curve (chance level)

    n_consecutive_trials : int, default=10
        Number of consecutive trials that must remain above chance to define learning
        Higher values = more conservative detection, fewer false positives

    Returns:
    -------
    table : pd.DataFrame
        Input table with added columns:
        - learning_curve_chance: interpolated chance level at each whisker trial time
        - learning_trial: trial number where learning occurred (NaN if no learning detected)

    Algorithm:
    ---------
    For each day 0 session:

    1. **Extract fitted curves**:
       - p_w(t): whisker learning curve at whisker trial times
       - p_ns(t): no-stim curve at no-stim trial times
       - CI_low(t): lower confidence bound for whisker curve

    2. **Interpolate chance level**:
       - Use cubic spline to interpolate p_ns to whisker trial timestamps
       - This accounts for temporal drift in false alarm rate

    3. **Identify trials above chance**:
       - Find all trials where CI_low(t) > interpolated_chance(t)
       - Using CI_low ensures statistical rigor (not just mean)

    4. **Find first sustained learning**:
       - Scan for first sequence of n consecutive trials above chance
       - Return the trial number of the first trial in this sequence
       - If no such sequence exists, return NaN (no learning detected)

    Notes:
    -----
    - Only processes day 0 sessions (initial learning day)
    - Interpolation uses cubic splines for smooth chance level estimation
    - The chance level varies over time to account for motivational drift
    - Conservative criterion (CI_low > chance) reduces false positives
    """
    for session in table.session_id.unique():
        # Only process day 0 (learning day)
        if table.loc[(table.session_id == session), 'day'].values[0] != 0:
            continue

        print(f'Defining learning trial for session {session}...')

        # Extract whisker trial data
        data_w = table[(table.session_id == session) & (table.whisker_stim == 1)].reset_index(drop=True)

        # Extract no-stim trial data (defines chance level)
        data_ns = table[(table.session_id == session) & (table.no_stim == 1)].reset_index(drop=True)

        # Get fitted learning curves
        p_mean_w = data_w.learning_curve_w.values  # Whisker learning curve
        p_low_w = data_w.learning_curve_w_ci_low.values  # Lower confidence bound
        p_high_w = data_w.learning_curve_w_ci_high.values  # Upper confidence bound
        p_mean_ns = data_ns.learning_curve_ns.values  # No-stim curve (chance)

        # Get trial timestamps for temporal alignment
        timestamps_no_stim = data_ns.start_time.values
        timestamps_whisker = data_w['start_time'].values

        # Define temporal bounds for interpolation
        bounds_w = (min(timestamps_whisker), max(timestamps_whisker))
        bounds_n = (min(timestamps_no_stim), max(timestamps_no_stim))

        # Create interpolation grid spanning both stimulus types
        interp_range = np.linspace(
            min(bounds_w[0], bounds_n[0]),
            max(bounds_w[1], bounds_n[1]),
            len(p_mean_ns)
        )

        # Interpolate no-stim curve using cubic spline
        # This provides smooth chance level estimation across time
        interp_func = sp.interpolate.CubicSpline(
            x=interp_range,
            y=p_mean_ns,
            extrapolate=False  # Don't extrapolate beyond data range
        )

        # Evaluate interpolated chance at whisker trial times
        interp_p_far = interp_func(timestamps_whisker)

        # Identify trials where performance exceeds chance
        # Use lower CI bound for conservative detection (accounts for uncertainty)
        trials_above_chance = np.where(p_low_w > interp_p_far)[0]
        trials_above_chance = data_w.loc[trials_above_chance, 'trial_w'].values

        # Find first sequence of n consecutive trials above chance
        learning_trial = np.nan  # Default: no learning detected

        for idx in range(len(trials_above_chance)):
            # Check if enough trials remain for consecutive sequence
            if idx + n_consecutive_trials - 1 < len(trials_above_chance):
                # Check if next n trials are consecutive (diff = 1 between all pairs)
                consecutive_diffs = np.diff(trials_above_chance[idx:idx + n_consecutive_trials])
                if np.all(consecutive_diffs == 1):
                    # Found first sustained learning sequence
                    learning_trial = trials_above_chance[idx]
                    break

        # Store results in table
        table.loc[(table.session_id == session) & (table.whisker_stim == 1),
                  'learning_curve_chance'] = interp_p_far
        table.loc[(table.session_id == session), 'learning_trial'] = learning_trial

    return table


def plot_learning_curves_pdf(table, session_list, pdf_path):
    """
    Generate a multi-page PDF with learning curve plots for each session.

    This function creates one plot per session showing:
    - Raw block-averaged performance (dashed line)
    - Fitted learning curve with confidence intervals (solid line + shaded area)
    - Interpolated chance level (no-stim curve)
    - Detected learning trial (vertical dashed line)

    Parameters:
    ----------
    table : pd.DataFrame
        Behavioral data table with fitted learning curves
        Required columns:
        - session_id: unique session identifier
        - day: training day
        - reward_group: reward contingency ('R+' or 'R-')
        - whisker_stim: binary flag for whisker trials
        - trial_w: whisker trial number
        - hr_w: block-averaged hit rate for whisker trials
        - learning_curve_w: fitted whisker learning curve
        - learning_curve_w_ci_low: lower credible interval
        - learning_curve_w_ci_high: upper credible interval
        - learning_curve_chance: interpolated chance level
        - learning_trial: detected learning trial number

    session_list : list
        List of session IDs to plot
        Each session will get one page in the PDF

    pdf_path : str
        Output path for PDF file (e.g., '/path/to/learning_curves.pdf')

    Visualization Details:
    ---------------------
    For each session:
    - X-axis: Whisker trial number
    - Y-axis: Lick probability [0, 1]
    - Color: R+ = green, R- = red (from reward_palette)
    - Shaded area: 80% credible interval
    - Dashed line: block-averaged performance (raw data)
    - Solid line: fitted learning curve (smoothed)
    - Gray line: chance level (no-stim performance)
    - Black dashed vertical line: learning trial (if detected)

    Notes:
    -----
    - Only plots day 0 sessions (skips other days)
    - Uses seaborn theme for consistent styling
    - Automatically closes figures to save memory
    - Progress not printed (assumed session_list is already filtered)
    """
    with PdfPages(pdf_path) as pdf:
        for session_id in session_list:
            # Extract data for this session
            data = table.loc[table.session_id == session_id]

            # Skip if not day 0 (learning day)
            if data.day.iloc[0] != 0:
                continue

            # Get reward group and determine color
            reward_group = data.reward_group.values[0]
            color = reward_palette[0] if reward_group == 'R-' else reward_palette[1]

            # Extract whisker trial data
            d = data.loc[data.whisker_stim == 1].reset_index(drop=True)

            # Get fitted learning curve and confidence intervals
            learning_curve_w = d.learning_curve_w.values.astype(float)
            learning_ci_low = d.learning_curve_w_ci_low.values.astype(float)
            learning_ci_high = d.learning_curve_w_ci_high.values.astype(float)
            learning_chance = d.learning_curve_chance.astype(float)

            # Plot fitted learning curve (solid line)
            plt.plot(d.trial_w, learning_curve_w,
                    label='Whisker (fitted curve)', color=color, linewidth=2)

            # Plot confidence interval (shaded area)
            plt.fill_between(d.trial_w, learning_ci_low, learning_ci_high,
                            color=color, alpha=0.2, label='80% CI')

            # Plot raw block-averaged performance (dashed line)
            sns.lineplot(data=d, x='trial_w', y='hr_w',
                        color=color, legend=False, linestyle='--', linewidth=1)

            # Plot chance level (no-stim curve)
            plt.plot(d.trial_w, learning_chance,
                    label='Chance (no-stim)', color=stim_palette[2], linewidth=1.5)

            # Mark learning trial if detected
            learning_trial = data.learning_trial.values[0]
            if not pd.isna(learning_trial):
                plt.axvline(x=learning_trial, color='black', linestyle='--',
                           label=f'Learning trial ({int(learning_trial)})', linewidth=1)

            # Format plot
            plt.title(f'Session {session_id} - {reward_group}')
            plt.ylim([0, 1])
            plt.xlabel('Whisker trial')
            plt.ylabel('Lick probability')
            plt.legend(frameon=False, fontsize=8)
            sns.despine()

            # Save page and close figure
            pdf.savefig()
            plt.close()


# =============================================================================
# Main execution: Fit learning curves and save results
# =============================================================================

if __name__ == '__main__':

    # Configure plot style
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    # Load configuration paths
    db_path = io.db_path
    nwb_dir = io.nwb_dir
    stop_flag_yaml = io.stop_flags_yaml
    trial_indices_yaml = io.trial_indices_yaml

    # Select imaging mice (exclude pharmacology and optogenetics)
    experimenters = ['AR', 'GF', 'MI']
    mice_imaging = io.select_mice_from_db(
        db_path, nwb_dir,
        experimenters=experimenters,
        exclude_cols=['exclude', 'two_p_exclude'],
        optogenetic=['no', np.nan],
        pharmacology=['no', np.nan],
        two_p_imaging='yes'
    )

    # Load behavioral data table
    print("Loading behavioral data table...")
    table_file = io.adjust_path_to_host(
        r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
        'behavior_imagingmice_table_5days_cut.csv'
    )
    table = pd.read_csv(table_file)

    # Fit learning curves to all sessions
    # -----------------------------------
    print("\nFitting learning curves...")
    print("This may take several minutes per session (MCMC sampling)...\n")
    table = compute_learning_curves(table, days_to_fit=[-2, -1, 0, 1, 2])

    # Identify learning trials
    # ------------------------
    print("\nIdentifying learning trials...")
    table = compute_learning_trial(table, n_consecutive_trials=10)

    # Save updated table with fitted curves
    # -------------------------------------
    save_path = io.adjust_path_to_host(
        r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
        'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    )
    print(f"\nSaving fitted curves to:\n{save_path}")
    table.to_csv(save_path, index=False)

    # Generate PDF with learning curve plots
    # ---------------------------------------
    pdf_path = io.adjust_path_to_host(
        '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/'
        'day0_learning/behavior/learning_curves_day0.pdf'
    )
    session_list = table.session_id.unique()

    print(f"\nGenerating PDF plots:\n{pdf_path}")
    plot_learning_curves_pdf(table, session_list, pdf_path)

    print("\n" + "="*70)
    print("Learning curve fitting complete!")
    print("="*70)
    print(f"Total sessions processed: {len(session_list)}")
    print(f"Output table: {save_path}")
    print(f"Output plots: {pdf_path}")
    print("="*70)
