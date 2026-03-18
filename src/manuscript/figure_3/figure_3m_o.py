"""
Figure 3m-o: Decoding analyses

This script generates Panels m-o for Figure 3:
- Panel m: Pre vs post learning decoding accuracy per reward group
- Panel n: Pairwise day decoding with fixed pre/post decoder (accuracy matrix)
- Panel o: Does day 0 activity resemble pre or post learning? (pre vs day0 / day0 vs post)

For each panel, two CSV files are saved alongside this script:
- figure_3X_data.csv: data points displayed in the panel
- figure_3X_stats.csv: statistical test results
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

SAMPLING_RATE = 30
WIN = (0, 0.300)
BASELINE_WIN = (-1, 0)
DAYS = [-2, -1, 0, 1, 2]
N_MAP_TRIALS = 40

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_3', 'output')
RESULTS_DIR = os.path.join(io.processed_dir, 'decoding')


# ============================================================================
# Data loading and processing
# ============================================================================

def load_and_process_data(
    select_lmi=False,
    projection_type=None,
    n_min_proj=5,
):
    """
    Load imaging data for decoding analyses.

    Args:
        select_lmi: If True, restrict to LMI-significant cells
        projection_type: Cell type filter ('wS2', 'wM1', or None for all cells)
        n_min_proj: Minimum number of projection cells required to include a mouse

    Returns:
        vectors_rew: List of xarrays (one per R+ mouse), shape (cells, trials)
        vectors_nonrew: List of xarrays (one per R- mouse), shape (cells, trials)
        mice_rew: List of R+ mouse IDs
        mice_nonrew: List of R- mouse IDs
    """
    _, _, mice, db = io.select_sessions_from_db(io.db_path, io.nwb_dir, two_p_imaging='yes')
    print(mice)

    selected_cells = None
    if select_lmi:
        processed_folder = io.solve_common_paths('processed_data')
        lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))
        selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

    vectors_rew, vectors_nonrew = [], []
    mice_rew, mice_nonrew = [], []

    for mouse in mice:
        print(f"Processing mouse: {mouse}")
        folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        xarray = utils_imaging.load_mouse_xarray(mouse, folder, 'tensor_xarray_mapping_data.nc')
        # Manual baseline subtraction
        xarray = xarray - np.nanmean(
            xarray.sel(time=slice(BASELINE_WIN[0], BASELINE_WIN[1])).values,
            axis=2, keepdims=True
        )
        rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

        xarray = xarray.sel(trial=xarray['day'].isin(DAYS))

        if select_lmi and selected_cells is not None:
            selected_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
            xarray = xarray.sel(cell=xarray['roi'].isin(selected_for_mouse))

        if projection_type is not None:
            xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
            if xarray.sizes['cell'] < n_min_proj:
                print(f"Not enough cells of type {projection_type} for mouse {mouse}.")
                continue

        n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
        if np.any(n_trials < N_MAP_TRIALS):
            print(f'Not enough mapping trials for {mouse}.')
            continue

        d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-N_MAP_TRIALS, None)))
        d = d.sel(time=slice(WIN[0], WIN[1])).mean(dim='time')
        d = d.fillna(0)

        if rew_gp == 'R-':
            vectors_nonrew.append(d)
            mice_nonrew.append(mouse)
        elif rew_gp == 'R+':
            vectors_rew.append(d)
            mice_rew.append(mouse)

    print(f"Loaded {len(vectors_rew)} R+ mice and {len(vectors_nonrew)} R- mice")
    return vectors_rew, vectors_nonrew, mice_rew, mice_nonrew


# ============================================================================
# Decoding helper
# ============================================================================

def _per_mouse_cv_accuracy(vectors, label_encoder, seed=42, n_shuffles=100, n_jobs=-1):
    """
    Compute cross-validated pre vs post learning decoding accuracy per mouse.

    Uses logistic regression with 10-fold stratified CV. Chance level is
    estimated by shuffling labels n_shuffles times.

    Returns:
        accuracies: Array of CV accuracies, one per mouse
        chance_accuracies: Array of mean shuffle accuracies, one per mouse
    """
    rng = np.random.default_rng(seed)
    accuracies = []
    chance_accuracies = []

    for d in vectors:
        days_per_trial = d['day'].values
        mask = np.isin(days_per_trial, [-2, -1, 1, 2])
        X = d.values[:, mask].T
        labels = np.array(['pre' if day in [-2, -1] else 'post' for day in days_per_trial[mask]])
        y = label_encoder.transform(labels)

        clf = LogisticRegression(max_iter=5000, random_state=seed)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        fold_scores = []
        for train_idx, test_idx in cv.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf.fit(X_train, y[train_idx])
            fold_scores.append(np.mean(clf.predict(X_test) == y[test_idx]))
        accuracies.append(np.mean(fold_scores))

        def _shuffle_score(clf, X, y, cv, rng):
            y_shuff = rng.permutation(y)
            scores = []
            for train_idx, test_idx in cv.split(X, y_shuff):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                clf.fit(X_train, y_shuff[train_idx])
                scores.append(np.mean(clf.predict(X_test) == y_shuff[test_idx]))
            return np.mean(scores)

        shuffle_scores = Parallel(n_jobs=n_jobs)(
            delayed(_shuffle_score)(clf, X, y, cv, rng) for _ in range(n_shuffles)
        )
        chance_accuracies.append(np.mean(shuffle_scores))

    return np.array(accuracies), np.array(chance_accuracies)


def _significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


# ============================================================================
# Decoder weight saving (used by figure_4b and figure_4c)
# ============================================================================

def train_and_save_decoder_weights(
    vectors_rew, vectors_nonrew, mice_rew, mice_nonrew,
    pre_days=[-2, -1], post_days=[1, 2],
    seed=42,
    results_dir=RESULTS_DIR,
):
    """
    Train a fixed pre/post logistic regression decoder per mouse on mapping
    trials (Days -2/-1 vs +1/+2) and save the weights to results_dir.

    Saved file: decoder_weights.pkl
    Structure: {mouse_id: {'scaler': StandardScaler, 'clf': LogisticRegression,
                            'sign_flip': int, 'reward_group': str}}

    The sign_flip ensures that higher decision values always correspond to
    post-learning activity.
    """
    weights = {}

    for vectors, mice_list, reward_group in [
        (vectors_rew, mice_rew, 'R+'),
        (vectors_nonrew, mice_nonrew, 'R-'),
    ]:
        for d, mouse in zip(vectors, mice_list):
            day_per_trial = d['day'].values
            train_mask = np.isin(day_per_trial, pre_days + post_days)
            if np.sum(train_mask) < 4:
                print(f'  {mouse}: not enough training trials, skipping.')
                continue

            X_train = d.values[:, train_mask].T
            y_train = np.array([0 if day in pre_days else 1
                                 for day in day_per_trial[train_mask]])

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=5000, random_state=seed)
            clf.fit(X_train_scaled, y_train)

            # Ensure post > pre in decision value
            pre_mask = np.isin(day_per_trial, pre_days)
            post_mask = np.isin(day_per_trial, post_days)
            mean_dec_pre = np.mean(clf.decision_function(
                scaler.transform(d.values[:, pre_mask].T)))
            mean_dec_post = np.mean(clf.decision_function(
                scaler.transform(d.values[:, post_mask].T)))
            sign_flip = -1 if mean_dec_pre > mean_dec_post else 1

            weights[mouse] = {
                'scaler': scaler,
                'clf': clf,
                'sign_flip': sign_flip,
                'reward_group': reward_group,
            }
            print(f'  {mouse} ({reward_group}): decoder trained '
                  f'({X_train.shape[1]} cells, sign_flip={sign_flip})')

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'decoder_weights.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f'Decoder weights saved: {out_path}  ({len(weights)} mice)')
    return weights


# ============================================================================
# Panel m: Pre vs post learning decoding accuracy
# ============================================================================

def panel_m_decoding_accuracy(
    vectors_rew=None,
    vectors_nonrew=None,
    mice_rew=None,
    mice_nonrew=None,
    select_lmi=False,
    projection_type=None,
    n_shuffles=100,
    seed=42,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 3 Panel m: Pre vs post learning decoding accuracy.

    Trains a logistic regression classifier per mouse to distinguish pre-learning
    (days -2, -1) from post-learning (days +1, +2) population activity patterns.
    Compares cross-validated accuracy between R+ and R- groups.

    Saves:
        figure_3m_data.csv: per-mouse accuracy and chance accuracy
        figure_3m_stats.csv: Mann-Whitney U (R+ vs R-) and Wilcoxon (accuracy vs chance)
    """
    if vectors_rew is None or vectors_nonrew is None:
        vectors_rew, vectors_nonrew, mice_rew, mice_nonrew = load_and_process_data(
            select_lmi=select_lmi,
            projection_type=projection_type,
        )

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    le = LabelEncoder()
    le.fit(['pre', 'post'])

    print("Computing decoding accuracy for R+ mice...")
    accs_rew, chance_rew = _per_mouse_cv_accuracy(vectors_rew, le, seed=seed, n_shuffles=n_shuffles)
    print("Computing decoding accuracy for R- mice...")
    accs_nonrew, chance_nonrew = _per_mouse_cv_accuracy(vectors_nonrew, le, seed=seed, n_shuffles=n_shuffles)

    print(f"Mean accuracy R+: {np.nanmean(accs_rew):.3f} +/- {np.nanstd(accs_rew):.3f}")
    print(f"Mean accuracy R-: {np.nanmean(accs_nonrew):.3f} +/- {np.nanstd(accs_nonrew):.3f}")

    # Statistics
    stat_between, p_between = mannwhitneyu(accs_rew, accs_nonrew, alternative='two-sided')
    stat_rew, p_rew = wilcoxon(accs_rew, chance_rew, alternative='two-sided')
    stat_nonrew, p_nonrew = wilcoxon(accs_nonrew, chance_nonrew, alternative='two-sided')

    stats_rows = [
        {
            'test': 'Mann-Whitney U',
            'comparison': 'R+ vs R- accuracy',
            'statistic': stat_between,
            'p_value': p_between,
            'significance': _significance_stars(p_between),
        },
        {
            'test': 'Wilcoxon signed-rank',
            'comparison': 'R+ accuracy vs chance',
            'statistic': stat_rew,
            'p_value': p_rew,
            'significance': _significance_stars(p_rew),
        },
        {
            'test': 'Wilcoxon signed-rank',
            'comparison': 'R- accuracy vs chance',
            'statistic': stat_nonrew,
            'p_value': p_nonrew,
            'significance': _significance_stars(p_nonrew),
        },
    ]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))
    data_plot = [accs_rew, accs_nonrew]
    sns.barplot(data=data_plot, palette=reward_palette[::-1], estimator=np.nanmean,
                errorbar='ci', ax=ax)
    sns.swarmplot(data=data_plot, palette=reward_palette[::-1], alpha=0.7, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['R+', 'R-'])
    ax.set_ylabel('Cross-validated accuracy')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1)

    p_text = 'p<0.001' if p_between < 0.001 else f'p={p_between:.3f}' if p_between < 0.01 else f'p={p_between:.2f}'
    ax.text(0.5, 0.95, p_text, ha='center', va='bottom', transform=ax.transAxes, fontsize=9)

    sns.despine()
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'figure_3m.{save_format}'), format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 3m saved to: {os.path.join(output_dir, 'figure_3m.' + save_format)}")

    # Save CSVs
    data_df = pd.DataFrame({
        'mouse_id': mice_rew + mice_nonrew,
        'reward_group': ['R+'] * len(mice_rew) + ['R-'] * len(mice_nonrew),
        'accuracy': np.concatenate([accs_rew, accs_nonrew]),
        'chance_accuracy': np.concatenate([chance_rew, chance_nonrew]),
    })
    data_df.to_csv(os.path.join(OUTPUT_DIR, 'figure_3m_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(OUTPUT_DIR, 'figure_3m_stats.csv'), index=False)
    print(f"Figure 3m data/stats saved to: {OUTPUT_DIR}")


# ============================================================================
# Panel n: Pairwise day decoding with fixed pre/post decoder
# ============================================================================

def _make_symmetric_with_diag(mat):
    """Make an accuracy matrix symmetric, preserving the diagonal."""
    sym = np.full_like(mat, np.nan)
    iu = np.triu_indices_from(mat, k=1)
    sym[iu] = mat[iu]
    il = np.tril_indices_from(mat, k=-1)
    sym[il] = mat.T[il]
    sym[np.diag_indices_from(mat)] = np.diag(mat)
    return sym


def _pairwise_day_decoding_fixed_decoder_cv(vectors, days, seed=42, n_splits=10):
    """
    Train a pre vs post classifier (excluding day 0) then apply it to all day pairs.

    For each mouse, trains a logistic regression on days -2, -1 (pre) vs +1, +2 (post),
    then evaluates it on every (day_i, day_j) combination including day 0.

    Returns:
        acc_matrices: Array of shape (n_mice, n_days, n_days) with accuracy per pair
    """
    rng = np.random.default_rng(seed)
    acc_matrices = []

    for d in vectors:
        day_per_trial = d['day'].values
        day_indices = {day: np.where(day_per_trial == day)[0] for day in days}
        fold_accs = np.zeros((n_splits, len(days), len(days)))

        for k in range(n_splits):
            train_idx, test_idx = [], []
            for day in days:
                idx = day_indices[day]
                if len(idx) < n_splits:
                    continue
                idx_shuff = rng.permutation(idx)
                fold_size = len(idx) // n_splits
                test_fold = idx_shuff[k * fold_size:(k + 1) * fold_size] if k < n_splits - 1 else idx_shuff[k * fold_size:]
                train_fold = np.setdiff1d(idx, test_fold)
                train_idx.append((day, train_fold))
                test_idx.append((day, test_fold))

            # Train on pre (-2, -1) vs post (+1, +2) only
            train_days = [-2, -1, 1, 2]
            train_mask = np.concatenate([fold for day, fold in train_idx if day in train_days])
            train_labels = np.concatenate([[day] * len(fold) for day, fold in train_idx if day in train_days])
            if len(train_mask) < 2 or len(np.unique(train_labels)) < 2:
                continue
            X_train = d.values[:, train_mask].T
            y_train = np.array([0 if day <= -1 else 1 for day in train_labels])
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=5000, random_state=seed)
            clf.fit(X_train_proc, y_train)

            # Apply to all (day_i, day_j) pairs using test folds
            for i, day_i in enumerate(days):
                for j, day_j in enumerate(days):
                    test_i = [fold for day, fold in test_idx if day == day_i]
                    test_j = [fold for day, fold in test_idx if day == day_j]
                    if not test_i or not test_j:
                        continue
                    X_test = np.concatenate([d.values[:, test_i[0]].T, d.values[:, test_j[0]].T], axis=0)
                    y_test = np.array([0] * len(test_i[0]) + [1] * len(test_j[0]))
                    y_pred = clf.predict(scaler.transform(X_test))
                    fold_accs[k, i, j] = np.mean(y_pred == y_test)

        with np.errstate(invalid='ignore'):
            acc_matrices.append(np.nanmean(fold_accs, axis=0))

    return np.array(acc_matrices)


def _extract_pairwise_decoding_accuracy(acc_matrices):
    """
    Extract pre-vs-day0 and day0-vs-post summary accuracies from pairwise matrices.

    Indices: -2→0, -1→1, 0→2, +1→3, +2→4

    Returns:
        acc_pre_vs_day0: Array (n_mice,) — mean accuracy of pre days vs day 0
        acc_day0_vs_post: Array (n_mice,) — mean accuracy of day 0 vs post days
    """
    idx_pre = [0, 1]
    idx_post = [3, 4]
    idx_day0 = 2
    acc_pre_vs_day0 = np.array([
        np.nanmean([mat[i, idx_day0] for i in idx_pre]) for mat in acc_matrices
    ])
    acc_day0_vs_post = np.array([
        np.nanmean([mat[idx_day0, j] for j in idx_post]) for mat in acc_matrices
    ])
    return acc_pre_vs_day0, acc_day0_vs_post


def panel_n_pairwise_decoding(
    vectors_rew=None,
    vectors_nonrew=None,
    mice_rew=None,
    mice_nonrew=None,
    accs_rew_matrix=None,
    accs_nonrew_matrix=None,
    select_lmi=False,
    projection_type=None,
    seed=42,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 3 Panel n: Pairwise day decoding with fixed pre/post decoder.

    Trains a classifier to distinguish pre (-2, -1) from post (+1, +2) activity,
    then applies it to decode all pairs of days (including day 0) to show when
    activity transitions from pre-like to post-like.

    Saves:
        figure_3n_data.csv: per-mouse accuracy for each (day_i, day_j) pair
        figure_3n_stats.csv: Mann-Whitney U (R+ vs R-) for each day pair
    """
    if vectors_rew is None or vectors_nonrew is None:
        vectors_rew, vectors_nonrew, mice_rew, mice_nonrew = load_and_process_data(
            select_lmi=select_lmi,
            projection_type=projection_type,
        )

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    if accs_rew_matrix is None:
        print("Computing pairwise day decoding for R+ mice...")
        accs_rew_matrix = _pairwise_day_decoding_fixed_decoder_cv(vectors_rew, DAYS, seed=seed)
    if accs_nonrew_matrix is None:
        print("Computing pairwise day decoding for R- mice...")
        accs_nonrew_matrix = _pairwise_day_decoding_fixed_decoder_cv(vectors_nonrew, DAYS, seed=seed)

    mean_accs_rew = np.nanmean(accs_rew_matrix, axis=0)
    mean_accs_nonrew = np.nanmean(accs_nonrew_matrix, axis=0)
    mean_accs_rew_sym = _make_symmetric_with_diag(mean_accs_rew)
    mean_accs_nonrew_sym = _make_symmetric_with_diag(mean_accs_nonrew)

    day_labels = [str(d) for d in DAYS]

    # Statistics: Mann-Whitney U (R+ vs R-) for each day pair
    stats_rows = []
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            r_plus = accs_rew_matrix[:, i, j]
            r_minus = accs_nonrew_matrix[:, i, j]
            if np.all(np.isnan(r_plus)) or np.all(np.isnan(r_minus)):
                continue
            stat, p = mannwhitneyu(
                r_plus[~np.isnan(r_plus)], r_minus[~np.isnan(r_minus)],
                alternative='two-sided',
            )
            stats_rows.append({
                'test': 'Mann-Whitney U',
                'comparison': f'R+ vs R- day {day_i:+d} vs day {day_j:+d}',
                'day_i': day_i,
                'day_j': day_j,
                'statistic': stat,
                'p_value': p,
                'significance': _significance_stars(p),
            })

    # Plot
    vmin = 0.5
    vmax = max(np.nanmax(mean_accs_rew_sym), np.nanmax(mean_accs_nonrew_sym))

    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    sns.heatmap(mean_accs_rew_sym, annot=True, fmt='.2f',
                xticklabels=day_labels, yticklabels=day_labels,
                ax=ax0, cmap='viridis', vmin=vmin, vmax=vmax,
                mask=np.isnan(mean_accs_rew_sym), cbar=False)
    ax0.set_title(f'R+ (N={len(mice_rew)} mice)')
    ax0.set_xlabel('Day')
    ax0.set_ylabel('Day')

    sns.heatmap(mean_accs_nonrew_sym, annot=True, fmt='.2f',
                xticklabels=day_labels, yticklabels=day_labels,
                ax=ax1, cmap='viridis', vmin=vmin, vmax=vmax,
                mask=np.isnan(mean_accs_nonrew_sym), cbar=False)
    ax1.set_title(f'R- (N={len(mice_nonrew)} mice)')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Day')

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_ticks([vmin, vmax])
    cbar.set_label('Decoding accuracy')

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'figure_3n.{save_format}'), format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 3n saved to: {os.path.join(output_dir, 'figure_3n.' + save_format)}")

    # Save CSVs
    records = []
    for group, matrices, mice_ids in [('R+', accs_rew_matrix, mice_rew), ('R-', accs_nonrew_matrix, mice_nonrew)]:
        for m_idx, mouse_id in enumerate(mice_ids):
            for i, day_i in enumerate(DAYS):
                for j, day_j in enumerate(DAYS):
                    records.append({
                        'mouse_id': mouse_id,
                        'reward_group': group,
                        'day_i': day_i,
                        'day_j': day_j,
                        'accuracy': matrices[m_idx, i, j],
                    })
    pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, 'figure_3n_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(OUTPUT_DIR, 'figure_3n_stats.csv'), index=False)
    print(f"Figure 3n data/stats saved to: {OUTPUT_DIR}")


# ============================================================================
# Panel o: Does day 0 look more like pre or post?
# ============================================================================

def panel_o_day0_classification(
    vectors_rew=None,
    vectors_nonrew=None,
    mice_rew=None,
    mice_nonrew=None,
    accs_rew_matrix=None,
    accs_nonrew_matrix=None,
    select_lmi=False,
    projection_type=None,
    seed=42,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 3 Panel o: Does day 0 activity resemble pre or post learning?

    Extracts from the fixed-decoder pairwise matrix:
    - Pre vs Day 0: average decoding accuracy between pre days (-2, -1) and day 0
    - Day 0 vs Post: average decoding accuracy between day 0 and post days (+1, +2)
    Plots both comparisons per reward group.

    Saves:
        figure_3o_data.csv: per-mouse pre_vs_day0 and day0_vs_post accuracies
        figure_3o_stats.csv: Wilcoxon (each condition vs chance and paired) +
                             Mann-Whitney U (R+ vs R- per comparison)
    """
    if vectors_rew is None or vectors_nonrew is None:
        vectors_rew, vectors_nonrew, mice_rew, mice_nonrew = load_and_process_data(
            select_lmi=select_lmi,
            projection_type=projection_type,
        )

    if accs_rew_matrix is None:
        print("Computing pairwise day decoding for R+ mice...")
        accs_rew_matrix = _pairwise_day_decoding_fixed_decoder_cv(vectors_rew, DAYS, seed=seed)
    if accs_nonrew_matrix is None:
        print("Computing pairwise day decoding for R- mice...")
        accs_nonrew_matrix = _pairwise_day_decoding_fixed_decoder_cv(vectors_nonrew, DAYS, seed=seed)

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    acc_pre_vs_day0_rew, acc_day0_vs_post_rew = _extract_pairwise_decoding_accuracy(accs_rew_matrix)
    acc_pre_vs_day0_nonrew, acc_day0_vs_post_nonrew = _extract_pairwise_decoding_accuracy(accs_nonrew_matrix)

    # Statistics
    stats_rows = []
    for group, acc_pre, acc_post in [
        ('R+', acc_pre_vs_day0_rew, acc_day0_vs_post_rew),
        ('R-', acc_pre_vs_day0_nonrew, acc_day0_vs_post_nonrew),
    ]:
        stat, p = wilcoxon(acc_pre, acc_post, alternative='two-sided')
        stats_rows.append({'test': 'Wilcoxon signed-rank', 'reward_group': group,
                           'comparison': 'pre_vs_day0 vs day0_vs_post (paired)',
                           'statistic': stat, 'p_value': p, 'significance': _significance_stars(p)})
        stat, p = wilcoxon(acc_pre - 0.5, alternative='two-sided')
        stats_rows.append({'test': 'Wilcoxon signed-rank', 'reward_group': group,
                           'comparison': 'pre_vs_day0 vs chance',
                           'statistic': stat, 'p_value': p, 'significance': _significance_stars(p)})
        stat, p = wilcoxon(acc_post - 0.5, alternative='two-sided')
        stats_rows.append({'test': 'Wilcoxon signed-rank', 'reward_group': group,
                           'comparison': 'day0_vs_post vs chance',
                           'statistic': stat, 'p_value': p, 'significance': _significance_stars(p)})

    for comparison, rew_vals, nonrew_vals in [
        ('pre_vs_day0', acc_pre_vs_day0_rew, acc_pre_vs_day0_nonrew),
        ('day0_vs_post', acc_day0_vs_post_rew, acc_day0_vs_post_nonrew),
    ]:
        stat, p = mannwhitneyu(rew_vals, nonrew_vals, alternative='two-sided')
        stats_rows.append({'test': 'Mann-Whitney U', 'reward_group': 'R+ vs R-',
                           'comparison': comparison,
                           'statistic': stat, 'p_value': p, 'significance': _significance_stars(p)})

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True)

    for ax, group, acc_pre, acc_post, color in zip(
        axes,
        ['R+', 'R-'],
        [acc_pre_vs_day0_rew, acc_pre_vs_day0_nonrew],
        [acc_day0_vs_post_rew, acc_day0_vs_post_nonrew],
        reward_palette[::-1],
    ):
        df_plot = pd.DataFrame({
            'comparison': ['Pre vs Day 0'] * len(acc_pre) + ['Day 0 vs Post'] * len(acc_post),
            'accuracy': np.concatenate([acc_pre, acc_post]),
        })
        sns.barplot(data=df_plot, x='comparison', y='accuracy', errorbar='ci',
                    ax=ax, color=color, alpha=0.7)
        sns.swarmplot(data=df_plot, x='comparison', y='accuracy',
                      ax=ax, color=color, alpha=0.5, size=7)
        ax.set_title(f'{group} group')
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color='grey', linestyle='--', linewidth=1)
        ax.set_xlabel('')
        ax.set_ylabel('Decoding accuracy (CV)')

    sns.despine()
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'figure_3o.{save_format}'), format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 3o saved to: {os.path.join(output_dir, 'figure_3o.' + save_format)}")

    # Save CSVs
    data_df = pd.DataFrame({
        'mouse_id': mice_rew + mice_nonrew + mice_rew + mice_nonrew,
        'reward_group': (['R+'] * len(mice_rew) + ['R-'] * len(mice_nonrew)) * 2,
        'comparison': (
            ['pre_vs_day0'] * (len(mice_rew) + len(mice_nonrew)) +
            ['day0_vs_post'] * (len(mice_rew) + len(mice_nonrew))
        ),
        'accuracy': np.concatenate([
            acc_pre_vs_day0_rew, acc_pre_vs_day0_nonrew,
            acc_day0_vs_post_rew, acc_day0_vs_post_nonrew,
        ]),
    })
    data_df.to_csv(os.path.join(OUTPUT_DIR, 'figure_3o_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(OUTPUT_DIR, 'figure_3o_stats.csv'), index=False)
    print(f"Figure 3o data/stats saved to: {OUTPUT_DIR}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    SELECT_LMI = False
    PROJECTION_TYPE = None  # 'wS2', 'wM1', or None
    N_SHUFFLES = 100

    print("Loading data...")
    vectors_rew, vectors_nonrew, mice_rew, mice_nonrew = load_and_process_data(
        select_lmi=SELECT_LMI,
        projection_type=PROJECTION_TYPE,
    )

    shared = dict(
        vectors_rew=vectors_rew,
        vectors_nonrew=vectors_nonrew,
        mice_rew=mice_rew,
        mice_nonrew=mice_nonrew,
        select_lmi=SELECT_LMI,
        projection_type=PROJECTION_TYPE,
    )

    print("\nTraining and saving decoder weights (used by figure_4b/c)...")
    train_and_save_decoder_weights(vectors_rew, vectors_nonrew, mice_rew, mice_nonrew)

    print("\nGenerating panel m (decoding accuracy)...")
    panel_m_decoding_accuracy(**shared, n_shuffles=N_SHUFFLES)

    # Compute pairwise matrices once — shared by panels n and o
    print("\nComputing pairwise day decoding matrices (shared by panels n and o)...")
    accs_rew_matrix = _pairwise_day_decoding_fixed_decoder_cv(vectors_rew, DAYS)
    accs_nonrew_matrix = _pairwise_day_decoding_fixed_decoder_cv(vectors_nonrew, DAYS)

    print("\nGenerating panel n (pairwise day decoding matrix)...")
    panel_n_pairwise_decoding(
        **shared,
        accs_rew_matrix=accs_rew_matrix,
        accs_nonrew_matrix=accs_nonrew_matrix,
    )

    print("\nGenerating panel o (day 0 pre vs post classification)...")
    panel_o_day0_classification(
        **shared,
        accs_rew_matrix=accs_rew_matrix,
        accs_nonrew_matrix=accs_nonrew_matrix,
    )

    print("\nDone!")
