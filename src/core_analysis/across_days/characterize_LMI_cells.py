"""
Population-level PSTH analysis for LMI-defined cell populations.

This script generates comprehensive PSTH visualizations for positive and negative LMI cells,
averaging across mice to show population-level responses during learning.

Output: One PDF with pages for each reward group × LMI group combination.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *


# =============================================================================
# PARAMETERS
# =============================================================================

# Cell selection parameters
USE_TOP_N = False  # If True, use top N cells; if False, use all cells meeting threshold
TOP_N = 20  # Number of top cells to use per reward group (if USE_TOP_N=True)
LMI_POSITIVE_THRESHOLD = 0.975  # Percentile threshold for positive LMI cells
LMI_NEGATIVE_THRESHOLD = 0.025  # Percentile threshold for negative LMI cells

# Data parameters
DAYS_LEARNING = [-2, -1, 0, 1, 2]
WIN_SEC = (-1, 5)
SAMPLING_RATE = 30

# Output directory
OUTPUT_DIR = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
OUTPUT_DIR = io.adjust_path_to_host(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

print("Loading LMI results and session database...")

# Load session database
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes',
    experimenters=['AR', 'GF', 'MI']
)
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

# Load LMI results
lmi_df_path = os.path.join(io.processed_dir, 'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df_path)

# Add reward group to LMI dataframe
lmi_df['reward_group'] = lmi_df['mouse_id'].map(
    dict(mice_count[['mouse_id', 'reward_group']].values)
)

# Exclude problematic mice
all_mice = [m for m in all_mice if m != 'GF305']

# Filter cells based on LMI percentile thresholds
positive_lmi_cells = lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD].copy()
negative_lmi_cells = lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD].copy()

# If using top N, select top N cells per reward group
if USE_TOP_N:
    print(f"Using top {TOP_N} cells per reward group...")
    top_positive = []
    top_negative = []
    for group in lmi_df['reward_group'].unique():
        # Positive cells: highest LMI values
        group_positive = positive_lmi_cells[positive_lmi_cells['reward_group'] == group]
        group_positive = group_positive.sort_values('lmi', ascending=False).head(TOP_N)
        top_positive.append(group_positive)

        # Negative cells: lowest LMI values
        group_negative = negative_lmi_cells[negative_lmi_cells['reward_group'] == group]
        group_negative = group_negative.sort_values('lmi', ascending=True).head(TOP_N)
        top_negative.append(group_negative)

    positive_lmi_cells = pd.concat(top_positive)
    negative_lmi_cells = pd.concat(top_negative)

    pdf_name = f'lmi_population_analysis_top{TOP_N}.pdf'
else:
    print(f"Using all cells meeting threshold (p >= {LMI_POSITIVE_THRESHOLD} or p <= {LMI_NEGATIVE_THRESHOLD})...")
    pdf_name = 'lmi_population_analysis_all.pdf'

print(f"  Positive LMI cells: {len(positive_lmi_cells)}")
print(f"  Negative LMI cells: {len(negative_lmi_cells)}")

# Organize cells by mouse, reward group, and LMI group
cell_organization = {}
for mouse_id in all_mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    cell_organization[mouse_id] = {
        'reward_group': reward_group,
        'positive': positive_lmi_cells[positive_lmi_cells['mouse_id'] == mouse_id]['roi'].tolist(),
        'negative': negative_lmi_cells[negative_lmi_cells['mouse_id'] == mouse_id]['roi'].tolist()
    }


# =============================================================================
# FUNCTION: compute_population_psth_summary
# =============================================================================

def compute_population_psth_summary(mice_list, lmi_group, reward_group):
    """
    Compute PSTH summary data for a population of cells.

    Parameters
    ----------
    mice_list : list
        List of mouse IDs to include
    lmi_group : str
        'positive' or 'negative'
    reward_group : str
        'R+' or 'R-'

    Returns
    -------
    dict
        {trial_type: {mouse_id: averaged_xarray}}
    """
    print(f"\n  Computing PSTH summary for {reward_group} {lmi_group} cells...")

    psth_data = {}
    folder = os.path.join(io.processed_dir, 'mice')

    for mouse_id in mice_list:
        cell_list = cell_organization[mouse_id][lmi_group]
        if not cell_list:
            continue

        print(f"    Processing {mouse_id}: {len(cell_list)} cells")

        # Load mapping data for each day
        try:
            file_name_mapping = 'tensor_xarray_mapping_data.nc'
            xarr_mapping = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name_mapping, substracted=True)
            xarr_mapping = xarr_mapping.sel(cell=xarr_mapping['roi'].isin(cell_list))
            xarr_mapping.load()  # Load into memory and close file handle

            # Average trials per day, then across cells
            for day in DAYS_LEARNING:
                day_data = xarr_mapping.sel(trial=xarr_mapping['day'] == day)
                if len(day_data.trial) > 0:
                    day_avg = day_data.mean(dim='trial').mean(dim='cell')

                    key = f'mapping_day{day}'
                    if key not in psth_data:
                        psth_data[key] = {}
                    psth_data[key][mouse_id] = day_avg

        except Exception as e:
            print(f"      Warning: Could not load mapping data: {e}")

        # Load learning data for trial types
        try:
            file_name_learning = 'tensor_xarray_learning_data.nc'
            xarr_learning = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name_learning, substracted=True)
            xarr_learning = xarr_learning.sel(cell=xarr_learning['roi'].isin(cell_list))
            xarr_learning = xarr_learning.sel(trial=xarr_learning['day'].isin(DAYS_LEARNING))
            xarr_learning.load()  # Load into memory and close file handle

            # Define trial types
            trial_filters = {
                'whisker_hit': (xarr_learning['whisker_stim'] == 1) & (xarr_learning['lick_flag'] == 1),
                'whisker_miss': (xarr_learning['whisker_stim'] == 1) & (xarr_learning['lick_flag'] == 0),
                'false_alarm': (xarr_learning['no_stim'] == 1) & (xarr_learning['lick_flag'] == 1),
                'correct_rejection': (xarr_learning['no_stim'] == 1) & (xarr_learning['lick_flag'] == 0),
                'auditory_hit': (xarr_learning['auditory_stim'] == 1) & (xarr_learning['lick_flag'] == 1)
            }

            for trial_key, trial_filter in trial_filters.items():
                xarr_trial_type = xarr_learning.sel(trial=trial_filter)

                # Average trials per day, then across cells
                for day in DAYS_LEARNING:
                    # Skip whisker trials for pre-learning days (mistaken trials)
                    if trial_key in ['whisker_hit', 'whisker_miss'] and day in [-2, -1]:
                        continue

                    day_data = xarr_trial_type.sel(trial=xarr_trial_type['day'] == day)
                    if len(day_data.trial) > 0:
                        day_avg = day_data.mean(dim='trial').mean(dim='cell')

                        key = f'{trial_key}_day{day}'
                        if key not in psth_data:
                            psth_data[key] = {}
                        psth_data[key][mouse_id] = day_avg

        except Exception as e:
            print(f"      Warning: Could not load learning data: {e}")

    return psth_data


# =============================================================================
# FUNCTION: compute_population_lick_aligned_psth
# =============================================================================

def compute_population_lick_aligned_psth(mice_list, lmi_group, reward_group):
    """
    Compute lick-aligned PSTH data for a population of cells.
    Only includes trial types with licks: whisker_hit, false_alarm, auditory_hit.

    Parameters
    ----------
    mice_list : list
        List of mouse IDs to process
    lmi_group : str
        'positive' or 'negative'
    reward_group : str
        'R+' or 'R-'

    Returns
    -------
    dict
        {trial_type: {mouse_id: averaged_xarray}}
    """
    print(f"\n  Computing lick-aligned PSTH for {reward_group} {lmi_group} cells...")

    psth_data = {}
    folder = os.path.join(io.processed_dir, 'mice')

    for mouse_id in mice_list:
        cell_list = cell_organization[mouse_id][lmi_group]
        if not cell_list:
            continue

        print(f"    Processing {mouse_id}: {len(cell_list)} cells")

        # Load lick-aligned data for trial types with licks
        try:
            file_name_lick = 'lick_aligned_xarray.nc'
            xarr_lick = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name_lick, substracted=False)
            xarr_lick = xarr_lick.sel(cell=xarr_lick['roi'].isin(cell_list))
            xarr_lick = xarr_lick.sel(trial=xarr_lick['day'].isin(DAYS_LEARNING))
            xarr_lick.load()  # Load into memory and close file handle

            # Define trial types with licks only
            trial_filters = {
                'whisker_hit': (xarr_lick['whisker_stim'] == 1) & (xarr_lick['lick_flag'] == 1),
                'false_alarm': (xarr_lick['no_stim'] == 1) & (xarr_lick['lick_flag'] == 1),
                'auditory_hit': (xarr_lick['auditory_stim'] == 1) & (xarr_lick['lick_flag'] == 1)
            }

            for trial_key, trial_filter in trial_filters.items():
                xarr_trial_type = xarr_lick.sel(trial=trial_filter)

                # Average trials per day, then across cells
                for day in DAYS_LEARNING:
                    # Skip whisker trials for pre-learning days (mistaken trials)
                    if trial_key == 'whisker_hit' and day in [-2, -1]:
                        continue

                    day_data = xarr_trial_type.sel(trial=xarr_trial_type['day'] == day)
                    if len(day_data.trial) > 0:
                        day_avg = day_data.mean(dim='trial').mean(dim='cell')

                        key = f'{trial_key}_day{day}'
                        if key not in psth_data:
                            psth_data[key] = {}
                        psth_data[key][mouse_id] = day_avg

        except Exception as e:
            print(f"      Warning: Could not load lick-aligned data: {e}")

    return psth_data


# =============================================================================
# FUNCTION: compute_population_whisker_evolution
# =============================================================================

def compute_population_whisker_evolution(mice_list, lmi_group, reward_group):
    """
    Compute whisker evolution data for a population of cells (Day 0).

    Parameters
    ----------
    mice_list : list
        List of mouse IDs to include
    lmi_group : str
        'positive' or 'negative'
    reward_group : str
        'R+' or 'R-'

    Returns
    -------
    dict
        {data_key: {mouse_id: averaged_xarray}}
    """
    print(f"\n  Computing whisker evolution for {reward_group} {lmi_group} cells...")

    evolution_data = {}
    folder = os.path.join(io.processed_dir, 'mice')

    for mouse_id in mice_list:
        cell_list = cell_organization[mouse_id][lmi_group]
        if not cell_list:
            continue

        print(f"    Processing {mouse_id}: {len(cell_list)} cells")

        try:
            # Load Day 0 learning data
            file_name = 'tensor_xarray_learning_data.nc'
            xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=True)
            xarr = xarr.sel(cell=xarr['roi'].isin(cell_list))
            xarr = xarr.sel(trial=xarr['day'] == 0)
            xarr.load()  # Load into memory and close file handle

            # Get all whisker trials
            whisker_trials = xarr.sel(trial=xarr['whisker_stim'] == 1)

            if len(whisker_trials.trial) == 0:
                print(f"      Warning: No whisker trials for {mouse_id}")
                continue

            # Find first hit (first trial where whisker_stim==1 and lick_flag==1)
            lick_flags = whisker_trials['lick_flag'].values
            hit_indices = np.where(lick_flags == 1)[0]

            if len(hit_indices) == 0:
                print(f"      Warning: No whisker hits for {mouse_id}")
                continue

            first_hit_idx = hit_indices[0]
            print(f"      First hit at trial {first_hit_idx + 1}")

            # Extract misses before first hit
            if first_hit_idx > 0:
                misses_before = whisker_trials.isel(trial=slice(0, first_hit_idx))
                misses_avg = misses_before.mean(dim='trial').mean(dim='cell')
                if 'misses_before_first_hit' not in evolution_data:
                    evolution_data['misses_before_first_hit'] = {}
                evolution_data['misses_before_first_hit'][mouse_id] = misses_avg

            # Extract first hit
            first_hit = whisker_trials.isel(trial=first_hit_idx)
            first_hit_avg = first_hit.mean(dim='cell')
            if 'first_hit' not in evolution_data:
                evolution_data['first_hit'] = {}
            evolution_data['first_hit'][mouse_id] = first_hit_avg

            # Extract next 5 trials after first hit
            next_5_end = min(first_hit_idx + 6, len(whisker_trials.trial))
            if next_5_end > first_hit_idx + 1:
                next_5 = whisker_trials.isel(trial=slice(first_hit_idx + 1, next_5_end))
                next_5_avg = next_5.mean(dim='trial').mean(dim='cell')
                if 'next_5_after_hit' not in evolution_data:
                    evolution_data['next_5_after_hit'] = {}
                evolution_data['next_5_after_hit'][mouse_id] = next_5_avg

            # Compute percentiles for all whisker trials, hits, and misses
            n_whisker = len(whisker_trials.trial)
            first_20_idx = max(1, int(n_whisker * 0.20))
            last_20_idx = int(n_whisker * 0.80)

            # All whisker trials by percentile
            all_first_20 = whisker_trials.isel(trial=slice(0, first_20_idx))
            all_middle = whisker_trials.isel(trial=slice(first_20_idx, last_20_idx))
            all_last_20 = whisker_trials.isel(trial=slice(last_20_idx, n_whisker))

            if len(all_first_20.trial) > 0:
                avg = all_first_20.mean(dim='trial').mean(dim='cell')
                if 'all_first_20pct' not in evolution_data:
                    evolution_data['all_first_20pct'] = {}
                evolution_data['all_first_20pct'][mouse_id] = avg

            if len(all_middle.trial) > 0:
                avg = all_middle.mean(dim='trial').mean(dim='cell')
                if 'all_middle_60pct' not in evolution_data:
                    evolution_data['all_middle_60pct'] = {}
                evolution_data['all_middle_60pct'][mouse_id] = avg

            if len(all_last_20.trial) > 0:
                avg = all_last_20.mean(dim='trial').mean(dim='cell')
                if 'all_last_20pct' not in evolution_data:
                    evolution_data['all_last_20pct'] = {}
                evolution_data['all_last_20pct'][mouse_id] = avg

            # Hits and misses from same chronological positions
            hit_indices_all = np.where(lick_flags == 1)[0]
            miss_indices_all = np.where(lick_flags == 0)[0]

            # Hits
            hits_in_first_20 = hit_indices_all[hit_indices_all < first_20_idx]
            hits_in_middle = hit_indices_all[(hit_indices_all >= first_20_idx) & (hit_indices_all < last_20_idx)]
            hits_in_last_20 = hit_indices_all[hit_indices_all >= last_20_idx]

            if len(hits_in_first_20) > 0:
                hits = whisker_trials.isel(trial=hits_in_first_20)
                avg = hits.mean(dim='trial').mean(dim='cell')
                if 'hits_first_20pct' not in evolution_data:
                    evolution_data['hits_first_20pct'] = {}
                evolution_data['hits_first_20pct'][mouse_id] = avg

            if len(hits_in_middle) > 0:
                hits = whisker_trials.isel(trial=hits_in_middle)
                avg = hits.mean(dim='trial').mean(dim='cell')
                if 'hits_middle_60pct' not in evolution_data:
                    evolution_data['hits_middle_60pct'] = {}
                evolution_data['hits_middle_60pct'][mouse_id] = avg

            if len(hits_in_last_20) > 0:
                hits = whisker_trials.isel(trial=hits_in_last_20)
                avg = hits.mean(dim='trial').mean(dim='cell')
                if 'hits_last_20pct' not in evolution_data:
                    evolution_data['hits_last_20pct'] = {}
                evolution_data['hits_last_20pct'][mouse_id] = avg

            # Misses
            misses_in_first_20 = miss_indices_all[miss_indices_all < first_20_idx]
            misses_in_middle = miss_indices_all[(miss_indices_all >= first_20_idx) & (miss_indices_all < last_20_idx)]
            misses_in_last_20 = miss_indices_all[miss_indices_all >= last_20_idx]

            if len(misses_in_first_20) > 0:
                misses = whisker_trials.isel(trial=misses_in_first_20)
                avg = misses.mean(dim='trial').mean(dim='cell')
                if 'misses_first_20pct' not in evolution_data:
                    evolution_data['misses_first_20pct'] = {}
                evolution_data['misses_first_20pct'][mouse_id] = avg

            if len(misses_in_middle) > 0:
                misses = whisker_trials.isel(trial=misses_in_middle)
                avg = misses.mean(dim='trial').mean(dim='cell')
                if 'misses_middle_60pct' not in evolution_data:
                    evolution_data['misses_middle_60pct'] = {}
                evolution_data['misses_middle_60pct'][mouse_id] = avg

            if len(misses_in_last_20) > 0:
                misses = whisker_trials.isel(trial=misses_in_last_20)
                avg = misses.mean(dim='trial').mean(dim='cell')
                if 'misses_last_20pct' not in evolution_data:
                    evolution_data['misses_last_20pct'] = {}
                evolution_data['misses_last_20pct'][mouse_id] = avg

        except Exception as e:
            print(f"      Warning: Could not process whisker evolution: {e}")
            import traceback
            traceback.print_exc()

    return evolution_data


# =============================================================================
# FUNCTION: plot_population_psth_summary
# =============================================================================

def plot_population_psth_summary(psth_data, reward_group, lmi_group):
    """
    Plot PSTH summary page for population (6 rows × 5 columns).

    Parameters
    ----------
    psth_data : dict
        {trial_type: {mouse_id: averaged_xarray}}
    reward_group : str
        'R+' or 'R-'
    lmi_group : str
        'positive' or 'negative'

    Returns
    -------
    matplotlib.figure.Figure
    """
    print(f"\n  Plotting PSTH summary for {reward_group} {lmi_group} cells...")

    # Create figure
    fig = plt.figure(figsize=(25, 18))
    gs = fig.add_gridspec(6, 5, hspace=0.4, wspace=0.3)

    # Calculate global y-limits
    all_values = []
    for key, mouse_dict in psth_data.items():
        for mouse_id, xarr in mouse_dict.items():
            all_values.extend(xarr.values.flatten() * 100)

    if len(all_values) > 0:
        global_ymin = np.nanpercentile(all_values, 1)
        global_ymax = np.nanpercentile(all_values, 99.9)
        y_range = global_ymax - global_ymin
        global_ymin -= y_range * 0.1
        global_ymax += y_range * 0.2
    else:
        global_ymin, global_ymax = -5, 20

    # Collect all axes
    all_axes = []

    # Set reward-group-specific colors
    if reward_group == 'R+':
        mapping_color = trial_type_rew_palette[3]  # Green
        whisker_hit_color = trial_type_rew_palette[3]
        whisker_miss_color = trial_type_rew_palette[2]
    else:  # R-
        mapping_color = trial_type_nonrew_palette[3]  # Magenta
        whisker_hit_color = trial_type_nonrew_palette[3]
        whisker_miss_color = trial_type_nonrew_palette[2]

    # Row 0: Mapping trials (passive whisker) by day
    for col_idx, day in enumerate(DAYS_LEARNING):
        ax = fig.add_subplot(gs[0, col_idx])
        all_axes.append(ax)

        key = f'mapping_day{day}'

        if key in psth_data and psth_data[key]:
            # Combine all mice
            dfs = []
            for mouse_id, xarr in psth_data[key].items():
                df = xarr.to_dataframe(name='activity').reset_index()
                df['mouse_id'] = mouse_id
                dfs.append(df)
            df_combined = pd.concat(dfs)
            df_combined['activity'] = df_combined['activity'] * 100

            sns.lineplot(data=df_combined, x='time', y='activity', errorbar='ci',
                        ax=ax, color=mapping_color, linewidth=2.5, linestyle='-')

            ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
            n_mice = len(psth_data[key])
            ax.set_title(f'Day {day}\n(n={n_mice} mice)', fontsize=9, fontweight='bold')
            ax.set_ylabel('DF/F0 (%)', fontsize=10)

            if col_idx == 0:
                # Add trial type label on the left
                ax.text(-0.3, 0.5, 'Passive Whisker\nTrials', transform=ax.transAxes,
                       rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('DF/F0 (%)', fontsize=10)
            if col_idx == 0:
                ax.text(-0.3, 0.5, 'Passive Whisker\nTrials', transform=ax.transAxes,
                       rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

    # Rows 1-5: Trial types by day
    trial_types = [
        ('whisker_hit', 'Whisker Hit', whisker_hit_color, '-', stim_palette[1], 1),
        ('whisker_miss', 'Whisker Miss', whisker_miss_color, '--', stim_palette[1], 2),
        ('false_alarm', 'False Alarm', behavior_palette[5], '-', stim_palette[2], 3),
        ('correct_rejection', 'Correct Rejection', behavior_palette[4], '--', stim_palette[2], 4),
        ('auditory_hit', 'Auditory Hit', behavior_palette[1], '-', stim_palette[0], 5)
    ]

    for trial_key, trial_label, color, linestyle, t0_color, row_idx in trial_types:
        for col_idx, day in enumerate(DAYS_LEARNING):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            all_axes.append(ax)

            key = f'{trial_key}_day{day}'

            if key in psth_data and psth_data[key]:
                # Combine all mice
                dfs = []
                for mouse_id, xarr in psth_data[key].items():
                    df = xarr.to_dataframe(name='activity').reset_index()
                    df['mouse_id'] = mouse_id
                    dfs.append(df)
                df_combined = pd.concat(dfs)
                df_combined['activity'] = df_combined['activity'] * 100

                sns.lineplot(data=df_combined, x='time', y='activity', errorbar='ci',
                            ax=ax, color=color, linewidth=2.5, linestyle=linestyle)

                ax.axvline(0, color=t0_color, linestyle='-', linewidth=1.5)
                n_mice = len(psth_data[key])
                ax.set_title(f'Day {day}\n(n={n_mice} mice)', fontsize=9, fontweight='bold')
                ax.set_ylabel('DF/F0 (%)', fontsize=10)

                if col_idx == 0:
                    # Add trial type label
                    ax.text(-0.3, 0.5, trial_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

                if row_idx == 5:
                    ax.set_xlabel('Time (s)', fontsize=10)
                    if col_idx == 0:
                        ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel('DF/F0 (%)', fontsize=10)
                if col_idx == 0:
                    ax.text(-0.3, 0.5, trial_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

    # Set shared y-limits
    for ax in all_axes:
        ax.set_ylim(global_ymin, global_ymax)

    # Overall title
    lmi_label = 'Positive' if lmi_group == 'positive' else 'Negative'
    if reward_group == 'R-':
        title_color = reward_palette[0]
    else:
        title_color = reward_palette[1]

    fig.suptitle(f'{reward_group} - {lmi_label} LMI Cells - Population PSTH Summary',
                fontsize=16, fontweight='bold', color=title_color)

    sns.despine(fig=fig)

    return fig


# =============================================================================
# FUNCTION: plot_population_lick_aligned_psth
# =============================================================================

def plot_population_lick_aligned_psth(psth_data, reward_group, lmi_group):
    """
    Plot lick-aligned PSTH summary page for population (6 rows × 5 columns).
    Only plots trial types with licks: whisker_hit (row 1), false_alarm (row 3), auditory_hit (row 5).
    Rows 0, 2, 4 are left empty (mapping, whisker_miss, correct_rejection have no licks).

    Parameters
    ----------
    psth_data : dict
        {trial_type: {mouse_id: averaged_xarray}}
    reward_group : str
        'R+' or 'R-'
    lmi_group : str
        'positive' or 'negative'

    Returns
    -------
    matplotlib.figure.Figure
    """
    print(f"\n  Plotting lick-aligned PSTH for {reward_group} {lmi_group} cells...")

    # Create figure
    fig = plt.figure(figsize=(25, 18))
    gs = fig.add_gridspec(6, 5, hspace=0.4, wspace=0.3)

    # Calculate global y-limits
    all_values = []
    for key, mouse_dict in psth_data.items():
        for mouse_id, xarr in mouse_dict.items():
            all_values.extend(xarr.values.flatten() * 100)

    if len(all_values) > 0:
        global_ymin = np.nanpercentile(all_values, 1)
        global_ymax = np.nanpercentile(all_values, 99.9)
        y_range = global_ymax - global_ymin
        global_ymin -= y_range * 0.1
        global_ymax += y_range * 0.2
    else:
        global_ymin, global_ymax = -5, 20

    # Collect all axes
    all_axes = []

    # Set reward-group-specific colors
    if reward_group == 'R+':
        whisker_hit_color = trial_type_rew_palette[3]
    else:  # R-
        whisker_hit_color = trial_type_nonrew_palette[3]

    # Define trial types with licks only (rows 1, 3, 5)
    # Leave rows 0, 2, 4 empty (mapping, whisker_miss, correct_rejection)
    trial_types = [
        ('whisker_hit', 'Whisker Hit\n(lick-aligned)', whisker_hit_color, '-', 'black', 1),
        ('false_alarm', 'False Alarm\n(lick-aligned)', behavior_palette[5], '-', 'black', 3),
        ('auditory_hit', 'Auditory Hit\n(lick-aligned)', behavior_palette[1], '-', 'black', 5)
    ]

    # Row 0: Empty (mapping trials - no licks)
    for col_idx, day in enumerate(DAYS_LEARNING):
        ax = fig.add_subplot(gs[0, col_idx])
        all_axes.append(ax)
        ax.axis('off')
        if col_idx == 0:
            ax.text(-0.3, 0.5, 'Passive Whisker\nTrials\n(no licks)', transform=ax.transAxes,
                   rotation=90, va='center', ha='center', fontsize=10, fontweight='bold', color='gray')

    # Row 2: Empty (whisker miss - no licks)
    for col_idx, day in enumerate(DAYS_LEARNING):
        ax = fig.add_subplot(gs[2, col_idx])
        all_axes.append(ax)
        ax.axis('off')
        if col_idx == 0:
            ax.text(-0.3, 0.5, 'Whisker Miss\n(no licks)', transform=ax.transAxes,
                   rotation=90, va='center', ha='center', fontsize=10, fontweight='bold', color='gray')

    # Row 4: Empty (correct rejection - no licks)
    for col_idx, day in enumerate(DAYS_LEARNING):
        ax = fig.add_subplot(gs[4, col_idx])
        all_axes.append(ax)
        ax.axis('off')
        if col_idx == 0:
            ax.text(-0.3, 0.5, 'Correct Rejection\n(no licks)', transform=ax.transAxes,
                   rotation=90, va='center', ha='center', fontsize=10, fontweight='bold', color='gray')

    # Rows 1, 3, 5: Trial types with licks
    for trial_key, trial_label, color, linestyle, t0_color, row_idx in trial_types:
        for col_idx, day in enumerate(DAYS_LEARNING):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            all_axes.append(ax)

            key = f'{trial_key}_day{day}'

            if key in psth_data and psth_data[key]:
                # Combine all mice
                dfs = []
                for mouse_id, xarr in psth_data[key].items():
                    df = xarr.to_dataframe(name='activity').reset_index()
                    df['mouse_id'] = mouse_id
                    dfs.append(df)
                df_combined = pd.concat(dfs)
                df_combined['activity'] = df_combined['activity'] * 100

                sns.lineplot(data=df_combined, x='time', y='activity', errorbar='ci',
                            ax=ax, color=color, linewidth=2.5, linestyle=linestyle)

                # t=0 is lick time (black vertical line)
                ax.axvline(0, color=t0_color, linestyle='-', linewidth=1.5)
                n_mice = len(psth_data[key])
                ax.set_title(f'Day {day}\n(n={n_mice} mice)', fontsize=9, fontweight='bold')
                ax.set_ylabel('DF/F0 (%)', fontsize=10)

                if col_idx == 0:
                    # Add trial type label
                    ax.text(-0.3, 0.5, trial_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

                if row_idx == 5:
                    ax.set_xlabel('Time from lick (s)', fontsize=10)
                    if col_idx == 0:
                        ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel('DF/F0 (%)', fontsize=10)
                if col_idx == 0:
                    ax.text(-0.3, 0.5, trial_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

    # Set shared y-limits (only for non-empty axes)
    for ax in all_axes:
        if ax.get_visible() and not ax.axison == False:
            ax.set_ylim(global_ymin, global_ymax)

    # Overall title
    lmi_label = 'Positive' if lmi_group == 'positive' else 'Negative'
    if reward_group == 'R-':
        title_color = reward_palette[0]
    else:
        title_color = reward_palette[1]

    fig.suptitle(f'{reward_group} - {lmi_label} LMI Cells - Lick-Aligned PSTH Summary',
                fontsize=16, fontweight='bold', color=title_color)

    sns.despine(fig=fig)

    return fig


# =============================================================================
# FUNCTION: plot_population_whisker_evolution
# =============================================================================

def plot_population_whisker_evolution(evolution_data, reward_group, lmi_group):
    """
    Plot whisker evolution page for population (4 rows × 3 columns).

    Parameters
    ----------
    evolution_data : dict
        {data_key: {mouse_id: averaged_xarray}}
    reward_group : str
        'R+' or 'R-'
    lmi_group : str
        'positive' or 'negative'

    Returns
    -------
    matplotlib.figure.Figure
    """
    print(f"\n  Plotting whisker evolution for {reward_group} {lmi_group} cells...")

    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(20, 16), sharey=True)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Calculate global y-limits
    all_values = []
    for key, mouse_dict in evolution_data.items():
        for mouse_id, xarr in mouse_dict.items():
            all_values.extend(xarr.values.flatten() * 100)

    if len(all_values) > 0:
        global_ymin = np.nanpercentile(all_values, 1)
        global_ymax = np.nanpercentile(all_values, 99.9)
        y_range = global_ymax - global_ymin
        global_ymin -= y_range * 0.1
        global_ymax += y_range * 0.2
    else:
        global_ymin, global_ymax = -5, 20

    # Set reward-group-specific colors
    if reward_group == 'R+':
        trial_color_hit = trial_type_rew_palette[3]
        trial_color_miss = trial_type_rew_palette[2]
    else:  # R-
        trial_color_hit = trial_type_nonrew_palette[3]
        trial_color_miss = trial_type_nonrew_palette[2]

    # Get time values
    if 'first_hit' in evolution_data and evolution_data['first_hit']:
        time_vals = list(evolution_data['first_hit'].values())[0]['time'].values
    elif 'all_first_20pct' in evolution_data and evolution_data['all_first_20pct']:
        time_vals = list(evolution_data['all_first_20pct'].values())[0]['time'].values
    else:
        time_vals = np.linspace(-1, 5, 100)

    # Row 0: Trials around first hit

    # Panel 0,0: Averaged misses before first hit
    ax = axes[0, 0]
    if 'misses_before_first_hit' in evolution_data and evolution_data['misses_before_first_hit']:
        dfs = []
        for mouse_id, xarr in evolution_data['misses_before_first_hit'].items():
            df = xarr.to_dataframe(name='activity').reset_index()
            df['mouse_id'] = mouse_id
            dfs.append(df)
        df_combined = pd.concat(dfs)
        df_combined['activity'] = df_combined['activity'] * 100

        n_mice = len(evolution_data['misses_before_first_hit'])

        sns.lineplot(data=df_combined, x='time', y='activity', errorbar='ci',
                    ax=ax, color=trial_color_miss, linewidth=2.5, linestyle='--')

        ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
        ax.set_title(f'Avg misses before 1st hit\n(n={n_mice} mice)', fontsize=10, fontweight='bold')
        ax.set_ylabel('DF/F0 (%)', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No misses\nbefore 1st hit', ha='center', va='center',
               transform=ax.transAxes, fontsize=9)

    # Add row label
    ax.text(-0.22, 0.5, 'Trials Around\nFirst Hit', transform=ax.transAxes,
           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(global_ymin, global_ymax)

    # Panel 0,1: First hit
    ax = axes[0, 1]
    if 'first_hit' in evolution_data and evolution_data['first_hit']:
        dfs = []
        for mouse_id, xarr in evolution_data['first_hit'].items():
            df = xarr.to_dataframe(name='activity').reset_index()
            df['mouse_id'] = mouse_id
            dfs.append(df)
        df_combined = pd.concat(dfs)
        df_combined['activity'] = df_combined['activity'] * 100

        n_mice = len(evolution_data['first_hit'])

        sns.lineplot(data=df_combined, x='time', y='activity', errorbar='ci',
                    ax=ax, color=trial_color_hit, linewidth=2.5, linestyle='-')

        ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
        ax.set_title(f'First whisker hit\n(n={n_mice} mice)', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No first hit', ha='center', va='center',
               transform=ax.transAxes, fontsize=9)
    ax.set_ylim(global_ymin, global_ymax)

    # Panel 0,2: Next 5 trials after first hit
    ax = axes[0, 2]
    if 'next_5_after_hit' in evolution_data and evolution_data['next_5_after_hit']:
        dfs = []
        for mouse_id, xarr in evolution_data['next_5_after_hit'].items():
            df = xarr.to_dataframe(name='activity').reset_index()
            df['mouse_id'] = mouse_id
            dfs.append(df)
        df_combined = pd.concat(dfs)
        df_combined['activity'] = df_combined['activity'] * 100

        n_mice = len(evolution_data['next_5_after_hit'])

        sns.lineplot(data=df_combined, x='time', y='activity', errorbar='ci',
                    ax=ax, color=trial_color_hit, linewidth=2.5, linestyle='-')

        ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
        ax.set_title(f'Avg next 5 trials\nafter 1st hit (n={n_mice} mice)', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No trials\nafter 1st hit', ha='center', va='center',
               transform=ax.transAxes, fontsize=9)
    ax.set_ylim(global_ymin, global_ymax)

    # Rows 1-3: Averaged PSTHs by percentile
    percentile_panels = [
        (1, 'all', 'All Whisker Trials', trial_color_hit, '-'),
        (2, 'hits', 'Whisker Hits Only', trial_color_hit, '-'),
        (3, 'misses', 'Whisker Misses Only', trial_color_miss, '--')
    ]

    for row_idx, trial_type, row_label, color, linestyle in percentile_panels:
        # Define percentile labels based on trial type
        if trial_type == 'all':
            percentile_labels = [
                ('first_20pct', 'First 20%\nwhisker trials'),
                ('middle_60pct', 'Middle 20-80%\nwhisker trials'),
                ('last_20pct', 'Last 20%\nwhisker trials')
            ]
        elif trial_type == 'hits':
            percentile_labels = [
                ('first_20pct', 'Hits among\nfirst 20% whisker trials'),
                ('middle_60pct', 'Hits among\nmiddle 20-80% whisker trials'),
                ('last_20pct', 'Hits among\nlast 20% whisker trials')
            ]
        else:  # misses
            percentile_labels = [
                ('first_20pct', 'Misses among\nfirst 20% whisker trials'),
                ('middle_60pct', 'Misses among\nmiddle 20-80% whisker trials'),
                ('last_20pct', 'Misses among\nlast 20% whisker trials')
            ]

        for col_idx, (percentile, pct_label) in enumerate(percentile_labels):
            ax = axes[row_idx, col_idx]

            key = f'{trial_type}_{percentile}'

            if key in evolution_data and evolution_data[key]:
                dfs = []
                for mouse_id, xarr in evolution_data[key].items():
                    df = xarr.to_dataframe(name='activity').reset_index()
                    df['mouse_id'] = mouse_id
                    dfs.append(df)
                df_combined = pd.concat(dfs)
                df_combined['activity'] = df_combined['activity'] * 100

                n_mice = len(evolution_data[key])

                # Plot PSTH with confidence interval
                sns.lineplot(data=df_combined, x='time', y='activity', errorbar='ci',
                            ax=ax, color=color, linewidth=2.5, linestyle=linestyle)

                ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
                ax.set_title(f'{pct_label}\n(n={n_mice} mice)', fontsize=9, fontweight='bold')

                if col_idx == 0:
                    ax.set_ylabel('DF/F0 (%)', fontsize=10)
                    # Add row label on the left
                    ax.text(-0.22, 0.5, row_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
                if row_idx == 3:
                    ax.set_xlabel('Time (s)', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'No data\n{pct_label}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=9)
                if col_idx == 0:
                    ax.text(-0.22, 0.5, row_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

            ax.set_ylim(global_ymin, global_ymax)

    # Overall title
    lmi_label = 'Positive' if lmi_group == 'positive' else 'Negative'
    if reward_group == 'R-':
        title_color = reward_palette[0]
    else:
        title_color = reward_palette[1]

    fig.suptitle(f'{reward_group} - {lmi_label} LMI Cells - Whisker Evolution (Day 0)',
                fontsize=16, fontweight='bold', color=title_color, y=0.995)

    sns.despine(fig=fig)

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

print(f"\n{'='*70}")
print(f"GENERATING LMI POPULATION ANALYSIS PDF")
print(f"{'='*70}\n")

pdf_path = os.path.join(OUTPUT_DIR, pdf_name)

with PdfPages(pdf_path) as pdf:
    # Loop through reward groups and LMI groups
    for reward_group in ['R+', 'R-']:
        for lmi_group in ['positive', 'negative']:
            print(f"\n{'-'*70}")
            print(f"Processing {reward_group} - {lmi_group} LMI cells")
            print(f"{'-'*70}")

            # Filter mice for this reward group that have cells in this LMI group
            mice_list = [
                m for m in all_mice
                if cell_organization[m]['reward_group'] == reward_group
                and len(cell_organization[m][lmi_group]) > 0
            ]

            if not mice_list:
                print(f"  No mice with {lmi_group} LMI cells in {reward_group} group, skipping...")
                continue

            print(f"  Mice included: {mice_list}")

            # Compute and plot PSTH summary
            psth_data = compute_population_psth_summary(mice_list, lmi_group, reward_group)
            fig_psth = plot_population_psth_summary(psth_data, reward_group, lmi_group)
            pdf.savefig(fig_psth, bbox_inches='tight')
            plt.close(fig_psth)
            print(f"  ✓ PSTH summary page added to PDF")

            # Compute and plot lick-aligned PSTH
            lick_psth_data = compute_population_lick_aligned_psth(mice_list, lmi_group, reward_group)
            fig_lick_psth = plot_population_lick_aligned_psth(lick_psth_data, reward_group, lmi_group)
            pdf.savefig(fig_lick_psth, bbox_inches='tight')
            plt.close(fig_lick_psth)
            print(f"  ✓ Lick-aligned PSTH page added to PDF")

            # Compute and plot whisker evolution
            evolution_data = compute_population_whisker_evolution(mice_list, lmi_group, reward_group)
            fig_evolution = plot_population_whisker_evolution(evolution_data, reward_group, lmi_group)
            pdf.savefig(fig_evolution, bbox_inches='tight')
            plt.close(fig_evolution)
            print(f"  ✓ Whisker evolution page added to PDF")

print(f"\n{'='*70}")
print(f"PDF GENERATION COMPLETE!")
print(f"{'='*70}")
print(f"\nOutput saved to: {pdf_path}\n")
