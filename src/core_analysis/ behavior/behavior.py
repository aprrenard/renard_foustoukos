"""This script contains functions to analyze behavior data.
It assumes that a excel sheet contains sessions metadata
and that data is stored on the server with the usual folder structure.
This code will be refactored to read behavioral data from NWB files.
"""

import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pymc as pm 
import scipy as sp

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *
from nwb_wrappers import nwb_reader_functions as nwb_read
from scipy.stats import mannwhitneyu, wilcoxon
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
import matplotlib
from scipy.signal import find_peaks, hilbert
import matplotlib.cm as cm



# nwb_dir = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/NWB'
# session_list = ['MS039_20240305_085825.nwb',
#                 'MS041_20240305_123141.nwb',
#                 'MS065_20240421_150811.nwb',
#                 'MS067_20240422_152536.nwb',
#                 'MS069_20240422_154117.nwb',
#                 'MS128_20240926_111219.nwb',
#                 'MS129_20240926_112052.nwb',
#                 'MS130_20240925_112915.nwb',
#                 'MS131_20240926_120457.nwb',
#                 'MS135_20240925_133312.nwb']

# # session_list = ['MS061_20240421_144950.nwb',
# #                 'MS066_20240422_143733.nwb',
# #                 'MS066_20240422_143733.nwb',
# #                 'MS127_20240926_105210.nwb',
# #                 'MS132_20240925_115907.nwb',
# #                 'MS133_20240925_121916.nwb',
# #                 'MS134_20240925_114817.nwb']

# nwb_list = [os.path.join(nwb_dir, nwb) for nwb in session_list]
# reward_group = 'R+'

# table = []
# for nwb, session in zip(nwb_list, session_list):
#     df = nwb_read.get_trial_table(nwb)
#     df = df.reset_index()
#     df.rename(columns={'id': 'trial_id'}, inplace=True)
#     df = compute_performance(df, session, reward_group)
#     table.append(df)
# table = pd.concat(table)
# table['day'] = '0'
# table = table.astype({'mouse_id':str,
#                       'session_id':str,
#                       'day':str})

# table = table.loc[table.block_id<=17]
# sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1)
# plt.figure(figsize=(15,6))
# plot_perf_across_blocks(table, reward_group, palette, nmax_trials=300, ax=None)
# sns.despine()


# Create and save data tables.
# ############################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_behavior = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )
# Add mice with inactivation done after D0 learning.
mice_pharma_execution = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['yes'],
                                    pharma_inactivation_type = ['execution'],
                                    )
mice_opto_execution = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes', np.nan],
                                    pharmacology = ['no',np.nan],
                                    opto_inactivation_type = ['execution'],
                                    )
mice_opto_learning = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes', np.nan],
                                    pharmacology = ['no',np.nan],
                                    opto_inactivation_type = ['learning'],
                                    )
# Because some opto mice had both execution and learning inactivation.
mice_opto = list(set(mice_opto_execution) - set(mice_opto_learning))
# Combine and deduplicate mouse lists
mice_behavior = list(set(mice_behavior + mice_pharma_execution + mice_opto))
mice_behavior = sorted(mice_behavior)

len(mice_behavior)

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols = ['exclude',],
    day = ["-2", "-1", '0', '+1', '+2'],
    mouse_id = mice_behavior,
    )
table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_behaviormice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)


mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',
                                    )

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols = ['exclude',  'two_p_exclude'],
    day = ["-2", "-1", '0', '+1', '+2'],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_imaging,
    )
table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)

# Auditory days for imaging mice.
mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',
                                    )

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols = ['exclude',  'two_p_exclude'],
    day = [f"-{i}" for i in range(8,1,-1)],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_imaging,
    )
table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_pretraining_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes'],
                                    pharmacology = ['no',np.nan],
                                    )

# Read behavior results.

particle_test_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',)

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude', 'two_p_exclude'],
    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = particle_test_mice,
    )

table_particle_test = make_behavior_table(nwb_list, session_list, db_path, cut_session=False, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_particle_test.csv'
save_path = io.adjust_path_to_host(save_path)
table_particle_test.to_csv(save_path, index=False)


mice_behavior = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )

mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',
                                    )

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes'],
                                    pharmacology = ['no',np.nan],
                                    )

# # plot_single_session(table, 'GF311_19112020_160412')
# # table.loc[(table.reward_group=='R+') & (table.day==0), 'session_id'].unique()
# # sns.despine()    

# Plot lick trace for a trial.
# ------------------------

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude'],
    # day = ["-2", "-1", '0', '+1', '+2'],
    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    subject_id = mice_imaging,
    )


#     bin_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/data/AR184/Training/AR184_20250319_175559/log_continuous.bin"
#     bin_file = io.adjust_path_to_host(bin_file)
#     # Read binary file using numpy
#     bin_data = np.fromfile(bin_file)
#     ttl = bin_data[2::6]
#     lick_trace = bin_data[1::6]

table = make_behavior_table(nwb_list, session_list, db_path= db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)


# ############################################# 
# Day 0 for each mouse.
# #############################################

# table = pd.read_csv(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_table_muscimol.csv')
# table = table.loc[table.pharma_inactivation_type=='learning']

# session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
#     db_path, nwb_dir, experimenters=['AR'],
#     exclude_cols=['exclude'],
#     pharma_day = ['pre_-1', 'pre_-2', 'muscimol_1', 'muscimol_2', 'muscimol_3', 'recovery_1', 'recovery_2', 'recovery_3'],
#     subject_id = ['AR181', 'AR182', 'AR183', 'AR184',],
#     )

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_table_all_mice_5days.csv')
table = pd.read_csv(table_file)

session_list = table.loc[table.day==0].session_id.drop_duplicates().to_list()
pdf_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/behaviorsingle_sessions_allmice.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)

with PdfPages(pdf_path) as pdf:
    for session_id in session_list:
        print(session_id)
        plot_single_session(table, session_id, ax=None)
        pdf.savefig()
        plt.close()

# Average performance across days.
# ################################

pdf_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/behaviorsingle_sessions_muscimol.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)

with PdfPages(pdf_path) as pdf:
    for session_id in session_list:
        plot_single_session(table, session_id, ax=None)
        pdf.savefig()
        plt.close()


# All five days for a single mouse.
# ################################

table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

# mouse_id = 'AR180'
mouse_id = 'GF305'
# for  mouse_id in mice_imaging:
data = table.loc[table.mouse_id==mouse_id]
data  = data.loc[data.day.isin([-2, -1, 0, 1, 2])]
sessions = data.session_id.drop_duplicates().to_list()
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'xtick.major.width': .8,'ytick.major.width': .8,})

data = data.loc[data.trial_id<=180]

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, session in enumerate(sessions):
    ax = axes[i]
    plot_single_session(
        data, session, ax=ax, palette=behavior_palette, 
         do_scatter=False, linewidth=1.5,
    )
    
# Save the figure.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior/exemples'
plt.savefig(os.path.join(output_dir, f'behavior_single_mouse_{mouse_id}.svg'), dpi=300)



# ################################
# Average performance across days.
# ################################


# table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# # Save the table to a CSV file
# save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
# table.to_csv(save_path, index=False)

# Load table.
table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table = pd.read_csv(table_file)

# Remove spurious whisker trials coming mapping session.
table.loc[table.day.isin([-2, -1]), 'outcome_w'] = np.nan
table.loc[table.day.isin([-2, -1]), 'hr_w'] = np.nan

# Average performance.
table = table.groupby(['mouse_id','session_id','reward_group','day'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg(np.mean)
# Convert performance to percentage
table[['outcome_c', 'outcome_a', 'outcome_w']] = table[['outcome_c', 'outcome_a', 'outcome_w']] * 100

# Plot.
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'xtick.major.width': 1, 'ytick.major.width': 1, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
table['day'] = table['day'].astype(str)  # Convert 'day' column to string for categorical alignment

sns.lineplot(data=table, x='day', y='outcome_c', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[4:6], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)
sns.lineplot(data=table, x='day', y='outcome_a', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[0:2], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)
sns.lineplot(data=table, x='day', y='outcome_w', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[2:4], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)

sns.pointplot(data=table, x='day', y='outcome_c', estimator=np.mean, palette=behavior_palette[4:6], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)
sns.pointplot(data=table, x='day', y='outcome_a', estimator=np.mean, palette=behavior_palette[0:2], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)
sns.pointplot(data=table, x='day', y='outcome_w', estimator=np.mean, palette=behavior_palette[2:4], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)

plt.xlabel('Training days')
plt.ylabel('Lick probability (%)')
# plt.ylim([-0.2, 1.05])
plt.legend()
sns.despine(trim=True)

# Ensure tick thickness is set for SVG output
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_tick_params(width=1)

# Save plot.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'performance_across_days_imagingmice.svg'), dpi=300)
# Save the dataframe to CSV
table.to_csv(os.path.join(output_dir, 'performance_across_days_imagingmice_data.csv'), index=False)


# Histogram quantifying D0.
# -------------------------
# Select data for days 0, +1, and +2
days_of_interest = [0, 1, 2]
day_data = table[table['day'].isin(days_of_interest)]
avg_performance = day_data.groupby(['day', 'mouse_id', 'reward_group'])['outcome_w'].mean().reset_index()

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})

# Plot barplot for each day
plt.figure(figsize=(8, 6))
sns.barplot(
    data=avg_performance,
    x='day',
    y='outcome_w',
    hue='reward_group',
    palette=behavior_palette[2:4][::-1],
    width=0.3,
    dodge=True
)
sns.swarmplot(
    data=avg_performance,
    x='day',
    y='outcome_w',
    hue='reward_group',
    dodge=True,
    color='grey',
    alpha=0.6,
)
plt.xlabel('Day')
plt.ylabel('Lick probability (%)')
plt.ylim([0, 100])
plt.legend(title='Reward group')
sns.despine(trim=True)

# Test significance with Mann-Whitney U test for each day
stats = []
for day in days_of_interest:
    df_day = avg_performance[avg_performance['day'] == day]
    group_R_plus = df_day[df_day['reward_group'] == 'R+']['outcome_w']
    group_R_minus = df_day[df_day['reward_group'] == 'R-']['outcome_w']
    stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
    stats.append({'day': day, 'statistic': stat, 'p_value': p_value})
    # Add stars to the plot to indicate significance
    ax = plt.gca()
    xpos = days_of_interest.index(day)
    ypos = 95
    if p_value < 0.001:
        plt.text(xpos, ypos, '***', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.01:
        plt.text(xpos, ypos, '**', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.05:
        plt.text(xpos, ypos, '*', ha='center', va='bottom', color='black', fontsize=14)

# Save.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'performance_D0_D1_D2_barplot_imagingmice.svg'), dpi=300)
avg_performance.to_csv(os.path.join(output_dir, 'performance_D0_D1_D2_barplot_imagingmice_data.csv'), index=False)
pd.DataFrame(stats).to_csv(os.path.join(output_dir, 'performance_D0_D1_D2_barplot_imagingmice_stats.csv'), index=False)


# Plot first whisker hit for both reward group (sanity check).
# ------------------------------------------------------------
 
# Select data of day 0
day_0_data = table[(table['day'] == 0) & (table.whisker_stim == 1)]
f = lambda x: x.reset_index(drop=True).idxmax()+1
fh = day_0_data.groupby(['mouse_id', 'reward_group'], as_index=False)[['outcome_w']].agg(f)

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
                rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
plt.figure(figsize=(4, 6))
sns.barplot(data=fh, x='reward_group', y='outcome_w', palette=reward_palette[::-1], width=0.3)
sns.stripplot(data=fh, x='reward_group', y='outcome_w', color='grey', jitter=False, dodge=True, alpha=.4)
plt.ylabel('First hit trial')
sns.despine()
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'firsthit_D0.svg'), dpi=300)

# Save the first hit data to CSV
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
fh.to_csv(os.path.join(output_dir, 'firsthit_D0_data.csv'), index=False)

# Mann-Whitney U test for first hit trial between reward groups
stat, p_value = mannwhitneyu(
    fh[fh['reward_group'] == 'R+']['outcome_w'],
    fh[fh['reward_group'] == 'R-']['outcome_w'],
    alternative='two-sided'   
)
# Save stats to file
with open(os.path.join(output_dir, 'firsthit_D0_stats.csv'), 'w') as f:
    f.write(f'Mann-Whitney U Test: Statistic={stat}, P-value={p_value}')
 
# Particle test plots.
# ---------------------

table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_particle_test.csv'
table_path = io.adjust_path_to_host(table_path)
table_particle_test = pd.read_csv(table_path)

output_dir = io.adjust_path_to_host('/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
on_off_order = ['whisker_on_1', 'whisker_off', 'whisker_on_2']


def pval_to_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return 'n.s.'


def draw_stat_bracket(ax, x1, x2, y, p_value, h=3):
    """Draw a significance bracket with stars between positions x1 and x2."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color='black')
    ax.text((x1 + x2) / 2, y + h, pval_to_stars(p_value),
            ha='center', va='bottom', fontsize=8)


def paired_wilcoxon(df, group_col, value_col, group1, group2):
    """Wilcoxon signed-rank test on paired data sorted by mouse_id."""
    g1 = df[df[group_col] == group1].sort_values('mouse_id')[value_col].values
    g2 = df[df[group_col] == group2].sort_values('mouse_id')[value_col].values
    stat, p = wilcoxon(g1, g2, alternative='two-sided')
    return stat, p


# ── Prepare data ─────────────────────────────────────────────────────────────
df_w = table_particle_test.loc[table_particle_test.reward_group == 'R+'].copy()
df_w['outcome_w'] = df_w['outcome_w'] * 100
df_w = df_w.groupby(['mouse_id', 'behavior_type'])['outcome_w'].mean().reset_index()

df_ns = table_particle_test.loc[table_particle_test.reward_group == 'R+'].copy()
df_ns['outcome_c'] = df_ns['outcome_c'] * 100
df_ns = df_ns.groupby(['mouse_id', 'behavior_type'])['outcome_c'].mean().reset_index()

df_off = table_particle_test.loc[
    (table_particle_test.reward_group == 'R+') &
    (table_particle_test.behavior_type == 'whisker_off')
].copy()
df_off['outcome_w'] = df_off['outcome_w'] * 100
df_off['outcome_c'] = df_off['outcome_c'] * 100
df_off = df_off.groupby('mouse_id')[['outcome_w', 'outcome_c']].mean().reset_index()

df_off_long = df_off.melt(id_vars='mouse_id', value_vars=['outcome_w', 'outcome_c'],
                           var_name='trial_type', value_name='lick_rate')
df_off_long['trial_type'] = df_off_long['trial_type'].map(
    {'outcome_w': 'Whisker hit', 'outcome_c': 'False alarm'})

# ── Stats ─────────────────────────────────────────────────────────────────────
stat1, p1 = paired_wilcoxon(df_w,  'behavior_type', 'outcome_w', 'whisker_on_1', 'whisker_off')
stat2, p2 = paired_wilcoxon(df_w,  'behavior_type', 'outcome_w', 'whisker_off',  'whisker_on_2')
stat3, p3 = paired_wilcoxon(df_ns, 'behavior_type', 'outcome_c', 'whisker_on_1', 'whisker_off')
stat4, p4 = paired_wilcoxon(df_ns, 'behavior_type', 'outcome_c', 'whisker_off',  'whisker_on_2')
stat5, p5 = wilcoxon(df_off['outcome_w'].values, df_off['outcome_c'].values, alternative='two-sided')

# ── Combined figure: 3 panels ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

# Panel 1: whisker hit rate ON/OFF/ON
ax = axes[0]
sns.barplot(data=df_w, x='behavior_type', y='outcome_w', order=on_off_order,
            color=trial_type_rew_palette[3], ax=ax)
for mouse_id in df_w.mouse_id.unique():
    v = {bt: df_w.loc[(df_w.mouse_id == mouse_id) & (df_w.behavior_type == bt), 'outcome_w'].to_numpy()
         for bt in on_off_order}
    if all(len(x) > 0 for x in v.values()):
        ax.plot([0, 1], [v['whisker_on_1'][0], v['whisker_off'][0]], color='grey', linewidth=1, alpha=0.8)
        ax.plot([1, 2], [v['whisker_off'][0], v['whisker_on_2'][0]], color='grey', linewidth=1, alpha=0.8)
draw_stat_bracket(ax, 0, 1, 88, p1)
draw_stat_bracket(ax, 1, 2, 88, p2)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['ON', 'OFF', 'ON'])
ax.set_ylim(0, 100)
ax.set_ylabel('Lick rate (%)')
ax.set_xlabel('Particle test')
ax.set_title('Whisker hit rate')

# Panel 2: no-stim false alarm rate ON/OFF/ON
ax = axes[1]
sns.barplot(data=df_ns, x='behavior_type', y='outcome_c', order=on_off_order,
            color=trial_type_rew_palette[5], ax=ax)
for mouse_id in df_ns.mouse_id.unique():
    v = {bt: df_ns.loc[(df_ns.mouse_id == mouse_id) & (df_ns.behavior_type == bt), 'outcome_c'].to_numpy()
         for bt in on_off_order}
    if all(len(x) > 0 for x in v.values()):
        ax.plot([0, 1], [v['whisker_on_1'][0], v['whisker_off'][0]], color='grey', linewidth=1, alpha=0.8)
        ax.plot([1, 2], [v['whisker_off'][0], v['whisker_on_2'][0]], color='grey', linewidth=1, alpha=0.8)
draw_stat_bracket(ax, 0, 1, 88, p3)
draw_stat_bracket(ax, 1, 2, 88, p4)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['ON', 'OFF', 'ON'])
ax.set_ylim(0, 100)
ax.set_xlabel('Particle test')
ax.set_title('False alarm rate')

# Panel 3: OFF only — whisker hit vs false alarm
ax = axes[2]
sns.barplot(data=df_off_long, x='trial_type', y='lick_rate',
            order=['Whisker hit', 'False alarm'],
            palette={'Whisker hit': trial_type_rew_palette[3],
                     'False alarm': trial_type_rew_palette[5]},
            ax=ax)
for mouse_id in df_off.mouse_id.unique():
    row = df_off[df_off.mouse_id == mouse_id]
    ax.plot([0, 1], [row['outcome_w'].values[0], row['outcome_c'].values[0]],
            color='grey', linewidth=1, alpha=0.8)
draw_stat_bracket(ax, 0, 1, 88, p5)
ax.set_ylim(0, 100)
ax.set_xlabel('OFF period')
ax.set_title('OFF: whisker vs FA')

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'particle_test_imagingmice.svg'), dpi=300)

# ── Save data and stats ───────────────────────────────────────────────────────
table_particle_test.to_csv(os.path.join(output_dir, 'particle_test_imagingmice_data.csv'), index=False)
df_off.to_csv(os.path.join(output_dir, 'particle_test_off_imagingmice_data.csv'), index=False)

pd.DataFrame([
    {'comparison': 'whisker ON1 vs OFF',   'statistic': stat1, 'p_value': p1},
    {'comparison': 'whisker OFF vs ON2',   'statistic': stat2, 'p_value': p2},
    {'comparison': 'no-stim ON1 vs OFF',   'statistic': stat3, 'p_value': p3},
    {'comparison': 'no-stim OFF vs ON2',   'statistic': stat4, 'p_value': p4},
    {'comparison': 'OFF: whisker hit vs FA', 'statistic': stat5, 'p_value': p5},
]).to_csv(os.path.join(output_dir, 'particle_test_imagingmice_stats.csv'), index=False)



# #####################################################################
# Performance over blocks during whisker learning sessions across mice.
# #####################################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
# table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_behaviormice_table_5days_cut.csv')
table = pd.read_csv(table_file)
table.mouse_id.unique().size


# Performance over blocks.
# ------------------------


data = table.copy()    
data['block_id'] = data.block_id + 1  # Start block index at 1 for plot.

# Performance over blocks
fig, axes = plt.subplots(1, 5, sharey=True, figsize=(20, 4))
plot_perf_across_blocks(data, "R-", -2, behavior_palette, nmax_trials=240, ax=axes[0])
plot_perf_across_blocks(data, "R-", -1, behavior_palette, nmax_trials=240, ax=axes[1])
plot_perf_across_blocks(data, "R-", 0, behavior_palette, nmax_trials=240, ax=axes[2])
plot_perf_across_blocks(data, "R-", 1, behavior_palette, nmax_trials=240, ax=axes[3])
plot_perf_across_blocks(data, "R-", 2, behavior_palette, nmax_trials=240, ax=axes[4])

plot_perf_across_blocks(data, "R+", -2, behavior_palette, nmax_trials=240, ax=axes[0])
plot_perf_across_blocks(data, "R+", -1, behavior_palette, nmax_trials=240, ax=axes[1])
plot_perf_across_blocks(data, "R+", 0, behavior_palette, nmax_trials=240, ax=axes[2])
plot_perf_across_blocks(data, "R+", 1, behavior_palette, nmax_trials=240, ax=axes[3])
plot_perf_across_blocks(data, "R+", 2, behavior_palette, nmax_trials=240, ax=axes[4])


# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, 'performance_over_blocks_imagingmice.svg')
plt.savefig(output_file, format='svg', dpi=300)



# Performance during sessions across mice with fitted learning curves.
# --------------------------------------------------------------------

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

table.columns

# Parameter: maximum number of trials per type to plot
max_trials_per_type = 120

# Performance over blocks
fig, axes = plt.subplots(1, 1, sharey=True, figsize=(8, 4)); axes = [axes]


for i, day in enumerate([0]):
    ax = axes[i]
    for reward_group, color_w, color_a, color_ns in zip(
        ['R-', 'R+'],
        behavior_palette[2:4],  # whisker
        behavior_palette[0:2],  # auditory
        behavior_palette[4:6],  # no_stim
    ):
        d = table[(table.day == day) & (table.reward_group == reward_group)]
        # Cut to max_trials_per_type for each trial type
        d_whisker = d.loc[d.trial_w<max_trials_per_type]
        d_auditory = d.loc[d.trial_a<max_trials_per_type]
        d_nostim = d.loc[d.trial_c<max_trials_per_type]
        
        # # Plot no_stim learning curve
        # sns.lineplot(
        #     data=d_nostim,
        #     x='trial_c', y='learning_curve_ns',
        #     errorbar='ci', ax=ax, color=color_ns, label=f'{reward_group} no_stim', linewidth=2,
        #     err_kws={'edgecolor': 'none'}
        # )
        
        # Plot whisker learning curve
        sns.lineplot(
            data=d_whisker,
            x='trial_w', y='learning_curve_w',
            errorbar='ci', ax=ax, color=color_w, label=f'{reward_group} whisker', linewidth=2,
            err_kws={'edgecolor': 'none'}
        )
        # # Plot auditory learning curve
        # sns.lineplot(
        #     data=d_auditory,
        #     x='trial_a', y='learning_curve_a',
        #     errorbar='ci', ax=ax, color=color_a, label=f'{reward_group} auditory', linewidth=2
        # )
    ax.set_title(f'Day {day}')
    ax.set_xlabel('Whisker trial')
axes[0].set_ylabel('Lick probability')
sns.despine()

# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, 'performance_over_blocks_imagingmice.svg')
plt.savefig(output_file, format='svg', dpi=300)




# Performance during day 0: comparing reward groups (2x2 layout)
# ----------------------------------------------------------------

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

# Parameter: maximum number of trials per type to plot
max_trials_per_type = 100
day = 0

# Prepare non-realigned data (whisker trials on day 0)
df_non_realigned = table.loc[(table.whisker_stim == 1) & (table.day == day)]
df_non_realigned = df_non_realigned.loc[df_non_realigned.trial_w < max_trials_per_type]

# Prepare realigned data (aligned to first hit for each mouse)
def realign_to_first_hit(data):
    """Realign trial numbers to first hit for each mouse."""
    realigned_data = []

    for mouse in data['mouse_id'].unique():
        mouse_data = data[data['mouse_id'] == mouse].copy()
        mouse_data = mouse_data.sort_values('trial_w')

        # Find first hit
        first_hit_idx = mouse_data[mouse_data['outcome_w'] == 1].index
        if len(first_hit_idx) > 0:
            first_hit_trial = mouse_data.loc[first_hit_idx[0], 'trial_w']
            # Create realigned trial number
            mouse_data['trial_w_realigned'] = mouse_data['trial_w'] - first_hit_trial
            realigned_data.append(mouse_data)

    return pd.concat(realigned_data, ignore_index=True)

df_realigned = realign_to_first_hit(df_non_realigned)
df_realigned = df_realigned.loc[df_realigned.trial_w_realigned >= 0]
df_realigned = df_realigned.loc[df_realigned.trial_w_realigned < max_trials_per_type]

# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Create colormap for p-values
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('pval_cmap', ['black', 'white'])
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)

# Store all stats for saving
all_stats = {}

# Panel (0,0): Single trial performance, non-realigned
ax = axes[0, 0]
sns.lineplot(data=df_non_realigned, x='trial_w', y='outcome_w',
            palette=reward_palette[::-1], hue='reward_group',
            errorbar='ci', err_style='band', ax=ax, legend=False)

# Statistical test
p_values = []
for trial_w in sorted(df_non_realigned['trial_w'].unique()):
    group_R_plus = df_non_realigned[(df_non_realigned['trial_w'] == trial_w) & (df_non_realigned['reward_group'] == 'R+')]['outcome_w']
    group_R_minus = df_non_realigned[(df_non_realigned['trial_w'] == trial_w) & (df_non_realigned['reward_group'] == 'R-')]['outcome_w']
    if len(group_R_plus) > 0 and len(group_R_minus) > 0:
        stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
        p_values.append((trial_w, p_value))

# FDR correction
if p_values:
    trials, raw_pvals = zip(*p_values)
    _, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
    p_values = list(zip(trials, corrected_pvals))
    all_stats['non_realigned_single'] = p_values

    # Plot p-value rectangles
    for trial, p_value in p_values:
        color = cmap(norm(min(p_value, 0.05)))
        ax.add_patch(plt.Rectangle((trial - 0.4, 0.95), 0.8, 0.03, color=color, edgecolor='none'))

ax.set_title('Single trial (non-realigned)')
ax.set_xlabel('Whisker trial')
ax.set_ylabel('Hit rate')
ax.set_ylim([-0.1, 1])

# Panel (0,1): Fitted learning curve, non-realigned
ax = axes[0, 1]
sns.lineplot(data=df_non_realigned, x='trial_w', y='learning_curve_w',
            palette=reward_palette[::-1], hue='reward_group',
            errorbar='ci', err_style='band', ax=ax, legend=False)

# Statistical test
p_values = []
for trial_w in sorted(df_non_realigned['trial_w'].unique()):
    group_R_plus = df_non_realigned[(df_non_realigned['trial_w'] == trial_w) & (df_non_realigned['reward_group'] == 'R+')]['learning_curve_w']
    group_R_minus = df_non_realigned[(df_non_realigned['trial_w'] == trial_w) & (df_non_realigned['reward_group'] == 'R-')]['learning_curve_w']
    if len(group_R_plus) > 0 and len(group_R_minus) > 0:
        stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
        p_values.append((trial_w, p_value))

# FDR correction
if p_values:
    trials, raw_pvals = zip(*p_values)
    _, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
    p_values = list(zip(trials, corrected_pvals))
    all_stats['non_realigned_learning'] = p_values

    # Plot p-value rectangles
    for trial, p_value in p_values:
        color = cmap(norm(min(p_value, 0.05)))
        ax.add_patch(plt.Rectangle((trial - 0.4, 0.95), 0.8, 0.03, color=color, edgecolor='none'))

ax.set_title('Fitted learning curve (non-realigned)')
ax.set_xlabel('Whisker trial')
ax.set_ylabel('Lick probability')
ax.set_ylim([-0.1, 1])

# Panel (1,0): Single trial performance, realigned to first hit
ax = axes[1, 0]
sns.lineplot(data=df_realigned, x='trial_w_realigned', y='outcome_w',
            palette=reward_palette[::-1], hue='reward_group',
            errorbar='ci', err_style='band', ax=ax, legend=False)

# Statistical test
p_values = []
for trial_w in sorted(df_realigned['trial_w_realigned'].unique()):
    group_R_plus = df_realigned[(df_realigned['trial_w_realigned'] == trial_w) & (df_realigned['reward_group'] == 'R+')]['outcome_w']
    group_R_minus = df_realigned[(df_realigned['trial_w_realigned'] == trial_w) & (df_realigned['reward_group'] == 'R-')]['outcome_w']
    if len(group_R_plus) > 0 and len(group_R_minus) > 0:
        stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
        p_values.append((trial_w, p_value))

# FDR correction
if p_values:
    trials, raw_pvals = zip(*p_values)
    _, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
    p_values = list(zip(trials, corrected_pvals))
    all_stats['realigned_single'] = p_values

    # Plot p-value rectangles
    for trial, p_value in p_values:
        color = cmap(norm(min(p_value, 0.05)))
        ax.add_patch(plt.Rectangle((trial - 0.4, 0.95), 0.8, 0.03, color=color, edgecolor='none'))

ax.set_title('Single trial (realigned to first hit)')
ax.set_xlabel('Whisker trial (from first hit)')
ax.set_ylabel('Hit rate')
ax.set_ylim([-0.1, 1])

# Panel (1,1): Fitted learning curve, realigned to first hit
ax = axes[1, 1]
sns.lineplot(data=df_realigned, x='trial_w_realigned', y='learning_curve_w',
            palette=reward_palette[::-1], hue='reward_group',
            errorbar='ci', err_style='band', ax=ax)

# Statistical test
p_values = []
for trial_w in sorted(df_realigned['trial_w_realigned'].unique()):
    group_R_plus = df_realigned[(df_realigned['trial_w_realigned'] == trial_w) & (df_realigned['reward_group'] == 'R+')]['learning_curve_w']
    group_R_minus = df_realigned[(df_realigned['trial_w_realigned'] == trial_w) & (df_realigned['reward_group'] == 'R-')]['learning_curve_w']
    if len(group_R_plus) > 0 and len(group_R_minus) > 0:
        stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
        p_values.append((trial_w, p_value))

# FDR correction
if p_values:
    trials, raw_pvals = zip(*p_values)
    _, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
    p_values = list(zip(trials, corrected_pvals))
    all_stats['realigned_learning'] = p_values

    # Plot p-value rectangles
    for trial, p_value in p_values:
        color = cmap(norm(min(p_value, 0.05)))
        ax.add_patch(plt.Rectangle((trial - 0.4, 0.95), 0.8, 0.03, color=color, edgecolor='none'))

ax.set_title('Fitted learning curve (realigned to first hit)')
ax.set_xlabel('Whisker trial (from first hit)')
ax.set_ylabel('Lick probability')
ax.set_ylim([-0.1, 1])
ax.legend(frameon=False, title='Reward group')

sns.despine()
plt.tight_layout()

# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, 'performance_D0_comparison_2x2.svg')
plt.savefig(output_file, format='svg', dpi=300)

# Save all data and statistics
df_non_realigned.to_csv(os.path.join(output_dir, 'performance_D0_non_realigned_data.csv'), index=False)
df_realigned.to_csv(os.path.join(output_dir, 'performance_D0_realigned_data.csv'), index=False)

# Save all p-values
for key, pvals in all_stats.items():
    pd.DataFrame(pvals, columns=['trial_w', 'p_value']).to_csv(
        os.path.join(output_dir, f'performance_D0_{key}_stats.csv'), index=False)



# Performance during day 0: whisker, auditory, no-stim on a common time axis
# ---------------------------------------------------------------------------
# Single panel. R+ and R- share the same axis; color encodes stim type and
# reward group. X-axis is time (min) from session start, cut at the 100th
# whisker trial per mouse.

table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

day = 0
df_day0 = table.loc[table.day == day].copy()

stim_type_map = {
    'whisker':  ('whisker_stim',  'learning_curve_w'),
    'auditory': ('auditory_stim', 'learning_curve_a'),
    'no_stim':  ('no_stim',       'learning_curve_ns'),
}
stim_labels = {'whisker': 'Whisker', 'auditory': 'Auditory', 'no_stim': 'No stim'}
rg_stim_colors = {
    'R+': {'whisker': behavior_palette[3], 'auditory': behavior_palette[1], 'no_stim': behavior_palette[5]},
    'R-': {'whisker': behavior_palette[2], 'auditory': behavior_palette[0], 'no_stim': behavior_palette[4]},
}

time_resolution  = 5    # seconds
max_whisker_trials = 100
max_time = df_day0['stim_onset'].max()
time_grid = np.arange(0, max_time + time_resolution, time_resolution)

interp_records = []
for mouse in df_day0['mouse_id'].unique():
    mouse_data = df_day0[df_day0['mouse_id'] == mouse]
    reward_group = mouse_data['reward_group'].iloc[0]

    whisker_onsets = (mouse_data.loc[mouse_data['whisker_stim'] == 1, 'stim_onset']
                      .dropna().sort_values().reset_index(drop=True))
    if len(whisker_onsets) == 0:
        continue
    cutoff_time = whisker_onsets.iloc[min(max_whisker_trials, len(whisker_onsets)) - 1]

    for stim_name, (stim_col, lc_col) in stim_type_map.items():
        if lc_col not in mouse_data.columns:
            continue
        stim_data = mouse_data.loc[mouse_data[stim_col] == 1, ['stim_onset', lc_col]].dropna()
        if len(stim_data) < 2:
            continue

        stim_times = stim_data['stim_onset'].values
        lc_values  = stim_data[lc_col].values

        t_mouse = time_grid[time_grid <= cutoff_time]
        lc_interp = np.interp(t_mouse, stim_times, lc_values)

        for t, lc in zip(t_mouse, lc_interp):
            interp_records.append({
                'mouse_id': mouse, 'reward_group': reward_group,
                'stim_type': stim_name,
                'time': t / 60,
                'lick_probability': lc,
            })

df_interp = pd.DataFrame(interp_records)

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))

for rg in ['R+', 'R-']:
    df_rg = df_interp[df_interp['reward_group'] == rg]
    for stim_name in ['whisker', 'auditory', 'no_stim']:
        df_stim = df_rg[df_rg['stim_type'] == stim_name]
        sns.lineplot(
            data=df_stim, x='time', y='lick_probability',
            color=rg_stim_colors[rg][stim_name],
            errorbar='ci', err_style='band',
            label=f'{stim_labels[stim_name]} {rg}', ax=ax,
        )

ax.set_xlabel('Time from session start (min)')
ax.set_ylabel('Lick probability')
ax.set_ylim([-0.1, 1.05])
ax.set_xlim(right=50)

sns.despine()
plt.tight_layout()

output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, 'performance_D0_all_stim_time_axis.svg')
plt.savefig(output_file, format='svg', dpi=300)

df_interp.to_csv(os.path.join(output_dir, 'performance_D0_all_stim_time_axis_data.csv'), index=False)


# ############################################################
# Time of 22nd whisker trial during day 0 across mice.
# The 22nd whisker trial is the first trial at which R+ and R-
# diverge significantly in whisker performance.
# ############################################################

WHISKER_TRIAL_OF_INTEREST = 22

table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

df_day0_whisker = table.loc[(table.day == 0) & (table.whisker_stim == 1)].copy()
df_day0_whisker = df_day0_whisker.sort_values(['mouse_id', 'stim_onset'])

# For each mouse, find the stim_onset of the Nth whisker trial (1-indexed).
records = []
for mouse, mouse_data in df_day0_whisker.groupby('mouse_id'):
    mouse_data = mouse_data.reset_index(drop=True)
    if len(mouse_data) >= WHISKER_TRIAL_OF_INTEREST:
        trial_row = mouse_data.iloc[WHISKER_TRIAL_OF_INTEREST - 1]
        records.append({
            'mouse_id': mouse,
            'reward_group': trial_row['reward_group'],
            'stim_onset_s': trial_row['stim_onset'],
            'stim_onset_min': trial_row['stim_onset'] / 60,
        })

df_t22 = pd.DataFrame(records)

# Average across all mice.
mean_min = df_t22['stim_onset_min'].mean()
std_min = df_t22['stim_onset_min'].std()
n_mice = len(df_t22)
print(f"\nTime of whisker trial #{WHISKER_TRIAL_OF_INTEREST} from session start (day 0):")
print(f"  {mean_min:.2f} ± {std_min:.2f} min  (N={n_mice} mice)")

# Plot: single bar for all mice.
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(1, 1, figsize=(2, 4))
df_t22['all'] = ''
sns.barplot(data=df_t22, x='all', y='stim_onset_min', color='grey', width=0.4, ax=ax)
sns.stripplot(data=df_t22, x='all', y='stim_onset_min', color='black', jitter=False, alpha=0.6, ax=ax)
ax.set_xlabel('')
ax.set_ylabel(f'Time at whisker trial #{WHISKER_TRIAL_OF_INTEREST} (min)')
ax.set_title(f'Whisker trial #{WHISKER_TRIAL_OF_INTEREST}\n{mean_min:.1f} ± {std_min:.1f} min')
sns.despine()
plt.tight_layout()

# Save.
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, f'time_whisker_trial_{WHISKER_TRIAL_OF_INTEREST}.svg'), dpi=300)
df_t22.drop(columns='all').to_csv(os.path.join(output_dir, f'time_whisker_trial_{WHISKER_TRIAL_OF_INTEREST}_data.csv'), index=False)


# ############################################################
# Muscimol inactivation.
# ############################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_muscimol = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    )

# Read behavior results.
session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude'],
    pharma_inactivation_type = ['learning'],
    pharma_day = ["pre_-2", "pre_-1",
            "muscimol_1", "muscimol_2", "muscimol_3",
            "recovery_1", "recovery_2", "recovery_3"],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_muscimol,
    )

# table_muscimol_learning = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=io.stop_flags_yaml, trial_indices_yaml=io.trial_indices_yaml)
# # Save the table to a CSV file
# save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_muscimol.csv'
# save_path = io.adjust_path_to_host(save_path)
# table_muscimol_learning.to_csv(save_path, index=False)

# Load table.
table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_muscimol.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

fpS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    pharma_inactivation_type = 'learning',
                                    pharma_area = 'fpS1',
                                    )

wS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    pharma_inactivation_type = 'learning',
                                    pharma_area = 'wS1',
                                    )

table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1' 
table = pd.merge(
    table,
    db[['mouse_id', 'session_id', 'pharma_day']],
    on=['mouse_id', 'session_id'],
    how='left'
)
inactivation_labels = ['pre_-2', 'pre_-1', 'muscimol_1', 'muscimol_2', 'muscimol_3', 'recovery_1', 'recovery_2', 'recovery_3']

data = table.groupby(['mouse_id', 'session_id', 'pharma_day', 'area'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg('mean')
# Order the data by inactivation labels for each session
data['pharma_day'] = pd.Categorical(data['pharma_day'], categories=inactivation_labels, ordered=True)
data = data.sort_values(by=['mouse_id', 'pharma_day'])
# Convert performance to percentage.
data['outcome_c'] = data['outcome_c'] * 100
data['outcome_a'] = data['outcome_a'] * 100
data['outcome_w'] = data['outcome_w'] * 100


# Plot inactivation for wS1 and fpS1.
# -----------------------------------

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,5))

ax = axes[0]

for imouse in wS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

ax = axes[1]

for imouse in fpS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

for ax in axes:
    ax.set_yticks([0,20, 40, 60, 80, 100])
    ax.set_xticklabels(['-2', '-1', 'M 1', 'M 2', 'M 3', 'R 1', 'R 2', 'R 3'])
    ax.set_xlabel('Muscimol inactivation during learning')
    ax.set_ylabel('Lick probability (%)')
sns.despine(trim=True)

# Save figure
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'muscimol_learning.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data.
data.to_csv(os.path.join(output_dir, 'muscimol_learning_data.csv'), index=False)

# Bar plot and stats for Day 0 (muscimol_1), Day +1 (muscimol_2), and Day +2 (muscimol_3)
# ----------------------------------------------------------------------------------------

days_of_interest = ['muscimol_1', 'muscimol_2', 'muscimol_3']
day_labels = ['D0', 'D+1', 'D+2']

day_data = data[data['pharma_day'].isin(days_of_interest)].copy()
day_data['day_label'] = day_data['pharma_day'].map(dict(zip(days_of_interest, day_labels)))

plt.figure(figsize=(8, 6))
sns.barplot(
    data=day_data,
    x='day_label',
    y='outcome_w',
    hue='area',
    palette=[reward_palette[1]],
    width=0.3,
    dodge=True
)
sns.swarmplot(
    data=day_data,
    x='day_label',
    y='outcome_w',
    hue='area',
    dodge=True,
    color=stim_palette[2],
    alpha=0.6
)
plt.xlabel('Day')
plt.ylabel('Whisker Performance (%)')
plt.ylim([0, 100])
plt.legend(title='Area')
sns.despine()

# Perform Mann-Whitney U test for each day between wS1 and fpS1
stats = []
for day, label in zip(days_of_interest, day_labels):
    df_day = day_data[day_data['pharma_day'] == day]
    group_wS1 = df_day[df_day['area'] == 'wS1']['outcome_w']
    group_fpS1 = df_day[df_day['area'] == 'fpS1']['outcome_w']
    stat, p_value = mannwhitneyu(group_wS1, group_fpS1, alternative='two-sided')
    stats.append({'day': label, 'statistic': stat, 'p_value': p_value})
    # Add significance stars to the plot
    ax = plt.gca()
    xpos = day_labels.index(label)
    ypos = 95
    if p_value < 0.001:
        plt.text(xpos, ypos, '***', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.01:
        plt.text(xpos, ypos, '**', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.05:
        plt.text(xpos, ypos, '*', ha='center', va='bottom', color='black', fontsize=14)
    # Add p-value text below stars
    plt.text(xpos, 90, f'p={p_value:.3g}', ha='center', va='bottom', color='black', fontsize=10)

# Save the results to CSV files
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, 'muscimol_learning_day0_day1_day2.svg'), format='svg', dpi=300)
day_data.to_csv(os.path.join(output_dir, 'muscimol_learning_day0_day1_day2_data.csv'), index=False)
pd.DataFrame(stats).to_csv(os.path.join(output_dir, 'muscimol_learning_day0_day1_day2_stats.csv'), index=False)


# ###################
# Reaction time plot.
# ###################

# Reaction time: per stim type across days (bar) + across day 0 trials (line)
# ----------------------------------------------------------------------------

table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table = pd.read_csv(table_file)

table = table[table['lick_flag']==1]
# Reindex with iht trials only.
table['trial_w'] = table.groupby(['mouse_id', 'session_id', 'trial_type']).cumcount()
table['trial_a'] = table.groupby(['mouse_id', 'session_id', 'trial_type']).cumcount()
table['trial_c'] = table.groupby(['mouse_id', 'session_id', 'trial_type']).cumcount()

table['reaction_time'] = table['lick_time'] - table['stim_onset']
max_trials_rt = 100

# Correcting spurious lick time entries on catch trials (30 data points from two mice)).
# 1.3 because of artefact window.
table.loc[table['reaction_time'] > 1.3, 'reaction_time'] = np.nan

# Stim type definitions:
#   (filter_col, outcome_col, trial_col, label, days_to_plot, rp_idx, rm_idx)
# Color indices into trial_type_rew_palette / trial_type_nonrew_palette:
#   [0]=auditory miss (cyan)  [1]=auditory hit (blue)
#   [2]=whisker miss          [3]=whisker hit
#   [4]=no-stim CR (grey)     [5]=no-stim FA (dark)
# Auditory and no-stim share the same hit/CR colors in both palettes, so R-
# uses the lighter (miss/CR) variant from the nonrew palette for contrast.
stim_defs_rt = [
    ('auditory_stim', 'outcome_a', 'trial_a', 'Auditory', [-2, -1, 0, 1, 2], 1, 0),
    ('whisker_stim',  'outcome_w', 'trial_w', 'Whisker',  [-2, -1, 0, 1, 2],          3, 3),
    ('no_stim',       'outcome_c', 'trial_c', 'No stim',  [-2, -1, 0, 1, 2], 5, 4),
]

# ── Per-mouse mean reaction time per stim type × day (for bar plot) ──────────
rt_rows = []
for stim_col, outcome_col, trial_col, stim_label, days_plot, _, _ in stim_defs_rt:
    df_stim = table.loc[(table[stim_col] == 1) & (table[outcome_col] == 1)]
    for (mouse_id, reward_group, day), grp in df_stim.groupby(['mouse_id', 'reward_group', 'day']):
        if day not in days_plot:
            continue


        rt_rows.append({
            'mouse_id':      mouse_id,
            'reward_group':  reward_group,
            'day':           day,
            'stim_type':     stim_label,
            'reaction_time': grp['reaction_time'].mean(),
        })
rt_df = pd.DataFrame(rt_rows)

# ── Bar plot: mean RT per stim type across days ───────────────────────────────
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for ax, (stim_col, outcome_col, trial_col, stim_label, days_plot, rpi, rmi) in zip(axes, stim_defs_rt):
    color_rp = trial_type_rew_palette[rpi]
    color_rm = trial_type_nonrew_palette[rmi]
    palette  = {'R+': color_rp, 'R-': color_rm}

    df_plot = rt_df[rt_df['stim_type'] == stim_label]

    sns.barplot(
        data=df_plot, x='day', y='reaction_time',
        hue='reward_group', hue_order=['R+', 'R-'],
        palette=palette,
        order=days_plot,
        errorbar='ci', capsize=0.05,
        alpha=0.8, ax=ax,
    )

    # Individual mouse dots in grey
    day_positions = {d: i for i, d in enumerate(days_plot)}
    bar_width = 0.35
    group_offsets = {'R+': -bar_width / 2, 'R-': bar_width / 2}

    for mouse_id in df_plot['mouse_id'].unique():
        mouse_data = df_plot[
            (df_plot['mouse_id'] == mouse_id) & (df_plot['day'].isin(days_plot))
        ].sort_values('day')
        rg = mouse_data['reward_group'].iloc[0]
        xs = [day_positions[d] + group_offsets[rg] for d in mouse_data['day']]
        ax.scatter(xs, mouse_data['reaction_time'].values,
                   color='grey', s=8, alpha=0.5, zorder=5, linewidths=0)

    ax.set_title(stim_label)
    ax.set_xlabel('Day')
    ax.set_ylabel('Reaction time (s)' if ax is axes[0] else '')
    ax.legend(frameon=False)

sns.despine()
plt.tight_layout()
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, 'reaction_time_per_stim_across_days.svg'), format='svg', dpi=300)

# ── Line plot: mean RT across trials within day 0 ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for ax, (stim_col, outcome_col, trial_col, stim_label, _, rpi, rmi) in zip(axes, stim_defs_rt):
    color_rp = trial_type_rew_palette[rpi]
    color_rm = trial_type_nonrew_palette[rmi]

    df_stim = table.loc[
        (table[stim_col] == 1) &
        (table[outcome_col] == 1) &
        (table['day'] == 0) &
        (table[trial_col] < max_trials_rt)
    ]

    for rg, color in [('R+', color_rp), ('R-', color_rm)]:
        df_rg = df_stim[df_stim['reward_group'] == rg]
        mouse_counts = df_rg.groupby(trial_col)['mouse_id'].nunique()
        valid_trials = mouse_counts[mouse_counts >= 5].index
        df_rg = df_rg[df_rg[trial_col].isin(valid_trials)]
        sns.lineplot(
            data=df_rg,
            x=trial_col, y='reaction_time',
            color=color, errorbar='ci', err_style='band',
            label=rg, ax=ax,
        )
        # mouse_means = df_rg.groupby(['mouse_id', trial_col])['reaction_time'].mean().reset_index()
        # ax.scatter(
        #     mouse_means[trial_col], mouse_means['reaction_time'],
        #     color=color, s=8, alpha=0.3, zorder=3, linewidths=0,
        # )

    ax.set_title(stim_label)
    ax.set_xlabel('Hit trials')
    ax.set_ylabel('Reaction time (s)' if ax is axes[0] else '')
    ax.set_ylim(bottom=0, top=1)

    ax.legend(frameon=False)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reaction_time_day0_per_stim_line.svg'), format='svg', dpi=300)






# ############################################################
# Opto inactivation.
# ############################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols=['exclude', 'opto_exclude'],
                                    opto_inactivation_type = ['learning'],
                                    optogenetic = 'yes',
                                    )

# Read behavior results.
session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude', 'opto_exclude'],
    opto_inactivation_type = ['learning'],
    opto_day = ["pre_-2", "pre_-1",
            "opto", "recovery_1",],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_opto,
    )

table_opto_learning = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=io.stop_flags_yaml, trial_indices_yaml=io.trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_opto_learning.csv'
save_path = io.adjust_path_to_host(save_path)
table_opto_learning.to_csv(save_path, index=False)

# Load table.
table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_opto_learning.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

fpS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude', 'opto_exclude'],
                                    optogenetic = 'yes',
                                    opto_inactivation_type = 'learning',
                                    opto_area = 'fpS1',
                                    )

wS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude', 'opto_exclude'],
                                    optogenetic = 'yes',
                                    opto_inactivation_type = 'learning',
                                    opto_area = 'wS1',
                                    )

table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1' 
table = pd.merge(
    table,
    db[['mouse_id', 'session_id', 'opto_day']],
    on=['mouse_id', 'session_id'],
    how='left'
)
inactivation_labels = ['pre_-2', 'pre_-1', 'opto', 'recovery_1']

data = table.groupby(['mouse_id', 'session_id', 'opto_day', 'area'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg('mean')
# Order the data by inactivation labels for each session
data['opto_day'] = pd.Categorical(data['opto_day'], categories=inactivation_labels, ordered=True)
data = data.sort_values(by=['mouse_id', 'opto_day'])
# Convert performance to percentage.
data['outcome_c'] = data['outcome_c'] * 100
data['outcome_a'] = data['outcome_a'] * 100
data['outcome_w'] = data['outcome_w'] * 100


# Plot inactivation for wS1 and fpS1.
# -----------------------------------

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,5))

ax = axes[0]

for imouse in wS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

# Add dots for individual points
sns.stripplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
              y='outcome_c', order=inactivation_labels,
              color=stim_palette[2], ax=ax, jitter=False, dodge=True, alpha=0.5, size=4)
sns.stripplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
              y='outcome_a', order=inactivation_labels,
              color=stim_palette[0], ax=ax, jitter=False, dodge=True, alpha=0.5, size=4)
sns.stripplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
              y='outcome_w', order=inactivation_labels,
              color=stim_palette[1], ax=ax, jitter=False, dodge=True, alpha=0.5, size=4)

ax = axes[1]

for imouse in fpS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

for ax in axes:
    ax.set_yticks([0,20, 40, 60, 80, 100])
    ax.set_xticklabels(['-2', '-1', '0', '+1'])
    ax.set_xlabel('Optogenetic inactivation during learning')
    ax.set_ylabel('Lick probability (%)')
sns.despine(trim=True)

# Save figure
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'opto_learning.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data.
data.to_csv(os.path.join(output_dir, 'opto_learning_data.csv'), index=False)

# Bar plot and stats for Day 0 (opto), Day +1 (recovery_1)
# ----------------------------------------------------------

days_of_interest = ['opto', 'recovery_1']
day_labels = ['D0', 'D+1']

day_data = data[data['opto_day'].isin(days_of_interest)].copy()
day_data['day_label'] = day_data['opto_day'].map(dict(zip(days_of_interest, day_labels)))

plt.figure(figsize=(8, 6))
sns.barplot(
    data=day_data,
    x='day_label',
    y='outcome_w',
    hue='area',
    palette=[stim_palette[1]],
    width=0.3,
    dodge=True,
    order=day_labels,
    hue_order=['wS1', 'fpS1']
)
sns.swarmplot(
    data=day_data,
    x='day_label',
    y='outcome_w',
    hue='area',
    dodge=True,
    color='black',
    alpha=0.6,
    order=day_labels,
    hue_order=['wS1', 'fpS1']
)
plt.xlabel('Day')
plt.ylabel('Whisker Performance (%)')
plt.ylim([0, 100])
plt.legend(title='Area')
sns.despine()

# Perform Mann-Whitney U test for each day between wS1 and fpS1
stats = []
for day, label in zip(days_of_interest, day_labels):
    df_day = day_data[day_data['opto_day'] == day]
    group_wS1 = df_day[df_day['area'] == 'wS1']['outcome_w']
    group_fpS1 = df_day[df_day['area'] == 'fpS1']['outcome_w']
    stat, p_value = mannwhitneyu(group_wS1, group_fpS1, alternative='two-sided')
    stats.append({'day': label, 'statistic': stat, 'p_value': p_value})
    # Add significance stars to the plot
    ax = plt.gca()
    xpos = day_labels.index(label)
    ypos = 95
    if p_value < 0.001:
        plt.text(xpos, ypos, '***', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.01:
        plt.text(xpos, ypos, '**', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.05:
        plt.text(xpos, ypos, '*', ha='center', va='bottom', color='black', fontsize=14)
    # Add p-value text below stars
    plt.text(xpos, 90, f'p={p_value:.3g}', ha='center', va='bottom', color='black', fontsize=10)

# Save the results to CSV files
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, 'opto_learning_day0_day1.svg'), format='svg', dpi=300)
day_data.to_csv(os.path.join(output_dir, 'opto_learning_day0_day1_data.csv'), index=False)
pd.DataFrame(stats).to_csv(os.path.join(output_dir, 'opto_learning_day0_day1_stats.csv'), index=False)


# ##############################################
# Fit learning curves and define learning trial.
# ##############################################§

# Load behavior results.
# ----------------------

db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

experimenters = ['AR', 'GF', 'MI']
mice_imaging = io.select_mice_from_db(db_path, nwb_dir,
                                    experimenters = experimenters,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes'
                                    )
# Load the table from the CSV file.
# table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

# Fit learning curves and define learning trial.
table = compute_learning_curves(table)
table = compute_learning_trial(table, n_consecutive_trials=10)

# Save updated table.
save_path = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table.to_csv(save_path, index=False)

# table = pd.read_csv(save_path)


# d = table.loc[(table.day==0) & (table.reward_group=='R+') & (table.whisker_stim==1) & (table.mouse_id=='GF305')]

# dd = fit_learning_curve(d.outcome_w.values, alpha=1, beta=1)
# plt.plot(d.hr_w.reset_index(drop=True))
# plt.plot(dd[1])


# # Convert learning curve columns to float type to avoid dtype issues
# learning_curve_cols = [
#     'learning_curve_w', 'learning_curve_w_ci_low', 'learning_curve_w_ci_high',
#     'learning_curve_a', 'learning_curve_a_ci_low', 'learning_curve_a_ci_high',
#     'learning_curve_ns', 'learning_curve_ns_ci_low', 'learning_curve_ns_ci_high'
# ]
# for col in learning_curve_cols:
#     if col in table.columns:
#         table[col] = pd.to_numeric(table[col], errors='coerce')

# Day 0 learning curves pdf
pdf_path = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/behavior/learning_curves_day0.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)
session_list = table.session_id.unique()
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

def plot_learning_curves_pdf(table, session_list, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for session_id in session_list:
            data = table.loc[table.session_id == session_id]
            if data.day.iloc[0] != 0:
                continue
            reward_group = data.reward_group.values[0]
            color = reward_palette[0] if reward_group == 'R-' else reward_palette[1]

            d = data.loc[data.whisker_stim == 1].reset_index(drop=True)


            learning_curve_w = d.learning_curve_w.values.astype(float)
            learning_ci_low = d.learning_curve_w_ci_low.values.astype(float)
            learning_ci_high = d.learning_curve_w_ci_high.values.astype(float)
            learning_chance = d.learning_curve_chance.astype(float)

            plt.plot(d.trial_w, learning_curve_w, label='Whisker (learning curve)', color=color)
            plt.fill_between(d.trial_w, learning_ci_low, learning_ci_high, color=color, alpha=0.2)
            sns.lineplot(data=d, x='trial_w', y='hr_w', color=color, legend=False, linestyle='--')
            plt.plot(d.trial_w, learning_chance, label='No_stim', color=stim_palette[2])

            learning_trial = data.learning_trial.values[0]
            if not pd.isna(learning_trial):
                plt.axvline(x=learning_trial, color='black', linestyle='--', label='Learning trial')
            plt.title(f'Session {session_id} - {reward_group}')
            plt.ylim([0, 1])
            plt.xlabel('Whisker trial')
            plt.ylabel('Lick probability')
            plt.legend()
            sns.despine()

            pdf.savefig()
            plt.close()

plot_learning_curves_pdf(table, session_list, pdf_path)


# Distribution of learning trials across mice (day 0).
# ------------------------------------------------------

output_dir = io.adjust_path_to_host(
    r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/behavior'
)

# One row per session, day 0 only, take first row (learning_trial is session-level).
day0 = (
    table[table.day == 0]
    .groupby('session_id', as_index=False)
    .first()[['session_id', 'mouse_id', 'reward_group', 'learning_trial']]
)

fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
for ax, rg, color in zip(axes, ['R+', 'R-'], [reward_palette[1], reward_palette[0]]):
    data_rg = day0[day0.reward_group == rg]['learning_trial'].dropna()
    bin_edges = np.arange(0, data_rg.max() + 5, 5) if len(data_rg) > 0 else 15
    ax.hist(data_rg, bins=bin_edges, color=color, edgecolor='white')
    n_learned = len(data_rg)
    n_total = (day0.reward_group == rg).sum()
    ax.set_title(f'{rg}  (n learned = {n_learned} / {n_total})')
    ax.set_xlabel('Learning trial (whisker)')
    ax.set_ylabel('Number of mice')
    ax.set_ylim([0, 6])
    if rg == 'R+':
        ax.set_xlim([0, 70])
    else:
        ax.set_xlim([0, 120])
    sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(
    os.path.join(output_dir, 'behavior', 'learning_trial_distribution_day0.svg'),
    format='svg', dpi=300
)

# Example learning curves for two mice (day 0).
# -----------------------------------------------

example_mice = {'R+': 'GF305', 'R-': 'AR180'}
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

for ax, (rg, mouse_id) in zip(axes, example_mice.items()):
    color = reward_palette[1] if rg == 'R+' else reward_palette[0]

    session_row = table[(table.mouse_id == mouse_id) & (table.day == 0)]

    d = session_row[session_row.whisker_stim == 1].reset_index(drop=True)
    d = d.loc[d.trial_w < 100]  # Limit to first 100 whisker trials for better visualization

    learning_curve_w = d.learning_curve_w.values.astype(float)
    learning_ci_low  = d.learning_curve_w_ci_low.values.astype(float)
    learning_ci_high = d.learning_curve_w_ci_high.values.astype(float)
    learning_chance  = d.learning_curve_chance.values.astype(float)

    ax.plot(d.trial_w, learning_curve_w, color=color, linewidth=2,
            label='Fitted curve')
    ax.fill_between(d.trial_w, learning_ci_low, learning_ci_high,
                    color=color, alpha=0.2, label='80% CI')
    ax.plot(d.trial_w, learning_chance, color=stim_palette[2], linewidth=1.5,
            label='False alarm rate')

    learning_trial = session_row.learning_trial.values[0]
    if not pd.isna(learning_trial):
        ax.axvline(x=learning_trial, color='black', linestyle='--', linewidth=1,
                   label=f'Learning trial ({int(learning_trial)})')

    ax.set_ylim([0, 1])
    ax.set_xlabel('Whisker trial')
    ax.set_ylabel('Lick probability')
    ax.set_title(f'{mouse_id} – {rg}')
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)

fig.tight_layout()
fig.savefig(
    os.path.join(output_dir, 'example_learning_curves_day0.svg'),
    format='svg', dpi=300
)



# # Debug: day 0 performance for mice without a detected learning trial.
# # ---------------------------------------------------------------------

# no_learning = day0[day0.learning_trial.isna()]

# for rg in ['R+', 'R-']:
#     mice_no_learning = no_learning[no_learning.reward_group == rg].mouse_id.values
#     if len(mice_no_learning) == 0:
#         continue

#     color = reward_palette[1] if rg == 'R+' else reward_palette[0]
#     ncols = len(mice_no_learning)
#     fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.5), sharey=True)
#     if ncols == 1:
#         axes = [axes]

#     for ax, mouse_id in zip(axes, mice_no_learning):
#         session_row = table[(table.mouse_id == mouse_id) & (table.day == 0)]
#         d = session_row[session_row.whisker_stim == 1].reset_index(drop=True)

#         ax.plot(d.trial_w, d.learning_curve_w.values.astype(float),
#                 color=color, linewidth=2, label='Fitted curve')
#         ax.fill_between(d.trial_w,
#                         d.learning_curve_w_ci_low.values.astype(float),
#                         d.learning_curve_w_ci_high.values.astype(float),
#                         color=color, alpha=0.2)
#         ax.plot(d.trial_w, d.learning_curve_chance.values.astype(float),
#                 color=stim_palette[2], linewidth=1.5, label='Chance (no-stim)')
#         ax.set_ylim([0, 1])
#         ax.set_xlabel('Whisker trial')
#         ax.set_ylabel('Lick probability')
#         ax.set_title(f'{mouse_id} – {rg} – no learning trial')
#         ax.legend(frameon=False, fontsize=7)
#         sns.despine(ax=ax)

#     fig.tight_layout()



# ── Mice used in the study ────────────────────────────────────────────────────

mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols=['exclude', 'two_p_exclude'],
                                    optogenetic=['no', np.nan],
                                    pharmacology=['no', np.nan],
                                    two_p_imaging='yes',
                                    )
mice_imaging = sorted(mice_imaging)
print(f"Two-photon imaging mice (n={len(mice_imaging)}):")
print(mice_imaging)

mice_muscimol = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols=['exclude'],
                                    optogenetic=['no', np.nan],
                                    pharmacology=['yes'],
                                    pharma_inactivation_type=['learning'],
                                        pharma_day = ["pre_-2", "pre_-1",
            "muscimol_1", "muscimol_2", "muscimol_3",
            "recovery_1", "recovery_2", "recovery_3"],
                                    )
mice_muscimol = sorted(mice_muscimol)
print(f"\nMuscimol inactivation mice (n={len(mice_muscimol)}):")
print(mice_muscimol)

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols=['exclude', 'opto_exclude'],
                                    optogenetic=['yes'],
                                    pharmacology=['no', np.nan],
                                    opto_inactivation_type=['learning'],
                                    )
mice_opto = sorted(mice_opto)
print(f"\nOpto inactivation mice (n={len(mice_opto)}):")
print(mice_opto)

mice_particle_test = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols=['exclude', 'two_p_exclude'],
                                    day=['whisker_on_1', 'whisker_off', 'whisker_on_2'],
                                    optogenetic=['no', np.nan],
                                    pharmacology=['no', np.nan],
                                    two_p_imaging='yes',
                                    )
mice_particle_test = sorted(mice_particle_test)
print(f"\nParticle test mice (n={len(mice_particle_test)}):")
print(mice_particle_test)