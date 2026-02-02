import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from scipy.stats import bootstrap
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *



# # #############################################################################
# # Comparing xarray datasets with previous tensors.
# # #############################################################################

# io.processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
# mouse_id = 'AR180'
# session_id = 'AR180_20241217_160355'

# arr, mdata = utils_imaging.load_session_2p_imaging(mouse_id,
#                                                     session_id,
#                                                     io.processed_dir
#                                                     )
# # arr = utils_imaging.substract_baseline(arr, 3, ())
# arr = utils_imaging.extract_trials(arr, mdata, 'UM', n_trials=None)
# arr.shape

# # Load the xarray dataset.
# file_name = 'tensor_xarray_mapping_data.nc'
# xarray = utils_imaging.load_mouse_xarray(mouse_id, io.processed_dir, file_name)

# d = xarray.sel(trial=xarray['day'] == 2)


# #############################################################################
# 1. PSTH's.
# #############################################################################

# Parameters.
# -----------

sampling_rate = 30
win_sec = (-0.5, 1.5)  
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']


_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
# mice = [m for m in mice if m not in ['AR163']]
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()
# print(mice_count)
# print(mice_count.groupby('reward_group').count().reset_index())


# Load the data.
# --------------

psth = []

for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)
    
    # Keep days of interest.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))
    # Average trials per days.
    xarr = xarr.groupby('day').mean(dim='trial')
    
    xarr.name = 'psth'
    xarr = xarr.to_dataframe().reset_index()
    xarr['mouse_id'] = mouse_id
    xarr['reward_group'] = reward_group
    psth.append(xarr)
psth = pd.concat(psth)


# Grand average psth's for all cells and projection neurons.
# ##########################################################

# GF305 has baseline artefact on day -1 at auditory trials.
# data = data.loc[~data.mouse_id.isin(['GF305'])]

# mice_AR = [m for m in mice if m.startswith('AR')]
# mice_GF = [m for m in mice if m.startswith('GF') or m.startswith('MI')]
# data = data.loc[data.mouse_id.isin(mice_AR)]
# len(mice_GF)

variance = 'mice'  # 'mice' or 'cells'

# Separate PSTH data for all cells (ignoring cell_type) and for projection types (wS2, wM1)
if variance == "mice":
    min_cells = 3
    psth_filtered = utils_imaging.filter_data_by_cell_count(psth, min_cells)
    # All cells: ignore cell_type
    data_allcells = psth_filtered.groupby(['mouse_id', 'day', 'reward_group', 'time'])['psth'].mean().reset_index()
    # Projection types: only wS2 and wM1
    data_ctype = psth_filtered[psth_filtered['cell_type'].isin(['wS2', 'wM1'])]
    data_ctype = data_ctype.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type'])['psth'].mean().reset_index()
else:
    # All cells: ignore cell_type
    data_allcells = psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'roi'])['psth'].mean().reset_index()
    data_allcells = data_allcells.groupby(['mouse_id', 'day', 'reward_group', 'time'])['psth'].mean().reset_index()
    # Projection types: only wS2 and wM1
    data_ctype = psth[psth['cell_type'].isin(['wS2', 'wM1'])]
    data_ctype = data_ctype.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].mean().reset_index()
    data_ctype = data_ctype.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type'])['psth'].mean().reset_index()

# Convert to percent dF/F0
data_allcells['psth'] = data_allcells['psth'] * 100
data_ctype['psth'] = data_ctype['psth'] * 100

# Plot for all cells (ignoring cell_type)
fig, axes = plt.subplots(1, len(days), figsize=(18, 5), sharey=True)
for j, day in enumerate(days):
    d = data_allcells.loc[data_allcells['day'] == day]
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                 hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[j], legend=False)
    axes[j].axvline(0, color='#FF9600', linestyle='-')
    axes[j].set_title(f'Day {day} - All Cells')
    axes[j].set_ylabel('DF/F0 (%)')
plt.ylim(-1, 12)
plt.tight_layout()
sns.despine()

# Save figure for all cells
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_across_days_all_cells_{variance}_long.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)

# Plot for each cell type (wS2, wM1)
fig, axes = plt.subplots(2, len(days), figsize=(18, 10), sharey=True)
for i, cell_type in enumerate(['wS2', 'wM1']):
    for j, day in enumerate(days):
        d = data_ctype[(data_ctype['cell_type'] == cell_type) & (data_ctype['day'] == day)]
        sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                     hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[i, j], legend=False)
        axes[i, j].axvline(0, color='#FF9600', linestyle='-')
        axes[i, j].set_title(f'{cell_type} - Day {day}')
        axes[i, j].set_ylabel('DF/F0 (%)')
plt.ylim(-1, 16)
plt.tight_layout()
sns.despine()

# Save figure for projection types
svg_file = f'psth_across_days_projection_types_{variance}_long.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Individual mice PSTH's.
# -----------------------

output_dir = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
pdf_file = f'psth_individual_mice_baseline.pdf'

with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
    for mouse_id in mice:
        print(f"\rProcessing {mouse_id} ({mice.index(mouse_id) + 1}/{len(mice)})", end="")
        # Plot for all cells (ignoring cell_type)
        data_all = data_allcells.loc[(data_allcells['day'].isin([-2, -1, 0, 1, 2])) & (data_allcells['mouse_id'] == mouse_id)]
        # Plot for projection types (wS2, wM1)
        data_proj = data_ctype.loc[(data_ctype['day'].isin([-2, -1, 0, 1, 2])) & (data_ctype['mouse_id'] == mouse_id)]
        fig, axes = plt.subplots(3, len(days), figsize=(15, 10), sharey=True)
        # Plot for all cells
        for j, day in enumerate(days):
            d = data_all.loc[data_all['day'] == day]
            sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                        hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0, j], legend=False)
            axes[0, j].axvline(0, color='#FF9600', linestyle='-')
            axes[0, j].set_title('All Cells')
            axes[0, j].set_ylabel('DF/F0 (%)')
            axes[0, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))

        # Plot for each cell type
        for i, cell_type in enumerate(['wS2', 'wM1']):
            for j, day in enumerate(days):
                d = data_proj[(data_proj['cell_type'] == cell_type) & (data_proj['day'] == day)]
                sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                            hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[i + 1, j], legend=False)
                axes[i + 1, j].axvline(0, color='#FF9600', linestyle='-')
                axes[i + 1, j].set_title(f'{cell_type} - Day {day}')
                axes[i + 1, j].set_ylabel('DF/F0 (%)')
                axes[i + 1, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))
        plt.tight_layout()
        sns.despine()
        plt.suptitle(mouse_id)
        pdf.savefig(dpi=300)
        plt.close()



# ####################################################################
# Quantifying response before and after learning inside reward groups.
# ####################################################################

sampling_rate = 30
win_sec_amp = (0, 0.300)  
win_sec_psth = (-0.5, 1.5)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])

# Load the data.
# --------------

avg_resp = []
psth = []

for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)
    
    # Average response data.
    # Keep days of interest.
    avg = xarr.sel(trial=xarr['day'].isin(days))
    # Average of time points.
    avg = avg.sel(time=slice(win_sec_amp[0], win_sec_amp[1])).mean(dim='time')
    # # Average trials per days.
    # avg = avg.groupby('day').mean(dim='trial')
    # Convert to dataframe.
    avg.name = 'average_response'
    avg = avg.to_dataframe().reset_index()
    avg['mouse_id'] = mouse_id
    avg['reward_group'] = reward_group
    avg_resp.append(avg)
    
    # PSTH data.
    # Keep days of interest.
    p = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    p = p.sel(time=slice(win_sec_psth[0], win_sec_psth[1]))
    # Average trials per days.
    p = p.groupby('day').mean(dim='trial')
    
    p.name = 'psth'
    p = p.to_dataframe().reset_index()
    p['mouse_id'] = mouse_id
    p['reward_group'] = reward_group
    psth.append(p)
avg_resp = pd.concat(avg_resp)
psth = pd.concat(psth)
# Convert to percent dF/F0.
avg_resp['average_response'] = avg_resp['average_response'] * 100
psth['psth'] = psth['psth'] * 100

# Add a new column 'learning_period' to group days into 'pre' and 'post'
avg_resp['learning_period'] = avg_resp['day'].apply(lambda x: 'pre' if x in [-2,-1] else 'post')
psth['learning_period'] = psth['day'].apply(lambda x: 'pre' if x in [-2,-1] else 'post')


# Pre and post learning responses.
# --------------------------------

variance = 'mice'  # 'mice' or 'cells'
days_selected = [-2,-1, 1,2]
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)

# Select days of interest.
data_plot_avg = avg_resp[avg_resp['day'].isin(days_selected)]
data_plot_psth = psth[psth['day'].isin(days_selected)]

if variance == "mice":
    # Just filter by cell count for the projection types.
    min_cells = 3
    data_plot_avg = utils_imaging.filter_data_by_cell_count(data_plot_avg, min_cells)
    data_plot_psth = utils_imaging.filter_data_by_cell_count(data_plot_psth, min_cells)
    # Average for all cells and projection types independently.
    data_plot_avg_all = data_plot_avg.groupby(['mouse_id', 'learning_period', 'reward_group'])['average_response'].agg('mean').reset_index()
    data_plot_avg_proj = data_plot_avg.groupby(['mouse_id', 'learning_period', 'reward_group', 'cell_type'])['average_response'].agg('mean').reset_index()
    data_plot_psth_all = data_plot_psth.groupby(['mouse_id', 'learning_period', 'reward_group', 'time'])['psth'].agg('mean').reset_index()
    data_plot_psth_proj = data_plot_psth.groupby(['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type'])['psth'].agg('mean').reset_index()
else:
    data_plot_avg_all = data_plot_avg.groupby(['mouse_id', 'learning_period', 'reward_group', 'cell_type', 'roi'])['average_response'].agg('mean').reset_index()
    data_plot_psth_all = data_plot_psth.groupby(['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
    data_plot_avg_proj = data_plot_avg_all
    data_plot_psth_proj = data_plot_psth_all
 
# Create the figure with four subplots
fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=False, sharey=True)

# Top-left: PSTH for rewarded mice
rewarded_data = data_plot_psth_all[(data_plot_psth_all['reward_group'] == 'R+')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 0], legend=False )
axes[0, 0].set_title('PSTH (Rewarded Mice)')
axes[0, 0].set_ylabel('DF/F0 (%)')
axes[0, 0].axvline(0, color='orange', linestyle='-')

# Bottom-left: PSTH for non-rewarded mice
nonrewarded_data = data_plot_psth_all[(data_plot_psth_all['reward_group'] == 'R-')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 0], legend=False )
axes[1, 0].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 0].set_ylabel('DF/F0 (%)')
axes[1, 0].axvline(0, color='orange', linestyle='-')

# Top-right: Response amplitude for rewarded mice
rewarded_avg = data_plot_avg_all[data_plot_avg_all['reward_group'] == 'R+']
sns.barplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#1b9e77', ax=axes[0, 1])
sns.swarmplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='black', alpha=0.5, size=4, ax=axes[0, 1])
axes[0, 1].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 1].set_ylabel('Average Response (dF/F0)')

# Bottom-right: Response amplitude for non-rewarded mice
nonrewarded_avg = data_plot_avg_all[data_plot_avg_all['reward_group'] == 'R-']
sns.barplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#c959affe', ax=axes[1, 1])
sns.swarmplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='black', alpha=0.5, size=4, ax=axes[1, 1])
axes[1, 1].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 1].set_ylabel('Average Response (dF/F0)')

sns.despine()

# Save figure.
svg_file = f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_allcells.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)



# A4 width is 8.27 inches, so 2/3 of A4 is about 5.5 inches per row.
# For 4 columns, set width to 8.27 * (2/3) ≈ 5.5, height to keep aspect ratio.
fig, axes = plt.subplots(2, 4, figsize=(8.27 * 2 / 3, 5* 2 / 3), sharex=False, sharey=True)

# S2 PSTH
rewarded_data = data_plot_psth_proj[(data_plot_psth_proj['reward_group'] == 'R+') & (data_plot_psth_proj['cell_type'] == 'wS2')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 0], legend=False)
axes[0, 0].set_title('PSTH (Rewarded Mice)')
axes[0, 0].set_ylabel('DF/F0 (%)')
axes[0, 0].axvline(0, color='orange', linestyle='-')

nonrewarded_data = data_plot_psth_proj[(data_plot_psth_proj['reward_group'] == 'R-') & (data_plot_psth_proj['cell_type'] == 'wS2')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 0], legend=False)
axes[1, 0].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 0].set_ylabel('DF/F0 (%)')
axes[1, 0].axvline(0, color='orange', linestyle='-')

# S2 Response amplitude
rewarded_avg = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == 'R+') & (data_plot_avg_proj['cell_type'] == 'wS2')]
sns.barplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#1b9e77', ax=axes[0, 1])
sns.swarmplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='black', alpha=0.5, size=4, ax=axes[0, 1])
axes[0, 1].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 1].set_ylabel('Average Response (dF/F0)')
axes[0, 1].set_ylim(-2, 15)

nonrewarded_avg = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == 'R-') & (data_plot_avg_proj['cell_type'] == 'wS2')]
sns.barplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#c959affe', ax=axes[1, 1])
sns.swarmplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='black', alpha=0.5, size=4, ax=axes[1, 1])
axes[1, 1].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 1].set_ylabel('Average Response (dF/F0)')
axes[1, 1].set_ylim(-2, 15)

# M1 PSTH
rewarded_data = data_plot_psth_proj[(data_plot_psth_proj['reward_group'] == 'R+') & (data_plot_psth_proj['cell_type'] == 'wM1')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 2], legend=False)
axes[0, 2].set_title('PSTH (Rewarded Mice)')
axes[0, 2].set_ylabel('DF/F0 (%)')
axes[0, 2].axvline(0, color='orange', linestyle='-')

nonrewarded_data = data_plot_psth_proj[(data_plot_psth_proj['reward_group'] == 'R-') & (data_plot_psth_proj['cell_type'] == 'wM1')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
    palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 2], legend=False)
axes[1, 2].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 2].set_ylabel('DF/F0 (%)')
axes[1, 2].axvline(0, color='orange', linestyle='-')

# M1 Response amplitude
rewarded_avg = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == 'R+') & (data_plot_avg_proj['cell_type'] == 'wM1')]
sns.barplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#1b9e77', ax=axes[0, 3])
sns.swarmplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='black', alpha=0.5, size=4, ax=axes[0, 3])
axes[0, 3].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 3].set_ylabel('Average Response (dF/F0)')
axes[0, 3].set_ylim(-2, 15)

nonrewarded_avg = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == 'R-') & (data_plot_avg_proj['cell_type'] == 'wM1')]
sns.barplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#c959affe', ax=axes[1, 3])
sns.swarmplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='black', alpha=0.5, size=4, ax=axes[1, 3])
axes[1, 3].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 3].set_ylabel('Average Response (dF/F0)')
axes[1, 3].set_ylim(-2, 15)

sns.despine()

# Save figure.
svg_file = f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_proj.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Perform stats on response amplitude
results = []
for reward_group in ['R+', 'R-']:
    for cell_type in ['all', 'wS2', 'wM1']:
        if cell_type == 'all':
            data_stats = data_plot_avg_all[data_plot_avg_all['reward_group'] == reward_group]
        else:
            data_stats = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == reward_group) & (data_plot_avg_proj['cell_type'] == cell_type)]
        pre = data_stats[data_stats['learning_period'] == 'pre']['average_response']
        post = data_stats[data_stats['learning_period'] == 'post']['average_response']
        stat, p_value = wilcoxon(pre, post)
        results.append({
            'reward_group': reward_group,
            'cell_type': cell_type,
            'stat': stat,
            'p_value': p_value
        })
stats_df = pd.DataFrame(results)
print(stats_df)

# Save the figure and stats

stats_df.to_csv(os.path.join(output_dir, f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_stats.csv'), index=False)
data_plot_avg_all.to_csv(os.path.join(output_dir, f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_data_allcells.csv'), index=False)
data_plot_avg_proj.to_csv(os.path.join(output_dir, f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_data_proj.csv'), index=False)



# #############################################################################
# Population vectors and correlation matrices for individual mice.
# #############################################################################

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 180 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)

output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/mapping'
output_dir = io.adjust_path_to_host(output_dir)
pdf_file = 'pop_vectors_and_corrmat_individual_mice_300ms.pdf'

with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
    for mouse in mice:
        print(mouse)
        processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse)

        file_name = 'tensor_xarray_mapping_data.nc'
        xarr_mapping = utils_imaging.load_mouse_xarray(mouse, processed_dir, file_name, substracted=True)
        xarr_mapping = xarr_mapping.sel(trial=xarr_mapping['day'].isin(days))
        # Select the last n_map_trials mapping trials for each day
        xarr_mapping = xarr_mapping.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
        xarr_mapping = xarr_mapping.sel(time=slice(win[0], win[1])).mean(dim='time')

        # Population vector plot
        vectors_mapping = xarr_mapping.values
        whisker_trial_counts = [vectors_mapping.shape[1] // len(days)] * len(days)
        edges = np.cumsum(whisker_trial_counts)
        
        # Exclude diagonal for vmax calculation
        arr = vectors_mapping
        arr_nodiag = arr.copy()
        min_dim = min(arr_nodiag.shape[0], arr_nodiag.shape[1])
        for i in range(min_dim):
            arr_nodiag[i, i] = np.nan
        vmax = np.nanpercentile(arr_nodiag, 98)
        vmin = 0
        plt.figure(figsize=(10, 6))
        im = plt.imshow(vectors_mapping, cmap='viridis', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im)
        cbar.set_label('Activity')
        for i in edges[:-1] - 0.5:
            plt.axvline(x=i, color='white', linestyle='-', lw=0.5)
        plt.xticks(edges - 0.5, edges)
        plt.xlabel('Trials')
        plt.ylabel('Cells')
        plt.title(f'{mouse} - {reward_group} - Population Vectors')
        plt.tight_layout()
        pdf.savefig(dpi=300)
        plt.close()
        # Correlation matrix plot
        # Compute Spearman correlation matrix across trials
        corrmat, _ = spearmanr(vectors_mapping.T, axis=1)
        plt.figure(figsize=(8, 7))
        # Use same vmin and vmax as population vector plot
        im = sns.heatmap(corrmat, cmap='viridis', vmin=vmin, vmax=vmax, cbar_kws={'label': 'Spearman Correlation'})
        # Draw lines to separate days
        for edge in edges[:-1]:
            plt.axvline(x=edge - 0.5, color='white', linestyle='-', linewidth=.5)
            plt.axhline(y=edge - 0.5, color='white', linestyle='-', linewidth=.5)
        plt.title(f'{mouse} - {reward_group} - Spearman Correlation Matrix')
        plt.xlabel('Trial')
        plt.ylabel('Trial')
        plt.tight_layout()
        pdf.savefig(dpi=300)
        plt.close()
        print(f"Saved population vector and Spearman correlation matrix for mouse {mouse} to PDF.")



# # Illustrate pop vectors of AR127.
# # --------------------------------

# # Vectors during learning.
# file_name = 'tensor_xarray_mapping_data.nc'
# folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
# mouse = 'AR180'

# xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)

# # Select days.
# xarray = xarray.sel(trial=xarray['day'].isin([0]))

# xarray = xarray.sel(time=slice(0, 0.300)).mean(dim='time')


# # Plot
# vectors_rew = xarray.values
# vmax = np.percentile(vectors_rew, 98)
# vmin = np.percentile(vectors_rew, 3)
# edges = np.cumsum([50 for _ in range(5)])
# f = plt.figure(figsize=(10, 6))
# im = plt.imshow(vectors_rew, cmap='viridis', vmin=vmin, vmax=vmax)

# # Add colorbar
# cbar = plt.colorbar(im)
# cbar.set_label('Activity')

# for i in edges[:-1] - 0.5:
#     plt.axvline(x=i, color='white', linestyle='-', lw=0.5)
# plt.xticks(edges - 0.5, edges)


# file_name = 'tensor_xarray_learning_data.nc'
# folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
# mouse = 'AR180'
# xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
# xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
# # Select days
# xarray = xarray.sel(trial=xarray['day'].isin([0]))
# xarray = xarray.sel(time=slice(0, 0.300)).mean(dim='time')
# xarray = xarray.sel(trial=xarray['trial_type'] == 'whisker_trial')

# # Plot
# vectors_rew = xarray.values
# # vmax = np.percentile(vectors_rew, 98)
# # vmin = np.percentile(vectors_rew, 2)

# # edges = np.cumsum([50 for _ in range(5)])

# f = plt.figure(figsize=(2, 6))
# im = plt.imshow(vectors_rew, cmap='viridis', vmin=vmin, vmax=vmax)
# # # Add colorbar
# # cbar = plt.colorbar(im)
# # cbar.set_label('Activity')

# # for i in edges[:-1] - 0.5:
# #     plt.axvline(x=i, color='white', linestyle='-', lw=0.5)
# # plt.xticks(edges - 0.5, edges)

# # Save the figure
# output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/examples'
# output_dir = io.adjust_path_to_host(output_dir)
# svg_file = 'AR180_pop_vectors_learning_day0.svg'
# plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)



# # Example traces 

# dff = "/mnt/lsens-analysis/Anthony_Renard/data/AR180/AR180_20241215_190049/suite2p/plane0/dff.npy"
# dff = np.load(dff)

# dff.shape

# sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=2)

# cells = [2, 4, 5, 8, 10, 11, 18, 21, 22, 25, 27, 28, 35, 36, 49, 54, 55, 67, 69, 77]
# counter = 0
# plt.figure(figsize=(16, 12))
# for icell in cells:
#     if icell == 31:
#         continue
#     plt.plot(dff[icell, 1000:15000] + counter * 3)
#     counter += 1

# # Save plot.
# output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/illustrations'
# file_name = 'AR180_example_traces.svg'
# plt.savefig(os.path.join(output_dir, file_name), format='svg', dpi=300)





# ###################################################
# Projection of whisker trials on learning dimension.
# ###################################################

# Compute a learning dimension as the difference vector between pre and post training vectors.
# Project whisker trials during learning on the learning dimension. 
sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = True
zscore = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data for mapping trials and whisker trials
results = []

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] >= 0.975)]

for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Select cells.
    if select_responsive_cells or select_lmi:
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))
    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue
    
    # Select pre, post mapping trials and day 0 whisker trials.
    pre = xarray.sel(trial=xarray['day'].isin([-2, -1,]))
    pre = pre.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    post = xarray.sel(trial=xarray['day'].isin([1, 2]))
    post = post.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))

    # Select whisker trials for Day 0
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    xarray = xarray.sel(time=slice(win[0], 0.180)).mean(dim='time')
    
    if select_responsive_cells or select_lmi:
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))

    day0 = xarray.sel(trial=(xarray['day'] == 0) & (xarray['trial_type'] == 'whisker_trial'))
    day0 = day0[:, :100]  # Select all trials

    learning_dim = post.mean(dim='trial') - pre.mean(dim='trial')
    # Project mapping trials onto the learning dimension (scalar projection)
    pre_mapping_proj = np.dot(pre.values.T, learning_dim.values) / np.linalg.norm(learning_dim.values)
    post_mapping_proj = np.dot(post.values.T, learning_dim.values) / np.linalg.norm(learning_dim.values)
    # Project whisker trials onto the learning dimension (scalar projection)
    day0_mapping_proj = np.dot(day0.values.T, learning_dim.values) / np.linalg.norm(learning_dim.values)

    # Compute cosine similarity between each trial and the learning dimension
    def cosine_sim(trials, ref):
        # trials: shape (n_cells, n_trials)
        # ref: shape (n_cells,)
        norm_trials = np.linalg.norm(trials, axis=0)
        norm_ref = np.linalg.norm(ref)
        # Avoid division by zero
        norm_trials[norm_trials == 0] = 1e-10
        if norm_ref == 0:
            norm_ref = 1e-10
        return np.dot(ref, trials) / (norm_ref * norm_trials)

    pre_mapping_cosine = cosine_sim(pre.values, learning_dim.values)
    post_mapping_cosine = cosine_sim(post.values, learning_dim.values)
    day0_mapping_cosine = cosine_sim(day0.values, learning_dim.values)
    
    pre_index = np.arange(pre_mapping_proj.shape[0])
    post_index = np.arange(post_mapping_proj.shape[0])
    day0_index = np.arange(day0_mapping_proj.shape[0])
    
    # Store the results in a common dataframe
    for proj, sim, period, index in zip(
        [pre_mapping_proj, post_mapping_proj, day0_mapping_proj],
        [pre_mapping_cosine, post_mapping_cosine, day0_mapping_cosine],
        ['pre', 'post', 'day0'],
        [pre_index, post_index, day0_index]
    ):
        results.append(pd.DataFrame({
            'mouse_id': mouse,
            'reward_group': rew_gp,
            'period': period,
            'projection': proj,
            'cosine_similarity': sim,
            'trial_index': index,
        }))
        
# Combine all results into a single dataframe
results_df = pd.concat(results, ignore_index=True)

# Add a column for block index (group trials into blocks of 10)
results_df['block_index'] = results_df['trial_index'] // 10

# Compute the mean projection for each block
block_results = results_df.groupby(['mouse_id', 'reward_group', 'period', 'block_index'])['cosine_similarity'].mean().reset_index()

# Plot projections across mice, averaged in blocks of 10
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=True)

# Pre mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'pre'], x='block_index', y='cosine_similarity', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0])
axes[0].set_title('Pre Mapping Trials')
axes[0].set_xlabel('Block Index (10 Trials per Block)')
# axes[0].set_ylabel('Projection on Learning Dimension')
axes[0].set_ylabel('Cosine similarity with Learning Dimension')

# Day 0 mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'day0'], x='block_index', y='cosine_similarity', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[1])
axes[1].set_title('Day 0 Mapping Trials')
axes[1].set_xlabel('Block Index (10 Trials per Block)')

# Post mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'post'], x='block_index', y='cosine_similarity', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[2])
axes[2].set_title('Post Mapping Trials')
axes[2].set_xlabel('Block Index (10 Trials per Block)')

sns.despine()

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/learning_dim'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'learning_dim_projection_lmi.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save results.
results_df.to_csv(os.path.join(output_dir, 'learning_dim_projection_lmi.csv'), index=False)




# ###################################################
# Correlation of whisker trials with post-learning vector.
# ###################################################

# Compute a post-learning vector as the average response of post-training trials.
# Compute the correlation of whisker trials during learning with the post-learning vector.
sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = True
zscore = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data for mapping trials and whisker trials
results = []

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Select cells.
    if select_responsive_cells or select_lmi:
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))
    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue
    
    # Select pre and post mapping trials
    pre = xarray.sel(trial=xarray['day'].isin([-1]))
    pre = pre.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    post = xarray.sel(trial=xarray['day'].isin([2]))
    post = post.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))

    # Compute the post-learning vector
    post_learning_vector = post.mean(dim='trial')

    # Compute correlation of pre and post trials with the post-learning vector
    pre_corr = np.array([np.corrcoef(trial, post_learning_vector.values)[0, 1] for trial in pre.values.T])
    post_corr = np.array([np.corrcoef(trial, post_learning_vector.values)[0, 1] for trial in post.values.T])

    # Select whisker trials for Day 0
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    xarray = xarray.sel(time=slice(win[0], 0.180)).mean(dim='time')
    
    if select_responsive_cells or select_lmi:
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))

    # day0 = xarray.sel(trial=(xarray['day'] == 0) & (xarray['trial_type'] == 'whisker_trial'))
    # day0 = day0[:, :145]  # Select all trials
    if rew_gp == 'R-':
        day0 = xarray.sel(trial=(xarray['day'] == 0) & (xarray['trial_type'] == 'whisker_trial'))
        day0 = day0[:, :145]  # Select all trials
    elif rew_gp == 'R+':
        day0 = xarray.sel(trial=(xarray['day'] == 0) & (xarray['trial_type'] == 'whisker_trial') & (xarray['outcome_w'] == 1))
        day0 = day0[:, :145]

    # Compute correlation of whisker trials with the post-learning vector
    day0_corr = np.array([np.corrcoef(trial, post_learning_vector.values)[0, 1] for trial in day0.values.T])
    
    pre_index = np.arange(pre_corr.shape[0])
    post_index = np.arange(post_corr.shape[0])
    day0_index = np.arange(day0_corr.shape[0])
    
    # Store the results in a common dataframe
    for corr, period, index in zip(
        [pre_corr, post_corr, day0_corr],
        ['pre', 'post', 'day0'],
        [pre_index, post_index, day0_index]
    ):
        results.append(pd.DataFrame({
            'mouse_id': mouse,
            'reward_group': rew_gp,
            'period': period,
            'correlation': corr,
            'trial_index': index,
        }))
        
# Combine all results into a single dataframe
results_df = pd.concat(results, ignore_index=True)

# Add a column for block index (group trials into blocks of 10)
results_df['block_index'] = results_df['trial_index'] // 7

# Compute the mean correlation for each block
block_results = results_df.groupby(['mouse_id', 'reward_group', 'period', 'block_index'])['correlation'].mean().reset_index()

# Plot correlations across mice, averaged in blocks of 10
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=True)

# Pre mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'pre'], x='block_index', y='correlation', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0])
axes[0].set_title('Pre Mapping Trials')
axes[0].set_xlabel('Block Index (10 Trials per Block)')
axes[0].set_ylabel('Correlation with Post-Learning Vector')

# Day 0 whisker trials
sns.pointplot(data=block_results[block_results['period'] == 'day0'], x='block_index', y='correlation', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[1])
axes[1].set_title('Day 0 Whisker Trials')
axes[1].set_xlabel('Block Index (10 Trials per Block)')

# Post mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'post'], x='block_index', y='correlation', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[2])
axes[2].set_title('Post Mapping Trials')
axes[2].set_xlabel('Block Index (10 Trials per Block)')

sns.despine()

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/learning_dim'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'correlation_with_post_learning_vector_whhitR+.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save results.
results_df.to_csv(os.path.join(output_dir, 'correlation_with_post_learning_vector_whhitR+.csv'), index=False)
















# Learning-Mapping Representational Similarity
# ----------------------------------------------
# Instead of CCA (which fails with n_cells >> n_samples), we compute
# representational similarity using correlation of average patterns and
# trial-by-trial pattern correlations

from sklearn.cross_decomposition import CCA

def compute_representational_similarity(X, Y):
    """
    Compute representational similarity between two sets of trials.
    
    Instead of CCA (which overfits with n_cells >> n_samples), we use:
    1. Correlation of average population patterns
    2. Average pairwise correlation between trials
    
    Parameters:
    -----------
    X : array (n_cells, n_trials_X)
        First set of trials (e.g., learning trials)
    Y : array (n_cells, n_trials_Y)
        Second set of trials (e.g., mapping trials)
    
    Returns:
    --------
    avg_pattern_corr : float
        Correlation between average patterns
    avg_trial_corr : float
        Average correlation between individual trials
    """
    # Method 1: Correlation of average patterns
    X_mean = np.mean(X, axis=1)  # (n_cells,)
    Y_mean = np.mean(Y, axis=1)  # (n_cells,)
    avg_pattern_corr = np.corrcoef(X_mean, Y_mean)[0, 1]
    
    # Method 2: Average pairwise trial-by-trial correlation
    # Sample subset of trials to avoid too many comparisons
    n_X = min(X.shape[1], 20)  # Use max 20 trials from each
    n_Y = min(Y.shape[1], 20)
    
    X_sample = X[:, -n_X:]  # Take last n_X trials
    Y_sample = Y[:, -n_Y:]  # Take last n_Y trials
    
    trial_corrs = []
    for i in range(n_X):
        for j in range(n_Y):
            corr = np.corrcoef(X_sample[:, i], Y_sample[:, j])[0, 1]
            trial_corrs.append(corr)
    
    avg_trial_corr = np.mean(trial_corrs)
    
    return avg_pattern_corr, avg_trial_corr


# Compute representational similarity for each mouse, each day
print("\nComputing representational similarity between learning and mapping trials...")

similarity_results = []

for mouse_data in mouse_data_rew + mouse_data_nonrew:
    mouse_id = mouse_data['mouse_id']
    reward_group = mouse_data['reward_group']
    
    # For each learning day (0, +1, +2), compare learning vs mapping
    for day in [0, 1, 2]:
        learning_data = mouse_data[f'learning_day{day}']  # (n_cells, 80)
        mapping_data = mouse_data[f'map_day{day}']  # (n_cells, 40)
        
        # Compute representational similarity
        pattern_corr, trial_corr = compute_representational_similarity(learning_data, mapping_data)
        
        # Store results
        similarity_results.append({
            'mouse_id': mouse_id,
            'reward_group': reward_group,
            'day': day,
            'pattern_correlation': pattern_corr,
            'trial_correlation': trial_corr,
        })
        
        print(f"  {reward_group} {mouse_id} Day {day}: Pattern corr = {pattern_corr:.3f}, Trial corr = {trial_corr:.3f}")

similarity_df = pd.DataFrame(similarity_results)

# Print summary statistics
print("\n" + "="*60)
print("REPRESENTATIONAL SIMILARITY: LEARNING vs MAPPING")
print("="*60)
for group in ['R+', 'R-']:
    print(f"\n{group} Group:")
    for day in [0, 1, 2]:
        data = similarity_df[(similarity_df['reward_group'] == group) & (similarity_df['day'] == day)]
        print(f"  Day {day}: Pattern corr = {data['pattern_correlation'].mean():.3f} ± {data['pattern_correlation'].std():.3f}, " +
              f"Trial corr = {data['trial_correlation'].mean():.3f} ± {data['trial_correlation'].std():.3f}")

# Statistical comparison
print("\n" + "-"*60)
print("STATISTICAL COMPARISONS (Mann-Whitney U test)")
print("-"*60)



for metric in ['pattern_correlation', 'trial_correlation']:
    print(f"\n{metric.replace('_', ' ').title()}:")
    for day in [0, 1, 2]:
        r_plus = similarity_df[(similarity_df['reward_group'] == 'R+') & (similarity_df['day'] == day)][metric]
        r_minus = similarity_df[(similarity_df['reward_group'] == 'R-') & (similarity_df['day'] == day)][metric]
        stat, p_value = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
        print(f"  Day {day}: U={stat:.1f}, p={p_value:.4f} {sig}")


# Visualize Representational Similarity results
# ---------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Pattern Correlation (average population patterns)
sns.pointplot(data=similarity_df, x='day', y='pattern_correlation', hue='reward_group',
              hue_order=['R+', 'R-'], palette=reward_palette[::-1], 
              markers=['o', 's'], ax=axes[0], errorbar='ci')
sns.stripplot(data=similarity_df, x='day', y='pattern_correlation', hue='reward_group',
              hue_order=['R+', 'R-'], palette=reward_palette[::-1],
              dodge=True, alpha=0.3, ax=axes[0], legend=False, size=3)
axes[0].set_xlabel('Learning Day')
axes[0].set_ylabel('Pattern Correlation')
axes[0].set_title('Average Population Pattern Similarity')
axes[0].legend(title='Group', loc='best')
axes[0].set_xticklabels(['Day 0', 'Day +1', 'Day +2'])

# Add significance markers
for day in [0, 1, 2]:
    r_plus = similarity_df[(similarity_df['reward_group'] == 'R+') & (similarity_df['day'] == day)]['pattern_correlation']
    r_minus = similarity_df[(similarity_df['reward_group'] == 'R-') & (similarity_df['day'] == day)]['pattern_correlation']
    stat, p_value = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
    
    y_max = max(r_plus.max(), r_minus.max())
    y_pos = y_max * 1.05
    
    if p_value < 0.001:
        p_text = '***'
    elif p_value < 0.01:
        p_text = '**'
    elif p_value < 0.05:
        p_text = '*'
    else:
        p_text = 'n.s.'
    
    axes[0].text(day, y_pos, p_text, ha='center', va='bottom', fontsize=10)

# Panel 2: Trial Correlation (trial-by-trial similarity)
sns.pointplot(data=similarity_df, x='day', y='trial_correlation', hue='reward_group',
              hue_order=['R+', 'R-'], palette=reward_palette[::-1],
              markers=['o', 's'], ax=axes[1], errorbar='ci')
sns.stripplot(data=similarity_df, x='day', y='trial_correlation', hue='reward_group',
              hue_order=['R+', 'R-'], palette=reward_palette[::-1],
              dodge=True, alpha=0.3, ax=axes[1], legend=False, size=3)
axes[1].set_xlabel('Learning Day')
axes[1].set_ylabel('Trial Correlation')
axes[1].set_title('Average Trial-by-Trial Similarity')
axes[1].legend(title='Group', loc='best')
axes[1].set_xticklabels(['Day 0', 'Day +1', 'Day +2'])

# Add significance markers
for day in [0, 1, 2]:
    r_plus = similarity_df[(similarity_df['reward_group'] == 'R+') & (similarity_df['day'] == day)]['trial_correlation']
    r_minus = similarity_df[(similarity_df['reward_group'] == 'R-') & (similarity_df['day'] == day)]['trial_correlation']
    stat, p_value = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
    
    y_max = max(r_plus.max(), r_minus.max())
    y_pos = y_max * 1.05
    
    if p_value < 0.001:
        p_text = '***'
    elif p_value < 0.01:
        p_text = '**'
    elif p_value < 0.05:
        p_text = '*'
    else:
        p_text = 'n.s.'
    
    axes[1].text(day, y_pos, p_text, ha='center', va='bottom', fontsize=10)

sns.despine()
plt.tight_layout()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/learning_mapping'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'similarity_learning_mapping_alignment_{celltype_str}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, svg_file.replace('.svg', '.png')), format='png', dpi=300)

# Save data
similarity_df.to_csv(os.path.join(output_dir, f'similarity_learning_mapping_alignment_{celltype_str}.csv'), index=False)

print(f"\nSaved representational similarity results to: {os.path.join(output_dir, svg_file)}")










# ##################################################
# Decoding before and after learning.
# ##################################################


# Similar to the previous section, but compute a correlation matrix for
# each mouse and then average across mice.

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '+1', '+2']
days = [-2, -1,  1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = False
projection_type = None  # 'wS2', 'wM1' or None

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data.
vectors_rew = []
vectors_nonrew = []
mice_rew = []
mice_nonrew = []

# Load responsive cells.
# Responsiveness df.
# test_df = os.path.join(io.processed_dir, f'response_test_results_alldaystogether_win_180ms.csv')
# test_df = pd.read_csv(test_df)
# test_df = test_df.loc[test_df['mouse_id'].isin(mice)]
# selected_cells = test_df.loc[test_df['pval_mapping'] <= 0.05]

if select_responsive_cells:
    test_df = os.path.join(io.processed_dir, f'response_test_results_win_180ms.csv')
    test_df = pd.read_csv(test_df)
    test_df = test_df.loc[test_df['day'].isin(days)]
    # Select cells as responsive if they pass the test on at least one day.
    selected_cells = test_df.groupby(['mouse_id', 'roi', 'cell_type'])['pval_mapping'].min().reset_index()
    selected_cells = selected_cells.loc[selected_cells['pval_mapping'] <= 0.05/5]

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]


for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin(days))
    
    # Select responsive cells.
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue
        
    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day.
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    # Average bins.
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # # Equalize the mean activity of each cell across days by shifting (additive constant).
    # # For each cell, compute its mean activity on each day.
    # # Use the mean across all days as the reference mean.
    # # For each day, add a constant so that its mean matches the reference mean.
    # cell_axis = 0  # axis for cells
    # trial_axis = 1  # axis for trials
    # if 'cell' in d.dims:
    #     cell_dim = 'cell'
    # else:
    #     cell_dim = 'roi'
    # trial_dim = 'trial'
    # day_per_trial = d['day'].values
    # unique_days = np.unique(day_per_trial)
    # arr = d.values  # shape: (n_cells, n_trials)
    # arr_eq = arr.copy()
    # for icell in range(arr.shape[cell_axis]):
    #     # For each cell, get trial indices for each day
    #     cell_vals = arr[icell, :]
    #     # Compute mean across all days (reference)
    #     ref_mean = np.nanmean(cell_vals)
    #     for day in unique_days:
    #         day_mask = (day_per_trial == day)
    #         if np.any(day_mask):
    #             day_mean = np.nanmean(cell_vals[day_mask])
    #             shift = ref_mean - day_mean
    #             arr_eq[icell, day_mask] = cell_vals[day_mask] + shift
    # # Replace d.values with the shifted array
    # d.values[:] = arr_eq

    # Remove artefacts by setting them at 0. To avoid NaN values and
    # mismatches (that concerns a single cell).
    print(np.isnan(d.values).sum(), 'NaN values in the data.')
    d = d.fillna(0)
    
    if rew_gp == 'R-':
        vectors_nonrew.append(d)
        mice_nonrew.append(mouse)
    elif rew_gp == 'R+':
        vectors_rew.append(d)
        mice_rew.append(mouse)


# Decoding accuracy between reward groups.
# ----------------------------------------
# Train a single classifier per mouse and plot average cross-validated accuracy
# Convoluted function because I test mean equalization without leaks to test sets.

def per_mouse_cv_accuracy(vectors, label_encoder, seed=42, n_shuffles=100, return_weights=False, equalize=False, debug=False, n_jobs=20):
    accuracies = []
    chance_accuracies = []
    weights_per_mouse = []
    rng = np.random.default_rng(seed)

    for d in vectors:
        days_per_trial = d['day'].values
        mask = np.isin(days_per_trial, [-2, -1, 1, 2])
        trials = d.values[:, mask].T
        labels = np.array(['pre' if day in [-2, -1] else 'post' for day in days_per_trial[mask]])
        X = trials
        y = labels

        y_enc = label_encoder.transform(y)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        fold_scores = []
        all_true = []
        all_pred = []
        for train_idx, test_idx in cv.split(X, y_enc):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]
            # Equalization
            if equalize:
                ref_means = np.nanmean(X_train, axis=0)
                pre_label = label_encoder.transform(['pre'])[0]
                post_label = label_encoder.transform(['post'])[0]
                pre_mask_train = y_train == pre_label
                post_mask_train = y_train == post_label
                for icell in range(X_train.shape[1]):
                    pre_mean = np.nanmean(X_train[pre_mask_train, icell])
                    post_mean = np.nanmean(X_train[post_mask_train, icell])
                    X_train[pre_mask_train, icell] += ref_means[icell] - pre_mean
                    X_train[post_mask_train, icell] += ref_means[icell] - post_mean
                pre_mask_test = y_test == pre_label
                post_mask_test = y_test == post_label
                for icell in range(X_test.shape[1]):
                    pre_mean = np.nanmean(X_test[pre_mask_test, icell])
                    post_mean = np.nanmean(X_test[post_mask_test, icell])
                    X_test[pre_mask_test, icell] += ref_means[icell] - pre_mean
                    X_test[post_mask_test, icell] += ref_means[icell] - post_mean
            # Scaling (always applied)
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_test_proc = scaler.transform(X_test)
            clf.fit(X_train_proc, y_train)
            y_pred = clf.predict(X_test_proc)
            fold_scores.append(np.mean(y_pred == y_test))
            all_true.extend(y_test)
            all_pred.extend(y_pred)
        acc = np.mean(fold_scores)
        accuracies.append(acc)

        if debug:
            print("Accuracy:", acc)
            print("True labels:", all_true)
            print("Predicted labels:", all_pred)
            print("Label counts (true):", np.bincount(all_true))
            print("Label counts (pred):", np.bincount(all_pred))

        # Estimate chance level by shuffling labels n_shuffles times (parallelized)
        def shuffle_score(clf, X, y_enc, cv):
            y_shuff = rng.permutation(y_enc)
            fold_scores = []
            for train_idx, test_idx in cv.split(X, y_shuff):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_shuff[train_idx], y_shuff[test_idx]
                # Equalization
                if equalize:
                    ref_means = np.nanmean(X_train, axis=0)
                    pre_label = label_encoder.transform(['pre'])[0]
                    post_label = label_encoder.transform(['post'])[0]
                    pre_mask_train = y_train == pre_label
                    post_mask_train = y_train == post_label
                    for icell in range(X_train.shape[1]):
                        pre_mean = np.nanmean(X_train[pre_mask_train, icell])
                        post_mean = np.nanmean(X_train[post_mask_train, icell])
                        X_train[pre_mask_train, icell] += ref_means[icell] - pre_mean
                        X_train[post_mask_train, icell] += ref_means[icell] - post_mean
                    pre_mask_test = y_test == pre_label
                    post_mask_test = y_test == post_label
                    for icell in range(X_test.shape[1]):
                        pre_mean = np.nanmean(X_test[pre_mask_test, icell])
                        post_mean = np.nanmean(X_test[post_mask_test, icell])
                        X_test[pre_mask_test, icell] += ref_means[icell] - pre_mean
                        X_test[post_mask_test, icell] += ref_means[icell] - post_mean
                # Scaling (always applied)
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(X_train)
                X_test_proc = scaler.transform(X_test)
                clf.fit(X_train_proc, y_train)
                y_pred = clf.predict(X_test_proc)
                fold_scores.append(np.mean(y_pred == y_test))
            return np.mean(fold_scores)

        shuffle_scores = Parallel(n_jobs=n_jobs)(
            delayed(shuffle_score)(clf, X, y_enc, cv) for _ in range(n_shuffles)
        )
        chance_accuracies.append(np.mean(shuffle_scores))

        if return_weights:
            # Train classifier on full dataset and return weights
            scaler_full = StandardScaler()
            X_full = scaler_full.fit_transform(X)
            clf_full = LogisticRegression(max_iter=5000, random_state=seed)
            clf_full.fit(X_full, y_enc)
            weights_per_mouse.append(clf_full.coef_.flatten())

    if return_weights:
        return np.array(accuracies), np.array(chance_accuracies), weights_per_mouse
    else:
        return np.array(accuracies), np.array(chance_accuracies)


le = LabelEncoder()
le.fit(['pre', 'post'])

accs_rew, chance_rew, weights_rew = per_mouse_cv_accuracy(vectors_rew, le, n_shuffles=1, return_weights=True, equalize=False)
accs_nonrew, chance_nonrew, weights_nonrew = per_mouse_cv_accuracy(vectors_nonrew, le, n_shuffles=1, return_weights=True, equalize=False)

for w in weights_rew:
    plt.plot(w, label='R+')

print(f"Mean accuracy R+: {np.nanmean(accs_rew):.3f} +/- {np.nanstd(accs_rew):.3f}")
print(f"Mean accuracy R-: {np.nanmean(accs_nonrew):.3f} +/- {np.nanstd(accs_nonrew):.3f}")

# Plot
plt.figure(figsize=(4, 5))
# Plot chance levels in grey
sns.pointplot(data=[chance_rew, chance_nonrew], color='grey', estimator=np.nanmean, errorbar='ci', linestyles="none")
# Plot actual accuracies
sns.swarmplot(data=[accs_rew, accs_nonrew], palette=reward_palette[::-1], alpha=0.7)
sns.pointplot(data=[accs_rew, accs_nonrew], palette=reward_palette[::-1], linestyle=None, estimator=np.nanmean, errorbar='ci')
plt.xticks([0, 1], ['R+', 'R-'])
plt.ylabel('Cross-validated accuracy')
plt.title('Pre vs post learning classification accuracy across mice')
plt.ylim(0, 1)
sns.despine()

# Statistical test: Mann-Whitney U test between R+ and R- accuracies
stat, p_value = mannwhitneyu(accs_rew, accs_nonrew, alternative='two-sided')
print(f"Mann-Whitney U test: stat={stat:.3f}, p-value={p_value:.4f}")

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'decoding_accuracy.svg'
if projection_type is not None:
    svg_file = f'decoding_accuracy_{projection_type}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save results.
results_df = pd.DataFrame({
    'accuracy': np.concatenate([accs_rew, accs_nonrew]),
    'reward_group': ['R+'] * len(accs_rew) + ['R-'] * len(accs_nonrew),
    'chance_accuracy': np.concatenate([chance_rew, chance_nonrew]),
})
results_csv_file = 'decoding_accuracy_data.csv'
if projection_type is not None:
    results_csv_file = f'decoding_accuracy_data_{projection_type}.csv'
results_df.to_csv(os.path.join(output_dir, results_csv_file), index=False)
# Save stats.

stat_file = 'decoding_accuracy_stats.csv'
if projection_type is not None:
    stat_file = f'decoding_accuracy_stats_{projection_type}.csv'
pd.DataFrame({'stat': [stat], 'p_value': [p_value]}).to_csv(os.path.join(output_dir, stat_file), index=False)



# Relationship between classifier weights and learning modualtion index.
# ----------------------------------------------------------------------

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)

# Merge classifier weights and LMI for each mouse
weights_all = []
lmi_all = []
mouse_ids = []

for i, mouse in enumerate(mice_rew + mice_nonrew):
    # Get classifier weights for this mouse
    if mouse in mice_rew:
        w = weights_rew[mice_rew.index(mouse)]
    else:
        w = weights_nonrew[mice_nonrew.index(mouse)]
    # Get cell IDs
    d = vectors_rew[mice_rew.index(mouse)] if mouse in mice_rew else vectors_nonrew[mice_nonrew.index(mouse)]
    cell_ids = d['roi'].values if 'roi' in d.coords else d['cell'].values
    # Get LMI for this mouse
    lmi_mouse = lmi_df[(lmi_df['mouse_id'] == mouse) & (lmi_df['roi'].isin(cell_ids))]
    lmi_mouse = lmi_mouse.set_index('roi').reindex(cell_ids)
    lmi_vals = lmi_mouse['lmi'].values
    # Only keep cells with non-nan LMI and weights
    mask = ~np.isnan(lmi_vals) & ~np.isnan(w)
    weights_all.append(w[mask])
    lmi_all.append(lmi_vals[mask])
    mouse_ids.extend([mouse] * np.sum(mask))
    # Flatten lists
    weights_flat = np.concatenate(weights_all)
    lmi_flat = np.concatenate(lmi_all)
    mouse_ids_flat = np.array(mouse_ids)

# Plot scatter and regression for each mouse
plt.figure(figsize=(4, 4))

for i, mouse in enumerate(np.unique(mouse_ids_flat)):
    mask = mouse_ids_flat == mouse
    sns.scatterplot(x=lmi_flat[mask], y=weights_flat[mask], alpha=0.5)
    # # Regression line for each mouse
    # if np.sum(mask) > 1:
    #     reg = LinearRegression().fit(lmi_flat[mask].reshape(-1, 1), weights_flat[mask])
    #     x_vals = np.linspace(np.nanmin(lmi_flat[mask]), np.nanmax(lmi_flat[mask]), 100)
    #     plt.plot(x_vals, reg.predict(x_vals.reshape(-1, 1)), color='grey', alpha=0.5)

# Main regression line for all mice
reg_all = LinearRegression().fit(lmi_flat.reshape(-1, 1), weights_flat)
x_vals = np.linspace(np.nanmin(lmi_flat), np.nanmax(lmi_flat), 100)
y_pred = reg_all.predict(x_vals.reshape(-1, 1))
plt.plot(x_vals, y_pred, color='#2d2d2d', linewidth=2)

# Bootstrap confidence interval for regression line
n_boot = 1000
y_boot = np.zeros((n_boot, len(x_vals)))
for i in range(n_boot):
    Xb, yb = resample(lmi_flat, weights_flat)
    regb = LinearRegression().fit(Xb.reshape(-1, 1), yb)
    y_boot[i] = regb.predict(x_vals.reshape(-1, 1))
ci_low = np.percentile(y_boot, 2.5, axis=0)
ci_high = np.percentile(y_boot, 97.5, axis=0)
plt.fill_between(x_vals, ci_low, ci_high, color='black', alpha=0.2, label='95% CI')

plt.xlabel('Learning Modulation Index (LMI)')
plt.ylabel('Classifier Weight')
plt.title('Classifier Weight vs LMI')
plt.tight_layout()
plt.ylim(-2.5, 2)
sns.despine()

# Save plot
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_mouse.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_mouse.png'), format='png', dpi=300)


# Fit linear regression
reg = LinearRegression()
reg.fit(X, y)
r2 = reg.score(X, y)
print(f"Linear regression R^2: {r2:.3f}")

# Bootstrapped confidence interval for R^2
n_boot = 1000
r2_boot = []
for _ in range(n_boot):
    Xb, yb = resample(X, y)
    regb = LinearRegression().fit(Xb, yb)
    r2_boot.append(regb.score(Xb, yb))
r2_ci = np.percentile(r2_boot, [2.5, 97.5])
print(f"Bootstrapped R^2 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")

# Save plot
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_group.svg'), format='svg', dpi=300)


# Accuracy as a function of percent most modulated cells removed.
# ---------------------------------------------------------------

# This is to show that without the cells modulated on average, no information
# can be decoded. The non-modulated cells could still carry some information
# similarly to non-place cells in the hippocampus.


# Accuracy as a function of percent most modulated cells removed.
# ---------------------------------------------------------------

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)

le = LabelEncoder()
le.fit(['pre', 'post'])
percentiles = np.arange(0, 91, 5)  # 0% to 60% in steps of 5%
accs_rew_curve = []
accs_nonrew_curve = []

for perc in percentiles:
    accs_rew_perc = []
    accs_nonrew_perc = []
    for group, vectors, mice_group in zip(
        ['R+', 'R-'],
        [vectors_rew, vectors_nonrew],
        [mice_rew, mice_nonrew]
    ):
        for i, mouse in enumerate(mice_group):
            # Get vector and cell LMI for this mouse
            d = vectors[i]
            cell_ids = d['roi'].values if 'roi' in d.coords else d['cell'].values
            lmi_mouse = lmi_df[(lmi_df['mouse_id'] == mouse) & (lmi_df['roi'].isin(cell_ids))]
            lmi_mouse = lmi_mouse.set_index('roi').reindex(cell_ids)
            abs_lmi = np.abs(lmi_mouse['lmi'].values)
            # Sort cells by abs(LMI)
            sorted_idx = np.argsort(-abs_lmi)  # descending
            n_cells = len(cell_ids)
            n_remove = int(np.round(n_cells * perc / 100))
            keep_idx = sorted_idx[n_remove:]
            # If less than 2 cells remain, skip
            if len(keep_idx) < 2:
                continue
            # Subset vector
            d_sub = d.isel({d.dims[0]: keep_idx})
            # Classification
            days_per_trial = d_sub['day'].values
            mask = np.isin(days_per_trial, [-2, -1, 1, 2])
            trials = d_sub.values[:, mask].T
            labels = np.array(['pre' if day in [-2, -1] else 'post' for day in days_per_trial[mask]])
            y_enc = le.transform(labels)
            clf = LogisticRegression(max_iter=50000)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scaler = StandardScaler()
            X = scaler.fit_transform(trials)
            scores = cross_val_score(clf, X, y_enc, cv=cv, n_jobs=1)
            acc = np.mean(scores)
            if group == 'R+':
                accs_rew_perc.append(acc)
            else:
                accs_nonrew_perc.append(acc)
    accs_rew_curve.append(accs_rew_perc)
    accs_nonrew_curve.append(accs_nonrew_perc)
# Plot with bootstrap confidence intervals
mean_rew = []
ci_rew = []
mean_nonrew = []
ci_nonrew = []
x_vals = 100 - percentiles  # 100% to 40% retained

for a in accs_rew_curve:
    mean_rew.append(np.nanmean(a))
    if len(a) > 1:
        res = bootstrap((np.array(a),), np.nanmean, confidence_level=0.95, n_resamples=1000, method='basic')
        ci_rew.append(res.confidence_interval)
    else:
        ci_rew.append((np.nan, np.nan))
for a in accs_nonrew_curve:
    mean_nonrew.append(np.nanmean(a))
    if len(a) > 1:
        res = bootstrap((np.array(a),), np.nanmean, confidence_level=0.95, n_resamples=1000, method='basic')
        ci_nonrew.append(res.confidence_interval)
    else:
        ci_nonrew.append((np.nan, np.nan))

# Prepare DataFrame for seaborn
df_plot = pd.DataFrame({
    'percent_cells_retained': np.tile(x_vals, 2),
    'accuracy': np.concatenate([mean_rew, mean_nonrew]),
    'ci_low': np.concatenate([np.array([ci.low for ci in ci_rew]), np.array([ci.low for ci in ci_nonrew])]),
    'ci_high': np.concatenate([np.array([ci.high for ci in ci_rew]), np.array([ci.high for ci in ci_nonrew])]),
    'reward_group': ['R+'] * len(x_vals) + ['R-'] * len(x_vals)
})

plt.figure(figsize=(6, 5))
sns.lineplot(data=df_plot, x='percent_cells_retained', y='accuracy', hue='reward_group', palette=reward_palette[::-1])
for group, color in zip(['R+', 'R-'], reward_palette[::-1]):
    sub = df_plot[df_plot['reward_group'] == group]
    plt.fill_between(sub['percent_cells_retained'], sub['ci_low'], sub['ci_high'], color=color, alpha=0.3)
plt.xlabel('Percent of cells retained')
plt.ylabel('Classification accuracy')
plt.title('Accuracy vs percent modulated cells retained')
plt.legend()
plt.ylim(0, 1)
plt.xlim(100, min(x_vals))  # Flip x-axis: start at 100% and go down
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'accuracy_vs_percent_modulated_cells.svg'), format='svg', dpi=300)

# Save data
curve_df = pd.DataFrame({
    'percent_cells_retained': np.tile(100 - percentiles, 2),
    'accuracy': np.concatenate([mean_rew, mean_nonrew]),
    'sem': np.concatenate([sem_rew, sem_nonrew]),
    'reward_group': ['R+'] * len(percentiles) + ['R-'] * len(percentiles)
})
curve_df.to_csv(os.path.join(output_dir, 'accuracy_vs_percent_modulated_cells.csv'), index=False)



# Stimulus encoding across days for cells with high LMI.
# #######################################################

# Load LMI and ROC on baseline vs stimulus results.
roc_df = os.path.join(io.processed_dir, 'roc_stimvsbaseline_results.csv')
roc_df = pd.read_csv(roc_df)
lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)
# Add reward_group column to roc_df using get_mouse_reward_group_from_db

roc_df = io.add_reward_col_to_df(roc_df)

# Select significant LMI.
# selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025), ['mouse_id', 'roi']]

# Plot the proportion of significant cells encoding stimulus across days for all cells (no LMI selection)
roc_df_all = roc_df.copy()
roc_df_all.loc[(roc_df_all['roc_p'] <= 0.025) | (roc_df_all['roc_p'] >= 0.975), 'significant'] = True
roc_df_all['significant'] = roc_df_all['significant'].fillna(False)
prop_roc_all = roc_df_all.groupby(['mouse_id', 'day', 'reward_group'], as_index=False)['significant'].mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
# Rewarded group (R+)
sns.barplot(x='day', y='significant', data=prop_roc_all[prop_roc_all['reward_group'] == 'R+'],
            estimator=np.mean, color=reward_palette[1], ax=axes[0])
axes[0].set_title('R+ (All Cells)')
axes[0].set_ylim(0, 1)
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Proportion Significant')

# Non-rewarded group (R-)
sns.barplot(x='day', y='significant', data=prop_roc_all[prop_roc_all['reward_group'] == 'R-'],
            estimator=np.mean, color=reward_palette[0], ax=axes[1])
axes[1].set_title('R- (All Cells)')
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Proportion Significant')
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'proportion_significant_cells_encoding_stimulus_all_cells.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data
prop_roc_all.to_csv(os.path.join(output_dir, 'proportion_significant_cells_encoding_stimulus_all_cells.csv'), index=False)



# PLot the proportion of significant cells encoding stimulus across days.

selected_cells = lmi_df.loc[(lmi_df['lmi_p'] >= 0.975), ['mouse_id', 'roi']]
roc_df_pos = roc_df.merge(selected_cells, on=['mouse_id', 'roi'])
roc_df_pos.loc[(roc_df_pos['roc_p'] <= 0.025) | (roc_df_pos['roc_p'] >= 0.975), 'significant'] = True
roc_df_pos['significant'] = roc_df_pos['significant'].fillna(False)
# Plot proportion of significant cells encoding stimulus for positive LMI
prop_roc_pos = roc_df_pos.groupby(['mouse_id', 'day', 'reward_group'], as_index=False)['significant'].mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
# Rewarded group (R+)
sns.barplot(x='day', y='significant', data=prop_roc_pos[prop_roc_pos['reward_group'] == 'R+'],
            estimator=np.mean, color=reward_palette[1], ax=axes[0])
axes[0].set_title('R+ (Positive LMI)')
axes[0].set_ylim(0, 1)
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Proportion Significant')

# Non-rewarded group (R-)
sns.barplot(x='day', y='significant', data=prop_roc_pos[prop_roc_pos['reward_group'] == 'R-'],
            estimator=np.mean, color=reward_palette[0], ax=axes[1])
axes[1].set_title('R- (Positive LMI)')
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Proportion Significant')
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'proportion_significant_cells_encoding_stimulus_positive_lmi.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data
prop_roc_pos.to_csv(os.path.join(output_dir, 'proportion_significant_cells_encoding_stimulus_positive_lmi.csv'), index=False)

# Now for negative LMI.
selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025), ['mouse_id', 'roi']]
roc_df_neg = roc_df.merge(selected_cells, on=['mouse_id', 'roi'])
roc_df_neg.loc[(roc_df_neg['roc_p'] <= 0.025) | (roc_df_neg['roc_p'] >= 0.975), 'significant'] = True
roc_df_neg['significant'] = roc_df_neg['significant'].fillna(False)

# Proportion of significant cells encoding stimulus for negative LMI
prop_roc_neg = roc_df_neg.groupby(['mouse_id', 'day', 'reward_group'], as_index=False)['significant'].mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Rewarded group (R+)
sns.barplot(x='day', y='significant', data=prop_roc_neg[prop_roc_neg['reward_group'] == 'R+'],
            estimator=np.mean, color=reward_palette[1], ax=axes[0])
axes[0].set_title('R+ (Negative LMI)')
axes[0].set_ylim(0, 1)
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Proportion Significant')

# Non-rewarded group (R-)
sns.barplot(x='day', y='significant', data=prop_roc_neg[prop_roc_neg['reward_group'] == 'R-'],
            estimator=np.mean, color=reward_palette[0], ax=axes[1])
axes[1].set_title('R- (Negative LMI)')
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Proportion Significant')
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'proportion_significant_cells_encoding_stimulus_negative_lmi.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data
prop_roc_neg.to_csv(os.path.join(output_dir, 'proportion_significant_cells_encoding_stimulus_negative_lmi.csv'), index=False)


