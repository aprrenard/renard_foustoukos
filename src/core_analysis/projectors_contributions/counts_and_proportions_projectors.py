import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as imaging_utils
import src.utils.utils_io as io
from src.utils.utils_plot import *


# #############################################################################
# Contribution of projection neurons to LMI and classifier weights.
# Simplified version: analyze LMI and weights independently (no merge).
# #############################################################################

print("\n" + "="*80)
print("PROJECTION NEURON CONTRIBUTIONS TO LMI AND CLASSIFIER WEIGHTS")
print("="*80 + "\n")

# Output directory
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/projectors_contributions'
output_dir = io.adjust_path_to_host(output_dir)
os.makedirs(output_dir, exist_ok=True)

cell_types = ['non_projector', 'wS2', 'wM1']
cell_type_colors = {
    'non_projector': '#808080',  # Gray
    'wS2': '#E74C3C',  # Red
    'wM1': '#3498DB'   # Blue
}

# =============================================================================
# PART 1: LMI ANALYSIS
# =============================================================================

print("\n" + "-"*80)
print("PART 1: LMI ANALYSIS")
print("-"*80 + "\n")

# Load LMI data
print("Loading LMI data...")
processed_folder = io.solve_common_paths('processed_data')
lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))

# Add reward group information
for mouse in lmi_df.mouse_id.unique():
    lmi_df.loc[lmi_df.mouse_id==mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(io.db_path, mouse)

print(f"LMI data: {len(lmi_df)} cells from {lmi_df['mouse_id'].nunique()} mice")

# Define cell type groups (non-projectors are cells with NaN in cell_type column)
lmi_df['cell_type_group'] = lmi_df['cell_type'].copy()
lmi_df.loc[lmi_df['cell_type'].isna(), 'cell_type_group'] = 'non_projector'
lmi_df.loc[lmi_df['cell_type'] == 'wS2', 'cell_type_group'] = 'wS2'
lmi_df.loc[lmi_df['cell_type'] == 'wM1', 'cell_type_group'] = 'wM1'

# Keep only the three groups we're interested in
lmi_df = lmi_df[lmi_df['cell_type_group'].isin(['non_projector', 'wS2', 'wM1'])]

print(f"Cell type distribution:\n{lmi_df['cell_type_group'].value_counts()}")

# Define LMI criteria (globally across all mice)
lmi_df['positive_lmi'] = lmi_df['lmi_p'] >= 0.975
lmi_df['negative_lmi'] = lmi_df['lmi_p'] <= 0.025

abs_lmi = np.abs(lmi_df['lmi'])
lmi_df['top_1pct_abs_lmi'] = abs_lmi >= np.percentile(abs_lmi, 99)
lmi_df['top_5pct_abs_lmi'] = abs_lmi >= np.percentile(abs_lmi, 95)
lmi_df['top_10pct_abs_lmi'] = abs_lmi >= np.percentile(abs_lmi, 90)

print(f"\nLMI criteria counts:")
print(f"  Positive LMI: {lmi_df['positive_lmi'].sum()}")
print(f"  Negative LMI: {lmi_df['negative_lmi'].sum()}")
print(f"  Top 1% abs LMI: {lmi_df['top_1pct_abs_lmi'].sum()}")
print(f"  Top 5% abs LMI: {lmi_df['top_5pct_abs_lmi'].sum()}")
print(f"  Top 10% abs LMI: {lmi_df['top_10pct_abs_lmi'].sum()}")

# Compute proportions for each reward group and cell type
lmi_criteria = ['positive_lmi', 'negative_lmi', 'top_1pct_abs_lmi', 'top_5pct_abs_lmi', 'top_10pct_abs_lmi']

lmi_proportions = []
for reward_group in ['R+', 'R-']:
    for cell_type in cell_types:
        data_subset = lmi_df[
            (lmi_df['reward_group'] == reward_group) &
            (lmi_df['cell_type_group'] == cell_type)
        ]

        n_total = len(data_subset)

        for criterion in lmi_criteria:
            n_met = data_subset[criterion].sum()
            proportion = n_met / n_total if n_total > 0 else 0

            lmi_proportions.append({
                'reward_group': reward_group,
                'cell_type': cell_type,
                'criterion': criterion,
                'n_total': n_total,
                'n_met': n_met,
                'proportion': proportion
            })

df_lmi_props = pd.DataFrame(lmi_proportions)

# Save LMI proportions
df_lmi_props.to_csv(os.path.join(output_dir, 'lmi_proportions.csv'), index=False)
print(f"\nSaved lmi_proportions.csv")

# =============================================================================
# PART 2: CLASSIFIER WEIGHTS ANALYSIS
# =============================================================================

print("\n" + "-"*80)
print("PART 2: CLASSIFIER WEIGHTS ANALYSIS")
print("-"*80 + "\n")

# Load classifier weights
print("Loading classifier weights...")
weights_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
weights_dir = io.adjust_path_to_host(weights_dir)
weights_df = pd.read_csv(os.path.join(weights_dir, 'classifier_weights.csv'))

print(f"Weights data: {len(weights_df)} cells from {weights_df['mouse_id'].nunique()} mice")

# Load cell type information from xarray for each mouse
print("Loading cell type information from xarray data...")
cell_type_info = []
for mouse_id in weights_df['mouse_id'].unique():
    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    try:
        data_xr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)
        rois = data_xr.coords['roi'].values
        cell_types_arr = data_xr.coords['cell_type'].values if 'cell_type' in data_xr.coords else [None] * len(rois)

        for roi, ct in zip(rois, cell_types_arr):
            cell_type_info.append({
                'mouse_id': mouse_id,
                'roi': roi,
                'cell_type_xr': ct
            })
    except Exception as e:
        print(f"  Warning: Could not load xarray for {mouse_id}: {e}")

df_cell_types = pd.DataFrame(cell_type_info)

# Merge weights with cell type info
weights_df = weights_df.merge(df_cell_types, on=['mouse_id', 'roi'], how='left')

# Define cell type groups
weights_df['cell_type_group'] = weights_df['cell_type_xr'].copy()
weights_df.loc[weights_df['cell_type_xr'].isna(), 'cell_type_group'] = 'non_projector'
weights_df.loc[weights_df['cell_type_xr'] == 'wS2', 'cell_type_group'] = 'wS2'
weights_df.loc[weights_df['cell_type_xr'] == 'wM1', 'cell_type_group'] = 'wM1'

# Keep only the three groups
weights_df = weights_df[weights_df['cell_type_group'].isin(['non_projector', 'wS2', 'wM1'])]

print(f"Cell type distribution:\n{weights_df['cell_type_group'].value_counts()}")

# Define weight criteria (globally across all mice, separate by sign)
pos_weights = weights_df[weights_df['classifier_weight'] > 0]['classifier_weight']
neg_weights = weights_df[weights_df['classifier_weight'] < 0]['classifier_weight']

weights_df['top_1pct_pos_weight'] = False
weights_df['top_5pct_pos_weight'] = False
weights_df['top_10pct_pos_weight'] = False
weights_df['top_1pct_neg_weight'] = False
weights_df['top_5pct_neg_weight'] = False
weights_df['top_10pct_neg_weight'] = False

if len(pos_weights) > 0:
    weights_df.loc[weights_df['classifier_weight'] > 0, 'top_1pct_pos_weight'] = \
        weights_df.loc[weights_df['classifier_weight'] > 0, 'classifier_weight'] >= np.percentile(pos_weights, 99)
    weights_df.loc[weights_df['classifier_weight'] > 0, 'top_5pct_pos_weight'] = \
        weights_df.loc[weights_df['classifier_weight'] > 0, 'classifier_weight'] >= np.percentile(pos_weights, 95)
    weights_df.loc[weights_df['classifier_weight'] > 0, 'top_10pct_pos_weight'] = \
        weights_df.loc[weights_df['classifier_weight'] > 0, 'classifier_weight'] >= np.percentile(pos_weights, 90)

if len(neg_weights) > 0:
    weights_df.loc[weights_df['classifier_weight'] < 0, 'top_1pct_neg_weight'] = \
        weights_df.loc[weights_df['classifier_weight'] < 0, 'classifier_weight'] <= np.percentile(neg_weights, 1)
    weights_df.loc[weights_df['classifier_weight'] < 0, 'top_5pct_neg_weight'] = \
        weights_df.loc[weights_df['classifier_weight'] < 0, 'classifier_weight'] <= np.percentile(neg_weights, 5)
    weights_df.loc[weights_df['classifier_weight'] < 0, 'top_10pct_neg_weight'] = \
        weights_df.loc[weights_df['classifier_weight'] < 0, 'classifier_weight'] <= np.percentile(neg_weights, 10)

print(f"\nWeight criteria counts:")
print(f"  Top 1% positive weights: {weights_df['top_1pct_pos_weight'].sum()}")
print(f"  Top 5% positive weights: {weights_df['top_5pct_pos_weight'].sum()}")
print(f"  Top 10% positive weights: {weights_df['top_10pct_pos_weight'].sum()}")
print(f"  Top 1% negative weights: {weights_df['top_1pct_neg_weight'].sum()}")
print(f"  Top 5% negative weights: {weights_df['top_5pct_neg_weight'].sum()}")
print(f"  Top 10% negative weights: {weights_df['top_10pct_neg_weight'].sum()}")

# Compute proportions for each reward group and cell type
weight_criteria = [
    'top_1pct_pos_weight', 'top_5pct_pos_weight', 'top_10pct_pos_weight',
    'top_1pct_neg_weight', 'top_5pct_neg_weight', 'top_10pct_neg_weight'
]

weight_proportions = []
for reward_group in ['R+', 'R-']:
    for cell_type in cell_types:
        data_subset = weights_df[
            (weights_df['reward_group'] == reward_group) &
            (weights_df['cell_type_group'] == cell_type)
        ]

        n_total = len(data_subset)

        for criterion in weight_criteria:
            n_met = data_subset[criterion].sum()
            proportion = n_met / n_total if n_total > 0 else 0

            weight_proportions.append({
                'reward_group': reward_group,
                'cell_type': cell_type,
                'criterion': criterion,
                'n_total': n_total,
                'n_met': n_met,
                'proportion': proportion
            })

df_weight_props = pd.DataFrame(weight_proportions)

# Save weight proportions
df_weight_props.to_csv(os.path.join(output_dir, 'weight_proportions.csv'), index=False)
print(f"\nSaved weight_proportions.csv")

# =============================================================================
# PART 3: VISUALIZATIONS
# =============================================================================

print("\n" + "-"*80)
print("PART 3: CREATING VISUALIZATIONS")
print("-"*80 + "\n")

# Figure 1: LMI Proportions
# ==========================
print("Creating LMI proportions figure...")

lmi_labels = ['Positive\nLMI', 'Negative\nLMI', 'Top 1%\nabs LMI', 'Top 5%\nabs LMI', 'Top 10%\nabs LMI']

fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True)

for reward_idx, reward_group in enumerate(['R+', 'R-']):
    for crit_idx, (criterion, label) in enumerate(zip(lmi_criteria, lmi_labels)):
        ax = axes[reward_idx, crit_idx]

        plot_data = df_lmi_props[
            (df_lmi_props['reward_group'] == reward_group) &
            (df_lmi_props['criterion'] == criterion)
        ]

        x_pos = np.arange(len(cell_types))
        proportions = [plot_data[plot_data['cell_type'] == ct]['proportion'].values[0]
                      if len(plot_data[plot_data['cell_type'] == ct]) > 0 else 0
                      for ct in cell_types]

        colors_list = [cell_type_colors[ct] for ct in cell_types]
        ax.bar(x_pos, proportions, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Non-proj', 'wS2', 'wM1'], rotation=45, ha='right')
        ax.set_ylim(0, max(0.5, max(proportions) * 1.2) if max(proportions) > 0 else 0.1)

        if reward_idx == 0:
            ax.set_title(label, fontsize=12, fontweight='bold')
        if crit_idx == 0:
            ax.set_ylabel(f'{reward_group}\nProportion', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_lmi_proportions.svg'), format='svg', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure_lmi_proportions.svg")

# Figure 2: Classifier Weight Proportions
# ========================================
print("Creating classifier weight proportions figure...")

weight_labels = [
    'Top 1%\nPos Weight', 'Top 5%\nPos Weight', 'Top 10%\nPos Weight',
    'Top 1%\nNeg Weight', 'Top 5%\nNeg Weight', 'Top 10%\nNeg Weight'
]

fig, axes = plt.subplots(2, 6, figsize=(22, 8), sharex=True, sharey=True)

for reward_idx, reward_group in enumerate(['R+', 'R-']):
    for crit_idx, (criterion, label) in enumerate(zip(weight_criteria, weight_labels)):
        ax = axes[reward_idx, crit_idx]

        plot_data = df_weight_props[
            (df_weight_props['reward_group'] == reward_group) &
            (df_weight_props['criterion'] == criterion)
        ]

        x_pos = np.arange(len(cell_types))
        proportions = [plot_data[plot_data['cell_type'] == ct]['proportion'].values[0]
                      if len(plot_data[plot_data['cell_type'] == ct]) > 0 else 0
                      for ct in cell_types]

        colors_list = [cell_type_colors[ct] for ct in cell_types]
        ax.bar(x_pos, proportions, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Non-proj', 'wS2', 'wM1'], rotation=45, ha='right')
        ax.set_ylim(0, max(0.5, max(proportions) * 1.2) if max(proportions) > 0 else 0.1)

        if reward_idx == 0:
            ax.set_title(label, fontsize=11, fontweight='bold')
        if crit_idx == 0:
            ax.set_ylabel(f'{reward_group}\nProportion', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_weight_proportions.svg'), format='svg', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure_weight_proportions.svg")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")
print(f"All outputs saved to: {output_dir}")
print(f"  - lmi_proportions.csv")
print(f"  - weight_proportions.csv")
print(f"  - figure_lmi_proportions.svg")
print(f"  - figure_weight_proportions.svg")
