import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as imaging_utils
import src.utils.utils_io as io
from src.utils.utils_plot import s2_m1_palette


# #############################################################################
# CDF analysis comparing LMI and classifier weights between wS2 and wM1.
# Uses Kolmogorov-Smirnov test to compare distributions.
# #############################################################################

print("\n" + "="*80)
print("CDF ANALYSIS: LMI AND CLASSIFIER WEIGHTS (wS2 vs wM1)")
print("="*80 + "\n")

# Output directory
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/projectors_contributions'
output_dir = io.adjust_path_to_host(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Cell types and colors
cell_types = ['wS2', 'wM1']
cell_type_colors = {
    'wS2': s2_m1_palette[0],
    'wM1': s2_m1_palette[1],
}


def pvalue_to_stars(p):
    """Convert p-value to significance notation."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# =============================================================================
# Load LMI data
# =============================================================================

print("Loading LMI data...")
processed_folder = io.solve_common_paths('processed_data')
lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))

# Add reward group
for mouse in lmi_df.mouse_id.unique():
    lmi_df.loc[lmi_df.mouse_id == mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(io.db_path, mouse)

# Define cell type groups
lmi_df['cell_type_group'] = lmi_df['cell_type'].replace({'na': 'non_projector'})

print(f"LMI data: {len(lmi_df)} cells from {lmi_df['mouse_id'].nunique()} mice")
print(f"Cell types: {lmi_df['cell_type_group'].value_counts().to_dict()}")


# =============================================================================
# Load classifier weights data
# =============================================================================

print("\nLoading classifier weights data...")
weights_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
weights_dir = io.adjust_path_to_host(weights_dir)
weights_df = pd.read_csv(os.path.join(weights_dir, 'classifier_weights.csv'))

print(f"Weights data: {len(weights_df)} cells from {weights_df['mouse_id'].nunique()} mice")

# Load cell types from xarray
cell_type_info = []
for mouse_id in weights_df['mouse_id'].unique():
    try:
        data_xr = imaging_utils.load_mouse_xarray(
            mouse_id,
            os.path.join(io.processed_dir, 'mice'),
            'tensor_xarray_mapping_data.nc'
        )
        rois = data_xr.coords['roi'].values
        cts = data_xr.coords['cell_type'].values if 'cell_type' in data_xr.coords else [None] * len(rois)
        for roi, ct in zip(rois, cts):
            cell_type_info.append({'mouse_id': mouse_id, 'roi': roi, 'cell_type_xr': ct})
    except Exception as e:
        print(f"Warning: Could not load cell types for {mouse_id}: {e}")

# Merge weights with cell types
if len(cell_type_info) > 0:
    weights_df = weights_df.merge(pd.DataFrame(cell_type_info), on=['mouse_id', 'roi'], how='left')
else:
    weights_df['cell_type_xr'] = None

# Define cell type groups
weights_df['cell_type_group'] = weights_df['cell_type_xr'].copy()
weights_df.loc[(weights_df['cell_type_xr'].isna()) | (weights_df['cell_type_xr'] == 'na'), 'cell_type_group'] = 'non_projector'

print(f"Cell types: {weights_df['cell_type_group'].value_counts().to_dict()}")


# =============================================================================
# KS test computation
# =============================================================================

def compute_ks_test(df, value_column, reward_groups, positive_only=False, negative_only=False):
    """Compute KS test comparing wS2 vs wM1 distributions."""
    results = []
    for reward_group in reward_groups:
        data_rg = df[df['reward_group'] == reward_group]

        if positive_only:
            data_rg = data_rg[data_rg[value_column] > 0]
        elif negative_only:
            data_rg = data_rg[data_rg[value_column] < 0]

        wS2_values = data_rg[data_rg['cell_type_group'] == 'wS2'][value_column].values
        wM1_values = data_rg[data_rg['cell_type_group'] == 'wM1'][value_column].values

        if len(wS2_values) > 0 and len(wM1_values) > 0:
            stat_2sided, pval_2sided = ks_2samp(wS2_values, wM1_values, alternative='two-sided')
            stat_greater, pval_greater = ks_2samp(wS2_values, wM1_values, alternative='greater')
            results.append({
                'reward_group': reward_group,
                'value_type': 'positive' if positive_only else ('negative' if negative_only else 'all'),
                'n_wS2': len(wS2_values),
                'n_wM1': len(wM1_values),
                'ks_pvalue_2sided': pval_2sided,
                'ks_stars_2sided': pvalue_to_stars(pval_2sided),
                'ks_pvalue_greater': pval_greater,
                'ks_stars_greater': pvalue_to_stars(pval_greater),
            })
    return pd.DataFrame(results)


print("\nComputing KS tests...")

# LMI KS tests
df_lmi_ks_pos = compute_ks_test(lmi_df, 'lmi', ['R+', 'R-'], positive_only=True)
df_lmi_ks_neg = compute_ks_test(lmi_df, 'lmi', ['R+', 'R-'], negative_only=True)
df_lmi_ks_pos.to_csv(os.path.join(output_dir, 'lmi_ks_test_positive.csv'), index=False)
df_lmi_ks_neg.to_csv(os.path.join(output_dir, 'lmi_ks_test_negative.csv'), index=False)

# Weights KS tests
df_weight_ks_pos = compute_ks_test(weights_df, 'classifier_weight', ['R+', 'R-'], positive_only=True)
df_weight_ks_neg = compute_ks_test(weights_df, 'classifier_weight', ['R+', 'R-'], negative_only=True)
df_weight_ks_pos.to_csv(os.path.join(output_dir, 'weight_ks_test_positive.csv'), index=False)
df_weight_ks_neg.to_csv(os.path.join(output_dir, 'weight_ks_test_negative.csv'), index=False)

print("Saved KS test results.")


# =============================================================================
# Figure 1: CDF - LMI
# =============================================================================

print("\nCreating CDF figures...")

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Positive LMI
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 0])
    data_rg = lmi_df[(lmi_df['reward_group'] == reward_group) & (lmi_df['lmi'] > 0)]

    for cell_type in cell_types:
        values = data_rg[data_rg['cell_type_group'] == cell_type]['lmi'].values
        if len(values) > 0:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=cell_type,
                    color=cell_type_colors[cell_type], linewidth=2, alpha=0.8)

    ax.set_xlabel('LMI Value' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{reward_group} - Positive LMI', fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.set_xlim(left=0, right=.8)
    ax.set_ylim(0,1)
    
    # Add KS test result
    ks_result = df_lmi_ks_pos[df_lmi_ks_pos['reward_group'] == reward_group]
    if len(ks_result) > 0:
        stars = ks_result.iloc[0]['ks_stars_2sided']
        pval = ks_result.iloc[0]['ks_pvalue_2sided']
        ax.text(0.98, 0.02, f'KS: {stars} (p={pval:.4f})',
                ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Negative LMI (absolute values)
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 1])
    data_rg = lmi_df[(lmi_df['reward_group'] == reward_group) & (lmi_df['lmi'] < 0)]

    for cell_type in cell_types:
        values = np.abs(data_rg[data_rg['cell_type_group'] == cell_type]['lmi'].values)
        if len(values) > 0:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=cell_type,
                    color=cell_type_colors[cell_type], linewidth=2, alpha=0.8)

    ax.set_xlabel('|LMI| Value' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{reward_group} - Negative LMI (abs)', fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.set_xlim(left=0, right=.8)
    ax.set_ylim(0,1)

    # Add KS test result
    ks_result = df_lmi_ks_neg[df_lmi_ks_neg['reward_group'] == reward_group]
    if len(ks_result) > 0:
        stars = ks_result.iloc[0]['ks_stars_2sided']
        pval = ks_result.iloc[0]['ks_pvalue_2sided']
        ax.text(0.98, 0.02, f'KS: {stars} (p={pval:.4f})',
                ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Cumulative Distribution Functions: LMI (wS2 vs wM1)',
             fontsize=14, fontweight='bold', y=0.98)
sns.despine()
plt.savefig(os.path.join(output_dir, 'figure_cdf_lmi.svg'), format='svg', dpi=300, bbox_inches='tight')
print("  Saved: figure_cdf_lmi.svg")


# =============================================================================
# Figure 2: CDF - Classifier Weights
# =============================================================================

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Positive weights
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 0])
    data_rg = weights_df[(weights_df['reward_group'] == reward_group) & (weights_df['classifier_weight'] > 0)]

    for cell_type in cell_types:
        values = data_rg[data_rg['cell_type_group'] == cell_type]['classifier_weight'].values
        if len(values) > 0:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=cell_type,
                    color=cell_type_colors[cell_type], linewidth=2, alpha=0.8)

    ax.set_xlabel('Classifier Weight' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{reward_group} - Positive Weights', fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.set_xlim(left=0, right=.8)
    ax.set_ylim(0,1)
    
    # Add KS test result
    ks_result = df_weight_ks_pos[df_weight_ks_pos['reward_group'] == reward_group]
    if len(ks_result) > 0:
        stars = ks_result.iloc[0]['ks_stars_2sided']
        pval = ks_result.iloc[0]['ks_pvalue_2sided']
        ax.text(0.98, 0.02, f'KS: {stars} (p={pval:.4f})',
                ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Negative weights (absolute values)
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 1])
    data_rg = weights_df[(weights_df['reward_group'] == reward_group) & (weights_df['classifier_weight'] < 0)]

    for cell_type in cell_types:
        values = np.abs(data_rg[data_rg['cell_type_group'] == cell_type]['classifier_weight'].values)
        if len(values) > 0:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=cell_type,
                    color=cell_type_colors[cell_type], linewidth=2, alpha=0.8)

    ax.set_xlabel('|Classifier Weight|' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{reward_group} - Negative Weights (abs)', fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.set_xlim(left=0, right=.8)
    ax.set_ylim(0,1)

    # Add KS test result
    ks_result = df_weight_ks_neg[df_weight_ks_neg['reward_group'] == reward_group]
    if len(ks_result) > 0:
        stars = ks_result.iloc[0]['ks_stars_2sided']
        pval = ks_result.iloc[0]['ks_pvalue_2sided']
        ax.text(0.98, 0.02, f'KS: {stars} (p={pval:.4f})',
                ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Cumulative Distribution Functions: Classifier Weights (wS2 vs wM1)',
             fontsize=14, fontweight='bold', y=0.98)
sns.despine()
plt.savefig(os.path.join(output_dir, 'figure_cdf_weights.svg'), format='svg', dpi=300, bbox_inches='tight')
print("  Saved: figure_cdf_weights.svg")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
