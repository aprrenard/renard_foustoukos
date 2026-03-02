import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(r'/Users/aprrenard/repos/NWB_analysis')
sys.path.append(r'/Users/aprrenard/repos/renard_foustoukos_2025')
import src.utils.utils_io as io
from nwb_wrappers.nwb_reader_functions import get_image_mask


# ---- Parameters ----
MOUSE_ID = 'GF314'
NWB_FILE = os.path.join(io.nwb_dir, 'GF314_28112020_171800.nwb')
OPS_PATH = io.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/Georgios_Foustoukos/Suite2PRois/GF314/ops.npy'
)
SEGMENTATION_INFO = ['ophys', 'all_cells', 'my_plane_segmentation']


# ---- Load LMI data ----
lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
lmi_df = lmi_df[lmi_df['mouse_id'] == MOUSE_ID].reset_index(drop=True)
n_lmi = len(lmi_df)
print(f"LMI entries for {MOUSE_ID}: {n_lmi}")


# ---- Load cell image masks from NWB ----
# Keys: module 'ophys' > ImageSegmentation 'all_cells' > PlaneSegmentation 'my_plane_segmentation'
image_masks = get_image_mask(NWB_FILE, SEGMENTATION_INFO)
n_cells_nwb = len(image_masks)
print(f"Cells in NWB: {n_cells_nwb}")
print(f"Match between NWB cells and LMI entries: {n_cells_nwb == n_lmi}")


# ---- Load mean image ----
ops = np.load(OPS_PATH, allow_pickle=True)
mean_img = ops.item()['meanImg']


# ---- Build LMI color overlay ----
lmi_values = lmi_df['lmi'].values
roi_indices = lmi_df['roi'].values.astype(int)

lmi_max = np.nanmax(np.abs(lmi_values))
norm = mcolors.Normalize(vmin=-lmi_max, vmax=lmi_max)
cmap = plt.cm.RdBu_r  # red = positive LMI, blue = negative LMI

# RGBA overlay canvas (same shape as meanImg)
overlay = np.zeros((*mean_img.shape, 4), dtype=float)

for roi, lmi_val in zip(roi_indices, lmi_values):
    if np.isnan(lmi_val):
        continue
    mask = image_masks[roi]
    rgba = cmap(norm(lmi_val))
    pixels = mask > 0
    overlay[pixels] = rgba


# ---- Plot ----
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(mean_img, cmap='gray', interpolation='none')
ax.imshow(overlay, interpolation='none', alpha=0.7)
ax.axis('off')

plt.tight_layout()
save_path = os.path.join(io.results_dir, f'fov_{MOUSE_ID}.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved to {save_path}")
