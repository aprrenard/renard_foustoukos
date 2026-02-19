import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io

# path = '/mnt/lsens-analysis/Anthony_Renard/Georgios_Foustoukos/Suite2PRois/GF314/stat.npy'
# stat = np.load(path, allow_pickle=True)
# stat
# stat[0].keys()

# path = '/mnt/lsens-analysis/Anthony_Renard/Georgios_Foustoukos/Suite2PRois/GF314/ops.npy'
# ops = np.load(path, allow_pickle=True)
# ops.item().keys()


# path = '/mnt/lsens-analysis/Anthony_Renard/Georgios_Foustoukos/Suite2PRois/GF314/iscell.npy'
# iscell = np.load(path, allow_pickle=True)
# iscell[:, 0].sum()

# plt.imshow(ops.item()['meanImg'], cmap='grey')

# mean_img = ops.item()['meanImg']


# lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
# lmi_df = lmi_df[lmi_df['mouse_id'] == 'GF314']
# lmi_df.count()

# stat.shape



