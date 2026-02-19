"""This script generates PSTH numpy arrays from lists of NWB files.
"""

import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import xarray as xr
import scipy.stats as stats

# sys.path.append('H:\\anthony\\repos\\NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from analysis.psth_analysis import make_events_aligned_array_3d
import nwb_wrappers.nwb_reader_functions as nwb_read
from src.utils.utils_behavior import make_behavior_table


# =============================================================================
# Create mice tensors with xarrays for session data (not mapping trials).
# =============================================================================

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
trial_indices_yaml = io.solve_common_paths('trial_indices')
stop_flag_yaml = io.solve_common_paths('stop_flags')
trial_indices_sensory_map_yaml = io.solve_common_paths('trial_indices_sensory_map')
stop_flag_sensory_map_yaml = io.solve_common_paths('stop_flags_sensory_map')
processed_data_dir = io.solve_common_paths('processed_data')

days = ['-2', '-1', '0', '+1', '+2']
_, nwb_list, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes')

with open(trial_indices_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
# # For "non motivated" sensory mapping trials at the end of the session.
# with open(trial_indices_sensory_map_yaml, 'r') as stream:
#     trial_indices_sensory_map = yaml.load(stream, yaml.Loader)
# trial_indices_sensory_map = pd.DataFrame(trial_indices_sensory_map.items(), columns=['session_id', 'trial_idx'])

mice_list = ['GF305',]
# nwb_list = [nwb for nwb in nwb_list if 'AR143' in nwb]

for mouse in mice_list:
    save_dir = os.path.join(processed_data_dir, 'mice', mouse)
    # Check if dataset is already created.
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tensor_xarray_learning_quietwindow.nc')
    # if os.path.exists(save_path_data):
    #     continue
    session_nwb = [nwb for nwb in nwb_list if mouse in nwb]
    
    # Get data and metadata for each session.
    sessions = []
    data = []
    metadatas = []

    for nwb_file in session_nwb:
        
        session_id = nwb_file[-25:-4]
        sessions.append(session_id)
        
        # Parameters for tensor array.
        cell_types = ['na', 'wM1', 'wS2']
        rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
        time_range = (5,1)
        epoch_name = None
        trial_selection = None

        idx_selection = trial_indices.loc[trial_indices.session_id==session_id, 'trial_idx'].values[0]
        # idx_sensory_map = trial_indices_sensory_map.loc[trial_indices_sensory_map.session_id==session_id, 'trial_idx'].values[0]

        # Generate a 3d array containing all trial types.
        print(f'Processing {session_id} {trial_selection}')
        traces, metadata = make_events_aligned_array_3d(nwb_file,
                                                        rrs_keys,
                                                        time_range,
                                                        trial_selection,
                                                        epoch_name,
                                                        cell_types,
                                                        idx_selection)

        data.append(traces)
        metadatas.append(metadata)
    
    # Sessions are concatenated on the trial dim.
    tensor = np.concatenate(data, axis=1)

    # Load trial table and compute performance for those sessions.
    print('Make behavior table')
    behav_table = make_behavior_table(session_nwb, sessions, db_path,
                                      cut_session=True,
                                      stop_flag_yaml=stop_flag_yaml,
                                      trial_indices_yaml=trial_indices_yaml)
        
    time = np.linspace(-time_range[0], time_range[1], traces.shape[2])
    # Create xarray.
    ds = xr.DataArray(tensor, dims=['cell', 'trial', 'time'],
                        coords={'roi': ('cell', metadata['rois']),
                                'cell_type': ('cell', metadata['cell_types']),
                                'time': time,
                                })
    for col in behav_table.columns:
        ds[col] = ('trial', behav_table[col].values)
    ds.attrs['session_ids'] = sessions
    ds.attrs['mouse_id'] = mouse

    # Save dataset.
    print(f'Saving {mouse}')
    ds.to_netcdf(save_path)


# =============================================================================
# Create mice tensors with xarrays for mapping trials.
# =============================================================================

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
trial_indices_sensory_map_yaml = io.solve_common_paths('trial_indices_sensory_map')
stop_flag_sensory_map_yaml = io.solve_common_paths('stop_flags_sensory_map')
processed_data_dir = io.solve_common_paths('processed_data')

days = ['-2', '-1', '0', '+1', '+2']
_, nwb_list, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

# For "non motivated" sensory mapping trials at the end of the session.
with open(trial_indices_sensory_map_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])

# mice_list = [mouse for mouse in mice_list if 'AR163'==mouse]
# mice_list = ['AR144']

for mouse in mice_list:
    save_dir = os.path.join(processed_data_dir, 'mice', mouse)
    # Check if dataset is already created.
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tensor_xarray_mapping_data.nc')
    # if os.path.exists(save_path_data):
    #     continue
    session_nwb = [nwb for nwb in nwb_list if mouse in nwb]
    
    # Get data and metadata for each session.
    sessions = []
    data = []
    metadatas = []
    behavior_days = []

    for nwb_file in session_nwb:
        
        session_id = nwb_file[-25:-4]
        sessions.append(session_id)
        
        # Parameters for tensor array.
        cell_types = ['na', 'wM1', 'wS2']
        rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
        time_range = (2,7)
        epoch_name = None
        trial_selection = None

        idx_selection = trial_indices.loc[trial_indices.session_id==session_id, 'trial_idx'].values[0]

        # Generate a 3d array containing all trial types.
        print(f'Processing {session_id} {trial_selection}')
        traces, metadata = make_events_aligned_array_3d(nwb_file,
                                                        rrs_keys,
                                                        time_range,
                                                        trial_selection,
                                                        epoch_name,
                                                        cell_types,
                                                        idx_selection)

        print(f'{np.isnan(traces).sum()} nan values in tensor.')
        
        data.append(traces)
        metadatas.append(metadata)
        # Get session day.
        # I don't load the trial table with the mapping trials, so get it for
        # nwb file.
        d = nwb_read.get_session_metadata(nwb_file)['day']
        behavior_days.extend([d for _ in range(traces.shape[1])])
        
    # Sessions are concatenated on the trial dim.
    tensor = np.concatenate(data, axis=1)
    
    time = np.linspace(-time_range[0], time_range[1], traces.shape[2])
    # Create xarray.
    ds = xr.DataArray(tensor, dims=['cell', 'trial', 'time'],
                        coords={'roi': ('cell', metadata['rois']),
                                'cell_type': ('cell', metadata['cell_types']),
                                'time': time,
                                'day': ('trial', behavior_days),
                                })
    ds.attrs['session_ids'] = sessions
    ds.attrs['mouse_id'] = mouse

    # Save dataset.
    print(f'Saving {mouse}')
    ds.to_netcdf(save_path)


# #############################################################################
#  Load xarrays and substract baseline.
# #############################################################################

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
trial_indices_sensory_map_yaml = io.solve_common_paths('trial_indices_sensory_map')
stop_flag_sensory_map_yaml = io.solve_common_paths('stop_flags_sensory_map')
processed_data_dir = io.solve_common_paths('processed_data')
days = ['-2', '-1', '0', '+1', '+2']
sampling_rate = 30  # Hz, for imaging data.
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))

_, nwb_list, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes')

# mice_list = ['GF305',]
for mouse_id in mice_list:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
    xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)

    # Save the xarray.
    save_path = os.path.join(folder, mouse_id, 'tensor_xarray_learning_data_baselinesubstracted.nc')
    xarr.to_netcdf(save_path)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')

    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
    xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win)
    # Save the xarray.
    save_path = os.path.join(folder, mouse_id, 'tensor_xarray_mapping_data_baselinesubstracted.nc')
    xarr.to_netcdf(save_path)




# # #############################################################################
# # Lick-aligned xarrays.
# # #############################################################################

# db_path = io.solve_common_paths('db')
# db = io.read_excel_db(db_path)
# nwb_path = io.solve_common_paths('nwb')
# processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')

# days = ['-2', '-1', '0', '+1', '+2']
# _, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
#                                                 exclude_cols=['exclude', 'two_p_exclude'],
#                                                 experimenters=['AR', 'GF', 'MI'],
#                                                 day=days,
#                                                 two_p_imaging='yes',)

# # mouse = 'GF305'
# # mice_list = mice_list[-8:]
# # mice_list = mice_list[-8:]

# for mouse in mice_list:
#     print(f'Processing lick aligned array for {mouse}')
#     file_name = 'tensor_xarray_learning_data.nc'
#     xarray = utils_imaging.load_mouse_xarray(mouse, processed_dir, file_name, substracted=True)
#     rew_gp = io.get_mouse_reward_group_from_db(db_path, mouse, db)

#     lick_times = xarray.coords['lick_time'].values
#     stim_onset = xarray.coords['stim_onset'].values
#     reaction_time = lick_times - stim_onset
#     # GF333 and GF334 have a few strange reaction times.
#     reaction_time[reaction_time>=1.25] = np.nan
#     # plt.plot(reaction_time)
#     # plt.show()
        
#     # Create a new xarray for lick-aligned traces
#     time = xarray.coords['time'].values
#     aligned_time = np.linspace(-1, 5, 180)
#     aligned_traces = []

#     for itrial in range(xarray.shape[1]):
#         lick_onset = reaction_time[itrial]
#         # Trials with no lick are set to nan.
#         if np.isnan(lick_onset):
#             aligned_traces.append(np.full((xarray.shape[0], 180), np.nan))
#             continue 
#         lick_onset_idx =  (np.abs(time - lick_onset)).argmin()
#         data = xarray[:, itrial, :].values
#         aligned_data = data[:, lick_onset_idx-30:lick_onset_idx+150]
        
#         # Two samples missing sometimes for late licks.
#         if aligned_data.shape[1] < 180:
#             pad_width = 180 - aligned_data.shape[1]
#             pad_values = np.repeat(aligned_data[:, -1][:, np.newaxis], pad_width, axis=1)
#             aligned_data = np.concatenate([aligned_data, pad_values], axis=1)
#         aligned_traces.append(aligned_data)

#     aligned_traces = np.stack(aligned_traces, axis=1)
#     aligned_traces.shape
#     aligned_xarray = xr.DataArray(aligned_traces,
#                                 dims=['cell', 'trial', 'time'],
#                                 coords={'time': ('time', aligned_time),
#                                         'reaction_time': ('trial', reaction_time),}
#                                 )

#     # Add all other coordinates from the original xarray
#     for coord in xarray.coords:
#         if coord not in aligned_xarray.coords:
#             aligned_xarray.coords[coord] = xarray.coords[coord]

#     # Save the aligned xarray
#     save_path = os.path.join(processed_dir, mouse, 'lick_aligned_xarray.nc')
#     aligned_xarray.to_netcdf(save_path)
#     print(f'Saved {save_path}')
