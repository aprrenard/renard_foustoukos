import io
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import warnings
from pathlib import Path


def read_excel_db(db_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        database = pd.read_excel(db_path, converters={'day': str})

    # Remove empty lines.
    database = database.loc[~database.isna().all(axis=1)]

    return database


def select_sessions_from_db(db_path, nwb_path, experimenters=None,
                            exclude_cols=['exclude', 'two_p_exclude'],
                            **filters):
    """Select sessions from excel database on filters.
    Return a list of session ids or nwb file paths.

    Args:
        db_path (_type_): _description_
        experimenters (_type_): _description_
        exclude_cols (_type_): _description_
        nwb_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    db = read_excel_db(db_path)
    for key, val in filters.items():
        if isinstance(val, list):
            db = db.loc[(db[key].isin(val))]
        else:
            db = db.loc[(db[key]==val)]
    # Remove excluded sessions.
    for col in exclude_cols:
        db = db.loc[(db[col]!='exclude')]
        
    mice_list = list(db.mouse_id.unique())
    session_list = list(db.session_id)
    if experimenters:
        session_list = [session for session in session_list
                        if session[:2] in experimenters]
        mice_list = [mouse for mouse in mice_list
                     if mouse[:2] in experimenters]
    nwb_paths = [os.path.join(nwb_path, f + '.nwb') for f in session_list]
    
    return session_list, nwb_paths, mice_list, db   


def select_mice_from_db(db_path, nwb_path, experimenters=None,
                            exclude_cols=['exclude', 'two_p_exclude'],
                            **filters):
    """Select mice from excel database on filters.
    Return a list of mouse ids.

    Args:
        db_path (_type_): _description_
        experimenters (_type_): _description_
        exclude_cols (_type_): _description_
        nwb_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    db = read_excel_db(db_path)
    for key, val in filters.items():
        if isinstance(val, list):
            db = db.loc[(db[key].isin(val))]
        else:
            db = db.loc[(db[key]==val)]
    # Remove excluded sessions.
    for col in exclude_cols:
        db = db.loc[(db[col]!='exclude')]
        
    mice_list = list(db.mouse_id.unique())
    if experimenters:
        mice_list = [mouse for mouse in mice_list
                     if mouse[:2] in experimenters]
    
    return mice_list


def get_reward_group_from_db(db_path, session_id):
    db = read_excel_db(db_path)
    reward_group = db.loc[db['session_id']==session_id, 'reward_group'].values[0]

    return reward_group


def get_mouse_reward_group_from_db(db_path, mouse_id, db=None):
    if db is None:
        db = read_excel_db(db_path)
    reward_group = db.loc[db['mouse_id']==mouse_id, 'reward_group'].values[0]

    return reward_group


def add_reward_col_to_df(df, mice_list=None):
    if mice_list is None:
        mice_list = df['mouse_id'].unique()
    reward_dict = {mouse: get_mouse_reward_group_from_db(db_path, mouse) for mouse in mice_list}
    df['reward_group'] = df['mouse_id'].map(reward_dict)
    return df


def read_group_yaml(group_yaml_path):
    with open(group_yaml_path, 'r') as file:
        group_yaml = yaml.safe_load(file)

    return group_yaml


def read_stop_flags_and_indices_yaml(stop_flag_yaml_path, trial_indices_path):
    with open(stop_flag_yaml_path, 'r') as file:
        stop_flags = yaml.load(file, Loader=yaml.FullLoader)
    with open(trial_indices_path, 'r') as file:
        trial_indices = yaml.load(file, Loader=yaml.FullLoader)
    trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
    
    return stop_flags, trial_indices



def adjust_path_to_host(path):
    import platform
    host = os.popen('hostname').read().strip()
    if 'haas' in host:
        # Linux analysis server: use /mnt mount points.
        if '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis' in path:
            path = path.replace('//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis', '/mnt/lsens-analysis')
        elif '//sv-nas1.rcp.epfl.ch/Petersen-Lab/data' in path:
            path = path.replace('//sv-nas1.rcp.epfl.ch/Petersen-Lab/data', '/mnt/lsens-data')
    elif platform.system() == 'Darwin':
        # macOS: server mounted under /Volumes/Petersen-Lab/.
        if '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis' in path:
            path = path.replace('//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis', '/Volumes/Petersen-Lab/analysis')
        elif '//sv-nas1.rcp.epfl.ch/Petersen-Lab/data' in path:
            path = path.replace('//sv-nas1.rcp.epfl.ch/Petersen-Lab/data', '/Volumes/Petersen-Lab/data')
        elif '/mnt/lsens-analysis' in path:
            path = path.replace('/mnt/lsens-analysis', '/Volumes/Petersen-Lab/analysis')
        elif '/mnt/lsens-data' in path:
            path = path.replace('/mnt/lsens-data', '/Volumes/Petersen-Lab/data')
    else:
        # Windows or other: use UNC paths.
        if '/mnt/lsens-analysis' in path:
            path = path.replace('/mnt/lsens-analysis', '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis')
        elif '/mnt/lsens-data' in path:
            path = path.replace('/mnt/lsens-data', '//sv-nas1.rcp.epfl.ch/Petersen-Lab/data')
    return path


def solve_common_paths(target):
    
    # Directories.
    data_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/data'
    analysis_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard'
    nwb_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/NWB'
    processed_data_dir = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed'
    results_dir = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/fast-learning'
    
    # Files.    
    db_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/session_metadata.xlsx'
    trial_indices_yaml = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/trial_indices_end_session.yaml'
    trial_indices_sensory_map_yaml = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/trial_indices_sensory_map.yaml'
    stop_flags_yaml = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/stop_flags_end_session.yaml'
    stop_flags_sensory_map_yaml = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/stop_flags_sensory_map.yaml'
    
    if target == 'data':
        path = data_path
    elif target == 'analysis':
        path = analysis_path
    elif target == 'nwb':
        path = nwb_path
    elif target == 'processed_data':
        path = processed_data_dir
    elif target == 'db':
        path = db_path
    elif target == 'trial_indices':
        path = trial_indices_yaml
    elif target == 'trial_indices_sensory_map':
        path = trial_indices_sensory_map_yaml
    elif target == 'stop_flags':
        path = stop_flags_yaml
    elif target == 'stop_flags_sensory_map':
        path = stop_flags_sensory_map_yaml
    elif target == 'results':
        path = results_dir
        
    return adjust_path_to_host(path)


# Path to the directory containing the processed data.
results_dir = solve_common_paths('results')
processed_dir = solve_common_paths('processed_data')
nwb_dir = solve_common_paths('nwb')
db_path = solve_common_paths('db')
trial_indices_yaml = solve_common_paths('trial_indices')
trial_indices_sensory_map_yaml = solve_common_paths('trial_indices_sensory_map')
stop_flags_yaml = solve_common_paths('stop_flags')
stop_flags_sensory_map_yaml = solve_common_paths('stop_flags_sensory_map')