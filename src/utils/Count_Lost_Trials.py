"""
We count the amount of trials in the validation trials csv and then look for each stage how many Trials were lost after
Pose2Sim and Opensim

Then we count the Trials in the outlier csv and finally report the number for each Participant and Setting (SXXX_PXX)
for analysis.

"""
import os
import sys
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from iDrink import iDrinkUtilities

def check_if_HPE_done(used_cams, id_p, id_t, hpe_model, filtered, verbose=1):
    """
    Check if the HPE was done for the given trial
    """

    filt = '02_filtered' if filtered == 'filtered' else '01_unfiltered'

    trial_int = int(id_t.split('T')[1])

    for cam in used_cams:
        if  len(used_cams) == 1:
            files = glob.glob(os.path.join(root_val, '02_pose_estimation', filt, id_p, f'{id_p}_{cam}',
                                           hpe_model, 'single_cam', f'{id_p}_{id_t}_*.trc'))
        else:
            files = glob.glob(os.path.join(root_val, '02_pose_estimation', filt, id_p, f'{id_p}_{cam}',
                                           hpe_model, f'trial_{trial_int}_*', f'trial_{trial_int}_*.zip'))
        if len(files) == 0:
            if verbose >= 2:
                print(f"Pose Estimation for {id_p}_{id_t} not done for {cam}")
            return False

    return True






if __name__ == '__main__':

    drive = iDrinkUtilities.get_drivepath()

    root_iDrink = os.path.join(drive, 'iDrink')
    root_val = os.path.join(root_iDrink, "validation_root")
    root_stat = os.path.join(root_val, '04_Statistics')
    root_omc = os.path.join(root_val, '03_data', 'OMC_new', 'S15133')
    root_data = os.path.join(root_val, "03_data")
    root_logs = os.path.join(root_val, "05_logs")

    csv_val_trials = os.path.join(root_logs, 'validation_trials.csv')
    df_val_trials = pd.read_csv(csv_val_trials, sep=';')

    csv_settings = os.path.join(root_logs, 'validation_settings.csv')
    df_settings = pd.read_csv(csv_settings, sep=';')

    columns_out = ['id_s', 'id_p', 'id_t', 'affected', 'side', 'calib_error', 'HPE', 'P2S', 'OS', 'failed_at', 'reason']
    df_failed_trials = pd.DataFrame(columns=columns_out)

    list_identifier = sorted(df_val_trials['identifier'].tolist())

    for identifier in list_identifier:
        df_temp = pd.DataFrame(columns=columns_out, index=[0])

        id_s = df_val_trials.loc[df_val_trials['identifier'] == identifier, 'id_s'].values[0]
        id_p = df_val_trials.loc[df_val_trials['identifier'] == identifier, 'id_p'].values[0]
        id_t = df_val_trials.loc[df_val_trials['identifier'] == identifier, 'id_t'].values[0]

        df_temp['id_s'] = id_s
        df_temp['id_p'] = id_p
        df_temp['id_t'] = id_t
        df_temp ['affected'] = df_val_trials.loc[df_val_trials['identifier'] == identifier, 'affected'].values[0]
        df_temp['side'] = df_val_trials.loc[df_val_trials['identifier'] == identifier, 'measured_side'].values[0]

        used_cams = [f'cam{cam}' for cam in ast.literal_eval(df_val_trials.loc[df_val_trials['identifier'] == identifier, 'used_cams'].values[0])]
        pass

        hpe_model = df_settings.loc[df_settings['setting_id'] == int(id_s.split('S')[1]), 'pose_estimation'].values[0]
        filtered = df_settings.loc[df_settings['setting_id'] == int(id_s.split('S')[1]), 'filtered_2d_keypoints'].values[0]

        HPE_success = check_if_HPE_done(used_cams, id_p, id_t, hpe_model, filtered)

        df_temp['HPE'] = HPE_success

        if not HPE_success:
            df_temp['failed_at'] = 'HPE'
            continue




