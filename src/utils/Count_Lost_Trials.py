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

csv_calib_error = os.path.join(root_logs, 'calib_errors.csv')
df_calib_error = pd.read_csv(csv_calib_error, sep=';')

csv_murphy = os.path.join(root_stat, '02_categorical', 'murphy_measures.csv')
df_murphy = pd.read_csv(csv_murphy, sep=';')

df_failed_trials = None

list_identifier = sorted(df_val_trials['identifier'].tolist())

csv_failed_trials = os.path.join(root_stat, '04_failed_trials', 'failed_trials.csv')

def get_calib_error(id_p, id_t, used_cams, df_error):
    """
    Get the calibration error for the given trial
    """

    calib_error = -1

    cam_str = ', '.join(used_cams)

    df_error_reduced = df_error.loc[(df_error['id_p'] == id_p) & (df_error['cam_used'] == cam_str)]

    if len(df_error_reduced) == 0:
        print(f"Calibration Error for {id_p}_{id_t} and {cam_str} not found")
        return calib_error

    calib_error = df_error_reduced['error'].values[0]

    return calib_error

def check_if_HPE_done(used_cams, id_p, id_t, hpe_model, filtered, verbose=1):
    """
    Check if the HPE was done for the given trial
    """

    filt = '02_filtered' if filtered == 'filtered' else '01_unfiltered'

    trial_int = int(id_t.split('T')[1])

    for cam in used_cams:
        if  len(used_cams) == 1: # TODO: Debug Single Cam
            files = glob.glob(os.path.join(root_val, '02_pose_estimation', filt, id_p, f'{id_p}_{cam}',
                                           hpe_model, 'single_cam', f'{id_p}_{id_t}_*.trc'))
        else:
            files = glob.glob(os.path.join(root_val, '02_pose_estimation', filt, id_p, f'{id_p}_{cam}',
                                           hpe_model, f'trial_{trial_int}_*', f'trial_{trial_int}_*.zip'))
        if len(files) == 0:
            if verbose >= 2:
                print(f"Pose Estimation for {id_p}_{id_t} not done for {cam}")
            return 0

    return 1

def check_if_pose2sim_done(id_s, id_p, id_t, verbose=1):
    """
    Check if the Pose2Sim was done for the given trial
    """

    setting = f'setting_{id_s.split("S")[1]}'

    files = glob.glob(os.path.join(root_val, '03_data', setting, id_p, id_s, f'{id_s}_{id_p}', f'{id_s}_{id_p}_{id_t}',
                                   'pose-3d', f'{id_s}_{id_p}_{id_t}_*.trc'))

    if len(files) == 0:
        if verbose >= 2:
            print(f"Pose2Sim for {id_s}_{id_p}_{id_t} not done")
        return 0

    return 1

def check_if_opensim_done(id_s, id_p, id_t, verbose=1):
    """
    Check if the OpenSim was done for the given trial
    """

    setting = f'setting_{id_s.split("S")[1]}'

    files = glob.glob(os.path.join(root_val, '03_data', setting, id_p, id_s, f'{id_s}_{id_p}', f'{id_s}_{id_p}_{id_t}',
                                   'movement_analysis', 'kin_opensim_analyzetool',  f'{id_s}_{id_p}_{id_t}_*Vec3.sto'))

    if len(files) == 0:
        if verbose >= 2:
            print(f"OpenSim for {id_s}_{id_p}_{id_t} not done")
        reason = 'Vec3.sto not found in movement_analysis\\kin_opensim_analyzetool'
        return 0, reason

    files = glob.glob(os.path.join(root_val, '03_data', setting, id_p, id_s, f'{id_s}_{id_p}', f'{id_s}_{id_p}_{id_t}',
                                   'movement_analysis', 'ik_tool',  f'{id_s}_{id_p}_{id_t}_*.csv'))

    if len(files) < 3:
        if verbose >= 2:
            print(f"OpenSim for {id_s}_{id_p}_{id_t} not done")
        reason = 'Joint velocity .csv not found in movement_analysis\\ik_tool'
        return 0, reason

    return 1, 'all_good'

def check_if_murphy_done(id_s, id_p, id_t, df_murphy, verbose=1):
    """
    Check if the Murphy Measures were done for the given trial
    """

    ts_files = glob.glob(os.path.join(root_val, '03_data', 'preprocessed_data', '01_murphy_out', f'{id_s}_{id_p}_{id_t}_*.csv'))

    if len(ts_files) == 0:
        if verbose >= 2:
            print(f"Murphy Measures for {id_s}_{id_p}_{id_t} not done")

        reason = 'time-series for measures not found in preprocessed_data\\01_murphy_out'
        return 0, reason

    df_trial_only = df_murphy.loc[(df_murphy['id_s'] == id_s) & (df_murphy['id_p'] == id_p) & (df_murphy['id_t'] == id_t)]

    if len(df_trial_only) == 0:
        if verbose >= 2:
            print(f"Murphy Measures for {id_s}_{id_p}_{id_t} not done")
        reason = 'trial not found in murphy_measures.csv'
        return 0, reason

    return 1, 'all_good'

def check_if_murphy_has_reference_data(id_s, id_p, id_t, df_murphy, verbose=1):
    """
    Check if the Murphy Measures were done for the Opensim trial
    """

    id_s_omc = 'S15133'

    ts_reference_files = glob.glob(
        os.path.join(root_val, '03_data', 'preprocessed_data', '01_murphy_out', f'{id_s_omc}_{id_p}_{id_t}_*.csv'))

    if len(ts_reference_files) == 0:
        if verbose >= 2:
            print(f"Murphy Measures for {id_s}_{id_p}_{id_t} not done")
        reason = 'time-series for OMC not found in preprocessed_data\\01_murphy_out'
        return 0, reason

    df_omc_only = df_murphy.loc[
        (df_murphy['id_s'] == id_s_omc) & (df_murphy['id_p'] == id_p) & (df_murphy['id_t'] == id_t)]

    if len(df_omc_only) == 0:
        if verbose >= 2:
            print(f"Murphy Measures for {id_s}_{id_p}_{id_t} not done")
        reason = 'OMC trial not found in murphy_measures.csv'
        return 0, reason

    return 1, 'all_good'

def get_lost_trials():

    global df_failed_trials

    columns_out = ['id_s', 'id_p', 'id_t', 'affected', 'side', 'calib_error', 'HPE', 'P2S', 'OS', 'murphy', 'murphy_omc',
                   'failed_at', 'reason']
    df_failed_trials = pd.DataFrame(columns=columns_out)

    progbar = tqdm(total=len(list_identifier), desc='Processing')

    for identifier in list_identifier:
        progbar.set_description(f"Processing {identifier}: ")
        progbar.update(1)

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

        if len(used_cams) > 1:
            calib_error = get_calib_error(id_p, id_t, used_cams, df_calib_error)
        else:
            calib_error = -2  # -2 represents single cam

        df_temp['calib_error'] = calib_error

        HPE_success = check_if_HPE_done(used_cams, id_p, id_t, hpe_model, filtered)

        df_temp['HPE'] = HPE_success

        if not HPE_success:
            df_temp['failed_at'] = 'HPE'

            df_failed_trials = pd.concat([df_failed_trials, df_temp], ignore_index=True)
            continue

        P2S_success = check_if_pose2sim_done(id_s, id_p, id_t)

        df_temp['P2S'] = P2S_success

        if not P2S_success:
            df_temp['failed_at'] = 'P2S'

            df_failed_trials = pd.concat([df_failed_trials, df_temp], ignore_index=True)
            continue

        opensim_success, reason = check_if_opensim_done(id_s, id_p, id_t)

        df_temp['OS'] = opensim_success

        if not opensim_success:
            df_temp['failed_at'] = 'OS'
            df_temp['reason'] = reason

            df_failed_trials = pd.concat([df_failed_trials, df_temp], ignore_index=True)
            continue

        murphy_success, reason = check_if_murphy_done(id_s, id_p, id_t, df_murphy)

        df_temp['murphy'] = murphy_success

        if not murphy_success:
            df_temp['failed_at'] = 'Murphy'
            df_temp['reason'] = reason

            murphy_omc_success, _ = check_if_murphy_has_reference_data(id_s, id_p, id_t, df_murphy)
            df_temp['murphy_omc'] = murphy_omc_success

            if not murphy_omc_success:
                df_temp_reason = 'Murphy Measures for OMC and MMC not done'

            df_failed_trials = pd.concat([df_failed_trials, df_temp], ignore_index=True)
            continue

        murphy_omc_success, reason = check_if_murphy_has_reference_data(id_s, id_p, id_t, df_murphy)

        df_temp['murphy_omc'] = murphy_omc_success

        if not murphy_omc_success:
            df_temp['failed_at'] = 'Murphy OMC'
            df_temp['reason'] = reason

            df_failed_trials = pd.concat([df_failed_trials, df_temp], ignore_index=True)
            continue

        df_failed_trials = pd.concat([df_failed_trials, df_temp], ignore_index=True)



    progbar.close()

    df_failed_trials.to_csv(csv_failed_trials, sep=';', index=False)


if __name__ == '__main__':

    get_lost_trials()

    if os.path.isfile(csv_failed_trials):
        df_failed_trials = pd.read_csv(csv_failed_trials, sep=';')

    count_lost_trials = df_failed_trials.groupby(['id_p', 'id_s', 'failed_at']).size().reset_index(name='count')



















