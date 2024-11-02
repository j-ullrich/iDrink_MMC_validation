"""
We compare each MMC-Trial Objects time series and Murphy Measures to the same metrics of the OMC-trials.

"""
import os
import sys
import shutil
import glob
import re
from tqdm import tqdm
from fuzzywuzzy import process

from trc import TRCData

import math
import pandas as pd
import numpy as np
import scipy as sp

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from iDrinkOpenSim import read_opensim_file
import iDrinkValPlots as iDrinkVP

murphy_measures = ["PeakVelocity_mms",
                   "elbowVelocity",
                   "tTopeakV_s",
                   "tToFirstpeakV_s",
                   "tTopeakV_rel",
                   "tToFirstpeakV_rel",
                   "NumberMovementUnits",
                   "InterjointCoordination",
                   "trunkDisplacementMM",
                   "trunkDisplacementDEG",
                   "ShoulderFlexionReaching",
                   "ElbowExtension",
                   "shoulderAbduction"]

def resample_dataframes(df1, df2, max_time=None):
    """
    Resamples two DataFrames to the same amount of rows.

    """
    # TODO: Rework this function when Data of OMC and MMC of the same trial are available.
    # Resample DataFrames to the same amount of frames
    if max_time is None:
        # Get the max time that is in both DataFrames. Last timestamp we can compare
        max_time = math.ceil(min(df1['time'].iloc[-1], df2['time'].iloc[-1]))

        # Cut DataFrames so that none exceeds the max_time, the shorter DF will be lower recording frequency
        df1 = df1[df1['time'] <= max_time]
        df2 = df2[df2['time'] <= max_time]

    if df1.shape[0] > df2.shape[0]:
        # make sure, df1 is always the shorter DataFrame when working on it. --> lower frequency
        df2, df1 = resample_dataframes(df2, df1, max_time)

    idx = []
    for i in df1["time"]:
        # Look for id with closest timestamp in df2 and add id to list.
        idx.append(df2["time"].sub(i).abs().idxmin())

    # Use Ids from list to create the new DataFrame
    df2 = df2.iloc[list(map(int, idx))].reset_index(drop=True)

    return df1, df2

def run_stat_murphy(df, id_s, root_stat_cat, verbose=1):
    """
    Calculates statistical measures for the given DataFrame.

    It creates a subfolder in the root_stat folder with the id_s as name.
    In this folder are .csv files containing the data for each trial in the setting.

    One .csv File contains the average difference between MMC and OMC for all trials.

    :param df:
    :param id_s:
    :param root_stat_cat:
    :param verbose:
    :return:
    """
    global murphy_measures

    csv_out = os.path.join(root_stat_cat, id_s, f'{id_s}_stat.csv')

    df_s = pd.DataFrame(columns=['identifier', 'id_s', 'id_p', 'id_t'] + murphy_measures)

    id_s_omc = 'S15133'
    idx_p = df['id_p'].unique()

    for id_p in idx_p:
        idx_t = sorted(list(df[(df['id_p']==id_p) & (df['id_s']==id_s )]['id_t'].unique()))

        for id_t in idx_t:
            identifier = f"{id_s}_{id_p}_{id_t}"

            path_trial_stat_csv = os.path.join(root_stat_cat, id_s, f'{identifier}_stat.csv')

            row_mmc = df.loc[df['identifier'] == identifier, murphy_measures].values[0]
            row_omc = df.loc[(df['id_s'] == id_s_omc) & (df['id_p'] == id_p) & (df['id_t'] == id_t), murphy_measures].values[0]

def get_mmc_omc_difference(df, root_stat_cat, thresh_PeakVelocity_mms=3000, verbose=1):
    """
    Creates DataFrame containing the difference between OMC and MMC measurments for each trial.

    :param df:
    :param verbose:
    :return:
    """
    global murphy_measures

    df_diff = pd.DataFrame(columns=['identifier', 'id_s', 'id_p', 'id_t'] + murphy_measures)

    id_s_omc = 'S15133'
    idx_s = sorted(df['id_s'].unique())
    for id_s in idx_s:
        idx_p = sorted(list(df[df['id_s'] == id_s]['id_p'].unique()))
        for id_p in idx_p:
            idx_t = sorted(list(df[(df['id_p'] == id_p) & (df['id_s'] == id_s)]['id_t'].unique()))

            for id_t in idx_t:
                identifier = f"{id_s}_{id_p}_{id_t}"

                path_trial_stat_csv = os.path.join(root_stat_cat, id_s, f'{identifier}_stat.csv')
                got_mmc = False
                try:
                    row_mmc = df.loc[df['identifier'] == identifier, murphy_measures].values[0]
                    got_mmc = True
                    row_omc = df.loc[(df['id_s'] == id_s_omc) & (df['id_p'] == id_p) & (df['id_t'] == id_t), murphy_measures].values[0]
                except IndexError:
                    if verbose >= 1:
                        if got_mmc:
                            print(f"Error in {os.path.basename(__file__)}.{get_mmc_omc_difference.__name__}\n"
                                  f"Trial: {id_s_omc}_{id_p}_{id_t} not found in OMC")
                        else:
                            print(f"Error in {os.path.basename(__file__)}.{get_mmc_omc_difference.__name__}\n"
                                  f"Trial: {id_s}_{id_p}_{id_t} not found in MMC")
                    continue

                # Check if PeakVelocity_mms is beyond threshold
                if thresh_PeakVelocity_mms is not None:
                    if row_omc[murphy_measures.index("PeakVelocity_mms")] >= thresh_PeakVelocity_mms:
                        if verbose >= 2:
                            print(f"Error in {os.path.basename(__file__)}.{get_mmc_omc_difference.__name__}\n"
                                  f"PeakVelocity_mms beyond threshold for trial {id_s_omc}_{id_p}_{id_t}\n"
                                  f"Value: {row_omc[murphy_measures.index('PeakVelocity_mms')]}\n"
                                  f"Threshold: {thresh_PeakVelocity_mms}")
                        continue
                    elif row_mmc[murphy_measures.index("PeakVelocity_mms")] >= thresh_PeakVelocity_mms:
                        if verbose >= 2:
                            print(f"Error in {os.path.basename(__file__)}.{get_mmc_omc_difference.__name__}\n"
                                  f"PeakVelocity_mms beyond threshold for trial {id_s}_{id_p}_{id_t}\n"
                                  f"Value: {row_mmc[murphy_measures.index('PeakVelocity_mms')]}\n"
                                  f"Threshold: {thresh_PeakVelocity_mms}")
                        continue

                diff = row_mmc - row_omc

                row_diff = [identifier, id_s, id_p, id_t] + list(diff)

                df_diff.loc[df_diff.shape[0]] = row_diff

    return df_diff

def get_datlists(df_murphy, measure, id_s, id_p=None):
    """
    Returns two lists containing the data of the OMC and MMC of the given measure.

    if id_p is None, the function returns the data of all participants for the setting and measure

    The data are sorted by the id_t in ascending order. If an id_t has to exist for omc and mmc to be added to the datalists.

    :param df_murphy:
    :param measure:
    :param id_s:
    :param id_p:
    :return:
    """



    if id_p is None:
        idx_p = sorted(list(df_murphy[df_murphy['id_s'] == id_s]['id_p'].unique()))
        idx_p = [id_p for id_p in idx_p if id_p in df_murphy[(df_murphy['id_s'] == 'S15133')]['id_p'].values]
    else:
        idx_p = [id_p]

    dat_ref = []
    dat_meas = []
    for id_p in idx_p:
        idx_t = sorted(list(df_murphy[(df_murphy['id_p'] == id_p) & (df_murphy['id_s'] == id_s)]['id_t'].unique()))
        idx_t = [id_t for id_t in idx_t if id_t in df_murphy[(df_murphy['id_s'] == 'S15133') &
                                                             (df_murphy['id_p'] == id_p)]['id_t'].values]
        for id_t in idx_t:
            dat_ref.append(df_murphy[(df_murphy['id_s'] == 'S15133') & (df_murphy['id_p'] == id_p) & (df_murphy['id_t'] == id_t)][measure].values[0])
            dat_meas.append(df_murphy[(df_murphy['id_s'] == id_s) & (df_murphy['id_p'] == id_p) & (df_murphy['id_t'] == id_t)][measure].values[0])

    return np.array(dat_ref), np.array(dat_meas)

def save_plots_murphy(df_murphy, root_stat_cat, filetype = '.png', verbose=1):
    """
    Creates plots for the Murphy Measures of the MMC and OMC and saves them in the Statistics Folder.

    :param df_murphy:
    :param root_stat_cat:
    :param verbose:
    :return:
    """

    global murphy_measures

    idx_s = df_murphy['id_s'].unique()
    idx_s_mmc = np.delete(idx_s, np.where(idx_s == 'S15133'))

    root_plots = os.path.join(root_stat_cat, 'plots')

    if not os.path.exists(root_plots):
        os.makedirs(root_plots)

    if verbose >= 1:
        progress = tqdm(total=sum(len(sublist) for sublist in list(df_murphy[df_murphy['id_s'] == id_s]['id_p'].unique() for id_s in idx_s)), desc="Creating Plots")
    for id_s in idx_s_mmc:
        idx_p = sorted(list(df_murphy[df_murphy['id_s'] == id_s]['id_p'].unique()))
        fullsettingplotted = False

        for id_p in idx_p:
            if verbose >= 1:
                progress.set_description(f"Creating Plots for {id_s}_{id_p}")
            idx_t = sorted(list(df_murphy[(df_murphy['id_p'] == id_p) & (df_murphy['id_s'] == id_s)]['id_t'].unique()))

            for measure in murphy_measures:
                filename = os.path.join(root_plots,  f'bland_altman_{id_s}_{id_p}_{measure}')
                iDrinkVP.plot_blandaltman(df_murphy, measure, id_s, id_p, filename=filename,
                                          filetype=filetype, show_id_t=False, verbose=verbose, show_plots=False)

                filename = os.path.join(root_plots, f'residuals_vs_mmc_{id_s}_{id_p}_{measure}')
                iDrinkVP.plot_blandaltman(df_murphy, measure, id_s, id_p,
                                          plot_to_val=True, filename=filename, show_id_t=False, verbose=verbose,
                                          filetype=filetype, show_plots=False)

                if not fullsettingplotted:
                    filename = os.path.join(root_plots,  f'bland_altman_all_{id_s}_{measure}')
                    iDrinkVP.plot_blandaltman(df_murphy, measure, id_s, filename=filename,
                                              filetype=filetype, show_id_t=False, verbose=verbose, show_plots=False)

                    filename = os.path.join(root_plots, f'residuals_vs_mmc_all_{id_s}_{measure}')
                    iDrinkVP.plot_blandaltman(df_murphy, measure, id_s, filename=filename,
                                              filetype=filetype, plot_to_val=True, show_id_t=False, verbose=verbose,
                                              show_plots=False)

            fullsettingplotted = True

            if verbose >= 1:
                progress.update(1)

    if verbose >= 1:
        progress.close()



    pass

def runs_statistics_discrete(path_csv_murphy, root_stat, thresh_PeakVelocity_mms = None, thresh_elbowVelocity=None, verbose=1):
    """
    Takes Murphy Measures of MMC and OMC and compares them. Then plots the results and saves data and plots in the Statistics Folder.
    :param df_mmc:
    :param df_omc:
    :return:
    """
    global murphy_measures

    df_murphy = pd.read_csv(path_csv_murphy, sep=';')
    root_stat_cat = os.path.join(root_stat, '02_categorical')

    idx_s = df_murphy['id_s'].unique()
    idx_s_mmc = np.delete(idx_s, np.where(idx_s == 'S15133'))

    # delete rows in df_murphy if PeakVelocity_mms is beyond threshold and if
    if thresh_PeakVelocity_mms is not None:
        df_murphy = df_murphy[df_murphy['PeakVelocity_mms'] < thresh_PeakVelocity_mms]
    if thresh_elbowVelocity is not None:
        df_murphy = df_murphy[df_murphy['elbowVelocity'] < thresh_elbowVelocity]


    # Create subset of DataFrame containing all trials that are also in OMC
    df = pd.DataFrame(columns=df_murphy.columns)
    for id_s in idx_s_mmc:

        df_s = df_murphy[df_murphy['id_s'] == id_s]
        df_omc = df_murphy[df_murphy['id_s'] == 'S15133']
        idx_p = sorted(list(df_murphy[df_murphy['id_s']==id_s]['id_p'].unique()))

        for id_p in idx_p:
            idx_t = sorted(list(df_murphy[(df_murphy['id_p']==id_p) & (df_murphy['id_s']==id_s )]['id_t'].unique()))
            for id_t in idx_t:
                df = pd.concat([df, df_omc[(df_omc['id_p'] == id_p) & (df_omc['id_t'] == id_t)]])

        df = pd.concat([df_s, df])
    df_diff = get_mmc_omc_difference(df, root_stat_cat, thresh_PeakVelocity_mms=thresh_PeakVelocity_mms, verbose=verbose)

    # Create DataFrame containing absolute values of the differences
    df_abs_diff = df_diff.copy()
    df_abs_diff.iloc[:, 4:] = np.abs(df_abs_diff.iloc[:, 4:])

    # Create DataFrame and calculate mean of each column
    df_mean = pd.DataFrame(columns=['id_s', 'id_p'] + murphy_measures)
    df_rmse = pd.DataFrame(columns=['id_s', 'id_p'] + murphy_measures)
    for id_s in idx_s_mmc:
        idx_p = sorted(list(df_murphy[df_murphy['id_s'] == id_s]['id_p'].unique()))

        for id_p in idx_p:
            # mean Error
            df_mean.loc[len(df_mean), 'id_s'] = id_s
            df_mean.loc[len(df_mean)-1, 'id_p'] = id_p
            df_mean.iloc[len(df_mean)-1, 2:] = np.mean(df_diff.loc[(df_diff['id_s'] == id_s) & (df_diff['id_p'] == id_p), df_diff.columns[4:]], axis=0)


            # Root Mean Squared Error
            df_rmse.loc[len(df_rmse), 'id_s'] = id_s
            df_rmse.loc[len(df_rmse) - 1, 'id_p'] = id_p
            df_rmse.iloc[len(df_rmse) - 1, 2:] = np.sqrt(np.mean(df_diff.loc[(df_diff['id_s'] == id_s) & (df_diff['id_p'] == id_p), df_diff.columns[4:]]**2, axis=0))

        # mean for setting over all participants
        df_mean.loc[len(df_mean), 'id_s'] = id_s
        df_mean.loc[len(df_mean) - 1, 'id_p'] = ''
        df_mean.iloc[len(df_mean) - 1, 2:] = np.mean(df_diff.loc[df_diff['id_s'] == id_s, df_diff.columns[4:]], axis=0)

        df_rmse.loc[len(df_rmse), 'id_s'] = id_s
        df_rmse.loc[len(df_rmse) - 1, 'id_p'] = ''
        df_rmse.iloc[len(df_rmse) - 1, 2:] = np.sqrt(np.mean(df_diff.loc[df_diff['id_s'] == id_s, df_diff.columns[4:]]**2, axis=0))

    # Write to  csv
    path_csv_murphy_diff = os.path.join(root_stat_cat, f'stat_murphy_diff.csv')
    path_csv_murphy_abs_diff = os.path.join(root_stat_cat, f'stat_murphy_abs_diff.csv')
    path_csv_murphy_mean = os.path.join(root_stat_cat, f'stat_murphy_mean.csv')
    path_csv_murphy_rmse = os.path.join(root_stat_cat, f'stat_murphy_rmse.csv')

    df_diff.to_csv(path_csv_murphy_diff, sep=';')
    df_abs_diff.to_csv(path_csv_murphy_abs_diff, sep=';')
    df_mean.to_csv(path_csv_murphy_mean, sep=';')
    df_rmse.to_csv(path_csv_murphy_rmse, sep=';')

    save_plots_murphy(df_murphy, root_stat_cat, filetype=['.html', '.png'], verbose=verbose)

    # Create DataFrame for each trial
    run_stat_murphy(df, id_s, root_stat_cat, verbose=verbose)



    pass
    """constructed_identifier = f'S15133_{trial.id_p}_{trial.id_t}'
    if constructed_identifier not in df_murphy['identifier'].values:
        raise ValueError(f"Error in {os.path.basename(__file__)}.{runs_statistics_discrete.__name__}\n"
                         f"Identifier {constructed_identifier} not found in DataFrame.")

    # Access rows
    df_mmc = df_murphy[df_murphy['id_s'] == trial.id_s]
    df_omc = df_murphy[df_murphy['identifier'] == constructed_identifier]"""



def get_paired_rom(df_stat, roms):
    """
    Calculates the paired Range of Motion Difference and adds it to the DataFrame

    :param df_stat:
    :return: DataFrame
    """

    df_paired_rom_diff = pd.DataFrame(columns=['hip_flexion', 'hip_adduction', 'hip_rotation',
                                               'arm_flex', 'arm_add', 'arm_rot', 'elbow_flex',
                                               'pro_sup', 'wrist_flex', 'wrist_dev'],
                                      index=['paired_rom_diff'])

    for column in df_paired_rom_diff.columns:

        left = df_stat.loc['rom_error', f'{column}_l']
        right = df_stat.loc['rom_error', f'{column}_r']

        paired_error = np.sqrt((left-right) ** 2)

        df_paired_rom_diff[column] = paired_error



    return df_paired_rom_diff

def omc_trial_present(dirs_trial, dirs_omc, verbose = 1):
    """
    Iterates over list of directories and checks if the directory contains a valid trial.

    If .sto files generated by Opensim are present, the trial is considered valid.

    Returns list of directories of valid trials.
    :param dirs_trial:
    :return:
    """
    valid_dirs_trial = []
    valid_dirs_omc = []

    for dir_trial, dir_omc in zip(dirs_trial, dirs_omc):
        dir_analyzetool = os.path.join(dir_trial, 'movement_analysis', 'kin_opensim_analyzetool')
        dir_analyzetool_omc = os.path.join(dir_omc, 'movement_analysis', 'kin_opensim_analyzetool')

        # get list of .sto files in directory
        sto_files = glob.glob(os.path.join(dir_analyzetool, '*.sto'))

        sto_files_omc = glob.glob(os.path.join(dir_analyzetool_omc, '*.sto'))

        # No. of files needs to be 4
        # If more than 4, old files left in folder --> no influence on validity.


        #Check that patient and trial id are the same
        p_id = os.path.basename(dir_trial).split('_')[1]
        t_id = os.path.basename(dir_trial).split('_')[2]

        p_id_omc = os.path.basename(dir_omc).split('_')[1]
        t_id_omc = os.path.basename(dir_omc).split('_')[2]

        if p_id != p_id_omc or t_id != t_id_omc:
            if verbose >= 1:
                print(f"Error in {os.path.basename(__file__)}.{omc_trial_present.__name__}\n"
                      f"Patient or Trial ID do not match for trial {dir_trial} and {dir_omc}")
            continue


        if len(sto_files) >= 4 and len(sto_files_omc) >= 4:
            valid_dirs_trial.append(dir_trial)
            valid_dirs_omc.append(dir_omc)

    return valid_dirs_trial, valid_dirs_omc


def get_omc_dirs(dirs_trial, root_omc):
    """
    iterates over list of trial directories and returns a list of corresponding omc_trial_directories.

    dirs_trial[i] <==> dirs_omc[i]


    :param dirs_trial:
    :return dirs_omc:
    """

    dirs_omc = []
    id_s = 'S15133'

    for dir_trial in dirs_trial:
        pass

        id_p = os.path.basename(dir_trial).split('_')[1]
        id_t = os.path.basename(dir_trial).split('_')[2]

        dirs_omc.append(os.path.join(root_omc, f'{id_s}_{id_p}', f'{id_s}_{id_p}_{id_t}'))

    return dirs_omc


def calc_mean_stat_p(dir_csv_p, dir_csv_t, id_s, id_p, verbose=1):
    """
    Iterate over .csv files containing statistical measures for each trial and calculate mean for each patient.

    Add standard deviation for each measure


    :param dir_csv_p:
    :param dir_csv_t:
    :param id_s:
    :param id_p:
    :param verbose:
    :return:
    """
    pass

    list_csv_t = glob.glob(os.path.join(dir_csv_t, f'{id_s}_{id_p}_*'))
    df_sum = None
    all_dfs = []

    for csv_t in list_csv_t:
        df_t = pd.read_csv(csv_t, sep=';', index_col=0)

        if df_sum is None:
            df_sum = df_t
        else:
            df_sum = df_sum + df_t

        all_dfs.append(df_t)

    df_mean = df_sum / len(list_csv_t)
    df_std = pd.concat(all_dfs).groupby(level=0).std()

    path_csv_mean = os.path.join(dir_csv_p, f'{id_s}_{id_p}_mean.csv')
    df_mean.to_csv(path_csv_mean, sep=';')

    path_csv_std = os.path.join(dir_csv_p, f'{id_s}_{id_p}_std.csv')
    df_std.to_csv(path_csv_std, sep=';')





def calc_mean_stat_s(dir_csv_s, dir_csv_t, id_s, verbose=1):
    """
    Iterate over .csv files containing statistical measures for each trial and calculate mean for each setting.

    Add standard deviation for each measure


    :param dir_csv_s:
    :param dir_csv_p:
    :param id_s:
    :param verbose:
    :return:
    """
    pass

    list_csv_t = glob.glob(os.path.join(dir_csv_t, f'{id_s}_*'))
    df_sum = None
    all_dfs = []

    for csv_t in list_csv_t:
        df_t = pd.read_csv(csv_t, sep=';', index_col=0)

        if df_sum is None:
            df_sum = df_t
        else:
            df_sum = df_sum + df_t

        all_dfs.append(df_t)

    df_mean = df_sum / len(list_csv_t)
    df_std = pd.concat(all_dfs).groupby(level=0).std()

    path_csv_mean = os.path.join(dir_csv_s, f'{id_s}_mean.csv')
    df_mean.to_csv(path_csv_mean, sep=';')

    path_csv_std = os.path.join(dir_csv_s, f'{id_s}_std.csv')
    df_std.to_csv(path_csv_std, sep=';')

def run_statistics_time_series(root_data, root_omc, dir_stat_ts, df_settings, path_csv_murphy_timestamps,
                               joints_of_interest, body_parts_of_interest, body_parts_of_interest_no_axes, verbose=1):
    """
    Calculate Statistics for time-series Data.

    Iterates over setting, patient, and trial folders, then calculates the statistical values for each trial.

    .csv files are written for each Participant and Setting.

    patient_csv: average for each participant within each Setting. no. files gernerated #P * #S

    setting_csv: average over all trials for each setting. no. files generated #S

    :param root_data:
    :param dir_stat_ts:
    :param df_settings:
    :param joints_of_interest:
    :param body_parts_of_interest:
    :param verbose:
    :return:
    """
    if joints_of_interest is None: # Default Joints of Interest
        joints_of_interest = ['arm_flex_r','arm_add_r','arm_rot_r',
                              'elbow_flex_r','pro_sup_r',
                              'arm_flex_l','arm_add_l','arm_rot_l',
                              'elbow_flex_l','pro_sup_l',]

    if body_parts_of_interest is None: # Default Body Parts of Interest
        body_parts_of_interest = ['hand_r_x','hand_r_y','hand_r_z', 'hand_l_x','hand_l_y','hand_l_z']

    if body_parts_of_interest_no_axes is None: # Default Body Parts of Interest without axes
        body_parts_of_interest_no_axes = ['hand_r', 'hand_l']

    df_timestamps = pd.read_csv(path_csv_murphy_timestamps, sep=';')

    dirs_setting = glob.glob(os.path.join(root_data, 'setting*'))

    dir_csv_t = os.path.join(dir_stat_ts, '01_results', '01_trial')
    dir_csv_p = os.path.join(dir_stat_ts, '01_results', '02_patient')
    dir_csv_s = os.path.join(dir_stat_ts, '01_results', '03_setting')

    for dir in [dir_csv_t, dir_csv_p, dir_csv_s]:
        os.makedirs(dir, exist_ok=True)

    idx_s = ['S' + os.path.basename(d).split('_')[1] for d in dirs_setting]

    for i, id_s in enumerate(idx_s):
        dirs_p = glob.glob(os.path.join(dirs_setting[i], 'P*'))
        idx_p = [os.path.basename(d) for d in dirs_p]

        for j , id_p in enumerate(idx_p):

            dirs_trial_mmc = glob.glob(os.path.join(dirs_p[j], id_s, f'{id_s}_{id_p}', f'{id_s}_{id_p}_T*'))
            dirs_omc = get_omc_dirs(dirs_trial_mmc, root_omc)

            dirs_trial_mmc, dirs_omc = omc_trial_present(dirs_trial_mmc, dirs_omc)
            if verbose >=1:
                progress = tqdm(total=len(dirs_trial_mmc), desc=f"Calculating Statistics for {id_s}_{id_p}")
            for dir_trial_mmc, dir_omc in zip(dirs_trial_mmc, dirs_omc):
                id_t = os.path.basename(dir_trial_mmc).split('_')[-1]
                valid = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['valid'].values[0]

                if valid == 0:
                    continue

                side = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['side'].values[0].lower()
                condition = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['condition'].values[0]

                if verbose >= 1:
                    progress.set_description(f"Calculating Statistics for {os.path.basename(dir_trial_mmc)}")

                # Load Data
                file_bodyparts_mmc = glob.glob(os.path.join(dir_trial_mmc, 'movement_analysis', 'kin_opensim_analyzetool', '*vel_global.sto'))[0]
                df_mmc_body_vel = get_dataframes(file_bodyparts_mmc)
                files_joints_mmc = glob.glob(os.path.join(dir_trial_mmc, 'movement_analysis', 'ik_tool', '*.csv'))
                df_mmc_joint_pos, df_mmc_joint_vel, _ = get_dataframes(files_joints_mmc)

                file_bodyparts_omc = glob.glob(os.path.join(dir_omc, 'movement_analysis', 'kin_opensim_analyzetool', '*vel_global.sto'))[0]
                df_omc_body_vel = get_dataframes(file_bodyparts_omc)
                files_joints_omc = glob.glob(os.path.join(dir_omc, 'movement_analysis', 'ik_tool', '*.csv'))
                df_omc_joint_pos, df_omc_joint_vel, _ = get_dataframes(files_joints_omc)

                if any([df_mmc_body_vel is None,
                        df_mmc_joint_pos is None, df_mmc_joint_vel is None,
                        df_omc_body_vel is None,
                        df_omc_joint_pos is None, df_omc_joint_vel is None]):
                    if verbose >= 1:
                        print(f"Error in {os.path.basename(__file__)}.{run_statistics_time_series.__name__}\n"
                              f"DataFrames could not be loaded for {os.path.basename(dir_trial_mmc)}"
                              f"df_mmc_body_vel: \t{df_mmc_body_vel is None}\n"
                              f"df_mmc_joint_pos: \t{df_mmc_joint_pos is None}\n"
                              f"df_mmc_joint_vel: \t{df_mmc_joint_vel is None}\n"
                              f"df_omc_body_vel: \t{df_omc_body_vel is None}\n"
                              f"df_omc_joint_pos: \t{df_omc_joint_pos is None}\n"
                              f"df_omc_joint_vel: \t{df_omc_joint_vel is None}\n")
                    continue
                pass

                df_stat_body_vel = compare_timeseries_for_trial(df_mmc_body_vel, df_omc_body_vel,
                                                                body_parts_of_interest, isjoint=True,
                                                                 id_s=id_s, id_p=id_p, verbose=verbose)

                df_stat_joint_pos = compare_timeseries_for_trial(df_mmc_joint_pos, df_omc_joint_pos,
                                                                 joints_of_interest, isjoint=True,
                                                                  verbose=verbose)
                df_stat_joint_vel = compare_timeseries_for_trial(df_mmc_joint_vel, df_omc_joint_vel,
                                                                 joints_of_interest, isjoint=True,
                                                                 id_s=id_s, id_p=id_p, verbose=verbose)

                pass
                # Write to csv

                path_csv_stat_body_vel = os.path.join(dir_csv_t, f'{id_s}_{id_p}_{os.path.basename(dir_trial_mmc)}_vel.csv')
                df_stat_body_vel.to_csv(path_csv_stat_body_vel, sep=';')

                path_csv_stat_joint_pos = os.path.join(dir_csv_t, f'{id_s}_{id_p}_{os.path.basename(dir_trial_mmc)}_pos.csv')
                path_csv_stat_joint_vel = os.path.join(dir_csv_t, f'{id_s}_{id_p}_{os.path.basename(dir_trial_mmc)}_vel.csv')
                df_stat_joint_pos.to_csv(path_csv_stat_joint_pos, sep=';')
                df_stat_joint_vel.to_csv(path_csv_stat_joint_vel, sep=';')

                if verbose >= 1:
                    progress.update(1)

            pass
            if verbose >= 1:
                progress.close()

            # Calculate mean for each participant
            calc_mean_stat_p(dir_csv_p, dir_csv_t, id_s, id_p, verbose=verbose)

        calc_mean_stat_s(dir_csv_s, dir_csv_p, id_s, verbose=verbose)







    pass

def compare_timeseries_for_trial(df_mmc, df_omc, to_compare, isjoint, id_s=None, id_p=None, verbose=1):
    """
    Compares two Dataframes containing positional data, velocity or acceleration data.

    This function works with endeffector Pos/Vel/acc and Joint Pos/Vel/acc.

    it calculates:

    - Standard Deviation
    - Root Mean Squared Error
    - Pearson Correlation Coefficient
    - Pearson Correlation P-Value
    - Range of Motion Error (Only for joint data)

    - Min Error
    - Max Error
    - Median Error
    - Mean Error


    It then plots the

    :param df_mmc:
    :param df_mmc_vel:
    :param df_omc:
    :param df_omc_vel:
    :param isjoint:
    :return:
    """

    # Make sure DataFrames contain same amount of frames. (Rows)

    if df_mmc.shape[0] != df_omc.shape[0]:
        if verbose>=2:
            print("DataFrames for Position do not contain the same amount of frames.\n"
                  "Resampling DataFrames to the same amount of frames.")

        y_label = 'Joint Velocity [deg/s]' if isjoint else 'Endeffector Velocity [mm/s]'
        plot_condition = id_s is not None and id_p is not None
        if plot_condition:
            for y in to_compare:
                # Use longer time for plot
                time = df_mmc['time'] if df_mmc.shape[0] > df_omc.shape[0] else df_omc['time']
                plot_dir = os.path.join(root_stat, '01_continuous', '02_plots')

                iDrinkVP.plot_two_timeseries(df_omc[['time', y]], df_mmc[['time', y]], xaxis_label='Time (s)', yaxis_label = y_label,
                                             title = f'{y} of {id_s}_{id_p} before resampling',
                                             path_dst=os.path.join(plot_dir, f'{id_s}_{id_p}_{y}_before_resampling.png'))

        df_mmc, df_omc = resample_dataframes(df_mmc, df_omc)

        if plot_condition:
            for y in to_compare:
                # Use longer time for plot
                time = df_mmc['time'] if df_mmc.shape[0] > df_omc.shape[0] else df_omc['time']

                iDrinkVP.plot_two_timeseries(df_omc[['time', y]], df_mmc[['time', y]], xaxis_label='Time (s)', yaxis_label = y_label,
                                             title=f'{y} of {id_s}_{id_p} after resampling',
                                             path_dst=os.path.join(plot_dir,
                                                                   f'{id_s}_{id_p}_{y}_after_resampling.png'))



    time_mmc = df_mmc['time']
    time_omc = df_omc['time']

    df_stat_cont = pd.DataFrame(columns=to_compare, index=['std', 'rmse', 'pearson', 'pearson_pval', 'rom_error', 'min_error', 'max_error', 'median_error', 'mean_error'])
    roms = pd.DataFrame(columns = to_compare, index = ['mmc', 'omc'])

    for column in to_compare:
        if column in df_omc.columns:
            pass
        else:
            raise ValueError(f"Data incompatible:\n"
                             f"Column {column} not in df_omc_pos")
        arr_mmc = df_mmc[column].to_numpy()
        arr_omc = df_omc[column].to_numpy()
        if isjoint:
            roms.loc['mmc', column] = np.sum(np.abs(np.diff(arr_mmc)))
            roms.loc['omc', column] = np.sum(np.abs(np.diff(arr_omc)))

        # Calculate Standard Deviation
        std = np.std(arr_mmc - arr_omc)

        # Calculate Root Mean Squared Error
        rmse = np.sqrt(np.mean((arr_mmc - arr_omc)**2))

        # Calculate Coefficient of Multiple Correlation
        correlation = sp.stats.pearsonr(arr_mmc, arr_omc)
        pearson = correlation[0]
        pearson_pval = correlation[1]

        if isjoint:
            # Calculate Range of Motion Error
            rom_error = roms.loc['omc', column] - roms.loc['mmc', column]
        else:
            rom_error = None

        # Calculate Min Error
        min_error = np.min(np.abs(arr_mmc - arr_omc))

        # Calculate Max Error
        max_error = np.max(np.abs(arr_mmc - arr_omc))

        # Calculate Median Error
        median_error = np.median(np.abs(arr_mmc - arr_omc))

        # Calculate Mean Error
        mean_error = np.mean(np.abs(arr_mmc - arr_omc))

        df_stat_cont[column] = [std, rmse, pearson, pearson_pval, rom_error, min_error, max_error, median_error, mean_error]

    """shapiro_rom = sp.stats.shapiro(df_stat_cont.loc['rom_error'].values)
    df_paired_rom_diff = get_paired_rom(df_stat_cont, roms)"""

    #sp.stats.tukey_hsd(df_stat_cont.loc['rom_error'].values)

    df_stat_cont = pd.concat([df_stat_cont, roms])
    return df_stat_cont


def run_comparison_trial_OMC(trial_list, root_stat, root_omc, joints_of_interest=None, body_parts_of_interest=None,
                             body_parts_of_interest_no_axes=None):
    """
    Iterates over trials, creates DataFrames containing the statistical values for each trial.

    Loads files of OMC recording and MMC recording and runs statistical analysis on them.

    For each Trial a DataFrame & .csv file is created for position, velocity and acceleration of joint and end effector values.

    The csv files are saved in the root_stat folder.

    :param trial_list:
    :param list_df_stat_cont:
    :return:
    """

    def reduce_axes(data):
        """
        This function gets absolute velocity based on velocity in 3 Dimensions
        """

        return np.sqrt(np.sum(np.array([axis ** 2 for axis in data]), axis=0))

    if joints_of_interest is None: # Default Joints of Interest
        joints_of_interest = ['arm_flex_r','arm_add_r','arm_rot_r',
                              'elbow_flex_r','pro_sup_r',
                              'arm_flex_l','arm_add_l','arm_rot_l',
                              'elbow_flex_l','pro_sup_l',]

    if body_parts_of_interest is None: # Default Body Parts of Interest
        body_parts_of_interest = ['time', 'hand_r_x','hand_r_y','hand_r_z', 'hand_l_x','hand_l_y','hand_l_z']

    if body_parts_of_interest_no_axes is None: # Default Body Parts of Interest without axes
        body_parts_of_interest_no_axes = ['hand_r', 'hand_l']


    # Sort values of interest so its more organized
    joints_of_interest = sorted(joints_of_interest)
    body_parts_of_interest = sorted(body_parts_of_interest)

    dir_destination = os.path.join(root_stat, '01_continuous', '01_on_trial')

    for trial in trial_list:

        """get OMC dataframes"""
        omc_endeff_pos = get_omc_file(trial, root_omc, endeff='pos')
        omc_endeff_vel = get_omc_file(trial, root_omc, endeff='vel')
        omc_endeff_acc = get_omc_file(trial, root_omc, endeff='acc')
        omc_joint_pos = get_omc_file(trial, root_omc, joint='pos')
        omc_joint_vel = get_omc_file(trial, root_omc, joint='vel')
        omc_joint_acc = get_omc_file(trial, root_omc, joint='acc')

        """get MMC dataframes"""
        mmc_endeff_pos = get_dataframes(trial.opensim_ana_pos)
        mmc_endeff_vel = get_dataframes(trial.opensim_ana_vel)
        mmc_endeff_acc = get_dataframes(trial.opensim_ana_acc)
        mmc_joint_pos = get_dataframes(trial.opensim_ik_ang_pos)
        mmc_joint_vel = get_dataframes(trial.opensim_ik_ang_vel)
        mmc_joint_acc = get_dataframes(trial.opensim_ik_ang_acc)


        """endeffector statistics"""
        df_stat_endeff_pos = compare_timeseries_for_trial(mmc_endeff_pos, omc_endeff_pos,
                                                    body_parts_of_interest, isjoint=False,
                                                    root_stat=root_stat)
        df_stat_endeff_vel = compare_timeseries_for_trial(mmc_endeff_vel, omc_endeff_vel,
                                                    body_parts_of_interest, isjoint=False,
                                                    root_stat=root_stat)
        df_stat_endeff_acc = compare_timeseries_for_trial(mmc_endeff_acc, omc_endeff_acc,
                                                    body_parts_of_interest, isjoint=False,
                                                    root_stat=root_stat)
        csv_stat_endeff_pos = os.path.join(dir_destination, f'{trial.identifier}_stat_endeff_pos.csv')
        csv_stat_endeff_vel = os.path.join(dir_destination, f'{trial.identifier}_stat_endeff_vel.csv')
        csv_stat_endeff_acc = os.path.join(dir_destination, f'{trial.identifier}_stat_endeff_acc.csv')
        df_stat_endeff_pos.to_csv(csv_stat_endeff_pos, sep=';')
        df_stat_endeff_vel.to_csv(csv_stat_endeff_vel, sep=';')
        df_stat_endeff_acc.to_csv(csv_stat_endeff_acc, sep=';')

        """endeffector statistics vel & acc magnitude"""
        df_stat_endeff_vel_magnitude = reduce_axes(df_stat_endeff_vel)
        df_stat_endeff_acc_magnitude = reduce_axes(df_stat_endeff_acc)

        csv_stat_endeff_vel_magnitude = os.path.join(dir_destination, f'{trial.identifier}_stat_end_vel_magnitude.csv')
        csv_stat_endeff_acc_magnitude = os.path.join(dir_destination, f'{trial.identifier}_stat_end_acc_magnitude.csv')
        df_stat_endeff_vel_magnitude.to_csv(csv_stat_endeff_vel_magnitude, sep=';')
        df_stat_endeff_acc_magnitude.to_csv(csv_stat_endeff_acc_magnitude, sep=';')

        """Joint statistics"""
        df_stat_joint_pos = compare_timeseries_for_trial(mmc_joint_pos, omc_joint_pos,
                                                    joints_of_interest, isjoint=True,
                                                    root_stat=root_stat)
        df_stat_joint_vel = compare_timeseries_for_trial(mmc_joint_vel, omc_joint_vel,
                                                    joints_of_interest, isjoint=True,
                                                    root_stat=root_stat)
        df_stat_joint_acc = compare_timeseries_for_trial(mmc_joint_acc, omc_joint_acc,
                                                    joints_of_interest, isjoint=True,
                                                    root_stat=root_stat)
        csv_stat_joint_pos = os.path.join(dir_destination, f'{trial.identifier}_stat_joint_pos.csv')
        csv_stat_joint_vel = os.path.join(dir_destination, f'{trial.identifier}_stat_joint_vel.csv')
        csv_stat_joint_acc = os.path.join(dir_destination, f'{trial.identifier}_stat_joint_acc.csv')
        df_stat_joint_pos.to_csv(csv_stat_joint_pos, sep=';')
        df_stat_joint_vel.to_csv(csv_stat_joint_vel, sep=';')
        df_stat_joint_acc.to_csv(csv_stat_joint_acc, sep=';')



def standardize_data(df, metadata=None, verbose=1):
    """
    gets a DataFrame containing data of joints or endeffector positions.

    It checks for the type of data and then renames the columns to a standardized set for later functions.

    metadata either contains a list or a string ('Speeds', 'Coordinates', 'Accelerations')

    :param verbose:
    :param metadata:
    :param df:
    :return: Datatype, DataFrame
    """
    def standardize_columns(columns_old, columns_stand, verbose=1):
        """
        This function takes a list of columns and a list of standardized names and renames the columns to the standardized names.

        :param verbose:
        :param columns_old:
        :param columns_stand:
        :return: columns_new
        """


        columns_old = [col.lower() for col in columns_old]  # make all columns lowercase
        columns_old = [col.replace(" ", "") for col in columns_old]  # Get rid of all whitespaces
        if any('rot' in col for col in columns_old):  # Check if 'rot' is contained in any of the columns
            columns_old = [col.replace('rot', 'o') for col in columns_old]  # Replace 'rot' with 'ox'
        if '#times' in columns_old:  # Check if '#times' is in the columns and rename to 'time'"""
            columns_old[columns_old.index('#times')] = 'time'

        # Safety check for columns that are not in the standardized list
        columns_new = []
        for col_old in columns_old:
            if col_old not in columns_stand:
                # Finde element in columns_stand that is most similar to col_old
                if verbose >= 2:
                    print(f"old: {col_old}\tnew: {process.extractOne(col_old, columns_stand)}")
                columns_new.append(process.extractOne(col_old, columns_stand)[0])  # Look for the most similar element in columns_stand
            else:
                columns_new.append(col_old)

        return columns_new

    stand_rawkps =  [] # List containing the standardized names of the raw keypoints
    stand_bodypart =  ['time','pelvis_x','pelvis_y','pelvis_z','pelvis_ox','pelvis_oy','pelvis_oz',
                       'sacrum_x','sacrum_y','sacrum_z','sacrum_ox','sacrum_oy','sacrum_oz',
                       'femur_r_x','femur_r_y','femur_r_z','femur_r_ox','femur_r_oy','femur_r_oz',
                       'patella_r_x','patella_r_y','patella_r_z','patella_r_ox','patella_r_oy','patella_r_oz',
                       'tibia_r_x','tibia_r_y','tibia_r_z','tibia_r_ox','tibia_r_oy','tibia_r_oz',
                       'talus_r_x','talus_r_y','talus_r_z','talus_r_ox','talus_r_oy','talus_r_oz',
                       'calcn_r_x','calcn_r_y','calcn_r_z','calcn_r_ox','calcn_r_oy','calcn_r_oz',
                       'toes_r_x','toes_r_y','toes_r_z','toes_r_ox','toes_r_oy','toes_r_oz',
                       'femur_l_x','femur_l_y','femur_l_z','femur_l_ox','femur_l_oy','femur_l_oz',
                       'patella_l_x','patella_l_y','patella_l_z','patella_l_ox','patella_l_oy','patella_l_oz',
                       'tibia_l_x','tibia_l_y','tibia_l_z','tibia_l_ox','tibia_l_oy','tibia_l_oz',
                       'talus_l_x','talus_l_y','talus_l_z','talus_l_ox','talus_l_oy','talus_l_oz',
                       'calcn_l_x','calcn_l_y','calcn_l_z','calcn_l_ox','calcn_l_oy','calcn_l_oz',
                       'toes_l_x','toes_l_y','toes_l_z','toes_l_ox','toes_l_oy','toes_l_oz',
                       'lumbar5_x','lumbar5_y','lumbar5_z','lumbar5_ox','lumbar5_oy','lumbar5_oz',
                       'lumbar4_x','lumbar4_y','lumbar4_z','lumbar4_ox','lumbar4_oy','lumbar4_oz',
                       'lumbar3_x','lumbar3_y','lumbar3_z','lumbar3_ox','lumbar3_oy','lumbar3_oz',
                       'lumbar2_x','lumbar2_y','lumbar2_z','lumbar2_ox','lumbar2_oy','lumbar2_oz',
                       'lumbar1_x','lumbar1_y','lumbar1_z','lumbar1_ox','lumbar1_oy','lumbar1_oz',
                       'torso_x','torso_y','torso_z','torso_ox','torso_oy','torso_oz',
                       'head_x','head_y','head_z','head_ox','head_oy','head_oz',
                       'abdomen_x','abdomen_y','abdomen_z','abdomen_ox','abdomen_oy','abdomen_oz',
                       'humerus_r_x','humerus_r_y','humerus_r_z','humerus_r_ox','humerus_r_oy','humerus_r_oz',
                       'ulna_r_x','ulna_r_y','ulna_r_z','ulna_r_ox','ulna_r_oy','ulna_r_oz',
                       'radius_r_x','radius_r_y','radius_r_z','radius_r_ox','radius_r_oy','radius_r_oz',
                       'hand_r_x','hand_r_y','hand_r_z','hand_r_ox','hand_r_oy','hand_r_oz',
                       'humerus_l_x','humerus_l_y','humerus_l_z','humerus_l_ox','humerus_l_oy','humerus_l_oz',
                       'ulna_l_x','ulna_l_y','ulna_l_z','ulna_l_ox','ulna_l_oy','ulna_l_oz',
                       'radius_l_x','radius_l_y','radius_l_z','radius_l_ox','radius_l_oy','radius_l_oz',
                       'hand_l_x','hand_l_y','hand_l_z','hand_l_ox','hand_l_oy','hand_l_oz',
                       'center_of_mass_x','center_of_mass_y','center_of_mass_z']
    stand_joints = ['time','pelvis_tilt','pelvis_list','pelvis_rotation','pelvis_tx','pelvis_ty','pelvis_tz',
                    'hip_flexion_r','hip_adduction_r','hip_rotation_r',
                    'knee_angle_r','knee_angle_r_beta','ankle_angle_r','subtalar_angle_r','mtp_angle_r',
                    'hip_flexion_l','hip_adduction_l','hip_rotation_l',
                    'knee_angle_l','knee_angle_l_beta','ankle_angle_l','subtalar_angle_l','mtp_angle_l',
                    'L5_S1_Flex_Ext','L5_S1_Lat_Bending','L5_S1_axial_rotation','L4_L5_Flex_Ext','L4_L5_Lat_Bending',
                    'L4_L5_axial_rotation','L3_L4_Flex_Ext','L3_L4_Lat_Bending','L3_L4_axial_rotation',
                    'L2_L3_Flex_Ext','L2_L3_Lat_Bending','L2_L3_axial_rotation','L1_L2_Flex_Ext','L1_L2_Lat_Bending',
                    'L1_L2_axial_rotation','L1_T12_Flex_Ext','L1_T12_Lat_Bending','L1_T12_axial_rotation',
                    'Abs_r3','Abs_r2','Abs_r1','Abs_t1','Abs_t2','neck_flexion','neck_bending','neck_rotation',
                    'arm_flex_r','arm_add_r','arm_rot_r','elbow_flex_r','pro_sup_r','wrist_flex_r','wrist_dev_r',
                    'arm_flex_l','arm_add_l','arm_rot_l','elbow_flex_l','pro_sup_l','wrist_flex_l','wrist_dev_l']


    # List containing the standardized names of the bodypart values
    stand_joint =  [] # List containing the standardized names of the joints

    kinematicState = ""

    if metadata is not None:
        if 'Coordinates' in metadata or 'Positions' in metadata:
            if verbose >= 2:
                print("Data is Position Data")

            kinematicState = 'pos'

        if 'Speeds' in metadata or 'Velocities' in metadata:
            if verbose >= 2:
                print("Data is Speed Data")

            kinematicState = 'vel'

        if 'Accelerations' in metadata:
            if verbose >= 2:
                print("Data is Acceleration Data")

            kinematicState = 'acc'

        pass

    columns = df.columns
    dat_type = ""
    if 'elbow_flex_l' in df.columns:

        if verbose >=2:
            print("Standardizing: \tJoint Data.")

        """Change columns to standardized names"""
        df.columns = standardize_columns(df.columns, stand_joints, verbose)



    elif any( i in df.columns for i in [' hand_l_x', 'hand_l_x', 'hand_l_X']):
        if verbose >=2:
            print("Standardizing:\tEndeffector Data")
        df.columns = standardize_columns(df.columns, stand_bodypart, verbose)

    else:
        raise ValueError("Error in iDrinkStatisticalAnalysis.standardize_data\n"
                         "Neither 'elbow_flex_l' nor 'hand_l_x' are in Data.\n"
                         "Please check the data and try again.")

    return df, kinematicState

def get_dataframes(files):
    """
    Reads files containing position, velocity or acceleration of endeffector or joint data.



    :param files: list of filepaths, can be string
    :return: df_pos, df_vel, df_acc, if path of either of these is not given, it returns None
    """

    df_pos = None
    df_vel = None
    df_acc = None
    single_file = False
    kinematicState = ""

    if type(files) is not dict and type(files) is not list:
        single_file = True
        files = [files]

    for file in files:
        metadata = None
        if os.path.splitext(file)[1] == '.csv':
            df = pd.read_csv(file)
            df,_ = standardize_data(df)

            if 'pos' in file:
                kinematicState = 'pos'
            elif 'vel' in file:
                kinematicState = 'vel'
            elif 'acc' in file:
                kinematicState = 'acc'
        else:
            metadata, df = read_opensim_file(file)
            df, kinematicState = standardize_data(df, metadata)

        if single_file:
            return df
        else:
            match kinematicState:
                case 'pos':
                    df_pos = df
                case 'vel':
                    df_vel = df
                case 'acc':
                    df_acc = df
                case _:
                    raise ValueError("Error in iDrinkStatisticalAnalysis\n"
                                     "Kinematic State not recognized.")
    return df_pos, df_vel, df_acc

def get_omc_file(id_p, id_t, root_omc, endeff=None, joint=None):
    """
    Gets the OMC file for the given trial.

    :param trial:
    :param endeff: 'pos', 'vel', 'acc'
    :param joint: 'pos', 'vel', 'acc'
    :return:
    """
    if all([endeff, joint]):
        raise ValueError("Error in iDrinkStatisticalAnalysis.get_omc_file\n"
                         "Both endeff and joint are not None.\n"
                         "Please specify only one.")

    id_s = 'S15133'

    root_mov_ana = os.path.join(root_omc, f'{id_s}_{id_p}', f'{id_s}_{id_p}_{id_t}', 'movement_analysis')

    if endeff:
        match endeff:
            case 'pos':
                path_file = os.path.join(root_mov_ana, 'kin_opensim_analyzetool', f'{id_s}_{id_p}_{id_t}_BodyKinematics_pos_global.sto')
            case 'vel':
                path_file = os.path.join(root_mov_ana, 'kin_opensim_analyzetool', f'{id_s}_{id_p}_{id_t}_BodyKinematics_vel_global.sto')
            case 'acc':
                path_file = os.path.join(root_mov_ana, 'kin_opensim_analyzetool', f'{id_s}_{id_p}_{id_t}_BodyKinematics_acc_global.sto')
            case _:
                raise ValueError(f"Error in iDrinkStatisticalAnalysis.get_omc_file\n"
                                 f"Endeff not recognized.\n"
                                 f"value given: {endeff}")
    elif joint:
        match joint:
            case 'pos':
                path_file = os.path.join(root_mov_ana, 'ik_tool', f'{id_s}_{id_p}_{id_t}_Kinematics_pos.csv')
            case 'vel':
                path_file = os.path.join(root_mov_ana, 'ik_tool', f'{id_s}_{id_p}_{id_t}_Kinematics_vel.csv')
            case 'acc':
                path_file = os.path.join(root_mov_ana, 'ik_tool', f'{id_s}_{id_p}_{id_t}_Kinematics_acc.csv')
            case _:
                raise ValueError(f"Error in iDrinkStatisticalAnalysis.get_omc_file\n"
                                 f"Joint not recognized.\n"
                                 f"value given: {joint}")
    else:
        raise ValueError("Error in iDrinkStatisticalAnalysis.get_omc_file\n"
                         "Neither endeff nor joint are specified.\n"
                         "Please specify one.")

    df_omc = get_dataframes(path_file)

    return df_omc


def get_omc_mmc_error(dir_root, df_timestamps, verbose=1):
    """
    Writes csv file containing error for all trials and participants of a setting.
    S*_omc_mmc_error.csv

    Write second csv file containing RSME instead of error
    S*_omc_mmc_error_rmse.csv

    Write third csv file containing mean and std of error for each participant and trial

    Write fourth

    Columns contain:
        - time
        - affected
        - time_P*_T*
        - P*_T*
        - P*_mean_affected
        - P*_std_affected
        - P*_mean_unaffected
        - P*_std_unaffected

    Rows are Frame

    The resulting DataFrames are used for the respecting plots

    :param dir_dat_filt:
    :param id_s:
    :param verbose:
    :return:
    """
    if verbose >=1:
        print(f"Running iDrinkStatisticalAnalysis.get_omc_mmc_error")

    # TODO: Make changes after preprocessed data exists
    # Check if timestamps is Dataframe
    if type(df_timestamps) is not pd.DataFrame:
        df_timestamps = pd.read_csv(df_timestamps, sep=';')

    dir_dat_out = os.path.join(dir_root, '04_statistics', '01_continuous', '01_results', '01_omc_to_mmc_error_per_s')

    dir_dat_in = os.path.join(dir_root, '03_data', 'preprocessed_data', '02_fully_preprocessed')

    # get all omc_trials
    omc_csvs = glob.glob(os.path.join(dir_dat_in, 'S15133_P*_T*.csv'))

    # get all s_ids except 'S15133'
    s_ids = sorted(list(set([os.path.basename(file).split('_')[0] for file in os.listdir(dir_dat_in)])))
    s_ids.remove('S15133')

    # retrieve all p_ids and t_ids present in omc data.
    p_ids = sorted(list(set([os.path.basename(omc_csv).split('_')[1] for omc_csv in omc_csvs])))

    if verbose >= 1:
        progress = tqdm(total=len(s_ids)*len(p_ids), desc='Calculating OMC-MMC Error')
    for id_s in s_ids:

        csv_s_error = os.path.join(dir_dat_out, f'{id_s}_omc_mmc_error.csv')
        csv_s_rse = os.path.join(dir_dat_out, f'{id_s}_omc_mmc_error_rse.csv')

        csv_s_error_mean = os.path.join(dir_dat_out, f'{id_s}_omc_mmc_error_mean_std.csv')
        csv_s_rmse_mean = os.path.join(dir_dat_out, f'{id_s}_omc_mmc_rmse_mean_std.csv')

        df_s_error = None
        df_s_rse  = None
        df_s_error_mean = None
        df_s_rmse_mean = None


        for id_p in p_ids:
            if verbose >= 1:
                progress.set_description(f'Calculating OMC-MMC Error for {id_s}_{id_p}')

            omc_csvs_p = sorted([omc_csv for omc_csv in omc_csvs if id_p in os.path.basename(omc_csv)])

            dict_error_p_mean_aff = None
            dict_error_p_mean_unaff = None
            dict_rmse_p_mean_aff = None
            dict_rmse_p_mean_unaff = None

            found_files = []
            t_ids = []
            for id_t in sorted(list(set([os.path.basename(omc_csv).split('_')[2] for omc_csv in omc_csvs_p]))):
                found_files.extend(glob.glob(os.path.join(dir_dat_in, f'{id_s}*{id_p}*{id_t}*.csv')))
                if len(found_files) > 0:
                    t_ids.append(id_t)

            if not t_ids:
                if verbose >= 2:
                    print(f"No files found for {id_s}_{id_p}")
                if verbose >= 1:
                    progress.update(1)

                continue

            for id_t in t_ids:
                mmc_csv = os.path.join(dir_dat_in, f'{id_s}_{id_p}_{id_t}_preprocessed.csv')
                omc_csv = os.path.join(dir_dat_in, f'S15133_{id_p}_{id_t}_preprocessed.csv')

                try:
                    df_omc = pd.read_csv(omc_csv, sep=';')
                    df_mmc = pd.read_csv(mmc_csv, sep=';')
                except Exception as e:
                    print(f"Error in iDrinkStatisticalAnalysis.get_omc_mmc_error while reading csv file:\n"
                          f"OMC-File:\t{omc_csv} \n"
                          f"MMC-File:\t{mmc_csv}\n"
                          f"\n"
                          f"Error:\t{e}")
                    continue

                time_t = df_omc['time']

                #check if affected in timestamps
                condition = df_timestamps[(df_timestamps['id_p'] == id_p) & (df_timestamps['id_t'] == id_t)]['condition'].values[0]

                # get columns without 'time'
                columns_old = [f'{col}' for col in df_omc.columns if col not in ['time', 'Unnamed: 0']]
                columns_new = [f'{id_p}_{id_t}_{col}' for col in df_omc.columns if col not in ['time', 'Unnamed: 0']]
                columns_new_full = [f'{id_p}_{id_t}_{col}' for col in df_omc.columns if col not in['Unnamed: 0']]


                if dict_error_p_mean_aff is None:
                    # Create dictionaries with columns_old as keys containing empty lists
                    dict_error_p_mean_aff = {col: [] for col in columns_old}
                    dict_error_p_mean_unaff = {col: [] for col in columns_old}
                    dict_rmse_p_mean_aff = {col: [] for col in columns_old}
                    dict_rmse_p_mean_unaff = {col: [] for col in columns_old}

                    dict_error_p_std_aff = {col: [] for col in columns_old}
                    dict_error_p_std_unaff = {col: [] for col in columns_old}
                    dict_rmse_p_std_aff = {col: [] for col in columns_old}
                    dict_rmse_p_std_unaff = {col: [] for col in columns_old}

                # Iterate over all columns and calculate error of all timepoints
                df_error = pd.DataFrame(columns=columns_new_full)
                df_rse = pd.DataFrame(columns=columns_new_full)

                df_error[f'{id_p}_{id_t}_time'] = time_t
                df_rse[f'{id_p}_{id_t}_time'] = time_t

                dict_error_t_mean = {col: 0 for col in columns_old}
                dict_rmse_t_mean = {col: 0 for col in columns_old}
                dict_error_t_std = {col: 0 for col in columns_old}
                dict_rmse_t_std = {col: 0 for col in columns_old}

                for column, column_new in zip(columns_old, columns_new):
                    error = df_mmc[column] - df_omc[column]
                    rse = np.sqrt(error**2)

                    dict_error_t_mean[column] = np.mean(error)
                    dict_rmse_t_mean[column] = np.sqrt(np.mean(error**2))
                    dict_error_t_std[column] = np.std(error)
                    dict_rmse_t_std[column] = np.std(rse)

                    df_error[column_new] = error
                    df_rse[column_new] = rse

                    # Add to dicts for mean over id_p
                    if condition == 'affected':
                        dict_error_p_mean_aff[column].extend(error)
                        dict_rmse_p_mean_aff[column].extend(rse)
                    else:
                        dict_error_p_mean_unaff[column].extend(error)
                        dict_rmse_p_mean_unaff[column].extend(rse)

                if df_s_error is None:
                    df_s_error = df_error
                    df_s_rse = df_rse
                else:
                    df_s_error = pd.concat([df_s_error, df_error], axis=1)
                    df_s_rse = pd.concat([df_s_rse, df_rse], axis=1)

                csv_t_error = os.path.join(dir_dat_out, f'{id_s}_{id_p}_{id_t}_error.csv')
                csv_t_rse = os.path.join(dir_dat_out, f'{id_s}_{id_p}_{id_t}_rse.csv')
                df_error.to_csv(csv_t_error, sep=';')
                df_rse.to_csv(csv_t_rse, sep=';')

                if condition == 'affected':
                    idx = [f'{id_p}_{id_t}_aff', f'{id_p}_{id_t}_aff_std']
                else:
                    idx = [f'{id_p}_{id_t}_unaff', f'{id_p}_{id_t}_unaff_std']
                if df_s_error_mean is None:
                    # Create with dicts
                    df_s_error_mean = pd.DataFrame([dict_error_t_mean, dict_error_t_std],
                                                   index=idx)
                    df_s_rmse_mean = pd.DataFrame([dict_rmse_t_mean, dict_rmse_t_std],
                                                  index=idx)

                else:  # Add dicts to existing DataFrames as Rows
                    df_s_error_mean = pd.concat([df_s_error_mean, pd.DataFrame([dict_error_t_mean, dict_error_t_std],
                                                             index=idx)])
                    df_s_rmse_mean = pd.concat([df_s_rmse_mean, pd.DataFrame([dict_rmse_t_mean, dict_rmse_t_std],
                                                            index=idx)])


            # iterate over keys in dictionary and calculate std and then mean
            try:
                for key in dict_error_p_mean_aff.keys():

                    dict_error_p_std_aff[key] = np.nanstd(dict_error_p_mean_aff[key], axis=0)
                    dict_error_p_std_unaff[key] = np.nanstd(dict_error_p_mean_unaff[key], axis=0)
                    dict_rmse_p_std_aff[key] = np.nanstd(dict_rmse_p_mean_aff[key], axis=0)
                    dict_rmse_p_std_unaff[key] = np.nanstd(dict_rmse_p_mean_unaff[key], axis=0)

                    dict_error_p_mean_aff[key] = np.nanmean(dict_error_p_mean_aff[key], axis=0)
                    dict_error_p_mean_unaff[key] = np.nanmean(dict_error_p_mean_unaff[key], axis=0)
                    dict_rmse_p_mean_aff[key] = np.nanmean(dict_rmse_p_mean_aff[key], axis=0)
                    dict_rmse_p_mean_unaff[key] = np.nanmean(dict_rmse_p_mean_unaff[key], axis=0)
            except Exception as e:
                if verbose >= 1:
                    print(f"Error in iDrinkStatisticalAnalysis.get_omc_mmc_error while calculating mean and std:\n"
                          f"{e}")

            if verbose >= 1:
                progress.update(1)




            columns = [f'{col}' for col in df_omc.columns if col != 'time']
            df_error_mean = pd.DataFrame(columns=columns)
            idx = [f'{id_p}_aff', f'{id_p}_aff_std',
                   f'{id_p}_unaff', f'{id_p}_unaff_std']
            if df_s_error_mean is None:
                # Create with dicts
                df_s_error_mean = pd.DataFrame([dict_error_p_mean_aff, dict_error_p_std_aff,
                                                dict_error_p_mean_unaff, dict_error_p_std_unaff],
                                               index=idx)
                df_s_rmse_mean = pd.DataFrame([dict_rmse_p_mean_aff, dict_rmse_p_std_aff,
                                               dict_rmse_p_mean_unaff, dict_rmse_p_std_unaff],
                                              index=idx)

            else: # Add dicts to existing DataFrames as Rows
                df_s_error_mean = pd.concat([df_s_error_mean, pd.DataFrame([dict_error_p_mean_aff, dict_error_p_std_aff,
                                                          dict_error_p_mean_unaff, dict_error_p_std_unaff],
                                                         index=idx)])
                df_s_rmse_mean = pd.concat([df_s_rmse_mean, pd.DataFrame([dict_rmse_p_mean_aff, dict_rmse_p_std_aff,
                                               dict_rmse_p_mean_unaff, dict_rmse_p_std_unaff],
                                                        index=idx)])

        pass

        os.makedirs(dir_dat_out, exist_ok=True)

        df_s_error_mean.to_csv(csv_s_error_mean, sep=';')
        df_s_rmse_mean.to_csv(csv_s_rmse_mean, sep=';')

        df_s_error.to_csv(csv_s_error, sep=';')
        df_s_rse.to_csv(csv_s_rse, sep=';')

def preprocess_timeseries(dir_root, downsample = True, drop_last_frame = False, detect_outliers = False,
                          joint_vel_thresh = 5, hand_vel_thresh = 3000, verbose=1):
    """
    Preprocess timeseries data for statistical analysis.

    - Downsample OMC to 60Hz
    - Detect outliers based on endeffector and elbow velocity.

    Write Log-csv for outlier detection with reason for taking a trial out.

    :param dir_root:
    :param verbose:
    :return:
    """
    pass

    def downsample_dataframe(df_, fps=60):
        """
        Taken from Marwens Masterthesis

        https://github.com/cf-project-delta/Delta_3D_reconstruction/blob/d73315f5cac31be1c1fe621fdfa7cdce24a2cc2a/processing/reading_storing_processed_data.py#L410

        :param df_:
        :param fps:
        :return:
        """

        # Convert the Timestamp column to a datetime format and set it as the index
        df = df_.copy()
        try:
            df['time'] = pd.to_timedelta(df['time'], unit='s')
        except:
            df['time'] = pd.to_timedelta(df['time'], unit='s')
        df = df.set_index('time')
        # Get the numerical and non-numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        # Resample the numerical columns to the desired fps and compute the mean for each interval
        # num_df = df[num_cols].resample(f'{1000/fps:.6f}ms').mean()
        num_df = df[num_cols].resample(f'{1000 / fps:.6f}ms').mean().interpolate()  # Interpolate missing values
        # Resample the non-numerical columns to the desired fps and compute the mode for each interval
        non_num_df = df[non_num_cols].resample(f'{1000 / fps:.6f}ms').agg(lambda x: x.mode()[0])
        # Merge the numerical and non-numerical dataframes
        df_downsampled = pd.concat([num_df, non_num_df], axis=1)
        # Reset the index to make the Timestamp column a regular column again
        df_downsampled.reset_index(inplace=True)
        return df_downsampled

    def drop_last_rows_if_needed(df1, df2):
        """
        Taken from Marwens Masterthesis

        https://github.com/cf-project-delta/Delta_3D_reconstruction/blob/d73315f5cac31be1c1fe621fdfa7cdce24a2cc2a/processing/reading_storing_processed_data.py#L410

        :param df1:
        :param df2:
        :return:
        """
        len_diff = len(df1) - len(df2)
        # Drop the last row from the dataframe with more rows
        if len_diff > 0:
            df1 = df1[:-len_diff]
        elif len_diff < 0:
            df2 = df2[:len(df1)]

        return df1.copy(), df2.copy()

    def drop_nan(df1, df2):
        """Iterate over rows and drop rows where either df1 or df2 contains NaN"""

        for idx, row in df1.iterrows():
            if row.isnull().values.any():
                df1.drop(idx, inplace=True)
                df2.drop(idx, inplace=True)

        for idx, row in df2.iterrows():
            if row.isnull().values.any():
                df1.drop(idx, inplace=True)
                df2.drop(idx, inplace=True)

        return df1, df2

    def cut_to_same_timeframe(df_1, df_2):
        """
        Cut Dataframes so that start und endtime are the same.

        :param df_1:
        :param df_2:
        :return:
        """

        max_time = min(max(df_1['time']), max(df_2['time']))
        min_time = max(min(df_1['time']), min(df_2['time']))

        df_1_out = df_1[(df_1['time'] <= max_time) & (df_1['time'] >= min_time)]
        df_2_out = df_2[(df_2['time'] <= max_time) & (df_2['time'] >= min_time)]

        return df_1_out, df_2_out

    dir_dat_in = os.path.join(dir_root, '03_data', 'preprocessed_data', '01_murphy_out')
    dir_dat_out = os.path.join(dir_root, '03_data', 'preprocessed_data', '02_fully_preprocessed')

    csv_outliers = os.path.join(dir_root, '05_logs', 'outliers.csv')
    if os.path.isfile(csv_outliers):
        df_outliers = pd.read_csv(csv_outliers, sep=';')
    else:
        df_outliers = pd.DataFrame(columns=['id_s', 'id_p', 'id_t', 'reason'])

    # get all omc_trials
    omc_csvs = glob.glob(os.path.join(dir_dat_in, 'S15133_P*_T*.csv'))
    # retrieve all p_ids and t_ids present in omc data.
    p_ids = list(set([os.path.basename(omc_csv).split('_')[1] for omc_csv in omc_csvs]))

    # get all s_ids except 'S15133'
    s_ids = list(set([os.path.basename(file).split('_')[0] for file in os.listdir(dir_dat_in)]))
    s_ids.remove('S15133')
    if verbose >=1:
        process = tqdm(p_ids, desc='Preprocessing', leave=True)

    for id_p in p_ids:
        if verbose >= 1:
            process.set_description(f'Preprocessing {id_p}')
        omc_csvs_p = [omc_csv for omc_csv in omc_csvs if id_p in os.path.basename(omc_csv)]

        t_ids = [os.path.basename(omc_csv).split('_')[2] for omc_csv in omc_csvs_p]

        # check if MMC recording exists for id_p
        found_files = []
        for id_s in s_ids:
            found_files.extend(glob.glob(os.path.join(dir_dat_in, f'{id_s}*{id_p}_*.csv')))

        if len(found_files) == 0:
            continue


        for id_t in t_ids:

            omc_csv = glob.glob(os.path.join(dir_dat_in, f'S15133_{id_p}_{id_t}*.csv'))[0]

            mmc_files = []
            for id_s in s_ids:
                found_file = glob.glob(os.path.join(dir_dat_in, f'{id_s}*{id_p}_{id_t}*.csv'))
                if len(found_file) > 0:
                    mmc_files.append(found_file[0])

            for mmc_csv in mmc_files:
                id_s = os.path.basename(mmc_csv).split('_')[0]

                try:
                    df_omc = pd.read_csv(omc_csv, sep=';')
                    df_mmc = pd.read_csv(mmc_csv, sep=';')
                except Exception as e:
                    print(f"Error in iDrinkStatisticalAnalysis.get_omc_mmc_error while reading csv file:\n"
                          f"OMC-File:\t{omc_csv} \n"
                          f"MMC-File:\t{mmc_csv}\n"
                          f"\n"
                          f"Error:\t{e}")
                    continue

                omc_nframes_before = len(df_omc)
                mmc_nframes_before = len(df_mmc)

                if downsample:
                    # Resample both DataFrames to 60Hz
                    df_omc = downsample_dataframe(df_omc)
                    df_mmc = downsample_dataframe(df_mmc)


                    #df_omc, df_mmc = drop_nan(df_omc, df_mmc)

                if drop_last_frame:
                    # Drop last rows if needed
                    df_omc, df_mmc = drop_last_rows_if_needed(df_omc, df_mmc)

                # Cut DataFrames to same timeframe
                df_omc, df_mmc = cut_to_same_timeframe(df_omc, df_mmc)

                # Detect Outliers
                if detect_outliers:
                    max_omc = max(df_omc['elbow_vel'])
                    max_mmc = max(df_mmc['elbow_vel'])
                    if max_omc > max_mmc:
                        reason = 'omc'
                        max_val = max_omc
                    else:
                        reason = 'mmc'
                        max_val = max_mmc

                    if max_val > joint_vel_thresh:
                        print(f"Trial {id_p}_{id_t} is an outlier due to high elbow velocity.")
                        df = pd.DataFrame({'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'reason': f'elbow_vel of {reason} {max_val} > {joint_vel_thresh}'})
                        pd.concat([df_outliers, df])
                        continue

                    max_omc = max(df_omc['hand_vel'])
                    max_mmc = max(df_mmc['hand_vel'])

                    if max_omc > max_mmc:
                        reason = 'omc'
                        max_val = max_omc
                    else:
                        reason = 'mmc'
                        max_val = max_mmc

                    if max_val > hand_vel_thresh:
                        print(f"Trial {id_p}_{id_t} is an outlier due to high endeffector velocity.")
                        df = pd.DataFrame({'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'reason': f'hand_vel {reason} {max_val} > {hand_vel_thresh}'})
                        pd.concat([df_outliers, df])
                        continue


                omc_nframes_after = len(df_omc)
                mmc_nframes_after = len(df_mmc)

                # Write to new csv
                os.makedirs(dir_dat_out, exist_ok=True)

                path_omc_out = os.path.join(dir_dat_out, f'S15133_{id_p}_{id_t}_preprocessed.csv')
                path_mmc_out = os.path.join(dir_dat_out, f'{id_s}_{id_p}_{id_t}_preprocessed.csv')

                df_omc.to_csv(path_omc_out, sep=';')
                df_mmc.to_csv(path_mmc_out, sep=';')

                if verbose >= 1:
                    print(f"Preprocessed:\t{path_omc_out}\n"
                          f"Preprocessed:\t{path_mmc_out}\n"
                          f"Dropped Frames:\n"
                          f"OMC: {omc_nframes_before - omc_nframes_after}\n"
                          f"MMC: {mmc_nframes_before - mmc_nframes_after}")



        if verbose >= 1:
            process.update(1)

    df_outliers.to_csv(csv_outliers, sep=';')

    if verbose >= 1:
        process.close()


if __name__ == '__main__':
    # this part is for Development and Debugging

    if sys.gettrace() is not None:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

    """Set Root Paths for Processing"""
    drives = ['C:', 'D:', 'E:', 'F:', 'G:', 'I:']
    if os.name == 'posix':  # Running on Linux
        drive = '/media/devteam-dart/Extreme SSD'
    else:
        drive = drives[1] + '\\'

    root_iDrink = os.path.join(drive, 'iDrink')
    root_val = os.path.join(root_iDrink, "validation_root")
    root_stat = os.path.join(root_val, '04_Statistics')
    root_omc = os.path.join(root_val, '03_data', 'OMC_new', 'S15133')
    root_data = os.path.join(root_val, "03_data")
    root_logs = os.path.join(root_val, "05_logs")

    # Prepare Logging Paths
    log_val_settings = os.path.join(root_logs, "validation_settings.csv")
    log_val_trials = os.path.join(root_logs, "validation_trials.csv")
    log_val_errors = os.path.join(root_logs, "validation_errors.csv")

    # prepare statistic paths
    path_csv_murphy_timestamps = os.path.join(root_stat, '02_categorical', 'murphy_timestamps.csv')
    path_csv_murphy_measures = os.path.join(root_stat, '02_categorical', 'murphy_measures.csv')

    df_settings = pd.read_csv(log_val_settings, sep=';')  # csv containing information for the various settings in use.

    test_timeseries = True



    if test_timeseries:

        preprocess_timeseries(root_val,
                              downsample=True, drop_last_frame=False, detect_outliers=False,
                              joint_vel_thresh=5, hand_vel_thresh=3000,
                              verbose=0)
        get_omc_mmc_error(root_val, path_csv_murphy_timestamps, verbose=1)


    else:

        runs_statistics_discrete(csv_murphy, root_stat, thresh_PeakVelocity_mms=None, thresh_elbowVelocity=None)
