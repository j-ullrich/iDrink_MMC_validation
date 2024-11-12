"""
We compare each MMC-Trial Objects time series and Murphy Measures to the same metrics of the OMC-trials.

"""
import os
import sys
import shutil
import glob
import re
from tqdm import tqdm
import platform
import pandas as pd
import numpy as np
import scipy as sp

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from iDrinkOpenSim import read_opensim_file
import iDrinkValPlots as iDrinkVP
import iDrinkUtilities

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

    save_plots_murphy(df_murphy, root_stat_cat, filetype=['.html'], verbose=verbose)

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


def get_omc_mmc_error_old(dir_root, df_timestamps, correct='fixed', verbose=1):
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

    if correct == 'fixed':
        dir_dat_out = os.path.join(dir_root, '04_statistics', '01_continuous', '01_results', '01_omc_to_mmc_error_per_s')
        dir_dat_in = os.path.join(dir_root, '03_data', 'preprocessed_data', '02_fully_preprocessed')
    elif correct == 'dynamic':
        dir_dat_out = os.path.join(dir_root, '04_statistics', '01_continuous', '01_results', '02_omc_to_mmc_error_per_s_dynamic')
        dir_dat_in = os.path.join(dir_root, '03_data', 'preprocessed_data', '03_fully_preprocessed_dynamic')

    for dir in [dir_dat_out, dir_dat_in]:
        os.makedirs(dir, exist_ok=True)
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
                          f"Error:\t{e} \t {e.filename}")
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


def get_error_timeseries(dir_processed, dir_results, verbose = 1):
    """
    Writes the .csv files with omc-mmc error for all trials and participants.

    normalized Data get a flag as appendix to the filename.


    csv_out: - {id_p}_{id_t}_{condition}_{side}_tserror.csv
             - {id_p}_{id_t}_{condition}_{side}_tserror_norm.csv

    df_out:         columns: [id_s, dynamic, time, {metric}_omc, {metric}_mmc, {metric}_error, {metric}_rse]
    df_out_norm =   columns: [id_s, dynamic, time, {metric}_omc, {metric}_mmc, {metric}_error, {metric}_rse]

    :param dir_processed:
    :param dir_results:
    :param verbose:
    :return:
    """
    list_dir_src = [
        os.path.join(dir_processed, '02_fully_preprocessed'),
        os.path.join(dir_processed, '03_fully_preprocessed_dynamic')
    ]

    metrics = ['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang',
               'elbow_flex_pos', 'shoulder_flex_pos', 'shoulder_abduction_pos']

    df_out_rom_temp = pd.DataFrame(columns=['id_s', 'id_p', 'id_t', 'condition', 'dynamic'] +
                                           [f'{metric}_min_omc' for metric in metrics] +
                                           [f'{metric}_max_omc' for metric in metrics] +
                                           [f'{metric}_rom_omc' for metric in metrics] +
                                           [f'{metric}_min_mmc' for metric in metrics] +
                                           [f'{metric}_max_mmc' for metric in metrics] +
                                           [f'{metric}_rom_mmc' for metric in metrics] +
                                           [f'{metric}_rom_error' for metric in metrics])

    csv_out_rom = os.path.join(dir_results, 'omc_mmc_rom.csv')

    for dir_src in list_dir_src:
        dir_dst = os.path.join(dir_results, '01_ts_error')
        os.makedirs(dir_dst, exist_ok=True)

        id_s_omc = 'S15133'

        csvs_in = glob.glob(os.path.join(dir_src, '*.csv'))
        list_csvs_in_mmc = sorted([csv for csv in csvs_in if id_s_omc not in os.path.basename(csv)])
        list_csvs_in_omc = sorted([csv for csv in csvs_in if id_s_omc in os.path.basename(csv)])

        # get all s_ids from list_csvs_in_mmc
        idx_s = sorted(list(set([os.path.basename(file).split('_')[0] for file in list_csvs_in_mmc])))

        progbar = tqdm(total=len(list_csvs_in_mmc), desc='Calculating Timeseries Error', disable=verbose<1)

        for csv_mmc in list_csvs_in_mmc:
            id_s = os.path.basename(csv_mmc).split('_')[0]
            id_p = os.path.basename(csv_mmc).split('_')[1]
            id_t = os.path.basename(csv_mmc).split('_')[2]
            condition = os.path.basename(csv_mmc).split('_')[3]
            side = os.path.basename(csv_mmc).split('_')[4]
            dynamic = os.path.basename(csv_mmc).split('_')[5]

            progbar.set_description(f'Calculating Timeseries Error for {id_s}_{id_p}_{id_t}')

            try:
                csv_omc = \
                [csv for csv in list_csvs_in_omc if id_p in os.path.basename(csv) and id_t in os.path.basename(csv)][0]
            except IndexError:
                print(f"No OMC-File found for {id_s}_{id_p}_{id_t}")
                if verbose >= 1:
                    progbar.update(1)
                continue

            df_mmc = pd.read_csv(csv_mmc, sep=';')
            df_omc = pd.read_csv(csv_omc, sep=';')

            csv_out = os.path.join(dir_dst, f'{id_p}_{id_t}_{condition}_{side}_tserror.csv')
            csv_out_norm = os.path.join(dir_dst, f'{id_p}_{id_t}_{condition}_{side}_tserror_norm.csv')

            df_out_temp = pd.DataFrame(columns=['id_s', 'dynamic', 'time'] + [f'{metric}_omc' for metric in metrics] +
                                       [f'{metric}_mmc' for metric in metrics] + [f'{metric}_error' for metric in metrics] +
                                       [f'{metric}_rse' for metric in metrics])

            df_out_norm_temp = pd.DataFrame(columns=['id_s', 'dynamic', 'time'] + [f'{metric}_omc' for metric in metrics] +
                                        [f'{metric}_mmc' for metric in metrics] + [f'{metric}_error' for metric in metrics] +
                                        [f'{metric}_rse' for metric in metrics])

            time = df_mmc['time']
            time_normalized = np.linspace(0, 1, num=len(time))

            df_out_temp['time'] = time
            df_out_temp['id_s'] = id_s
            df_out_temp['dynamic'] = dynamic

            df_out_norm_temp['time'] = time_normalized
            df_out_norm_temp['id_s'] = id_s
            df_out_norm_temp['dynamic'] = dynamic

            df_out_rom_temp['id_s'] = id_s
            df_out_rom_temp['id_p'] = id_p
            df_out_rom_temp['id_t'] = id_t
            df_out_rom_temp['condition'] = condition
            df_out_rom_temp['dynamic'] = dynamic


            # Get errors for all metrics and add to dataframes
            for metric in metrics:
                omc = df_omc[metric]
                mmc = df_mmc[metric]

                error = mmc - omc
                rse = np.sqrt(error**2)

                df_out_temp[f'{metric}_omc'] = omc
                df_out_temp[f'{metric}_mmc'] = mmc
                df_out_temp[f'{metric}_error'] = error
                df_out_temp[f'{metric}_rse'] = rse

                df_out_norm_temp[f'{metric}_omc'] = omc
                df_out_norm_temp[f'{metric}_mmc'] = mmc
                df_out_norm_temp[f'{metric}_error'] = error
                df_out_norm_temp[f'{metric}_rse'] = rse

                # Add rom values TODO: DEBUG
                df_out_rom_temp[f'{metric}_min_omc'] = min(omc)
                df_out_rom_temp[f'{metric}_max_omc'] = max(omc)
                df_out_rom_temp[f'{metric}_rom_omc'] = max(omc) - min(omc)
                df_out_rom_temp[f'{metric}_min_mmc'] = min(mmc)
                df_out_rom_temp[f'{metric}_max_mmc'] = max(mmc)
                df_out_rom_temp[f'{metric}_rom_mmc'] = max(mmc) - min(mmc)
                df_out_rom_temp[f'{metric}_rom_error'] = (max(mmc) - min(mmc)) - (max(omc) - min(omc))

            def get_Dataframe(path, columns):
                if os.path.isfile(path):
                    return pd.read_csv(path, sep=';')
                else:
                    return pd.DataFrame(columns=columns)

            columns = (['id_s', 'dynamic', 'time'] +
                       [f'{metric}_omc' for metric in metrics] +
                       [f'{metric}_mmc' for metric in metrics] +
                       [f'{metric}_error' for metric in metrics] +
                       [f'{metric}_rse' for metric in metrics])




            #write to .csv file
            df_out = get_Dataframe(csv_out, columns)
            df_out_norm = get_Dataframe(csv_out_norm, columns)

            columns = columns=(['id_s', 'id_p', 'id_t', 'condition', 'dynamic'] +
                               [f'{metric}_omc' for metric in metrics] +
                               [f'{metric}_mmc' for metric in metrics] +
                               [f'{metric}_error' for metric in metrics] +
                               [f'{metric}_rse' for metric in metrics])
            df_out_rom = get_Dataframe(csv_out_rom, columns)

            df_out = pd.concat([df_out, df_out_temp], axis=0, ignore_index=True)
            df_out_norm = pd.concat([df_out_norm, df_out_norm_temp], axis=0, ignore_index=True)
            df_out_rom = pd.concat([df_out_rom, df_out_rom_temp], axis=0, ignore_index=True)

            df_out.to_csv(csv_out, sep=';', index=False)
            df_out_norm.to_csv(csv_out_norm, sep=';', index=False)
            df_out_rom.to_csv(csv_out_rom, sep=';', index=False)


def get_error_mean_rmse(dir_results, verbose=1):
    """
    Writes .csv file with mean and median error, std and rmse for each setting, participant and trial.

    First the values for each trial are calculated and added to the DataFrame.

    Then, using the trial-values, the values for Participants and Settings are calculated and added to the DataFrame.

    csv_out:    omc_mmc_error.csv
    df_out:     columns: [id, condition, dynamic, normalized, {metric}_mean, {metric}_median, {metric}_std, {metric}_rmse, {metric}_rmse_std]

    condition: affected, unaffected
    dynamic: 0, 1
    normalized: 0, 1

    :param dir_results:
    :param verbose:
    :return:
    """
    dir_src = os.path.join(dir_results, '01_ts_error')

    csv_out = os.path.join(dir_results, 'omc_mmc_error.csv')

    metrics = ['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang',
               'elbow_flex_pos', 'shoulder_flex_pos', 'shoulder_abduction_pos']

    if os.path.isfile(csv_out):
        df_out = pd.read_csv(csv_out, sep=';')
    else:
        df_out = pd.DataFrame(columns=['id', 'condition', 'dynamic', 'normalized'] + [f'{metric}_mean' for metric in metrics] +
                               [f'{metric}_median' for metric in metrics] + [f'{metric}_std' for metric in metrics] +
                               [f'{metric}_rmse' for metric in metrics] + [f'{metric}_rmse_std' for metric in metrics])

    dir_dst = os.path.join(dir_results, '02_ts_error_mean_rmse')
    os.makedirs(dir_dst, exist_ok=True)

    id_s_omc = 'S15133'
    csvs_in = glob.glob(os.path.join(dir_src, '*.csv'))
    list_csv = sorted([csv for csv in csvs_in if id_s_omc not in os.path.basename(csv)])

    for csv in list_csv:
        id_p = os.path.basename(csv).split('_')[0]
        id_t = os.path.basename(csv).split('_')[1]
        condition = os.path.basename(csv).split('_')[2]
        side = os.path.basename(csv).split('_')[3]
        normalized = 'normalized' if 'norm' in os.path.basename(csv) else 'original'

        df = pd.read_csv(csv, sep=';')
        for id_s in df['id_s'].unique():
            df_s = df[df['id_s'] == id_s]
            id = f'{id_s}_{id_p}_{id_t}'

            for dynamic in df_s['dynamic'].unique():
                for metric in metrics:
                    df_metric = df_s[[f'{metric}_error', f'{metric}_rse']]
                    mean = np.mean(df_metric[f'{metric}_error'])
                    median = np.median(df_metric[f'{metric}_error'])
                    std = np.std(df_metric[f'{metric}_error'])
                    rmse = np.sqrt(np.mean(df_metric[f'{metric}_error']**2))
                    rmse_std = np.std(df_metric[f'{metric}_rse'])

                    df_mean = pd.DataFrame({
                        'id': id,
                        'condition': condition,
                        'dynamic': dynamic,
                        'normalized': normalized,
                        f'{metric}_mean': mean,
                        f'{metric}_median': median,
                        f'{metric}_std': std,
                        f'{metric}_rmse': rmse,
                        f'{metric}_rmse_std': rmse_std
                    }, index=[0])

                    df_out = pd.concat([df_out, df_mean], axis=0, ignore_index=True)

    # safetysave
    df_out.to_csv(csv_out, sep=';', index=False)

    idx = df_out['id'].unique()

    # get sets for id_s and id_p
    idx_s = sorted(list(set([id.split('_')[0] for id in idx])))
    idx_p = sorted(list(set([id.split('_')[1] for id in idx])))

    df_mean_s = pd.DataFrame(columns=['id', 'condition', 'dynamic', 'normalized'] + [f'{metric}_mean' for metric in metrics] +
                               [f'{metric}_median' for metric in metrics] + [f'{metric}_std' for metric in metrics] +
                               [f'{metric}_rmse' for metric in metrics] + [f'{metric}_rmse_std' for metric in metrics])

    df_mean_p = pd.DataFrame(columns=['id', 'condition', 'dynamic', 'normalized'] + [f'{metric}_mean' for metric in metrics] +
                               [f'{metric}_median' for metric in metrics] + [f'{metric}_std' for metric in metrics] +
                               [f'{metric}_rmse' for metric in metrics] + [f'{metric}_rmse_std' for metric in metrics])

    def get_mean_error_for_s_and_p(df_temp, df_mean, metrics, id ):
        if df_temp.shape[0] == 0:
            return df_mean

        for dynamic in df_temp['dynamic'].unique():
            df_temp = df_temp[df_temp['dynamic'] == dynamic]

            for normalized in df_temp['normalized'].unique():
                df_temp = df_temp[df_temp['normalized'] == normalized]

                for condition in df_temp['condition'].unique():
                    df_temp = df_temp[df_temp['condition'] == condition]

                    for metric in metrics:
                        df_metric = df_temp[[f'{metric}_mean', f'{metric}_median', f'{metric}_std', f'{metric}_rmse',
                                             f'{metric}_rmse_std']]
                        mean = np.mean(df_metric[f'{metric}_mean'])
                        median = np.mean(df_metric[f'{metric}_median'])
                        std = np.mean(df_metric[f'{metric}_std'])
                        rmse = np.mean(df_metric[f'{metric}_rmse'])
                        rmse_std = np.mean(df_metric[f'{metric}_rmse_std'])

                        df_mean = pd.concat([df_mean, pd.DataFrame({
                            'id': id,
                            'condition': condition,
                            'dynamic': dynamic,
                            'normalized': normalized,
                            f'{metric}_mean': mean,
                            f'{metric}_median': median,
                            f'{metric}_std': std,
                            f'{metric}_rmse': rmse,
                            f'{metric}_rmse_std': rmse_std
                        }, index=[0])], axis=0, ignore_index=True)

        return df_mean

    for id_s in idx_s:
        df_mean_s = get_mean_error_for_s_and_p(df_out[df_out['id'].str.contains(id_s)], df_mean_s, metrics, id_s)

        for id_p in idx_p:
            df_mean_p = get_mean_error_for_s_and_p(df_out[df_out['id'].str.contains(id_p) & df_out['id'].str.contains(id_s)],
                                                   df_mean_p, metrics, f'{id_s}_{id_p}')

    df_out = pd.concat([df_out, df_mean_s, df_mean_p], axis=0, ignore_index=True)
    df_out.to_csv(csv_out, sep=';', index=False)

    for csv_mmc in list_csv:
        id_s = os.path.basename(csv_mmc).split('_')[0]
        id_p = os.path.basename(csv_mmc).split('_')[1]
        id_t = os.path.basename(csv_mmc).split('_')[2]


def get_rom_rmse(dir_results, verbose=1):
    """
    Reads the .csv files with range of motion errors and writes rmse for following ids: '{id_s}', '{id_s}_{id_p}'

    :param dir_results:
    :param verbose:
    :return:
    """
    metrics = ['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang',
               'elbow_flex_pos', 'shoulder_flex_pos', 'shoulder_abduction_pos']

    csv_in = os.path.join(dir_results, 'omc_mmc_rom.csv')
    csv_out = os.path.join(dir_results, 'omc_mmc_rom_rmse.csv')

    df_p_temp = pd.DataFrame(columns=['id', 'dynamic', 'condition'] + [f'{metric}_rom_rmse' for metric in metrics])
    df_s_temp = pd.DataFrame(columns=['id', 'dynamic', 'condition'] + [f'{metric}_rom_rmse' for metric in metrics])

    if os.path.isfile(csv_in):
        df_in = pd.read_csv(csv_in, sep=';')
    else:
        raise FileNotFoundError(f"File not found: {csv_in}")


    def get_mean_error_for_s_and_p(df_temp, df_mean, metrics, id ):
        if df_temp.shape[0] == 0:
            return df_mean

        for dynamic in df_temp['dynamic'].unique():
            df_temp = df_temp[df_temp['dynamic'] == dynamic]

            for condition in df_temp['condition'].unique():
                df_temp = df_temp[df_temp['condition'] == condition]
                for metric in metrics:
                    df_metric = df_temp[[f'{metric}_min_omc', f'{metric}_max_omc', f'{metric}_rom_omc', f'{metric}_min_mmc', f'{metric}_max_mmc', f'{metric}_rom_mmc', f'{metric}_rom_error']]

                    error_max = df_metric[f'{metric}_max_mmc'] - df_metric[f'{metric}_max_omc']
                    error_min = df_metric[f'{metric}_min_mmc'] - df_metric[f'{metric}_min_omc']
                    error_rom = df_metric[f'{metric}_rom_mmc'] - df_metric[f'{metric}_rom_omc']

                    df_mean = pd.concat([df_mean, pd.DataFrame({
                        'id': id,
                        'condition': condition,
                        'dynamic': dynamic,
                        f'{metric}_max_rmse': np.sqrt(np.mean( error_max**2)),
                        f'{metric}_min_rmse': np.sqrt(np.mean( error_min**2)),
                        f'{metric}_rom_rmse': np.sqrt(np.mean( error_rom**2))
                    }, index=[0])], axis=0, ignore_index=True)

        return df_mean

    idx = df_in['id'].unique()
    # get sets for id_s and id_p
    idx_s = sorted(list(set([id.split('_')[0] for id in idx])))
    idx_p = sorted(list(set([id.split('_')[1] for id in idx])))

    for id_s in idx_s:
        df_s_temp = get_mean_error_for_s_and_p(df_in[df_in['id'].str.contains(id_s)], df_s_temp, metrics, id_s)

        for id_p in idx_p:
            df_p_temp = get_mean_error_for_s_and_p(df_in[df_in['id'].str.contains(id_p) & df_in['id'].str.contains(id_s)],
                                                   df_p_temp, metrics, f'{id_s}_{id_p}')

    if os.path.isfile(csv_out):
        df_out = pd.read_csv(csv_out, sep=';')
        df_out = pd.concat([df_out, df_s_temp, df_p_temp], axis=0, ignore_index=True)
    else:
        df_out = pd.concat([df_s_temp, df_p_temp], axis=0, ignore_index=True)

    df_out.to_csv(csv_out, sep=';', index=False)




def preprocess_timeseries(dir_root, downsample = True, drop_last_rows = False, correct='fixed', detect_outliers = [],
                          joint_vel_thresh = 5, hand_vel_thresh = 3000, verbose=1, plot_debug=False, print_able=False):
    """
    Preprocess timeseries data for statistical analysis.

    - Downsample OMC to 60Hz
    - Detect outliers based on endeffector and elbow velocity.

    Write Log-csv for outlier detection with reason for taking a trial out.

    :param dir_root:
    :param verbose:
    :return:
    """
    import matplotlib.pyplot as plt

    def fixing_decay_and_offset_trial_by_trial(df_omc, dm_mmc, kinematic_val, trial_identifier, print_able=False,
                                               plot_debug=False, offset=False):
        '''
        This Function is taken and modified from Marwen Moknis Masterthesis:
        https://github.com/cf-project-delta/Delta_3D_reconstruction/blob/main/processing/transform_coordinate.py#L322

        This function optimize the positioning of the kinematic trajectories

        By getting the time delay between the trials of two systems (sys 1 as reference)

        It will also get the offset present between both functions.

        We will try to  minimize the root mean square error for those two variables :

         - delta_t: time delay  (s)

         - offset: Vertical offset in the unit of trajectory

        '''

        from scipy.optimize import curve_fit
        ## creating two copies of both DataFrames
        work_df_omc = df_omc.copy()
        work_df_mmc = dm_mmc.copy()

        # Setting the time vector for the reference DataFrame
        end_time = work_df_omc['time'].iloc[-1].total_seconds()
        start_time = work_df_omc['time'].iloc[0].total_seconds()
        frames = len(work_df_omc)
        # time_delta = (end_time-start_time)/frames
        time_df_omc = np.linspace(start_time, end_time, frames)
        work_df_omc['time'] = time_df_omc

        ##Same for the second Dataframe

        end_time_2 = work_df_mmc['time'].iloc[-1].total_seconds()
        start_time_2 = work_df_mmc['time'].iloc[0].total_seconds()
        frames_2 = len(work_df_mmc)
        time_df_mmc = np.linspace(start_time_2, end_time_2, frames_2)
        work_df_mmc['time'] = time_df_mmc

        if offset:
            def reference_function(time, delay_time, offset):

                return np.interp(time + delay_time, work_df_omc['time'], work_df_omc[kinematic_val] + offset)
        else:
            def reference_function(time, delay_time):

                return np.interp(time + delay_time, work_df_omc['time'], work_df_omc[kinematic_val])

        popt, _ = curve_fit(reference_function, work_df_mmc['time'], work_df_mmc[kinematic_val])

        ## Getting the value we did the optimisation on
        optimal_delay_time = popt[0]
        if offset:
            optimal_offset_val = popt[1]

        if print_able:
            if offset:
                print(
                    f"Those are the optimal delay time {optimal_delay_time} and offset {optimal_offset_val} for trial {trial_identifier} ")
            else:
                print(f"This is the optimal delay time {optimal_delay_time} for trial {trial_identifier} ")

        if not (offset):
            ## We try the method of just taking ou the mean of sys_2 and adding by the mean of the reference system
            mean_omc = np.mean(work_df_omc[kinematic_val])
            mean_mmc = np.mean(work_df_mmc[kinematic_val])
            val_work_mmc = work_df_mmc[kinematic_val] - mean_mmc + mean_omc

        if plot_debug:
            plt.plot(work_df_omc['time'], work_df_omc[kinematic_val])
            if offset:
                plt.plot(work_df_mmc['time'] + optimal_delay_time,
                         work_df_mmc[kinematic_val] + optimal_offset_val)
            else:
                plt.plot(work_df_mmc['time'] + optimal_delay_time, val_work_mmc)
            plt.xlabel("Time (s)")
            plt.ylabel(f"{kinematic_val} (Unit of kinematic)")
            plt.title(f"Optimisation of the time delay for trial {trial_identifier}")
            plt.show()

        if offset:
            return optimal_delay_time, optimal_offset_val
        else:
            return optimal_delay_time

    def handling_vertical_offset(df_omc, df_mmc):
        '''
        Taken and adapted from Marwen Moknis Masterthesis:

        Function that centers the graphs relative to each other (vertically)

        Centers mmc to omc by adding the mean of omc to mmc

        Reference system : df_omc

        '''
        ##Copy of both DataFrames
        df_omc_cp = df_omc.copy()
        df_mmc_cp = df_mmc.copy()

        ##Getting only the kinematic labels
        kinematics = list(df_mmc_cp.columns)[1:] # TODO check that kinematics list is correct

        for kinematic in kinematics:
            val_omc = df_omc_cp[kinematic]
            val_mmc = df_mmc_cp[kinematic]

            mean_omc = np.mean(val_omc)
            mean_mmc = np.mean(val_mmc)

            df_mmc_cp[kinematic] = df_mmc_cp[kinematic] - mean_mmc + mean_omc

        return df_mmc_cp


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
    dir_dat_out = os.path.join(dir_root, '03_data', 'preprocessed_data', '03_fully_preprocessed_dynamic') if correct == 'dynamic' \
        else os.path.join(dir_root, '03_data', 'preprocessed_data', '02_fully_preprocessed')


    csv_outliers = os.path.join(dir_root, '05_logs', 'outliers.csv')

    df_outliers = pd.DataFrame(columns=['id_s', 'id_p', 'id_t', 'reason'])

    if type(detect_outliers) is str:
        detect_outliers = [detect_outliers]
    else:
        detect_outliers = detect_outliers

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

                # Drop last rows if needed
                df_omc, df_mmc = drop_last_rows_if_needed(df_omc, df_mmc)

                if correct == "dynamic":
                    ## Correct on Elbow Angle:
                    kinematic_value = "elbow_flex_pos"
                    delay_time = fixing_decay_and_offset_trial_by_trial(df_omc, df_mmc,
                                                                        kinematic_value, trial_identifier = f'{id_s}_{id_p}_{id_t}',print_able=print_able,
                                                                        plot_debug=plot_debug)

                # Taken from Marwen Moknis Masterthesis
                df_omc['time'] = np.around(
                    np.linspace(0.01, len(df_omc) / 60, num=len(df_omc)), decimals=3)
                df_mmc['time'] = np.around(
                    np.linspace(0.01, len(df_omc) / 60, num=len(df_mmc)), decimals=3)

                if correct == "dynamic":
                    ## shifting the values of kinematics by the nb frames needed
                    time_step = df_omc["time"].iloc[1] - df_omc["time"].iloc[0]
                    frames = int(delay_time / time_step)

                    if frames < 0:
                        nb_frames = -frames
                        df_omc = df_omc[:-nb_frames]
                        df_mmc = df_mmc[nb_frames:]
                        df_mmc["time"] = np.array(df_omc["time"])
                    elif frames > 0:
                        nb_frames = frames
                        df_mmc = df_mmc[:-nb_frames]
                        df_omc = df_omc[nb_frames:]
                        df_omc["time"] = np.array(df_mmc["time"])

                    ## Function that handle the offset
                    df_mmc = handling_vertical_offset(df_omc, df_mmc)
                    # df_mmc = df_mmc

                else:
                    df_mmc = df_mmc

                """# Cut DataFrames to same timeframe
                df_omc, df_mmc = cut_to_same_timeframe(df_omc, df_mmc)"""

                # Detect Outliers
                for detect in  detect_outliers:
                    if detect == 'elbow':
                        max_omc = max(df_omc['elbow_vel'])
                        max_mmc = max(df_mmc['elbow_vel'])
                        if max_omc > max_mmc:
                            reason = 'omc'
                            max_val = max_omc
                        else:
                            reason = 'mmc'
                            max_val = max_mmc

                        if max_val > joint_vel_thresh:
                            print(f"Trial {id_s}_{id_p}_{id_t} is an outlier due to high elbow velocity.")
                            df = pd.DataFrame({'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'reason': f'elbow_vel of {reason} {max_val} > {joint_vel_thresh}'}, index=[0])
                            pd.concat([df_outliers, df], ignore_index=True)
                            continue

                    if detect == 'endeff':
                        max_omc = max(df_omc['hand_vel'])
                        max_mmc = max(df_mmc['hand_vel'])

                        if max_omc > max_mmc:
                            reason = 'omc'
                            max_val = max_omc
                        else:
                            reason = 'mmc'
                            max_val = max_mmc

                        if max_val > hand_vel_thresh:
                            print(f"Trial {id_s}_{id_p}_{id_t} is an outlier due to high endeffector velocity.")
                            df = pd.DataFrame({'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'reason': f'hand_vel {reason} {max_val} > {hand_vel_thresh}'}, index = [0])
                            pd.concat([df_outliers, df], ignore_index=True)
                            continue

                if plot_debug:
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
                    axes[0].plot(df_omc["time"], df_omc["hand_vel"])
                    axes[0].plot(df_mmc["time"], df_mmc["hand_vel"])
                    axes[0].text(np.mean(df_omc["time"]),
                                 np.max(df_mmc["hand_vel"]), f'{id_s}_{id_p}_{id_t}')
                    axes[0].set_xlabel("Time (s)")
                    axes[0].set_ylabel("Hand Velocity mm/s")
                    axes[0].legend(['OMC', 'MMC'])

                    axes[1].plot(df_omc["time"], df_omc["elbow_vel"])
                    axes[1].plot(df_mmc["time"], df_mmc["elbow_vel"])
                    axes[1].text(np.mean(df_omc["time"]),
                                 np.max(df_mmc["elbow_vel"]), f'{id_s}_{id_p}_{id_t}')
                    axes[1].set_xlabel("Time (s)")
                    axes[1].set_ylabel("Elbow Velocity deg/s")
                    axes[1].legend(['OMC', 'MMC'])
                    plt.tight_layout()
                    plt.show()


                omc_nframes_after = len(df_omc)
                mmc_nframes_after = len(df_mmc)

                # Write to new csv
                os.makedirs(dir_dat_out, exist_ok=True)

                appendix = '_dynamic_preprocessed.csv' if correct == 'dynamic' else '_preprocessed.csv'

                path_omc_out = os.path.join(dir_dat_out, f'S15133_{id_p}_{id_t}_preprocessed.csv')
                path_mmc_out = os.path.join(dir_dat_out, f'{id_s}_{id_p}_{id_t}_preprocessed.csv')

                df_omc.to_csv(path_omc_out, sep=';')
                df_mmc.to_csv(path_mmc_out, sep=';')

                if verbose >= 2:
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

def normalize_data(dir_src, dynamic=False, verbose=1):
    """
    iterates over .csv files in directory. It takes all time series data from a given metric. It normalizes the data to a timeframe of 1 and puts them into a new DataFrame.


    df_out: columns - [id_s, id_p, id_t, condition, side, time, omc, mmc]

    csv_out: [metric]_normalized.csv [metric] stands for the metric used in the time series data.

    :param dir_src:
    :return:
    """
    dir_dst = os.path.join(dir_src, '01_normalized')
    os.makedirs(dir_dst, exist_ok=True)

    csv_appendix = '_dynamic_normalized.csv' if dynamic else '_normalized.csv'

    metrics = ['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang',
               'elbow_flex_pos', 'shoulder_flex_pos', 'shoulder_abduction_pos']

    dict_csvs_out = {}
    dict_df_out = {}

    for metric in metrics:
        csv_out = os.path.join(dir_dst, f'{metric}{csv_appendix}')
        df_out = pd.DataFrame(columns=['id_s', 'id_p', 'id_t', 'condition', 'side', 'time_normalized', 'omc', 'mmc'])

        dict_csvs_out[metric] = csv_out
        dict_df_out[metric] = df_out

    id_s_omc = 'S15133'

    csvs_in = glob.glob(os.path.join(dir_src, '*.csv'))
    list_csvs_in_mmc = sorted([csv for csv in csvs_in if id_s_omc not in os.path.basename(csv)])
    list_csvs_in_omc = sorted([csv for csv in csvs_in if id_s_omc in os.path.basename(csv)])

    if verbose >= 1:
        progbar = tqdm(list_csvs_in_mmc, desc='Normalizing', leave=True)

    for csv_mmc in list_csvs_in_mmc:
        id_s = os.path.basename(csv_mmc).split('_')[0]
        id_p = os.path.basename(csv_mmc).split('_')[1]
        id_t = os.path.basename(csv_mmc).split('_')[2]
        condition = os.path.basename(csv_mmc).split('_')[3]
        side = os.path.basename(csv_mmc).split('_')[4]

        if verbose >= 1:
            progbar.set_description(f'Normalizing {id_s}_{id_p}_{id_t}')

        try:
            csv_omc = [csv for csv in list_csvs_in_omc if id_p in os.path.basename(csv) and id_t in os.path.basename(csv)][0]
        except IndexError:
            print(f"No OMC-File found for {id_s}_{id_p}_{id_t}")
            if verbose >= 1:
                progbar.update(1)
            continue

        df_mmc = pd.read_csv(csv_mmc, sep=';')
        df_omc = pd.read_csv(csv_omc, sep=';')

        time = df_mmc['time']
        time_normalized = np.linspace(0, 1, num=len(time))

        for metric in metrics:
            omc = df_omc[metric]
            mmc = df_mmc[metric]

            df_temp = pd.DataFrame({'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'condition': condition, 'side': side,
                                    'time_normalized': time_normalized, 'omc': omc, 'mmc': mmc})

            dict_df_out[metric] = pd.concat([dict_df_out[metric], df_temp], ignore_index=True)

        if verbose >= 1:
            progbar.update(1)

    if verbose >= 1:
        progbar.close()

    for metric in metrics:
        dict_df_out[metric].to_csv(dict_csvs_out[metric], sep=';')


if __name__ == '__main__':
    # this part is for Development and Debugging

    if sys.gettrace() is not None:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

    drive = iDrinkUtilities.get_drivepath()

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
    corrections = ['fixed', 'dynamic']


    if test_timeseries:

        for correct in corrections:

            preprocess_timeseries(root_val,
                                  downsample=True, drop_last_rows=False, detect_outliers= [],
                                  joint_vel_thresh=5, hand_vel_thresh=3000, correct=correct,
                                  verbose=1, plot_debug=False, print_able=False)
            get_omc_mmc_error_old(root_val, path_csv_murphy_timestamps, correct=correct, verbose=1)


    else:

        runs_statistics_discrete(path_csv_murphy_measures, root_stat, thresh_PeakVelocity_mms=None, thresh_elbowVelocity=None)
