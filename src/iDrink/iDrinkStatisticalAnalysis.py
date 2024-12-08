"""
We compare each MMC-Trial Objects time series and Murphy Measures to the same metrics of the OMC-trials.

"""
import os
import sys
import shutil
import glob
import re
from operator import index

from sympy.physics.units import sidereal_year
from tqdm import tqdm
import platform
import pandas as pd
import numpy as np
import scipy as sp
from vtkmodules.numpy_interface.algorithms import condition

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


def delete_existing_files(dir):
    """Deletes all files that are in the given directory"""
    if not os.path.exists(dir):
        return
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path):
            os.remove(os.path.join(dir, file))

def get_murphy_corrrelation(df, root_stat_cat, thresh_PeakVelocity_mms=None, thresh_elbowVelocity=None, verbose=1):
    """Calculates the correlation of Murphy Measures for each setting and participant

    When id_p is None, the correlation accounts for the whole setting

    """
    from scipy.stats import pearsonr

    #cor_columns = ['id_s', 'id_p', 'measure', 'condition', 'side', 'pearson', 'pearson_p']
    corr_columns = ['id_s', 'measure', 'condition', 'pearson', 'pearson_p']

    df_corr = pd.DataFrame(columns=corr_columns)

    id_s_omc = 'S15133'

    df_omc = df[df['id_s'] == id_s_omc]
    df_mmc = df[df['id_s'] != id_s_omc]

    total = len(df_mmc['id_s'].unique()) * len(murphy_measures)
    idx_s = sorted(df_mmc['id_s'].unique())

    progbar = tqdm(total=total, desc="Calculating Correlations", disable=verbose<1)

    for measure in murphy_measures:
        for id_s in idx_s:
            progbar.set_description(f'Calculating Correlations for {id_s}_{measure} \t \t \t \t \t')

            id_p_mmc = df_mmc[df_mmc['id_s'] == id_s]['id_p'].unique()
            id_p_omc = df_omc['id_p'].unique()
            idx_p = sorted(list(set(id_p_mmc).intersection(id_p_omc)))

            df_temp = pd.DataFrame(columns=corr_columns, index = [0])

            df_mmc_s = None
            df_omc_s = None

            for id_p in idx_p:
                df_mmc_p = df_mmc[(df_mmc['id_s']== id_s) & (df_mmc['id_p'] == id_p)]
                df_omc_p = df_omc[df_omc['id_p'] == id_p]

                id_t_mmc = df_mmc_p['id_t'].unique()
                id_t_omc = df_omc_p['id_t'].unique()
                idx_t = sorted(list(set(id_t_mmc).intersection(id_t_omc)))

                df_mmc_t = df_mmc_p[df_mmc_p['id_t'].isin(idx_t)]
                df_omc_t = df_omc_p[df_omc_p['id_t'].isin(idx_t)]

                # sort df_mmc_t and df_omc_t by id_t
                df_mmc_t = df_mmc_t.sort_values(by='id_t')
                df_omc_t = df_omc_t.sort_values(by='id_t')

                # delete duplicate rows
                df_mmc_t = df_mmc_t.drop_duplicates(subset='id_t', keep='first')
                df_omc_t = df_omc_t.drop_duplicates(subset='id_t', keep='first')

                df_mmc_s = df_mmc_t if df_mmc_s is None else pd.concat([df_mmc_s, df_mmc_t])
                df_omc_s = df_omc_t if df_omc_s is None else pd.concat([df_omc_s, df_omc_t])

            if df_mmc_s is not None and df_omc_s is not None:
                df_temp['id_s'] = id_s
                df_temp['measure'] = measure

                for condition in ['affected', 'unaffected']:
                    df_mmc_t = df_mmc_s[df_mmc_s['condition'] == condition]
                    df_omc_t = df_omc_s[df_omc_s['condition'] == condition]
                    df_temp['condition'] = condition

                    try:
                        correlation = pearsonr(df_mmc_t[measure], df_omc_t[measure])
                        df_temp['pearson'] = correlation[0]
                        df_temp['pearson_p'] = correlation[1]
                    except Exception as e:
                        if verbose >= 1:
                            print(f"Error in {os.path.basename(__file__)}.{get_murphy_corrrelation.__name__}\n"
                                  f"Error while calculating correlation for {id_s}_{condition}")
                            print(f"Error:\t{e}")
                            df_temp['pearson'] = None
                            df_temp['pearson_p'] = None

                    df_corr = pd.concat([df_corr, df_temp], ignore_index=True)
            progbar.update(1)
    progbar.close()

    csv_corr = os.path.join(root_stat_cat, 'stat_murphy_corr.csv')
    df_corr.to_csv(csv_corr, sep=';')

    return df_corr

def get_mmc_omc_difference(df, root_stat_cat, thresh_PeakVelocity_mms=3000, verbose=1):
    """
    Creates DataFrame containing the difference between OMC and MMC measurments for each trial.

    :param df:
    :param verbose:
    :return:
    """
    global murphy_measures

    df_diff = pd.DataFrame(columns=['identifier', 'id_s', 'id_p', 'id_t', 'condition', 'side'] + murphy_measures)

    id_s_omc = 'S15133'
    idx_s = sorted(df['id_s'].unique())


    progbar = tqdm(total=sum(len(sublist) for sublist in list(df[df['id_s'] == id_s]['id_p'].unique() for id_s in idx_s)), desc="Calculating Differences", disable=verbose<1)

    for id_s in idx_s:
        idx_p = sorted(list(df[df['id_s'] == id_s]['id_p'].unique()))
        for id_p in idx_p:
            idx_t = sorted(list(df[(df['id_p'] == id_p) & (df['id_s'] == id_s)]['id_t'].unique()))

            progbar.set_description(f'Calculating Differences for {id_s}_{id_p}')

            for id_t in idx_t:
                identifier = f"{id_s}_{id_p}_{id_t}"
                condition = df.loc[df['identifier'] == identifier, 'condition'].values[0]
                side = df.loc[df['identifier'] == identifier, 'side'].values[0]

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

                row_diff = [identifier, id_s, id_p, id_t, condition, side] + list(diff)

                df_diff.loc[df_diff.shape[0]] = row_diff

            progbar.update(1)

    progbar.close()

    return df_diff

def runs_statistics_discrete(path_csv_murphy, root_stat,
                             thresh_PeakVelocity_mms = None, thresh_elbowVelocity=None,
                             make_plots = False, verbose=1):
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

    if thresh_elbowVelocity or thresh_PeakVelocity_mms:
        outlier_corrected = '_outlier_corrected'
    else:
        outlier_corrected = ''


    # Create subset of DataFrame containing all trials that are also in OMC
    df = pd.DataFrame(columns=df_murphy.columns)

    if verbose >= 1:
        progbar = tqdm(total=len(idx_s_mmc), desc='Calculating Differences')
    for id_s in sorted(idx_s_mmc):

        if verbose >= 1:
            progbar.set_description(f'Joining MMC and OMC Data for {id_s}')


        df_s = df_murphy[df_murphy['id_s'] == id_s]
        df_omc = df_murphy[df_murphy['id_s'] == 'S15133']
        idx_p = sorted(list(df_murphy[df_murphy['id_s']==id_s]['id_p'].unique()))

        for id_p in idx_p:
            idx_t = sorted(list(df_murphy[(df_murphy['id_p']==id_p) & (df_murphy['id_s']==id_s )]['id_t'].unique()))
            for id_t in idx_t:
                df = pd.concat([df, df_omc[(df_omc['id_p'] == id_p) & (df_omc['id_t'] == id_t)]])

        df = pd.concat([df_s, df])
        if verbose >= 1:
            progbar.update(1)

    if verbose >= 1:
        progbar.close()

    # TODO: calculate correlation over idxs ans idxs_idxP
    df_corr = get_murphy_corrrelation(df, root_stat_cat,
                                      thresh_PeakVelocity_mms = thresh_PeakVelocity_mms, thresh_elbowVelocity=thresh_elbowVelocity,
                                      verbose=verbose)

    # Create DataFrame containing the differences between MMC and OMC
    df_diff = get_mmc_omc_difference(df, root_stat_cat, thresh_PeakVelocity_mms=thresh_PeakVelocity_mms, verbose=verbose)







    # Create DataFrame and calculate mean of each column
    col_identity = ['id_s', 'id_p', 'condition', 'side']
    id_first_measure = len(col_identity)

    # Create DataFrame containing absolute values of the differences
    df_abs_diff = df_diff.copy()
    df_abs_diff.iloc[:, id_first_measure+2:] = np.abs(df_abs_diff.iloc[:, id_first_measure+2:])

    df_mean = pd.DataFrame(columns=col_identity + murphy_measures)
    df_rmse = pd.DataFrame(columns=col_identity + murphy_measures)


    if verbose >= 1:
        progbar = tqdm(total=len(idx_s_mmc), desc='Calculating Means')


    for id_s in idx_s_mmc:

        if verbose >= 1:
            progbar.set_description(f'Calculating Means for {id_s}')

        idx_p = sorted(list(df_murphy[df_murphy['id_s'] == id_s]['id_p'].unique()))

        for id_p in idx_p:
            # mean Error
            for condition in ['affected', 'unaffected']:
                df_diff_temp = df_diff.loc[(df_diff['id_s'] == id_s) & (df_diff['id_p'] == id_p) & (df_diff['condition']==condition)]
                df_diff_temp_measures = df_diff_temp.iloc[:, id_first_measure+2:]

                side = df_murphy.loc[(df_murphy['id_s'] == id_s) & (df_murphy['id_p'] == id_p), 'side'].values[0]

                # mean for setting over all trials
                df_mean.loc[len(df_mean), 'id_s'] = id_s
                df_mean.loc[len(df_mean)-1, 'id_p'] = id_p
                df_mean.loc[len(df_mean)-1, 'condition'] = condition
                df_mean.loc[len(df_mean)-1, 'side'] = side
                df_mean.iloc[len(df_mean)-1, id_first_measure:] = np.mean(df_diff_temp_measures, axis=0)



                # Root Mean Squared Error
                df_rmse.loc[len(df_rmse), 'id_s'] = id_s
                df_rmse.loc[len(df_rmse) - 1, 'id_p'] = id_p
                df_rmse.loc[len(df_rmse) - 1, 'condition'] = condition
                df_rmse.loc[len(df_rmse) - 1, 'side'] = side
                df_rmse.iloc[len(df_rmse) - 1, id_first_measure:] = np.sqrt(np.mean(df_diff_temp_measures**2, axis=0))

        for condition in ['affected', 'unaffected']:
            df_diff_temp = df_diff.loc[(df_diff['id_s'] == id_s)  & (df_diff['condition'] == condition)]
            df_diff_temp_measures = df_diff_temp.iloc[:, id_first_measure+2:]

            # mean for setting over all participants
            df_mean.loc[len(df_mean), 'id_s'] = id_s
            df_mean.loc[len(df_mean) - 1, 'id_p'] = ''
            df_mean.loc[len(df_mean) - 1, 'condition'] = condition
            df_mean.loc[len(df_mean) - 1, 'side'] = ''
            df_mean.iloc[len(df_mean) - 1, id_first_measure:] = np.mean(df_diff_temp_measures, axis=0)

            df_rmse.loc[len(df_rmse), 'id_s'] = id_s
            df_rmse.loc[len(df_rmse) - 1, 'id_p'] = ''
            df_rmse.loc[len(df_rmse) - 1, 'condition'] = condition
            df_rmse.loc[len(df_rmse) - 1, 'side'] = ''
            df_rmse.iloc[len(df_rmse) - 1, id_first_measure:] = np.sqrt(np.mean(df_diff_temp_measures**2, axis=0))

            if verbose >= 1:
                progbar.update(1)

    if verbose >= 1:
        progbar.close()

    # Write to  csv
    path_csv_murphy_diff = os.path.join(root_stat_cat, f'stat_murphy_diff{outlier_corrected}.csv')
    path_csv_murphy_abs_diff = os.path.join(root_stat_cat, f'stat_murphy_abs_diff{outlier_corrected}.csv')
    path_csv_murphy_mean = os.path.join(root_stat_cat, f'stat_murphy_mean{outlier_corrected}.csv')
    path_csv_murphy_rmse = os.path.join(root_stat_cat, f'stat_murphy_rmse{outlier_corrected}.csv')

    df_diff.to_csv(path_csv_murphy_diff, sep=';')
    df_abs_diff.to_csv(path_csv_murphy_abs_diff, sep=';')
    df_mean.to_csv(path_csv_murphy_mean, sep=';')
    df_rmse.to_csv(path_csv_murphy_rmse, sep=';')

    # Create DataFrame for each trial
    #run_stat_murphy(df, id_s, root_stat_cat, verbose=verbose)

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


def get_error_timeseries(dir_processed, dir_results, empty_dst=False, verbose = 1, debug=False):
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

    if empty_dst:
        delete_existing_files(dir_results)

    for dir_src in list_dir_src:
        dir_dst = os.path.join(dir_results, '01_ts_error')
        os.makedirs(dir_dst, exist_ok=True)

        if empty_dst:
            delete_existing_files(dir_dst)

        id_s_omc = 'S15133'

        csvs_in = glob.glob(os.path.join(dir_src, '*.csv'))
        list_csvs_in_mmc = sorted([csv for csv in csvs_in if id_s_omc not in os.path.basename(csv)])
        list_csvs_in_omc = sorted([csv for csv in csvs_in if id_s_omc in os.path.basename(csv)])

        # get all s_ids from list_csvs_in_mmc
        idx_s = sorted(list(set([os.path.basename(file).split('_')[0] for file in list_csvs_in_mmc])))

        progbar = tqdm(total=len(list_csvs_in_mmc), desc='Calculating Timeseries Error', disable=verbose<1)

        for i, csv_mmc in enumerate(list_csvs_in_mmc):
            id_s = os.path.basename(csv_mmc).split('_')[0]
            id_p = os.path.basename(csv_mmc).split('_')[1]
            id_t = os.path.basename(csv_mmc).split('_')[2]
            condition = os.path.basename(csv_mmc).split('_')[3]
            side = os.path.basename(csv_mmc).split('_')[4]
            dynamic = os.path.basename(csv_mmc).split('_')[5]

            if debug and i > 50:
                break

            progbar.set_description(f'Calculating Timeseries Error for {dynamic} {id_s}_{id_p}_{id_t}')

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

            df_out_rom_temp = pd.DataFrame({
                'id_s': id_s,
                'id_p': id_p,
                'id_t': id_t,
                'condition': condition,
                'dynamic': dynamic,
                **{f'{metric}_min_omc': None for metric in metrics},
                **{f'{metric}_max_omc': None for metric in metrics},
                **{f'{metric}_rom_omc': None for metric in metrics},
                **{f'{metric}_min_mmc': None for metric in metrics},
                **{f'{metric}_max_mmc': None for metric in metrics},
                **{f'{metric}_rom_mmc': None for metric in metrics},
                **{f'{metric}_rom_error': None for metric in metrics},
            }, index = [0])


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

            if verbose >= 1:
                progbar.update(1)

        if verbose >= 1:
            progbar.close()


def get_error_mean_rmse(dir_results, overwrite_csvs=False, verbose=1):
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

    df_out = pd.DataFrame(columns=['id', 'id_s', 'id_p', 'id_t', 'condition', 'dynamic', 'normalized', 'metric', 'mean',
                                   'median', 'std', 'rmse', 'rmse_std'])



    dir_dst = os.path.join(dir_results, '02_ts_error_mean_rmse')
    os.makedirs(dir_dst, exist_ok=True)

    id_s_omc = 'S15133'
    csvs_in = glob.glob(os.path.join(dir_src, '*.csv'))
    list_csv = sorted([csv for csv in csvs_in if id_s_omc not in os.path.basename(csv)])

    if verbose >= 1:
        total_count = len(list_csv)
        progbar = tqdm(total=total_count, desc='Calculating Mean and RMSE', disable=verbose<1)

    for csv in list_csv:
        id_p = os.path.basename(csv).split('_')[0]
        id_t = os.path.basename(csv).split('_')[1]
        condition = os.path.basename(csv).split('_')[2]
        side = os.path.basename(csv).split('_')[3]
        normalized = 'normalized' if 'norm' in os.path.basename(csv) else 'original'

        df = pd.read_csv(csv, sep=';')


        if verbose >= 1:
            total_count += len(df['id_s'].unique())
            progbar.total = total_count
            progbar.refresh()

        for id_s in df['id_s'].unique():
            df_s = df[df['id_s'] == id_s]
            id = f'{id_s}_{id_p}_{id_t}'

            if verbose >= 1:
                progbar.set_description(f'Calculating Mean and RMSE: {id_s}_{id_p}_{id_t}')

            for dynamic in df_s['dynamic'].unique():



                for metric in metrics:

                    mean = np.nanmean(df_s[f'{metric}_error'])
                    median = np.nanmedian(df_s[f'{metric}_error'])
                    std = np.nanstd(df_s[f'{metric}_error'])
                    rmse = np.sqrt(np.nanmean(np.square(df_s[f'{metric}_error'])))
                    rmse_std = np.nanstd(df_s[f'{metric}_rse'])

                    df_mean = pd.DataFrame({
                        'id': id,
                        'id_s': id_s,
                        'id_p': id_p,
                        'id_t': id_t,
                        'condition': condition,
                        'dynamic': dynamic,
                        'normalized': normalized,
                        'metric': metric,
                        'mean': mean,
                        'median': median,
                        'std': std,
                        'rmse': rmse,
                        'rmse_std': rmse_std
                    }, index=[0])

                    df_out = pd.concat([df_out, df_mean], axis=0, ignore_index=True)

            if verbose >= 1:
                progbar.update(1)

    if verbose >= 1:
        progbar.close()

    # safetysave
    df_out.to_csv(csv_out, sep=';', index=False)

    idx = df_out['id'].unique()

    # get sets for id_s and id_p
    idx_s = sorted(list(set([id.split('_')[0] for id in idx if len(id.split('_')) == 3])))
    idx_p = sorted(list(set([id.split('_')[1] for id in idx if len(id.split('_')) == 3])))

    df_mean_s = pd.DataFrame(columns=['id', 'id_s', 'id_p', 'id_t', 'condition', 'dynamic', 'normalized', 'metric',
                                      'mean', 'median', 'std', 'rmse', 'rmse_std'])

    df_mean_p = pd.DataFrame(columns=['id', 'id_s', 'id_p', 'id_t', 'condition', 'dynamic', 'normalized', 'metric',
                                      'mean', 'median', 'std', 'rmse', 'rmse_std'])

    def get_mean_error_for_s_and_p(df_in, df_mean, metrics, id):
        if df_in.shape[0] == 0:
            return df_mean

        if len(id.split('_')) == 2:
            id_s = id.split('_')[0]
            id_p = id.split('_')[1]
            id_t = None
        else:
            id_s = id
            id_p = None
            id_t = None

        for dynamic in df_in['dynamic'].unique():
            df_dyn = df_in[df_in['dynamic'] == dynamic]

            for normalized in df_dyn['normalized'].unique():
                df_norm = df_dyn[df_dyn['normalized'] == normalized]

                for condition in df_norm['condition'].unique():
                    df_temp = df_norm[df_norm['condition'] == condition]

                    for metric in metrics:
                        df_temp_metric = df_temp[df_temp['metric'] == metric]
                        mean = np.nanmean(df_temp_metric[f'mean'])
                        median = np.nanmean(df_temp_metric[f'median'])
                        std = np.nanmean(df_temp_metric[f'std'])
                        rmse = np.nanmean(df_temp_metric[f'rmse'])
                        rmse_std = np.nanmean(df_temp_metric[f'rmse_std'])

                        df_t_out = pd.DataFrame(columns = df_mean.columns, index=[0])
                        df_t_out['id'] = id
                        df_t_out['id_s'] = id_s
                        df_t_out['id_p'] = id_p
                        df_t_out['id_t'] = id_t
                        df_t_out['condition'] = condition
                        df_t_out['dynamic'] = dynamic
                        df_t_out['normalized'] = normalized
                        df_t_out['metric'] = metric
                        df_t_out['mean'] = mean
                        df_t_out['median'] = median
                        df_t_out['std'] = std
                        df_t_out['rmse'] = rmse
                        df_t_out['rmse_std'] = rmse_std

                        df_mean = pd.concat([df_mean, df_t_out], axis=0, ignore_index=True)

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


def get_rom_rmse(dir_results, overwrite_csvs=False, verbose=1):
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


    def get_mean_error_for_s_and_p(df_in, df_mean, metrics, id):
        if df_in.shape[0] == 0:
            return df_mean

        for dynamic in df_in['dynamic'].unique():
            df_dyn = df_in[df_in['dynamic'] == dynamic]

            for condition in df_dyn['condition'].unique():
                df_con = df_dyn[df_dyn['condition'] == condition]

                df_t_out = pd.DataFrame(columns=df_mean.columns, index=[0])
                df_t_out['id'] = id
                df_t_out['condition'] = condition
                df_t_out['dynamic'] = dynamic

                for metric in metrics:
                    error_max = df_con[f'{metric}_max_mmc'] - df_con[f'{metric}_max_omc']
                    error_min = df_con[f'{metric}_min_mmc'] - df_con[f'{metric}_min_omc']
                    error_rom = df_con[f'{metric}_rom_mmc'] - df_con[f'{metric}_rom_omc']

                    df_t_out[f'{metric}_max_rmse'] = np.sqrt(np.nanmean( error_max**2))
                    df_t_out[f'{metric}_min_rmse'] = np.sqrt(np.nanmean( error_min**2))
                    df_t_out[f'{metric}_rom_rmse'] = np.sqrt(np.nanmean( error_rom**2))

                df_mean = pd.concat([df_mean, df_t_out], axis=0, ignore_index=True)

        return df_mean

    # get sets for id_s and id_p
    idx_s = sorted(df_in['id_s'].unique())
    idx_p = sorted(df_in['id_p'].unique())


    for id_s in idx_s:
        df_s_temp = get_mean_error_for_s_and_p(df_in[df_in['id_s'] == id_s], df_s_temp, metrics, id_s)

        for id_p in idx_p:
            df_p_temp = get_mean_error_for_s_and_p(df_in[(df_in['id_s'] == id_s) & (df_in['id_p'] == id_p) ],
                                                   df_p_temp, metrics, f'{id_s}_{id_p}')

    if os.path.isfile(csv_out) and not overwrite_csvs:
        df_out = pd.read_csv(csv_out, sep=';')
        df_out = pd.concat([df_out, df_s_temp, df_p_temp], axis=0, ignore_index=True)
    else:
        df_out = pd.concat([df_s_temp, df_p_temp], axis=0, ignore_index=True)

    df_out.to_csv(csv_out, sep=';', index=False)

def get_rom_rmse_old(dir_results, overwrite_csvs=False, verbose=1):
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


    def get_mean_error_for_s_and_p(df_in, df_mean, metrics, id):
        if df_in.shape[0] == 0:
            return df_mean

        for dynamic in df_in['dynamic'].unique():
            df_dyn = df_in[df_in['dynamic'] == dynamic]

            for condition in df_dyn['condition'].unique():
                df_con = df_dyn[df_dyn['condition'] == condition]

                df_t_out = pd.DataFrame(columns=df_mean.columns, index=[0])
                df_t_out['id'] = id
                df_t_out['condition'] = condition
                df_t_out['dynamic'] = dynamic

                for metric in metrics:
                    error_max = df_con[f'{metric}_max_mmc'] - df_con[f'{metric}_max_omc']
                    error_min = df_con[f'{metric}_min_mmc'] - df_con[f'{metric}_min_omc']
                    error_rom = df_con[f'{metric}_rom_mmc'] - df_con[f'{metric}_rom_omc']

                    df_t_out[f'{metric}_max_rmse'] = np.sqrt(np.nanmean( error_max**2))
                    df_t_out[f'{metric}_min_rmse'] = np.sqrt(np.nanmean( error_min**2))
                    df_t_out[f'{metric}_rom_rmse'] = np.sqrt(np.nanmean( error_rom**2))

                df_mean = pd.concat([df_mean, df_t_out], axis=0, ignore_index=True)

        return df_mean

    # get sets for id_s and id_p
    idx_s = sorted(df_in['id_s'].unique())
    idx_p = sorted(df_in['id_p'].unique())


    for id_s in idx_s:
        df_s_temp = get_mean_error_for_s_and_p(df_in[df_in['id_s'] == id_s], df_s_temp, metrics, id_s)

        for id_p in idx_p:
            df_p_temp = get_mean_error_for_s_and_p(df_in[(df_in['id_s'] == id_s) & (df_in['id_p'] == id_p) ],
                                                   df_p_temp, metrics, f'{id_s}_{id_p}')

    if os.path.isfile(csv_out) and not overwrite_csvs:
        df_out = pd.read_csv(csv_out, sep=';')
        df_out = pd.concat([df_out, df_s_temp, df_p_temp], axis=0, ignore_index=True)
    else:
        df_out = pd.concat([df_s_temp, df_p_temp], axis=0, ignore_index=True)

    df_out.to_csv(csv_out, sep=';', index=False)

def get_rom_rmse(dir_results, overwrite_csvs=False, verbose=1):
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

    df_p_temp = pd.DataFrame(columns=['id_s', 'id_p', 'dynamic', 'condition', 'metric', 'rom_rmse'])
    df_s_temp = pd.DataFrame(columns=['id_s', 'id_p', 'dynamic', 'condition', 'metric', 'rom_rmse'])

    if os.path.isfile(csv_in):
        df_in = pd.read_csv(csv_in, sep=';')
    else:
        raise FileNotFoundError(f"File not found: {csv_in}")


    def get_mean_error_for_s_and_p(df_in, df_mean, metrics, id):
        if df_in.shape[0] == 0:
            return df_mean

        for dynamic in df_in['dynamic'].unique():
            df_dyn = df_in[df_in['dynamic'] == dynamic]

            for condition in df_dyn['condition'].unique():
                df_con = df_dyn[df_dyn['condition'] == condition]

                df_t_out = pd.DataFrame(columns=df_mean.columns, index=[0])
                df_t_out['id'] = id
                df_t_out['condition'] = condition
                df_t_out['dynamic'] = dynamic

                for metric in metrics:
                    error_max = df_con[f'{metric}_max_mmc'] - df_con[f'{metric}_max_omc']
                    error_min = df_con[f'{metric}_min_mmc'] - df_con[f'{metric}_min_omc']
                    error_rom = df_con[f'{metric}_rom_mmc'] - df_con[f'{metric}_rom_omc']

                    df_t_out[f'{metric}_max_rmse'] = np.sqrt(np.nanmean( error_max**2))
                    df_t_out[f'{metric}_min_rmse'] = np.sqrt(np.nanmean( error_min**2))
                    df_t_out[f'{metric}_rom_rmse'] = np.sqrt(np.nanmean( error_rom**2))

                df_mean = pd.concat([df_mean, df_t_out], axis=0, ignore_index=True)

        return df_mean

    df_in.dropna(axis=1, inplace=True)
    # get sets for id_s and id_p
    idx_s = sorted(df_in['id_s'].unique())


    if os.path.isfile(csv_out) and not overwrite_csvs:
        df_out = pd.read_csv(csv_out, sep=';')
        df_out = pd.concat([df_out, df_s_temp, df_p_temp], axis=0, ignore_index=True)
    else:
        df_out = pd.concat([df_s_temp, df_p_temp], axis=0, ignore_index=True)

    total = 0
    for id_s in idx_s:
        df_temp_s = df_in[df_in['id_s'] == id_s]
        idx_p = sorted(df_temp_s['id_p'].unique())
        total += len(idx_p) * len(metrics) * 2 * 2



    progbar = tqdm(total=total, desc='Calculating RMSE', disable=verbose<1)
    for id_s in idx_s:
        df_temp_s = df_in[df_in['id_s'] == id_s]

        idx_p = sorted(df_temp_s['id_p'].unique())

        for id_p in idx_p:
            progbar.set_description(f'Calculating RMSE for {id_s}_{id_p}')

            df_temp_p = df_temp_s[df_temp_s['id_p'] == id_p]
            for metric in metrics:
                col = f'{metric}_rom_error'
                for dynamic in ['fixed', 'dynamic']:
                    for condition in ['affected', 'unaffected']:
                        df_t = df_temp_p[(df_temp_p['dynamic'] == dynamic) & (df_temp_p['condition'] == condition)]

                        df_new_row = pd.DataFrame(columns = df_p_temp.columns, index=[0])
                        df_new_row['id_s'] = id_s
                        df_new_row['id_p'] = id_p
                        df_new_row['dynamic'] = dynamic
                        df_new_row['condition'] = condition
                        df_new_row['metric'] = metric
                        df_new_row['rom_rmse'] = np.sqrt(np.nanmean(df_t[col]**2))

                        df_out = pd.concat([df_out, df_new_row], axis=0, ignore_index=True)

                        progbar .update(1)

    progbar.close()
    df_out.to_csv(csv_out, sep=';', index=False)

def get_timeseries_correlations(dir_processed, dir_results,overwrite_csvs=False, verbose=1):
    """
    Uses normalized timeseries to calculate Pearson Correlation and CMC of mmc and omc for each kinematic metric


    csv_path: time_series_correlation.csv

    df_out: columns: [id_s, id_p, id_t, dynamic, {metric}_pearson , {metric}_pearson_p, {metric}_cmc, {metric}_cmc_p]


    :param dir_results:
    :param verbose:
    :return:
    """
    from scipy.stats import pearsonr

    def join_metric_dataFrames(dir_src, metrics, appendix):
        """
        Read in all metric DataFrames and join them in a single Dataframe based on id_s, id_p and id_t

        :param dir_src:
        :param metrics:
        :param appendix:
        :return:
        """

        def check_if_columns_are_same(columns, df1, df2):
            """
            Check if all rows of all columns are identical in two DataFrames

            :param columns:
            :param df1:
            :param df2:
            :return:
            """

            for column in columns:
                if not np.all(df1[column] == df2[column]):
                    return False
            return True

        columns_to_check = ['id_s', 'id_p', 'id_t', 'condition', 'side', 'time_normalized']

        df_out = pd.DataFrame(
            columns=['id_s', 'id_p', 'id_t', 'condition', 'side', 'time_normalized'] +
                    [f'{metric}_omc' for metric in metrics] +
                    [f'{metric}_mmc' for metric in metrics])

        df_old = None
        progbar = tqdm(total=len(metrics), desc='Joining Metric DataFrames', disable=verbose < 1)
        for metric in metrics:
            progbar.set_description(f'Joining Metric DataFrames: {metric}')
            csv_in = os.path.join(dir_src, f'{metric}{appendix}')

            if not os.path.isfile(csv_in):
                raise FileNotFoundError(f"File not found: {csv_in}")

            df_in = pd.read_csv(csv_in, sep=';')

            df_out[f'{metric}_omc'] = df_in['omc'].values
            df_out[f'{metric}_mmc'] = df_in['mmc'].values

            df_out['time_normalized'] = df_in['time_normalized'].values
            df_out['id_s'] = df_in['id_s'].values
            df_out['id_p'] = df_in['id_p'].values
            df_out['id_t'] = df_in['id_t'].values
            df_out['condition'] = df_in['condition'].values
            df_out['side'] = df_in['side'].values

            if df_old is not None:
                progbar.set_description(f'Joining Metric DataFrames: Checking Columns')
                if not check_if_columns_are_same(columns_to_check, df_out, df_old):
                    raise ValueError(f"Columns do not match: {columns_to_check}")

            df_old = df_out

            progbar.update(1)
        progbar.close()

        return df_out

    metrics = ['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang',
               'elbow_flex_pos', 'shoulder_flex_pos', 'shoulder_abduction_pos']

    dict_dir_src = {'fixed': os.path.join(dir_processed, '02_fully_preprocessed', '01_normalized'),
                   'dynamic': os.path.join(dir_processed, '03_fully_preprocessed_dynamic', '01_normalized')}

    csv_out = os.path.join(dir_results, 'time_series_correlation.csv')

    if os.path.isfile(csv_out) and not overwrite_csvs:
        if verbose >= 1:
            print(f"Reading existing csv file: {csv_out}")
        df_out = pd.read_csv(csv_out, sep=';')
    else:
        if verbose >= 1:
            print(f"Creating new csv file: {csv_out}")
        df_out = pd.DataFrame(columns=['id_s', 'id_p', 'id_t', 'condition', 'side', 'dynamic', 'metric', 'pearson', 'pearson_p'] )

    progbar = None

    for dynamic in ['fixed', 'dynamic']:

        dir_src = dict_dir_src[dynamic]

        if dynamic == 'dynamic':
            appendix = '_dynamic_normalized.csv'
        else:
            appendix = '_fixed_normalized.csv'

        df_in = join_metric_dataFrames(dir_src, metrics, appendix)

        if verbose >= 1:
            print("getting id_sets")

        id_sets = sorted(list(set(tuple(row) for row in df_in[['id_s', 'id_p', 'id_t']].to_records(index=False))))

        if verbose >= 1:
            if progbar is None:
                progbar = tqdm(total=len(id_sets), desc=f'Calculating Correlations for {dynamic}', disable=verbose < 1)

        for i, id_set in enumerate(id_sets):
            id_s = id_set[0]
            id_p = id_set[1]
            id_t = id_set[2]

            """df_out_temp = pd.DataFrame(columns=['id_s', 'id_p', 'id_t', 'dynamic'] + [f'{metric}_pearson' for metric in metrics] +
                                        [f'{metric}_pearson_p' for metric in metrics], index=[0])"""

            df_out_temp = pd.DataFrame(
                columns=['id_s', 'id_p', 'id_t', 'condition', 'side', 'dynamic', 'metric', 'pearson', 'pearson_p'], index=[0])

            df_out_temp['id_s'] = id_s
            df_out_temp['id_p'] = id_p
            df_out_temp['id_t'] = id_t
            df_out_temp['dynamic'] = dynamic

            for metric in metrics:
                progbar.set_description(f'Calculating Correlations for {dynamic} \t {id_s}_{id_p}_{id_t} \t{metric} \t \t \t \t {i}')

                # check if row already exists
                if len(df_out[(df_out['id_s'] == id_s) & (df_out['id_p'] == id_p) & (df_out['id_t'] == id_t) & (df_out['dynamic'] == dynamic) & (df_out['metric'] == metric)]) > 0:
                    continue

                df_temp = df_in[(df_in['id_s'] == id_s) & (df_in['id_p'] == id_p) & (df_in['id_t'] == id_t)]

                condition = df_temp['condition'].values[0]
                side = df_temp['side'].values[0]

                # Get correlation
                omc = df_temp[f'{metric}_omc'].values.tolist()
                mmc = df_temp[f'{metric}_mmc'].values.tolist()
                pearson = pearsonr(omc, mmc)

                df_out_temp['condition'] = condition
                df_out_temp['side'] = side
                df_out_temp['metric'] = metric
                df_out_temp['pearson'] = pearson[0]
                df_out_temp['pearson_p'] = pearson[1]

                df_out = pd.concat([df_out, df_out_temp], axis=0, ignore_index=True)

            progbar.update(1)

            progbar.set_description(
                f'Calculating Correlations for {dynamic} \t {id_s}_{id_p}_{id_t} \t {metric} \t {i} \t safetywrite')

            if i % 50 == 0:
                df_out.to_csv(csv_out, sep=';', index=False)
        progbar.close()
        progbar = None

    df_out.to_csv(csv_out, sep=';', index=False)


def get_multiple_correlations(dir_processed, dir_results, verbose=1):
    """
    Calculates Coefficient of multiple correlations (CMC) for all kinematic metrics.

    csv_path: time_series_multiple_correlation.csv

    df_out: columns: [id_s, id_p, id_t, dynamic, {metric}_pearson , {metric}_pearson_p, {metric}_cmc, {metric}_cmc_p]

    :param dir_processed:
    :param dir_results:
    :param verbose:
    :return:
    """
    metrics = ['hand_vel', 'elbow_vel', 'trunk_disp', 'trunk_ang',
               'elbow_flex_pos', 'shoulder_flex_pos', 'shoulder_abduction_pos']

    dict_dir_src = {'fixed': os.path.join(dir_processed, '02_fully_preprocessed', '01_normalized'),
                   'dynamic': os.path.join(dir_processed, '02_fully_preprocessed_dynamic', '01_normalized')}

    csv_out = os.path.join(dir_results, 'time_series_multiple_correlation.csv')

    if os.path.isfile(csv_out):
        df_out = pd.read_csv(csv_out, sep=';')
    else:
        df_out = pd.DataFrame(columns=['id_s', 'id_p', 'id_t', 'dynamic'] + [f'{metric}_pearson' for metric in metrics] +
                               [f'{metric}_pearson_p' for metric in metrics] )

    for dynamic in ['fixed', 'dynamic']:
        dir_src = dict_dir_src[dynamic]

        if dynamic == 'dynamic':
            appendix = '_dynamic_normalized.csv'
        else:
            appendix = '_normalized.csv'

        for metric in metrics:

            csv_in = os.path.join(dir_src, f'{metric}{appendix}.csv')

            if not os.path.isfile(csv_in):
                raise FileNotFoundError(f"File not found: {csv_in}")

            df_in = pd.read_csv(csv_in, sep=';')


            id_sets = list(set(tuple(row) for row in df_in[['ids', 'idp']].to_records(index=False)))

            idx_s = df_in['ids'].unique()

            # TODO: Write if there is enough time




def preprocess_timeseries(dir_root, downsample = True, drop_last_rows = False, correct='fixed', detect_outliers = [],
                          joint_vel_thresh = 5, hand_vel_thresh = 3000, fancy_offset = True,
                          verbose=1, plot_debug=False, print_able=False,
                          empty_dst=False,  debug=False, debug_c=20):
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
                print(f"Those are the optimal delay time {optimal_delay_time} and offset {optimal_offset_val} for trial {trial_identifier} ")
            else:
                print(f"This is the optimal delay time {optimal_delay_time} for trial {trial_identifier} ")

        if not (offset):
            ## We try the method of just taking out the mean of sys_2 and adding by the mean of the reference system
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
            plt.legend(["Reference system", "System to optimise"])
            plt.show()

        if offset:
            return optimal_delay_time, optimal_offset_val
        else:
            return optimal_delay_time

    def handling_vertical_offset(df_omc, df_mmc, fancy=False):
        '''
        Taken and adapted from Marwen Moknis Masterthesis:

        Function that centers the graphs relative to each other (vertically)

        Centers mmc to omc by adding the mean of omc to mmc

        Reference system : df_omc

        '''
        def do_it_fancy(work_df_mmc, work_df_omc, kinematics, dict_offsets):
            from scipy.optimize import curve_fit

            def reference_function(time, delay_time, offset):
                return np.interp(time + delay_time, work_df_omc['time'], work_df_omc[kinematic] + offset)

            for kinematic in kinematics:
                try:
                    popt, _ = curve_fit(reference_function, work_df_mmc['time'], work_df_mmc[kinematic], maxfev=5000)

                    ## Getting the value we did the optimisation on
                    optimal_delay_time = popt[0]
                    offset_val = popt[1]

                    dict_offsets[kinematic] = offset_val
                except Exception as e:
                    print(f"Error in curve_fit: {e}")

                    # If curve_fit fails, use unfancy_method
                    val_omc = work_df_omc[kinematic]
                    val_mmc = work_df_mmc[kinematic]

                    mean_omc = np.mean(val_omc)
                    mean_mmc = np.mean(val_mmc)

                    dict_offsets[kinematic] = mean_mmc + mean_omc

                work_df_mmc[kinematic] = work_df_mmc[kinematic] + offset_val

            return work_df_mmc, dict_offsets


        ##Copy of both DataFrames
        df_omc_cp = df_omc.copy()
        df_mmc_cp = df_mmc.copy()

        ##Getting only the kinematic labels
        kinematics = list(df_mmc_cp.columns)[1:] # TODO check that kinematics list is correct

        dict_offsets = {key: None for key in kinematics}

        if fancy:
            df_mmc_cp, dict_offsets = do_it_fancy(df_mmc_cp, df_omc, kinematics, dict_offsets)
        else:
            for kinematic in kinematics:
                val_omc = df_omc_cp[kinematic]
                val_mmc = df_mmc_cp[kinematic]

                mean_omc = np.mean(val_omc)
                mean_mmc = np.mean(val_mmc)

                df_mmc_cp[kinematic] = df_mmc_cp[kinematic] - mean_mmc + mean_omc

                dict_offsets[kinematic] = mean_mmc + mean_omc

        return df_mmc_cp, dict_offsets


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

    def update_offset_csv(csv_offset, id_s, id_p, id_t, dict_offsets):
        """
        Updates or writes csv file with offsets for each trial and kinematic

        :param csv_offset:
        :param id_s:
        :param id_p:
        :param id_t:
        :param dict_offsets:
        :return:
        """
        kinematics = list(dict_offsets.keys())

        if os.path.isfile(csv_offset):
            df_offset = pd.read_csv(csv_offset, sep=';')
        else:
            df_offset = pd.DataFrame(columns=['id_s', 'id_p', 'id_t'] + kinematics)

        dict_new_row = {'id_s': id_s, 'id_p': id_p, 'id_t': id_t, **dict_offsets}

        if len(df_offset[(df_offset['id_s'] == id_s) & (df_offset['id_p'] == id_p) & (df_offset['id_t'] == id_t)]) == 0:
            df_offset = pd.concat([df_offset, pd.DataFrame(dict_new_row, index=[0])], ignore_index=True)
        else:
            #df_offset.loc[(df_offset['id_s'] == id_s) & (df_offset['id_p'] == id_p) & (df_offset['id_t'] == id_t)] = list(dict_new_row.values())
            for key in dict_new_row.keys():
                df_offset.loc[(df_offset['id_s'] == id_s) & (df_offset['id_p'] == id_p) & (df_offset['id_t'] == id_t), key] = dict_new_row[key]

        df_offset.to_csv(csv_offset, sep=';', index=False)


    dir_dat_in = os.path.join(dir_root, '03_data', 'preprocessed_data', '01_murphy_out')
    dir_dat_out = os.path.join(dir_root, '03_data', 'preprocessed_data', '03_fully_preprocessed_dynamic') if correct == 'dynamic' \
        else os.path.join(dir_root, '03_data', 'preprocessed_data', '02_fully_preprocessed')

    if empty_dst:
        delete_existing_files(dir_dat_out)


    csv_outliers = os.path.join(dir_root, '05_logs', 'outliers.csv')
    csv_offset = os.path.join(dir_root, '03_data', 'preprocessed_data', 'offset.csv')

    if os.path.isfile(csv_outliers):
        df_outliers = pd.read_csv(csv_outliers, sep=';')
    else:
        df_outliers = pd.DataFrame(columns=['id_s', 'id_p', 'id_t', 'dynamic', 'detect_on', 'value_endeff',
                                            'value_elbow', 'thresh_endeff_vel', 'thresh_elbow_vel', 'reason_endeff',
                                            'reason_elbow'])

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

    total_count = len(p_ids)

    if verbose >=1:
        def get_total_for_process():
            total = 0
            for id_p in p_ids:
                omc_csvs_p = [omc_csv for omc_csv in omc_csvs if id_p in os.path.basename(omc_csv)]
                t_ids = [os.path.basename(omc_csv).split('_')[2] for omc_csv in omc_csvs_p]
                total += len(t_ids)
            return total

        if debug:
            total_count = len(p_ids) * debug_c
        else:

            total_count = get_total_for_process()

        process = tqdm(range(total_count), desc='Preprocessing', leave=True)

    for id_p in p_ids:
        omc_csvs_p = [omc_csv for omc_csv in omc_csvs if id_p in os.path.basename(omc_csv)]

        t_ids = [os.path.basename(omc_csv).split('_')[2] for omc_csv in omc_csvs_p]

        # check if MMC recording exists for id_p
        found_files = []
        for id_s in s_ids:
            found_files.extend(glob.glob(os.path.join(dir_dat_in, f'{id_s}*{id_p}_*.csv')))

        if len(found_files) == 0:
            continue

        if debug:
            debug_count = 0

        for id_t in t_ids:
            if verbose >= 1:
                process.set_description(f'Preprocessing {correct} - {id_p}_{id_t}')
                process.update(1)

            if debug:
                debug_count += 1
                if debug_count > debug_c:
                    break

            omc_csv = glob.glob(os.path.join(dir_dat_in, f'S15133_{id_p}_{id_t}*.csv'))[0]

            mmc_files = []
            for id_s in s_ids:
                found_file = glob.glob(os.path.join(dir_dat_in, f'{id_s}*{id_p}_{id_t}*.csv'))
                if len(found_file) > 0:
                    mmc_files.append(found_file[0])

            for mmc_csv in mmc_files:
                id_s = os.path.basename(mmc_csv).split('_')[0]

                condition = os.path.basename(mmc_csv).split('_')[3]
                side = os.path.basename(mmc_csv).split('_')[4]

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
                                                                        plot_debug=plot_debug, offset=False)

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

                    ## Function that handles the offset
                    df_mmc, dict_offsets = handling_vertical_offset(df_omc, df_mmc, fancy=fancy_offset)
                    # df_mmc = df_mmc

                    update_offset_csv(csv_offset, id_s, id_p, id_t, dict_offsets)

                else:
                    df_mmc = df_mmc

                """# Cut DataFrames to same timeframe
                df_omc, df_mmc = cut_to_same_timeframe(df_omc, df_mmc)"""

                # Detect Outliers
                if detect_outliers:
                    def update_outlier(df_outliers, dict_new_row):
                        """Updates the outlier DataFrame based on id_s, id_p, and id_t

                        If row with the Ids exists, it is overwritten, else a new row is added to the DataFrame

                        """
                        id_s = dict_new_row['id_s']
                        id_p = dict_new_row['id_p']
                        id_t = dict_new_row['id_t']
                        dynamic = dict_new_row['dynamic']

                        keys = ['value_endeff', 'value_elbow', 'thresh_endeff_vel', 'thresh_elbow_vel',
                                'reason_endeff', 'reason_elbow']

                        if len(df_outliers[(df_outliers['id_s'] == id_s) & (df_outliers['id_p'] == id_p) &
                                           (df_outliers['id_t'] == id_t) & (df_outliers['dynamic'] == dynamic)]) == 0:
                            df_outliers = pd.concat([df_outliers, pd.DataFrame(dict_new_row, index=[0])], ignore_index=True)
                        else:
                            for key in dict_new_row.keys():
                                df_outliers.loc[(df_outliers['id_s'] == id_s) & (df_outliers['id_p'] == id_p) &
                                           (df_outliers['id_t'] == id_t) & (df_outliers['dynamic'] == dynamic), key] = dict_new_row[key]

                        return df_outliers

                    max_omc = max(df_omc['elbow_vel'])
                    max_mmc = max(df_mmc['elbow_vel'])
                    if max_omc > max_mmc:
                        reason_elbow = 'omc'
                        max_val_elbow = max_omc
                    elif max_omc < max_mmc:
                        reason_elbow = 'mmc'
                        max_val_elbow = max_mmc

                    if max_omc > max_mmc:
                        reason_endeff = 'omc'
                        max_val_endeff = max_omc
                    else:
                        reason_endeff = 'mmc'
                        max_val_endeff = max_mmc

                    if 'elbow' in detect_outliers and 'endeff' in detect_outliers:
                        if max_val_elbow > joint_vel_thresh and max_val_endeff > hand_vel_thresh:
                            print(f"Trial {id_s}_{id_p}_{id_t} is an outlier due to high elbow and endeffector velocity.\n"
                                  f"Elbow:\tMax Value: {max_val_elbow} deg/s\tThreshold: {joint_vel_thresh} deg/s\n"
                                  f"Endeffector\tMax Value: {max_val_endeff} mm/s\tThreshold: {hand_vel_thresh} mm/s")
                            dict_new_row = {'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'dynamic': correct,
                                            'detect_on': 'elbow and endeffector', 'value_endeff': max_val_endeff,
                                            'value_elbow': max_val_elbow, 'thresh_endeff_vel': hand_vel_thresh,
                                            'thresh_elbow_vel': joint_vel_thresh, 'reason_endeff': reason_endeff,
                                            'reason_elbow': reason_elbow}
                            df_outliers = update_outlier(df_outliers, dict_new_row)

                    if 'elbow' in detect_outliers:
                        if max_val_elbow > joint_vel_thresh:
                            print(f"Trial {id_s}_{id_p}_{id_t} is an outlier due to high elbow velocity.\n"
                                  f"Max. Elbow Velocity: {max_val_elbow} deg/s\tThreshold: {joint_vel_thresh} deg/s")
                            dict_new_row = {'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'dynamic': correct,
                                            'detect_on': 'elbow', 'value_endeff': max_val_endeff,
                                            'value_elbow': max_val_elbow, 'thresh_endeff_vel': hand_vel_thresh,
                                            'thresh_elbow_vel': joint_vel_thresh, 'reason_endeff': '',
                                            'reason_elbow': reason_elbow}
                            df_outliers = update_outlier(df_outliers, dict_new_row)

                    if 'endeff' in detect_outliers:
                        if max_val_endeff > hand_vel_thresh:
                            print(f"Trial {id_s}_{id_p}_{id_t} is an outlier due to high endeffector velocity.\n"
                                  f"Max Value: {max_val_endeff} mm/s\tThreshold: {hand_vel_thresh} mm/s")
                            dict_new_row = {'id_s': id_s, 'id_p': id_p, 'id_t': id_t, 'dynamic': correct,
                                            'detect_on': 'endeffector', 'value_endeff': max_val_endeff,
                                            'value_elbow': max_val_elbow, 'thresh_endeff_vel': hand_vel_thresh,
                                            'thresh_elbow_vel': joint_vel_thresh, 'reason_endeff': reason_endeff,
                                            'reason_elbow': ''}
                            df_outliers = update_outlier(df_outliers, dict_new_row)

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

                path_omc_out = os.path.join(dir_dat_out, f'S15133_{id_p}_{id_t}_{condition}_{side}_{correct}_preprocessed.csv')
                path_mmc_out = os.path.join(dir_dat_out, f'{id_s}_{id_p}_{id_t}_{condition}_{side}_{correct}_preprocessed.csv')

                df_omc.to_csv(path_omc_out, sep=';')
                df_mmc.to_csv(path_mmc_out, sep=';')

                if verbose >= 2:
                    print(f"Preprocessed:\t{path_omc_out}\n"
                          f"Preprocessed:\t{path_mmc_out}\n"
                          f"Dropped Frames:\n"
                          f"OMC: {omc_nframes_before - omc_nframes_after}\n"
                          f"MMC: {mmc_nframes_before - mmc_nframes_after}")





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

    csv_appendix = '_dynamic_normalized.csv' if dynamic else '_fixed_normalized.csv'

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

    list_df_out = []

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

        maxtime = min(max(df_mmc['time']), max(df_omc['time'].values))


        #cut dataframes to same timeframe
        df_mmc = df_mmc[df_mmc['time'] <= maxtime]
        df_omc = df_omc[df_omc['time'] <= maxtime]
        time_normalized = np.linspace(0, 1, num=len(df_mmc['time']))

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

    if verbose >=1:
        progbar = tqdm(metrics, desc='Writing', leave=True)

    for metric in metrics:
        if verbose>=1:
            progbar.set_description(f'Writing {os.path.basename(dict_csvs_out[metric])}')

            progbar.update(1)
        df_debug = dict_df_out[metric]
        dict_df_out[metric].to_csv(dict_csvs_out[metric], sep=';')


if __name__ == '__main__':
    # this part is for Development and Debugging

    debug = False
    if sys.gettrace() is not None:
        print("Debug Mode is activated\n"
              "Starting debugging script.")
        debug=True

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

    dir_processed = os.path.join(root_data, 'preprocessed_data')
    dir_results = os.path.join(root_stat, '01_continuous', '01_results')
    det_outliers = ['elbow', 'endeff']
    hand_vel_thresh = 3000
    thresh_elbowVelocity = 200

    if test_timeseries:

        """for correct in corrections:
            debug = False

            preprocess_timeseries(root_val,
                                  downsample=True, drop_last_rows=False, detect_outliers= det_outliers,
                                  joint_vel_thresh=thresh_elbowVelocity, hand_vel_thresh=hand_vel_thresh, correct=correct, fancy_offset=False,
                                  verbose=1, plot_debug=False, print_able=False, empty_dst=True, debug=debug, debug_c=50)
            dir_src = '02_fully_preprocessed' if correct == 'fixed' else '03_fully_preprocessed_dynamic'
            dir_src = os.path.join(root_data, 'preprocessed_data', dir_src)
            normalize_data(dir_src=dir_src, dynamic = True if correct == 'dynamic' else False, verbose=1)

        get_error_timeseries(dir_processed = dir_processed, dir_results = dir_results, empty_dst=True, verbose=1, debug=debug)
        get_error_mean_rmse(dir_results, overwrite_csvs=True, verbose=1)
        get_rom_rmse_old(dir_results, overwrite_csvs=True, verbose=1)
        get_timeseries_correlations(dir_processed, dir_results, overwrite_csvs=False, verbose=1)"""

        get_rom_rmse(dir_results, overwrite_csvs=True, verbose=1)




    else:

        runs_statistics_discrete(path_csv_murphy_measures, root_stat, make_plots=True,
                                 thresh_PeakVelocity_mms=None, thresh_elbowVelocity=None)
        runs_statistics_discrete(path_csv_murphy_measures, root_stat, make_plots=True,
                                 thresh_PeakVelocity_mms=hand_vel_thresh, thresh_elbowVelocity=thresh_elbowVelocity)
