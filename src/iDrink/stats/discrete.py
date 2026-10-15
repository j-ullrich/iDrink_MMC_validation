"""discrete part of iDrinkStatisticalAnalysis.py (moved without changes, see iDrinkStatisticalAnalysis.py for the public names)."""
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, zscore
murphy_measures = ["PeakVelocity_mms",
                   "elbowVelocity",
                   "tTopeakV_s",
                   "tToFirstpeakV_s",
                   "tTopeakV_rel",
                   "tToFirstpeakV_rel",
                   "NumberMovementUnits",
                   "InterjointCoordination",
                   "ElbowExtension",
                   "ShoulderFlexionReaching",
                   "shoulderFlexionDrinking",
                   "shoulderAbduction",
                   "trunkDisplacementMM",
                   "trunkDisplacementDEG",
                   ]
ignore_id_p = ['P11', 'P19']
idx_s_singlecam = ['S017', 'S018', 'S019', 'S020', 'S021', 'S022', 'S023', 'S024', 'S025', 'S026']
idx_s_singlecam = ['S017', 'S018']
idx_s_multicam_reduced = ['S001', 'S002']
idx_s_reduced = idx_s_multicam_reduced + idx_s_singlecam
from iDrink.stats.correlation import get_mmc_omc_difference, get_murphy_corrrelation


def runs_statistics_discrete(path_csv_murphy, root_stat,
                             thresh_PeakVelocity_mms = None, thresh_elbowVelocity=None, outlier_iqr=False,
                             make_plots = False, reduced_analysis=False, verbose=1):
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


    df = pd.DataFrame(columns=df_murphy.columns)

    def detect_outliers_iqr(dataframe, columns):
        outlier_mask = pd.DataFrame(index=dataframe.index)
        for col in columns:
            Q1 = dataframe[col].quantile(0.25)
            Q3 = dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask[col] = (dataframe[col] > lower_bound) & (dataframe[col] < upper_bound)

        return outlier_mask.all(axis=1), dataframe[~outlier_mask.all(axis=1)].reindex()

    cols_error = ['TotalMovementTime', 'PeakVelocity_mms', 'elbowVelocity', 'tTopeakV_s',
                  'tToFirstpeakV_s', 'tTopeakV_rel', 'tToFirstpeakV_rel', 'NumberMovementUnits',
                  'InterjointCoordination', 'trunkDisplacementMM', 'trunkDisplacementDEG', 'ShoulderFlexionReaching',
                  'ElbowExtension', 'shoulderAbduction', 'shoulderFlexionDrinking']

    df_murphy_omc = df_murphy[df_murphy['id_s'] == 'S15133']
    df_murphy_mmc = df_murphy[df_murphy['id_s'] != 'S15133']

    if reduced_analysis:
        df_murphy_mmc = df_murphy_mmc[(df_murphy_mmc['id_s'].isin(idx_s_reduced)) & (~df_murphy_mmc['id_p'].isin(ignore_id_p))]
        df_murphy_omc = df_murphy_omc[(~df_murphy_omc['id_p'].isin(ignore_id_p))]

    if outlier_iqr:
        mask_iqr, df_murphy_mmc = detect_outliers_iqr(df_murphy_mmc, cols_error)
    else:
        # If after z_score Nan values are present, they are replaced by z0 to have no influence on the outcome
        df_murphy_mmc = df_murphy_mmc[(np.abs(zscore(df_murphy_mmc[cols_error]).fillna(0)) < 3).all(axis=1)].reindex()
        df_murphy_omc = df_murphy_omc[(np.abs(zscore(df_murphy_omc[cols_error])) < 3).all(axis=1)].reindex()

    df_murphy = pd.concat([df_murphy_omc, df_murphy_mmc])
    if verbose >= 1:
        progbar = tqdm(total=len(idx_s_mmc), desc='Joining MMC and OMC Data')
    for id_s in sorted(idx_s_mmc):

        if verbose >= 1:
            progbar.set_description(f'Joining MMC and OMC Data for {id_s}')


        df_s = df_murphy[df_murphy['id_s'] == id_s]
        df_omc = df_murphy[df_murphy['id_s'] == 'S15133']
        idx_p = sorted(list(df_murphy[df_murphy['id_s']==id_s]['id_p'].unique()))

        for id_p in idx_p:
            idx_t = sorted(list(df_murphy[(df_murphy['id_p']==id_p) & (df_murphy['id_s']==id_s )]['id_t'].unique()))
            for id_t in idx_t:
                if len(df[(df['id_s'] == 'S15133') & (df['id_p'] == id_p) & (df['id_t'] == id_t)]) == 0:
                    df = pd.concat([df, df_omc[(df_omc['id_p'] == id_p) & (df_omc['id_t'] == id_t)]], ignore_index=True)

        df = pd.concat([df_s, df], ignore_index=True)
        if verbose >= 1:
            progbar.update(1)

    if verbose >= 1:
        progbar.close()

    # TODO: calculate correlation over idxs ans idxs_idxP
    df_corr = get_murphy_corrrelation(df, root_stat_cat,
                                      thresh_PeakVelocity_mms = thresh_PeakVelocity_mms, thresh_elbowVelocity=thresh_elbowVelocity,
                                      reduced_analysis=reduced_analysis,   verbose=verbose)

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
    if reduced_analysis:
        file_app = '_reduced'
    else:
        file_app = ''




    path_csv_murphy_diff = os.path.join(root_stat_cat, f'stat_murphy_diff{outlier_corrected}{file_app}.csv')
    path_csv_murphy_abs_diff = os.path.join(root_stat_cat, f'stat_murphy_abs_diff{outlier_corrected}{file_app}.csv')
    path_csv_murphy_mean = os.path.join(root_stat_cat, f'stat_murphy_mean{outlier_corrected}{file_app}.csv')
    path_csv_murphy_rmse = os.path.join(root_stat_cat, f'stat_murphy_rmse{outlier_corrected}{file_app}.csv')

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
