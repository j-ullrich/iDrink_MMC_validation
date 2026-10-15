"""correlation part of iDrinkStatisticalAnalysis.py (moved without changes, see iDrinkStatisticalAnalysis.py for the public names)."""
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


def get_murphy_corrrelation(df, root_stat_cat, thresh_PeakVelocity_mms=None, thresh_elbowVelocity=None, reduced_analysis=False, verbose=1):
    """Calculates the correlation of Murphy Measures for each setting and participant

    When id_p is None, the correlation accounts for the whole setting

    """

    # Detect outlier for ROM errors
    """cols_error = ['TotalMovementTime', 'condition', 'PeakVelocity_mms', 'elbowVelocity', 'tTopeakV_s',
                  'tToFirstpeakV_s', 'tTopeakV_rel', 'tToFirstpeakV_rel', 'NumberMovementUnits',
                  'InterjointCoordination', 'trunkDisplacementMM', 'trunkDisplacementDEG', 'ShoulderFlexionReaching',
                  'ElbowExtension', 'shoulderAbduction', 'shoulderFlexionDrinking']
    df = df[(np.abs(zscore(df[cols_error])) < 3).all(axis=1)]"""

    #cor_columns = ['id_s', 'id_p', 'measure', 'condition', 'side', 'pearson', 'pearson_p']
    corr_columns = ['id_s', 'measure', 'condition', 'pearson', 'pearson_p']

    df_corr = pd.DataFrame(columns=corr_columns)

    id_s_omc = 'S15133'

    df_omc = df[df['id_s'] == id_s_omc]
    df_mmc = df[df['id_s'] != id_s_omc]

    if reduced_analysis:
        df_mmc = df_mmc[(df_mmc['id_s'].isin(idx_s_reduced)) & (~df_mmc['id_p'].isin(ignore_id_p))]
        df_omc = df_omc[(~df_omc['id_p'].isin(ignore_id_p))]

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

    if reduced_analysis:
        file_app = '_reduced'
    else:
        file_app = ''

    csv_corr = os.path.join(root_stat_cat, f'stat_murphy_corr{file_app}.csv')
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
        if id_s == id_s_omc:
            continue

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
