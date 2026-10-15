"""preprocessing part of iDrinkStatisticalAnalysis.py (moved without changes, see iDrinkStatisticalAnalysis.py for the public names)."""
import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np


def delete_existing_files(dir):
    """Deletes all files that are in the given directory"""
    if not os.path.exists(dir):
        return
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path):
            os.remove(os.path.join(dir, file))


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

                    dict_offsets[kinematic] = - mean_mmc + mean_omc

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

                path_omc_out = os.path.join(dir_dat_out,
                                            f'S15133_{id_p}_{id_t}_{condition}_{side}_{correct}_preprocessed.csv')
                path_mmc_out = os.path.join(dir_dat_out,
                                            f'{id_s}_{id_p}_{id_t}_{condition}_{side}_{correct}_preprocessed.csv')

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


def get_offset_unfancy(dir_root, downsample = True, drop_last_rows = False, correct='fixed', detect_outliers = [],
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

                    dict_offsets[kinematic] = -mean_mmc + mean_omc

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


    csv_outliers = os.path.join(dir_root, '05_logs', 'outliers.csv')
    csv_offset = os.path.join(dir_root, '03_data', 'preprocessed_data', 'offset.csv')

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

                # Resample both DataFrames to 60Hz
                df_omc = downsample_dataframe(df_omc)
                df_mmc = downsample_dataframe(df_mmc)
                #df_omc, df_mmc = drop_nan(df_omc, df_mmc)

                # Drop last rows if needed
                df_omc, df_mmc = drop_last_rows_if_needed(df_omc, df_mmc)

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

                df_mmc, dict_offsets = handling_vertical_offset(df_omc, df_mmc, fancy=fancy_offset)

                update_offset_csv(csv_offset, id_s, id_p, id_t, dict_offsets)

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
