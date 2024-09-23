import glob
import os
import sys
import time
import re

from tqdm import tqdm

import argparse
import pandas as pd

"""
Functions in this file are used to update the Log files of the iDrink Pipeline and to check the state of the pipeline.
"""

def get_new_row(trial, columns):
    """
    Returns a new row for the csv file.

    :param trial:
    :param columns:
    :return:
    """
    new_row = pd.Series(dtype='object')
    for column in columns:
        try:
            new_row[column] = getattr(trial, column)
        except:
            new_row[column] = ', '.join(getattr(trial, column))
    return new_row

def update_trial_csv(args, trial_list, csv_path, columns_to_add = None):
    """
    Updates the csv file containing the trial_informations.

    :param trial_list:
    :param csv_path:

    :return:
    """

    df_trials_old = pd.read_csv(csv_path, sep=';')

    # write safety file
    df_trials_old.to_csv(csv_path.split(".csv")[0] + "_safety.csv", sep=';', index=False)
    df_trials = pd.DataFrame(columns=df_trials_old.columns)

    if args.verbose >= 1:
        progress = tqdm(total=len(trial_list), desc="Updating CSV", unit="Trial")
    for trial in trial_list:

        df_trials = pd.concat([df_trials, get_new_row(trial, df_trials.columns).to_frame().T], axis=0, ignore_index=True)

        if columns_to_add is not None:
            df_trials = pd.concat([df_trials, get_new_row(trial, columns_to_add).to_frame().T], axis=0,
                                  ignore_index=True)

            """
            Old Version -- Keep until new version is tested
            for column in columns_to_add:
                try:
                    df_trials.loc[df_trials.identifier == trial.identifier, column] = getattr(trial, column)
                except:
                    df_trials.loc[df_trials.identifier == trial.identifier, column] = ', '.join(getattr(trial, column))
            """

        if args.verbose >= 1:
            progress.update(1)

    if args.verbose >= 1:
        progress.close()

    df_trials.to_csv(csv_path, sep=';', index=False)
    return df_trials



def trials_to_csv(args, trial_list, csv_path, get_df=False):
    """
    Writes the information of the trials to a csv file.
    The .csv  should have the following Columns:
        HPE_done: Boolean tells whether Pose estimation has been done.
        P2S_done: Boolean tells whether Pose2Sim has been done.
        OS_done: Boolean tells whether Opensim has been done.
        MM_done: Boolean tells whether Murphy Measures have been done.
        stat_done: Boolean tells whether Statistics have been done.
        id_s: Setting ID
        id_p: Patient ID
        id_t: Trial ID
        identifier: Identifier of the trial
        affected: Affected or Unaffected
        measured_side: Side of the body that is affected
        cam_setting: Camera Setting used for the trial
        used_cams: cams used for the trial
        video_files: List of video files used for the trial
        dir_session: Directory of the session
        dir_participant: Directory of the participant
        dir_trial: Directory of the trial
        dir_calib: Directory of the calibration files
        json_list: Original position of the json files
        TODO: Add other variables to csv when integrated into the pipeline
    :param args:
    :param trial_list:
    :param csv_path:
    :return:
    """
    df_trials = pd.DataFrame(columns=["HPE_done",
                                      "MMPose_done",
                                      "P2SPose_done",
                                      "Metrabs_multi_done",
                                      "Metrabs_single_done",
                                      "calib_done",
                                      "P2S_done",
                                      "OS_done",
                                      "MM_done",
                                      "stat_done",
                                      "id_s",
                                      "id_p",
                                      "id_t",
                                      "identifier",
                                      "affected",
                                      "measured_side",
                                      "cam_setting",
                                      "used_cams",
                                      "video_files",
                                      "n_frames",
                                      "dir_session",
                                      "dir_participant",
                                      "dir_trial",
                                      "dir_calib",
                                      "dir_calib_videos",
                                      "json_list"])
    for trial in trial_list:
        if args.verbose >= 2:
            print(f"adding trial {trial.identifier} to csv file.")
        df_trials = pd.concat([df_trials, get_new_row(trial, df_trials.columns).to_frame().T], axis=0, ignore_index=True)
        pass


    df_trials.to_csv(csv_path, sep=';', index=False)

    if get_df:
        return df_trials


def trials_from_csv(args, df_trials, df_settings, root_data, default_dir):
    """
    Creates the trial objects based on the csv file.

    :param args:
    :param df_trials: Dataframe containing the trial information
    :param df_settings: Dataframe containing the setting information
    :param root_data: Root directory of the data
    :param default_dir: Default directory

    :return: trial_list
    """
    from . import iDrinkTrial
    import ast

    def correct_drive(path, drive):
        if type(path) is list:
            path = [correct_drive(p, drive) for p in path]
            return path
        if os.path.splitdrive(path)[0] != drive:
            path = os.path.join(drive+os.path.splitdrive(path)[1])
        return path

    # get Drive in use
    drive = os.path.splitdrive(root_data)[0]

    trial_list = []
    if args.verbose >= 1:
        progress = tqdm(total=df_trials.shape[0], desc="Creating Trial Objects", unit="Trial")
    for index, row in df_trials.iterrows():
        id_s = row["id_s"]
        id_p = row["id_p"]
        id_t = row["id_t"]
        identifier = row["identifier"]
        used_cams = ast.literal_eval(row["used_cams"])
        affected = row["affected"]
        measured_side = row["measured_side"]

        pose_model = df_settings.loc[df_settings["setting_id"] == int(re.search("\d+", id_s).group()), "pose_model"].values[0]

        # Drive in path
        dir_session = correct_drive(row["dir_session"], drive)
        dir_participant = correct_drive(row["dir_participant"], drive)
        dir_trial = correct_drive(row["dir_trial"], drive)
        video_files = correct_drive(ast.literal_eval(row["video_files"]), drive)
        dir_calib = correct_drive(row["dir_calib"], drive)
        dir_calib_videos = correct_drive(row["dir_calib_videos"], drive)

        if os.name == 'posix' and '\\' in dir_trial:
            # Replace backslashes with forward slashes
            base = root_data.split('iDrink')[0]

            dir_trial = os.path.join(base, 'iDrink', dir_trial.replace('\\', '/').split('iDrink/')[1])
            dir_participant = os.path.join(base, 'iDrink', dir_participant.replace('\\', '/').split('iDrink/')[1])
            dir_session = os.path.join(base, 'iDrink', dir_session.replace('\\', '/').split('iDrink/')[1])
            dir_calib = os.path.join(base, 'iDrink', dir_calib.replace('\\', '/').split('iDrink/')[1])
            dir_calib_videos = os.path.join(base, 'iDrink', dir_calib_videos.replace('\\', '/').split('iDrink/')[1])
            video_files = [os.path.join(base, 'iDrink', video_file.replace('\\', '/').split('iDrink/')[1]) for video_file in video_files]
        elif os.name == 'nt' and '/' in dir_trial:
            # Replace forward slashes with backslashes
            base = root_data.split('iDrink')[0]

            dir_trial = os.path.join(base, 'iDrink', dir_trial.replace('/', '\\').split('iDrink\\')[1])
            dir_participant = os.path.join(base, 'iDrink', dir_participant.replace('/', '\\').split('iDrink\\')[1])
            dir_session = os.path.join(base, 'iDrink', dir_session.replace('/', '\\').split('iDrink\\')[1])
            dir_calib = os.path.join(base, 'iDrink', dir_calib.replace('/', '\\').split('iDrink\\')[1])
            dir_calib_videos = os.path.join(base, 'iDrink', dir_calib_videos.replace('/', '\\').split('iDrink\\')[1])
            video_files = [os.path.join(base, 'iDrink', video_file.replace('/', '\\').split('iDrink\\')[1]) for video_file in video_files]


        # Create the trial object
        trial = iDrinkTrial.Trial(identifier=identifier, id_s=id_s, id_p=id_p, id_t=id_t,
                                  dir_root=root_data, dir_default=default_dir,
                                  dir_trial=dir_trial, dir_participant=dir_participant, dir_session=dir_session,
                                  dir_calib=dir_calib, dir_calib_videos=dir_calib_videos,
                                  affected=affected, measured_side=measured_side,
                                  video_files=video_files, used_cams=used_cams,
                                  used_framework=args.poseback, pose_model=pose_model)
        trial.create_trial()
        trial.load_configuration()
        trial.config_dict["pose"]["videos"] = trial.video_files
        trial.config_dict["pose"]["cams"] = trial.used_cams
        trial.config_dict.get("project").update({"project_dir": trial.dir_trial})
        trial.config_dict['pose']['pose_framework'] = trial.used_framework
        trial.config_dict['pose']['pose_model'] = trial.pose_model
        trial.save_configuration()
        trial_list.append(trial)
        if args.verbose >= 1:
            progress.update(1)
    if args.verbose >= 1:
        progress.close()
        print(f"Number of Trials created: {len(trial_list)}")
    return trial_list


def does_json_exist(trial, pose_root, posebacks=["openpose", "metrabs", "mmpose", "pose2sim"]):
    """
    Checks if json files exist for all camera recordings of a trial.

    Quick reminder on folder structure

    - root_data
        - 01_unfiltered
            - P07
                - P07_cam1
                    - metrabs
                        - trial_1_[...]_json
                            - trial_1_[...]_000000.json
                            - trial_1_[...]_000001.json
                            - trial_1_[...]_[....].json
                        - trial_2_[...]_json
                        - trial_[.....]_json
                    - mmpose
                    - openpose
                    - pose2sim
                - P07_cam2
                - P07_cam...
            - P08
            - ...
        - 02_filtered

    :param id_t: Trial id
    :param id_p: Participant id
    :param pose_root: Root directory of the pose data
    :param cams: List of cameras used for the trial
    :param n_frames: number of frames in video

    :return: True if all json files exist, False otherwise
    """
    import cv2
    if type(posebacks) is str:
        posebacks = [posebacks]

    id_t = f"trial_{int(trial.id_t.split('T')[1])}"
    id_p = trial.id_p
    cams = [f'cam{cam}' for cam in trial.used_cams]

    if trial.n_frames == 0:
        # Get the number of frames in the video
        # We assume, all videos of a trial have the same number of frames
        video_file = trial.video_files[0]
        cap = cv2.VideoCapture(video_file)
        trial.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    # Check if the json files exist for all cameras
    for cam in cams:
        cam_dir = os.path.realpath(os.path.join(pose_root, '02_filtered', id_p, f"{id_p}_{cam}"))
        if not os.path.isdir(cam_dir):
            return False
        for poseback in posebacks:
            json_files = glob.glob(os.path.join(cam_dir, poseback, f"{id_t}_*_json", f"{id_t}_*_*.json"))
            if len(json_files) != trial.n_frames:
                return False  # If for one camera the number of json files is not equal to the number of frames return False
    return True


def all_2d_HPE_done(trial, root_HPE = None):
    """
    Checks if all 2D Pose Estimation Methods have been done for a trial.

    if root_HPE is given, it checks if the json files exist in the given directory.

    :param trial:
    :return:
    """

    if trial.MMPose_done and trial.P2SPose_done and trial.Metrabs_multi_done:
        return True
    else:
        if root_HPE is not None:
            all_done = does_json_exist(trial, root_HPE, posebacks=["metrabs", "mmpose", "pose2sim"])
            trial.MMPose_done = trial.P2SPose_done = trial.Metrabs_multi_done = all_done
            return all_done

        return False


def log_error(args, trial, exception, stage, pose, csv_path):
    """
    Logs exceptions that occur during the pipeline.


    """
    if os.path.isfile(csv_path):
        df_log = pd.read_csv(csv_path, sep=';')
    else:
        df_log = pd.DataFrame(columns=["date", "time", "identifier", "stage", "pose_estimation", "cams", "exception"])


    new_row = pd.Series(dtype='object')
    new_row["date"] = time.strftime("%d.%m.%Y")
    new_row["time"] = time.strftime("%H:%M:%S")
    new_row["identifier"] = trial.identifier
    new_row["stage"] = stage
    new_row["pose_estimation"] = pose
    new_row["cams"] = trial.used_cams
    new_row["exception"] = exception
    df_log = pd.concat([df_log, new_row.to_frame().T], axis=0, ignore_index=True)
    df_log.to_csv(csv_path, sep=';', index=False)

    return df_log


def files_exist(dir, file_type, verbose=0):
    """
    Checks if the .mot files exist for all cameras of the trial.

    :param trial:
    :return:
    """

    files = glob.glob(os.path.join(dir, f'*{file_type}'))

    if verbose >= 1:
        print(f"Checking if {file_type} files exist in {dir}\n")
        print(files)

    return len(files) > 0