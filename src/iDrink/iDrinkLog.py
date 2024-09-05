import glob
import os
import sys
import time
import re

from tqdm import tqdm

import argparse
import pandas as pd

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
def update_trial_csv(args, trial_list,  csv_path, columns_to_add = None):
    """
    Updates the csv file containing the trial_informations.

    :param trial_list:
    :param csv_path:

    :return:
    """

    df_trials_old = pd.read_csv(csv_path, sep=';')
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

    pass


def trials_to_csv(args, trial_list, csv_path):
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

    trial_list = []
    if args.verbose >= 1:
        progress = tqdm(total=df_trials.shape[0], desc="Creating Trial Objects", unit="Trial")
    for index, row in df_trials.iterrows():
        id_s = row["id_s"]
        id_p = row["id_p"]
        id_t = row["id_t"]
        identifier = row["identifier"]
        dir_session = row["dir_session"]
        dir_participant = row["dir_participant"]
        dir_trial = row["dir_trial"]
        video_files = row["video_files"].split(", ")
        used_cams = row["used_cams"].split(", ")
        affected = row["affected"]
        measured_side = row["measured_side"]
        dir_calib = row["dir_calib"]
        dir_calib_videos = row["dir_calib_videos"]
        pose_model = df_settings.loc[df_settings["setting_id"] == int(re.search("\d+", id_s).group()), "pose_model"].values[0]
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