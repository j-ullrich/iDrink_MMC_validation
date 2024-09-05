import glob
import os
import sys
import time
import re
import shutil
import subprocess

from tqdm import tqdm

import argparse
import pandas as pd

from iDrink import iDrinkTrial, iDrinkPoseEstimation, iDrinkLog
from iDrink.iDrinkCalibration import delta_calibration_val

from Pose2Sim import Pose2Sim

"""
This File is the starting Point of the iDrink Validation.


It creates all trial objects and then runs the pipeline for each trial and setting.

The pipeline is as follows:

- Pose Estimation
- Pose2Sim
- Opensim
- iDrink Analytics --> Calculation of Murphy Measures

The possible settings are:

- 29 different Camera Setups
- 2 Calibration Methods
- 3 HPE Methods (Openpose, MMPose and Pose2Sim Pose Estimation)
- 2 sets of 2D Keypoints (filtered and unfiltered)
- 4 ways to calculate the Murphy Measures (Analyzer only, Analyzer and keypoints, Opensim Invkin Tool and Analyzer, and Opensim invkin Tool and keypoints)

In Total we have 29 * 2 * 3 * 2 * 4 = 1392 different settings

When creating the trial objects, a csv file is created, which contains all the settings for each trial.
"""

""" The Folder Structure is as explained on this table: https://miro.com/app/board/uXjVKzgWI8c=/"""

# Command-line interface definition 6112qg
parser = argparse.ArgumentParser(description='Extract marker locations from C3D files and save in TRC format.')
parser.add_argument('--mode', metavar='m', default=None,
                    help='"pose_estimation", "pose2sim", "opensim", "murphy_measures", "statistics", or "full"')
parser.add_argument('--poseback', metavar='hpe', type=str, default='mmpose',
                    help='Method for Pose Estimation: "openpose", "mmpose", "pose2sim", "metrabs_multi", '
                         '"metrabs_single"')
parser.add_argument('--trial_id', metavar='t_id', type=str, default=None,
                    help='Trial ID if only one single trial should be processed')
parser.add_argument('--patient_id', metavar='p_id', type=str, default=None,
                    help='Patient ID if only one single patient should be processed')
parser.add_argument('--verbose', metavar='v', type=int, default=1,
                    help='Verbosity level: 0, 1, 2 default: 0')
parser.add_argument('--DEBUG', action='store_true', default=False,
                    help='Debug mode')

root_MMC = r"C:\iDrink\Test_folder_structures"  # Root directory of all MMC-Data --> Videos and Openpose json files
root_OMC = r"C:\iDrink\OMC_data_newStruct"  # Root directory of all OMC-Data --> trc of trials.
root_val = r"C:\iDrink\validation_root"  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
root_data = os.path.join(root_val, "03_data")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
default_dir = os.path.join(root_val, "01_default_files")  # Default Files for the iDrink Validation
metrabs_models_dir = os.path.join(root_val, "04_metrabs_models")  # Directory containing the Metrabs Models

df_settings = pd.read_csv(os.path.join(root_val, "validation_settings.csv"), sep=';')  # csv containing information for the various settings in use.

try:
    df_trials = pd.read_csv(os.path.join(root_val, "validation_trials.csv"), sep=';')  # csv containing information for the various settings in use.
except FileNotFoundError:
    print("No .csv file for trials found.")
    df_trials = None



def run_full_pipeline(trial_list, mode):
    """
    Runs the pipeline for the given trial list.

    :param trial_list:
    :return:
    """
    for trial in trial_list:
        print(f"Running Pipeline for {trial.identifier}")
        # Pose Estimation
        if mode == "pose_estimation":
            print("Running Pose Estimation")




def create_trial_objects():
    """
    TODO: Script writes a csv file, that contains information of which steps are done for which trial and setting.

    Creates the trial Objects that will be use in the following Pipelines.

    Depending on the mode, different files will be used to create the objects.

    Cases:
    Pose Estimation: Uses only the video_files
    Pose2Sim: Uses the json files created by Pose estimation. If the folder structure for the Pose2Sim execution is not yet build, the files will be copied into the right structure.
    Opensim: Uses the trc files created by Pose2Sim.
    Murphy Measures: Uses the Opensim files.
    statistics: Checks for all files and creates a csv file with the results.
    full: uses video_files as Pose Estimation does.


    :param:
    :return trial_list: List of Trial Objects
    """

    if sys.gettrace() is not None:  # If Debugger is in Use, limit settings to first 10
        n_settings = 10
    else:
        n_settings = df_settings["setting_id"].max()



    def trials_from_video():
        """
        Creates the trial objects for the Pose Estimation Pipeline.

        :return:
        """


        trials_df = pd.DataFrame(
            columns=["setting_id", "patient_id", "trial_id", "identifier", "affected", "side", "cam_setting",
                     "cams_used", "videos_used", "session_dir", "participant_dir", "trial_dir", "dir_calib", "valid"])
        if args.verbose >= 1:
            progress_bar = tqdm(total=n_settings, desc="Creating Trial-DataFrame", unit="Setting")

        for setting_id in range(1, n_settings+1):
            id_s = f"S{setting_id:03d}"  # get Setting ID for use as Session ID in the Pipeline
            cam_setting = df_settings.loc[df_settings["setting_id"] == setting_id, "cam_setting"].values[0]
            # Check whether Cam is used for Setting
            cams_tuple = eval(df_settings.loc[df_settings["setting_id"] == setting_id, "cams"].values[ 0])  # Get the tuple of cams for the setting


            # Create Setting folder if not yet done
            dir_setting = os.path.join(root_data, f"setting_{setting_id:03d}")
            if not os.path.exists(dir_setting):
                os.makedirs(dir_setting, exist_ok=True)

            for p in os.listdir(root_MMC):
                p_dir = os.path.join(root_MMC, p)
                # Get the patient_id
                id_p = re.search(r'(P\d+)', p_dir).group(1)  # # get Participant ID for use in the Pipeline

                part_dir = os.path.join(dir_setting, id_p)
                if not os.path.exists(part_dir):
                    os.makedirs(part_dir, exist_ok=True)
                dir_session = os.path.join(part_dir, id_s)
                if not os.path.exists(dir_session):
                    os.makedirs(dir_session, exist_ok=True)
                dir_calib = os.path.join(dir_session, f"{id_s}_Calibration")
                if not os.path.exists(dir_calib):
                    os.makedirs(dir_calib, exist_ok=True)
                part_dir = os.path.join(dir_session, f"{id_s}_{id_p}")
                if not os.path.exists(part_dir):
                    os.makedirs(part_dir, exist_ok=True)

                # Make sure, we only iterate over videos, that correspond to the correct camera
                pattern = "".join([f"{i}" for i in cams_tuple])

                pattern = re.compile(f'cam[{pattern}]*').pattern
                cam_folders = glob.glob(os.path.join(p_dir, "01_measurement", "04_Video", "03_Cut", "drinking", pattern))

                # Get all video files of patient
                video_files = []
                for cam_folder in cam_folders:
                    video_files.extend(glob.glob(os.path.join(cam_folder, "**", "*.mp4"), recursive=True))

                for video_file in video_files:
                    # Extract trial ID and format it
                    trial_number = int(re.search(r'trial_\d+', video_file).group(0).split('_')[1])
                    id_t = f'T{trial_number:03d}'
                    identifier = f"{id_s}_{id_p}_{id_t}"
                    if args.verbose >=2:
                        print("Creating: ", identifier)

                    trial_dir = os.path.join(part_dir, identifier)
                    if not os.path.exists(trial_dir):
                        os.makedirs(trial_dir, exist_ok=True)



                    if identifier not in trials_df["identifier"].values:
                        # Add new row to dataframe only containing the trial_id
                        identifier = f"{id_s}_{id_p}_{id_t}"
                        new_row = pd.Series({"setting_id": id_s, "patient_id": id_p,"trial_id": id_t, "identifier": identifier, "cams_used": "", "videos_used": ""})
                        trials_df = pd.concat([trials_df, new_row.to_frame().T], ignore_index=True)

                    try:
                        affected = re.search(r'unaffected', video_file).group(0)
                    except:
                        affected = 'affected'

                    side = re.search(r'[RL]', video_file).group(0)
                    cam = re.search(r'(cam\d+)', video_file).group(0)


                    if len(trials_df.loc[trials_df["identifier"] == identifier, "cams_used"].values[0]) == 0:
                        trials_df.loc[trials_df["identifier"] == identifier, "cams_used"] = str(re.search(r'\d+', cam).group())
                        trials_df.loc[trials_df["identifier"] == identifier, "videos_used"] = str(video_file)
                    else:
                        trials_df.loc[trials_df["identifier"] == identifier, "cams_used"] = (
                                trials_df.loc[trials_df["identifier"] == identifier, "cams_used"] +
                                ", " + str(re.search(r'\d+', cam).group()))
                        trials_df.loc[trials_df["identifier"] == identifier, "videos_used"] = (
                                trials_df.loc[trials_df["identifier"] == identifier, "videos_used"] +
                                ", " + str(video_file))


                    trials_df.loc[trials_df["identifier"] == identifier, "affected"] = affected
                    trials_df.loc[trials_df["identifier"] == identifier, "side"] = side
                    trials_df.loc[trials_df["identifier"] == identifier, "cam_setting"] = cam_setting

                    trials_df.loc[trials_df["identifier"] == identifier, "session_dir"] = dir_session
                    trials_df.loc[trials_df["identifier"] == identifier, "participant_dir"] = part_dir
                    trials_df.loc[trials_df["identifier"] == identifier, "trial_dir"] = trial_dir
                    trials_df.loc[trials_df["identifier"] == identifier, "dir_calib"] = dir_calib

                    """
                    If cameras needed for setup are not present, set trial to invalid
                    Trial object and folder will not be created.
                    """
                    if len(cam_folders) == len(cams_tuple):
                        trials_df.loc[trials_df["identifier"] == identifier, "valid"] = True
                    else:
                        trials_df.loc[trials_df["identifier"] == identifier, "valid"] = False

            if args.verbose >= 1:
                progress_bar.update(1)

        if args.verbose >= 1:
            progress_bar.close()

        if args.verbose >= 1:
            progress_bar = tqdm(total=trials_df["identifier"].shape[0], desc="Creating Trial Objects:",
                                unit="Trial")

        trial_list = []
        for identifier in trials_df["identifier"]:

            if args.verbose >= 2:
                print(f"Setting and Videos for {identifier} are: ", trials_df.loc[trials_df["identifier"] == identifier, "setting_id"].values[0] , trials_df.loc[trials_df["identifier"] == identifier, "videos_used"].values[0])

            # get list from strings of videos and cams
            videos = trials_df.loc[trials_df["identifier"] == identifier, "videos_used"].values[0].split(", ")
            cams = trials_df.loc[trials_df["identifier"] == identifier, "cams_used"].values[0].split(", ")
            affected = trials_df.loc[trials_df["identifier"] == identifier, "affected"].values[0]  # get Participant ID for use in the Pipeline
            side = trials_df.loc[trials_df["identifier"] == identifier, "side"].values[0]  # get Participant ID for use in the Pipeline

            id_s = trials_df.loc[trials_df["identifier"] == identifier, "setting_id"].values[0] # get Setting ID for use as Session ID in the Pipeline
            id_p = trials_df.loc[trials_df["identifier"] == identifier, "patient_id"].values[0]  # get Participant ID for use in the Pipeline
            id_t = trials_df.loc[trials_df["identifier"] == identifier, "trial_id"].values[0]  # get Trial ID for use in the Pipeline

            s_dir = trials_df.loc[trials_df["identifier"] == identifier, "session_dir"].values[0]  # get Session Directory for use in the Pipeline
            p_dir = trials_df.loc[trials_df["identifier"] == identifier, "participant_dir"].values[0]  # get Participant Directory for use in the Pipeline
            t_dir = trials_df.loc[trials_df["identifier"] == identifier, "trial_dir"].values[0]  # get Trial Directory for use in the Pipeline

            dir_calib = os.path.join(s_dir, f"{id_s}_Calibration")
            dir_calib_videos = os.path.realpath(os.path.join(videos[0], '..', '..', '..', '..', '05_Calib_before'))

            if trials_df.loc[trials_df["identifier"] == identifier, "valid"].values[0]:
                # Create the trial object
                trial = iDrinkTrial.Trial(identifier=identifier, id_s=id_s, id_p=id_p, id_t=id_t,
                                          dir_root=root_data, dir_default=default_dir,
                                          dir_trial=t_dir, dir_participant=p_dir, dir_session=s_dir,
                                          dir_calib=dir_calib, dir_calib_videos=dir_calib_videos,
                                          affected=affected, measured_side=side,
                                          video_files=videos, used_cams=cams,
                                          used_framework=args.poseback, pose_model="Coco17_UpperBody")

                trial.cam_setting = trials_df.loc[trials_df["identifier"] == identifier, "cam_setting"].values[0]

                trial.create_trial()
                trial.load_configuration()

                trial.config_dict["pose"]["videos"] = videos
                trial.config_dict["pose"]["cams"] = cams

                trial.config_dict.get("project").update({"project_dir": trial.dir_trial})
                trial.config_dict['pose']['pose_framework'] = trial.used_framework
                trial.config_dict['pose']['pose_model'] = trial.pose_model

                trial.save_configuration()

                trial_list.append(trial)

            if args.verbose >= 1:
                progress_bar.update(1)
        if args.verbose >= 1:
            progress_bar.close()
            print(f"Number of Trials created: {len(trial_list)}")

        # Return the trial_list
        return trial_list

    def trials_from_json():
        """
                Creates the trial objects for the Pose Estimation Pipeline.

                :return:
                """

        # get number of settings from setting folders present in root_val\03_data

        dirs_setting = glob.glob(os.path.join(root_data, "setting_*"))

        trials_df = pd.DataFrame(
            columns=["setting_id", "patient_id", "trial_id", "identifier", "affected", "side", "cam_setting",
                     "cams_used", "videos_used", "session_dir", "participant_dir", "trial_dir", "dir_calib", 'json_list'])

        n = len(dirs_setting)
        for dir_setting in dirs_setting:
            setting_id = int(os.path.basename(dir_setting).split('_')[1])

            # get Setting ID for use as Session ID in the Pipeline
            id_s = f"S{setting_id:03d}"
            # Check whether Cam is used for Setting
            cam_setting = df_settings.loc[df_settings["setting_id"] == setting_id, "cam_setting"].values[0]
            # Get the tuple of cams for the setting
            cams_tuple = eval(df_settings.loc[df_settings["setting_id"] == setting_id, "cams"].values[0])

            dirs_participant = glob.glob(os.path.join(dir_setting, "P[0-9]*"))

            n *= len(dirs_participant)
            for dir_participant in dirs_participant:


                dir_session = glob.glob(os.path.join(dir_participant, f"{id_s}"))[0]
                dir_calib = glob.glob(os.path.join(dir_participant, f"{id_s}_Calibration"))[0]

                id_p = re.search(r'(P\d+)', dir_participant).group()
                dir_p = glob.glob(os.path.join(dir_session, f"{id_s}_{id_p}"))[0]

                dirs_trial = glob.glob(os.path.join(dir_p, f"{id_s}_{id_p}_T[0-9]*"))
                n *= len(dirs_trial)




                for dir_trial in dirs_trial:
                    id_t = re.search(r'(T\d+)', dir_trial).group()
                    identifier = f"{id_s}_{id_p}_{id_t}"

                    try:
                        if args.verbose >= 1:
                            progress_bar.update(1)
                    except NameError:
                        if args.verbose >= 1:
                            progress_bar = tqdm(total=n, desc="Creating Trial-DataFrame", unit="Trial")

                    if args.verbose >= 2:
                        print("Creating: ", identifier)

                    if identifier not in trials_df["identifier"].values:
                        # Add new row to dataframe only containing the trial_id
                        identifier = f"{id_s}_{id_p}_{id_t}"
                        new_row = pd.Series(
                            {"setting_id": id_s, "patient_id": id_p, "trial_id": id_t, "identifier": identifier,
                             "cams_used": "", "videos_used": "",})
                        trials_df = pd.concat([trials_df, new_row.to_frame().T], ignore_index=True)

                    # Make sure, we only iterate over videos, that correspond to the correct camera
                    filt = df_settings.loc[df_settings["setting_id"] == setting_id, "filtered_2d_keypoints"].values[0]
                    filt = '01_unfiltered' if filt == 'unfiltered' else '02_filtered'

                    cam_root = os.path.join(root_val, "02_pose_estimation", filt, id_p)
                    pattern = "".join([f"{i}" for i in cams_tuple])

                    pattern = re.compile(f'cam[{pattern}]*').pattern
                    cam_folders = glob.glob(os.path.join(cam_root, pattern))
                    dir_pose = os.path.join(dir_trial, 'pose')
                    cams_used = []

                    for cam_folder in cam_folders:
                        trial_number = int(id_t.split('T')[1])
                        dir_json_pose = glob.glob(os.path.join(cam_folder, args.poseback, f'trial_{trial_number}_*'))[0]

                        try:
                            affected = re.search(r'unaffected', dir_json_pose).group(0)
                        except:
                            affected = 'affected'

                        side = re.search(r'[RL]', dir_json_pose).group(0)
                        cam = re.search(r'(cam\d+)', dir_json_pose).group(0)
                        cams_used.append(cam)

                        json_list = glob.glob(os.path.join(dir_json_pose, "*.json"))
                        # Create subfolder of trial\pose

                        target_dir = os.path.join(dir_pose, f"{cam}_{identifier}_json")
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir, exist_ok=True)

                        #Copy all json-files in json_list into target_dir
                        for json_file in json_list:
                            shutil.copy2(json_file, target_dir)

                    trials_df.loc[trials_df["identifier"] == identifier, "affected"] = affected
                    trials_df.loc[trials_df["identifier"] == identifier, "side"] = side
                    trials_df.loc[trials_df["identifier"] == identifier, "cam_setting"] = cam_setting

                    trials_df.loc[trials_df["identifier"] == identifier, "session_dir"] = dir_session
                    trials_df.loc[trials_df["identifier"] == identifier, "participant_dir"] = dir_p
                    trials_df.loc[trials_df["identifier"] == identifier, "trial_dir"] = dir_trial
                    trials_df.loc[trials_df["identifier"] == identifier, "dir_calib"] = dir_calib

                    trials_df.loc[trials_df["identifier"] == identifier, "cams_used"] = ", ".join(cams_used)
                    trials_df.loc[trials_df["identifier"] == identifier, "json_list"] = ", ".join(json_list)

                    root_video = os.path.join(root_MMC, id_p, "01_measurement", "04_Video", "03_Cut", "drinking")
                    video_files = []
                    for cam in cams_used:
                        video = glob.glob(os.path.join(root_video, f"{cam}", f"trial_{trial_number}_*"))[0]
                        video_files.append(video)


                    trials_df.loc[trials_df["identifier"] == identifier, "videos_used"] = (
                            trials_df.loc[trials_df["identifier"] == identifier, "videos_used"] +
                            ", " + str(video_files))



        if args.verbose >= 1:
            progress_bar.close()

        if args.verbose >= 1:
            progress_bar = tqdm(total=trials_df["identifier"].shape[0], desc="Creating Trial Objects",
                                unit="Trial")

        trial_list = []
        for identifier in trials_df["identifier"]:

            if args.verbose >= 2:
                print(f"Setting and Videos for {identifier} are: ",
                      trials_df.loc[trials_df["identifier"] == identifier, "setting_id"].values[0],
                      trials_df.loc[trials_df["identifier"] == identifier, "videos_used"].values[0])

            # get list from strings of videos and cams
            videos = trials_df.loc[trials_df["identifier"] == identifier, "videos_used"].values[0].split(", ")
            cams = trials_df.loc[trials_df["identifier"] == identifier, "cams_used"].values[0].split(", ")
            affected = trials_df.loc[trials_df["identifier"] == identifier, "affected"].values[
                0]  # get Participant ID for use in the Pipeline
            side = trials_df.loc[trials_df["identifier"] == identifier, "side"].values[
                0]  # get Participant ID for use in the Pipeline

            id_s = trials_df.loc[trials_df["identifier"] == identifier, "setting_id"].values[
                0]  # get Setting ID for use as Session ID in the Pipeline
            id_p = trials_df.loc[trials_df["identifier"] == identifier, "patient_id"].values[
                0]  # get Participant ID for use in the Pipeline
            id_t = trials_df.loc[trials_df["identifier"] == identifier, "trial_id"].values[
                0]  # get Trial ID for use in the Pipeline

            s_dir = trials_df.loc[trials_df["identifier"] == identifier, "session_dir"].values[
                0]  # get Session Directory for use in the Pipeline
            p_dir = trials_df.loc[trials_df["identifier"] == identifier, "participant_dir"].values[
                0]  # get Participant Directory for use in the Pipeline
            t_dir = trials_df.loc[trials_df["identifier"] == identifier, "trial_dir"].values[
                0]  # get Trial Directory for use in the Pipeline
            json_list = trials_df.loc[trials_df["identifier"] == identifier, "json_list"].values[0]

            # Create the trial object
            trial = iDrinkTrial.Trial(identifier=identifier, id_s=id_s, id_p=id_p, id_t=id_t,
                                      dir_root=root_data, dir_default=default_dir,
                                      dir_trial=t_dir, dir_participant=p_dir, dir_session=s_dir,
                                      affected=affected, measured_side=side,
                                      video_files=videos, used_cams=cams, json_list=json_list,
                                      used_framework=args.poseback, pose_model="Coco17_UpperBody")

            trial.cam_setting = trials_df.loc[trials_df["identifier"] == identifier, "cam_setting"].values[0]

            trial.create_trial()
            trial.load_configuration()

            trial.config_dict["pose"]["videos"] = videos
            trial.config_dict["pose"]["cams"] = cams

            trial.config_dict.get("project").update({"project_dir": trial.dir_trial})
            trial.config_dict['pose']['pose_framework'] = trial.used_framework
            trial.config_dict['pose']['pose_model'] = trial.pose_model

            trial.save_configuration()

            trial_list.append(trial)

            if args.verbose >= 1:
                progress_bar.update(1)
        if args.verbose >= 1:
            progress_bar.close()

        # Return the trial_list
        return trial_list


    def trials_from_p2s():
        pass

    def trials_from_opensim():
        pass

    def trials_from_murphy():
        pass

    trial_list = []

    if df_trials is not None:
        trial_list = iDrinkLog.trials_from_csv(args, df_trials, df_settings, root_data, default_dir)

    if not trial_list:
        # Create the trial objects depending on the mode
        match args.mode:
            case "pose_estimation":
                print("creating trial objects for Pose Estimation")
                trial_list = trials_from_video()

            case "pose2sim":
                print("creating trial objects for Pose2Sim")
                trial_list = trials_from_json()

            case "opensim":
                print("creating trial objects for Opensim")

            case "murphy_measures":
                print("creating trial objects for Murphy Measures")

            case "statistics":
                print("creating trial objects for Statistics")

            case "full":
                print("creating trial objects for Full Pipeline")
            case _:
                print("No Mode was given. Please specify a mode.")
                sys.exit(1)

        iDrinkLog.trials_to_csv(args, trial_list, os.path.join(root_val, "validation_trials.csv"))

    return trial_list





    # Get all video files
    video_files = glob.glob(os.path.join(root_MMC, "**", "*.mp4"), recursive=True)
    # Get all json files
    json_files = glob.glob(os.path.join(root_MMC, "**", "*.json"), recursive=True)
    # Get all trc files
    trc_files = glob.glob(os.path.join(root_OMC, "**", "*.trc"), recursive=True)

    """# Create the trial objects
    for video_file in video_files:
        # Get the trial_id
        trial_id = os.path.basename(video_file).split("_")[0]

        # Get the patient_id
        patient_id = os.path.basename(os.path.dirname(video_file))

        # Get the json file
        json_file = [j for j in json_files if trial_id in j and patient_id in j]

        # Get the trc file
        trc_file = [t for t in trc_files if trial_id in t and patient_id in t]

        # Create the trial object
        trial = Trial(trial_id, patient_id, video_file, json_file, trc_file)

        # Run the pipeline
        trial.run_pipeline(mode)"""



def run_mode():
    """
    Runs the pipeline for given mode.

    :param:
    :return:
    """
    # First create list of trials to iterate through
    trial_list = create_trial_objects()

    # before starting on any mode, make sure, each Trial has their respective calibration file generated.
    for trial in trial_list:
        if trial.calib == None:
            if args.verbose >= 2:
                print(f"Start calibration for {trial.identifier}")
            delta_calibration_val(trial, os.path.join(root_val, "calib_errors.csv"), args.verbose)

    iDrinkLog.update_trial_csv(args, trial_list, os.path.join(root_val, "validation_trials.csv"))
    # TODO: Implement trial csv update


    match args.mode:
        case "pose_estimation":  # Runs only the Pose Estimation
            print("Pose Estimaton Method: ", args.poseback)

            match args.poseback:
                case "openpose":
                    print("Running Openpose")
                    #trial_list = create_trial_objects(mode)
                    raise NotImplementedError("Openpose is not yet implemented")


                case "mmpose":
                    print("Pose Estimation mode: MMPose starting.")
                    for trial in trial_list:
                        if args.verbose >= 1:
                            print(f"starting Pose Estimation for: {trial.identifier}")

                        iDrinkPoseEstimation.validation_pose_estimation_2d(trial, root_val, writevideofiles=True,
                                                                           filter_2d=False, DEBUG=False)


                case "pose2sim":
                    from Pose2Sim import Pose2Sim
                    print("Pose Estimation mode: Pose2Sim starting.")
                    for trial in trial_list:
                        # Change the config_dict so that the correct pose model is used

                        if trial.pose_model == "Coco17_UpperBody":
                            trial.config_dict['pose']['pose_model'] = 'COCO_17'
                        Pose2Sim.poseEstimation(trial.config_dict)

                        trial.config_dict['pose']['pose_model'] = trial.pose_model

                case "metrabs_multi":
                    from Metrabs_PoseEstimation.metrabsPose2D_pt import metrabs_pose_estimation_2d_val
                    print("Pose Estimation mode: Metrabs Multi Cam starting.")
                    model_path = os.path.realpath(os.path.join(metrabs_models_dir, 'pytorch', 'metrabs_eff2l_384px_800k_28ds_pytorch'))

                    for trial in trial_list:
                        trial.pose_model = "bml_movi_87"

                        metrabs_pose_estimation_2d_val(curr_trial=trial, video_files=trial.video_files, calib_file=trial.config_dict, model_path=model_path, identifier=trial.identifier, root_val=root_val, skeleton=trial.pose_model)

                case "metrabs_single":
                    print("Pose Estimation mode: Metrabs Single Cam starting.")

                    model_path = os.path.realpath(
                        os.path.join(metrabs_models_dir, 'pytorch', 'metrabs_eff2l_384px_800k_28ds_pytorch'))
                    




                case _:  # If no mode is given
                    print("Invalid Mode was given. Please specify a valid mode.")
                    sys.exit(1)


        case "pose2sim":  # Runs only Pose2Sim

            from Pose2Sim import Pose2Sim
            p2s_progress = tqdm(total=len(trial_list), iterable=trial_list, desc="Running Pose2Sim", unit="Trial")
            for trial in trial_list:
                trial.run_pose2sim()

                p2s_progress.update(1)

            p2s_progress.close()



        case "opensim":  # Runs only Opensim
            print("Johann, take this out")

        case "murphy_measures":  # runs only the calculation of murphy measures
            print("Johann, take this out")

        case "statistics":  # runs only the statistic script
            print("Johann, take this out")

        case "full":  # runs the full pipeline
            print("Johann, take this out")

        case _:  # If no mode is given
            raise ValueError("Invalid Mode was given. Please specify a valid mode.")
            sys.exit(1)



if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()

    if args.DEBUG or sys.gettrace() is not None:
        print("Debug Mode is activated\n"
              "Starting debugging script.")
        args.mode = "pose_estimation"
        #args.mode = 'pose2sim'
        args.poseback = 'metrabs_multi'





    if args.mode is not None:
        print("Starting with Mode: ", args.mode)
        run_mode()
    else:
        print("No Mode was given. Please specify a mode.")
        sys.exit(1)