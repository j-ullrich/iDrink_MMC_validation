import glob
import os
import sys
import time
import re
import shutil
import platform

from tqdm import tqdm

import argparse
import pandas as pd

from iDrink import iDrinkConfig, iDrinkTrial, iDrinkPoseEstimation, iDrinkLog, iDrinkOpenSim, iDrinkUtilities, iDrinkMurphyMeasures, iDrinkStatisticalAnalysis
from iDrink.iDrinkCalibration import delta_calibration_val, delta_full_calibration_val
from iDrink.iDrinkPipeline import PipelineContext, STAGES, selected, steps_for_mode


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

# Command-line interface definition
parser = argparse.ArgumentParser(description='Run the iDrink validation pipeline.')
parser.add_argument('--stages', '--mode', dest='mode', metavar='S', default=None,
                    help='Comma separated stages to run, any of: ' + ', '.join(STAGES))
parser.add_argument('--poseback', '--pose-backend', dest='poseback', metavar='hpe', default='metrabs_multi',
                    help='Comma separated Methods for Pose Estimation: "mmpose", "pose2sim", "metrabs_multi", "all"')
parser.add_argument('--identifier', dest='single_identifier', metavar='id', default=False,
                    help='Trial identifier if only one single trial should be processed')
parser.add_argument('--patients', default=None,
                    help='Comma separated patient ids to process, e.g. P07,P08. Default: [run] patients of config.toml, else all.')
parser.add_argument('--settings', default=None,
                    help='Comma separated setting ids to process, e.g. S027,S028. Default: [run] settings of config.toml, else all.')
parser.add_argument('--only-single-cam', dest='only_single_cam_trials', action='store_true', default=False,
                    help='Only process trials recorded with a single camera')
parser.add_argument('--run-again', action='store_true', default=False,
                    help='Process trials again that are already done')
parser.add_argument('--verbose', metavar='v', type=int, default=1,
                    help='Verbosity level: 0, 1, 2 default: 1')
parser.add_argument('--config', default=None,
                    help='Path to config.toml (default: IDRINK_CONFIG or src/config.toml)')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='Only list the trials that would be processed')
parser.add_argument('--DEBUG', action='store_true', default=False,
                    help='Debug mode')


def _csv_list(value):
    return [v.strip() for v in value.split(',') if v.strip()] if value else None


def prepare_args(args, config):
    """Normalise the parsed arguments to the attributes the run_* functions read."""
    args.mode = _csv_list(args.mode)
    if args.mode is not None:
        invalid = [m for m in args.mode if m not in STAGES]
        if invalid:
            parser.error(f"Unknown stage(s) {invalid}. Choose from {STAGES}")
    args.poseback = _csv_list(args.poseback)
    args.single_patient = _csv_list(args.patients) or (list(config.patients) or False)
    args.single_setting = _csv_list(args.settings) or (list(config.settings) or False)
    return args


def setup(config_file=None):
    """
    Resolve paths, create the directory tree and load the settings/trials csv.

    :return: PipelineContext. (args are added by main())
    """
    config = iDrinkConfig.load_config(config_file)
    iDrinkConfig.ensure_dirs(config)

    os.environ['DATA_ROOT'] = str(config.root_iDrink)  # Set the DATA_ROOT Environment Variable for Metrabs
    os.environ['ROOT_DIR'] = str(config.root_iDrink)  # Set the ROOT_DIR Environment Variable for Metrabs

    df_settings = pd.read_csv(config.log_val_settings, sep=';')  # csv containing information for the various settings in use.
    try:
        df_trials = pd.read_csv(config.log_val_trials, sep=';')
    except FileNotFoundError:
        print("No .csv file for trials found.")
        df_trials = None

    return PipelineContext(paths=config, args=None, df_settings=df_settings, df_trials=df_trials)


def create_trial_objects(ctx):
    """
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
    args = ctx.args
    default_dir = ctx.default_dir
    df_settings = ctx.df_settings
    df_trials = ctx.df_trials
    log_val_trials = ctx.log_val_trials
    root_MMC = ctx.root_MMC
    root_data = ctx.root_data

    n_settings = df_settings["setting_id"].max()

    def create_trial_list(trials_df):
        """Create_Trial_list based on trials_dataframe"""

        if args.verbose >= 1:
            progress_bar = tqdm(total=trials_df["identifier"].shape[0], desc="Creating Trial Objects:",
                                unit="Trial")
        trial_list = []
        for identifier in trials_df["identifier"]:
            if args.verbose >= 1:
                progress_bar.set_description(f"Creating Trial Objects: {identifier}")

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
            cams_tuple = eval(df_settings.loc[df_settings["setting_id"] == setting_id, "cams"].values[0])  # Get the tuple of cams for the setting¨

            if type(cams_tuple) == int:
                cams_tuple = [cams_tuple]


            # Create Setting folder if not yet done
            dir_setting = os.path.join(root_data, f"setting_{setting_id:03d}")
            if not os.path.exists(dir_setting):
                os.makedirs(dir_setting, exist_ok=True)

            for p in sorted(os.listdir(root_MMC)):
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

                cam_folders = []
                for cam_number in cams_tuple:
                    cam_root = os.path.join(root_MMC, p, "01_measurement", "04_Video", "03_Cut", "drinking")
                    if f'cam{cam_number}' in os.listdir(cam_root):
                        cam_folders.append(os.path.join(cam_root, f'cam{cam_number}'))

                # Get all video files of patient
                video_files = []
                for cam_folder in cam_folders:
                    video_files.extend(glob.glob(os.path.join(cam_folder, "**", "*.mp4"), recursive=True))

                for video_file in video_files:
                    # Extract trial ID and format it
                    trial_number = int(re.search(r'trial_\d+', video_file).group(0).split('_')[1])
                    id_t = f'T{trial_number:03d}'

                    if args.verbose >=1:
                        #update name of progressbar
                        progress_bar.set_description(f"Adding Trials to DataFrame: {id_s}_{id_p}_{id_t}")

                    identifier = f"{id_s}_{id_p}_{id_t}"

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

        return trials_df


    trial_list = []
    if df_trials is not None:
        trial_list = iDrinkLog.trials_from_csv(args, df_trials, df_settings, root_data, default_dir)
    else:
        # Create the trial objects depending on the mode

        df_temp = trials_from_video()
        trial_list = create_trial_list(df_temp)

        ctx.df_trials = df_trials = iDrinkLog.trials_to_csv(args, trial_list, log_val_trials, get_df=True)

    return trial_list

def run_calibrations(ctx, trial_list):
    args = ctx.args
    df_settings = ctx.df_settings
    log_calib = ctx.log_calib
    log_calib_full = ctx.log_calib_full
    log_val_errors = ctx.log_val_errors
    log_val_trials = ctx.log_val_trials
    root_data = ctx.root_data

    failed_p = []
    failed_s = []
    failed_p_full = []
    s_done = []
    s_skipped_single_cam = []
    s_skipped = []
    s_all = []
    p_done = []
    p_all = []
    for trial in trial_list:

        # Calibrate for all cameras used in recording
        try:
            if trial.id_p in failed_p_full:
                if args.verbose >= 2:
                    print(f"Skip full calibration for {trial.identifier} as it failed before.")
            else:

                # Full Calibration is needed for Pose Estimation. There a calib file containing all cameras is needed.
                delta_full_calibration_val(trial, log_calib_full, args.verbose)
        except Exception as e:
            if args.verbose >= 2:
                print(f"Error in Full Calibration for {trial.identifier}")
                print(e)
            iDrinkLog.log_error(args, trial, e, 'calibration_full', '', log_val_errors)

            failed_p_full.append(trial.id_p)

        # Calibrate for cameras used in trial and setting
        try:

            if trial.id_p in failed_p and trial.id_s in failed_s:
                if args.verbose >= 2:
                    print(f"Skip calibration for {trial.identifier} as it failed before.")
                    s_skipped.append(trial.id_s)
            else:
                if args.verbose >= 2:
                    print(f"Start calibration for {trial.identifier}")
                if len(trial.used_cams) > 1:
                    delta_calibration_val(trial, log_calib, args.verbose, df_settings, root_data)
                    s_done.append(trial.id_s)
                    p_done.append(trial.id_p)
                else:
                    if args.verbose >= 2:
                        print(f"Skip calibration for {trial.identifier} as only one camera is used.")
                        s_skipped_single_cam.append(trial.id_s)

        except Exception as e:
            if args.verbose >= 2:
                print(f"Error in Calibration for {trial.identifier}")
                print(e)
            iDrinkLog.log_error(args, trial, e, 'calibration', '', log_val_errors)
            failed_p.append(trial.id_p)
            failed_s.append(trial.id_s)

        s_all.append(trial.id_s)
        p_all.append(trial.id_p)


    return iDrinkLog.update_trial_csv(trial_list, log_val_trials)

def _sync_done_from_csv(ctx, trial, flag):
    """If the trials csv already marks the trial as done for this backend, take over the flag and frame count."""
    df_trials = ctx.df_trials
    rows = (df_trials["id_t"] == trial.id_t) & (df_trials["id_p"] == trial.id_p)
    if any(df_trials.loc[rows, flag].values):
        setattr(trial, flag, True)
        trial.n_frames = max(df_trials.loc[rows, "n_frames"])
        df_trials.loc[rows, flag] = True


def _hpe_mmpose(ctx, trial):
    video_files = iDrinkPoseEstimation.get_all_trial_vids(trial)
    iDrinkPoseEstimation.validation_pose_estimation_2d(trial, ctx.root_val, video_files=video_files,
                                                       writevideofiles=False, filter_2d=False, DEBUG=False)
    trial.MMPose_done = True
    trial.HPE_done = iDrinkLog.all_2d_HPE_done(trial)


def _hpe_pose2sim(ctx, trial):
    from Pose2Sim import Pose2Sim

    # Prepare Pose Estimation
    trial.config_dict['pose']['pose_model'] = 'COCO_17'
    vids_used = trial.config_dict['pose']['videos']
    trial.config_dict['pose']['videos'] = iDrinkPoseEstimation.get_all_trial_vids(trial)
    trial.config_dict['pose']['save_video'] = 'none'

    # Run Pose Estimation
    Pose2Sim.poseEstimation(trial.config_dict)

    # Filter json_files and packing them to .zip file
    iDrinkPoseEstimation.filt_p2s_pose(trial, ctx.root_val, verbose=ctx.args.verbose)

    trial.config_dict['pose']['pose_model'] = trial.pose_model
    trial.config_dict['pose']['videos'] = vids_used
    trial.P2SPose_done = True
    trial.HPE_done = iDrinkLog.all_2d_HPE_done(trial)


def _hpe_metrabs_multi(ctx, trial):
    from Metrabs_PoseEstimation.metrabsPose2D_pt import metrabs_pose_estimation_2d_val

    model_path = os.path.realpath(
        os.path.join(ctx.metrabs_models_dir, 'pytorch', 'metrabs_eff2l_384px_800k_28ds_pytorch'))
    video_files = sorted(iDrinkPoseEstimation.get_all_trial_vids(trial))
    metrabs_pose_estimation_2d_val(curr_trial=trial, video_files=video_files,
                                   calib_file=os.path.join(trial.dir_calib_videos, f'Calib_full_{trial.id_p}.toml'),
                                   model_path=model_path,
                                   identifier=trial.identifier, root_val=ctx.root_val,
                                   skeleton=trial.pose_model)
    trial.Metrabs_multi_done = True
    trial.HPE_done = iDrinkLog.all_2d_HPE_done(trial, ctx.root_HPE)


# name: (pose model, trial flag, flag column in the trials csv or None, key of the zip files, function)
HPE_BACKENDS = {
    "mmpose": ("Coco17_UpperBody", "MMPose_done", "MMPose_done", "mmpose", _hpe_mmpose),
    "pose2sim": ("Coco17_UpperBody", "P2SPose_done", "P2SPose_done", "pose2sim", _hpe_pose2sim),
    "metrabs_multi": ("bml_movi_87", "Metrabs_multi_done", None, "metrabs", _hpe_metrabs_multi),
}


def run_HPE(ctx, trial_list):
    args = ctx.args

    print("Pose Estimaton Method: ", args.poseback)

    if args.poseback == "all":
        args.poseback = ["mmpose", "pose2sim", "metrabs_multi"]
    elif type(args.poseback) != list:
        args.poseback = [args.poseback]

    for poseback in args.poseback:
        if poseback == "openpose":
            # TODO: Move Openpose Data from DataNewStruc to pose estimation data folder
            raise NotImplementedError("Openpose is not yet implemented")
        if poseback not in HPE_BACKENDS:
            print(f"Invalid HPE-Mode: {poseback}\n"
                  f"Please specify a valid mode.")
            sys.exit(1)

        pose_model, flag, csv_flag, zip_key, run_backend = HPE_BACKENDS[poseback]
        print(f"Pose Estimation mode: {poseback} starting.")

        for trial in trial_list:
            if args.verbose >= 1:
                print(f"starting Pose Estimation for: {trial.identifier}")

            trial.pose_model = pose_model

            if csv_flag:
                _sync_done_from_csv(ctx, trial, csv_flag)

            # Safety Check on the zip file with the json files
            setattr(trial, flag, iDrinkLog.does_HPE_zip_exist(trial, ctx.root_HPE, posebacks=[zip_key]))

            if getattr(trial, flag):
                if args.verbose >= 2:
                    print(f"Pose Estimation for {trial.identifier} already done.")
                continue

            try:
                run_backend(ctx, trial)
            except Exception as e:
                if args.verbose >= 2:
                    print(f"Error in Pose Estimation\n"
                          f"Trial: {trial.identifier}\tPoseback: {poseback}\n"
                          f"{e}")

                iDrinkLog.log_error(args, trial, e, '2D_pose_estimation', poseback, ctx.log_val_errors)
                setattr(trial, flag, False)

        ctx.df_trials = iDrinkLog.update_trial_csv(trial_list, ctx.log_val_trials)


def move_metrabs3d_trc(trial, filt, root_HPE):
    if filt == 'filtered':
        filt = '02_filtered'
        buttered = 'iDrinkbutterfilt'
    else:
        filt = '01_unfiltered'
        buttered = 'iDrink_unfilt'

    cam = trial.used_cams[0]

    paths_found = glob.glob(
        os.path.join(root_HPE, filt, trial.id_p, f'{trial.id_p}_cam{cam}', 'metrabs', 'single-cam',
                     f'*{trial.id_t}*.trc'))

    if len(paths_found) > 0:
        path_src = paths_found[0]
        path_dst = os.path.join(trial.dir_trial, 'pose-3d', f'{trial.identifier}_cam{cam}_{buttered}.trc')

        old_files = glob.glob(os.path.join(trial.dir_trial, 'pose-3d', f'.trc'))
        for file in old_files:
            os.remove(file)

        shutil.copy2(path_src, path_dst)


def run_triangulation(ctx, trial_list):
    args = ctx.args
    df_settings = ctx.df_settings
    df_trials = ctx.df_trials
    log_val_errors = ctx.log_val_errors
    log_val_trials = ctx.log_val_trials
    root_HPE = ctx.root_HPE
    root_logs = ctx.root_logs
    root_val = ctx.root_val
    from Pose2Sim import Pose2Sim


    if args.verbose >= 1:
        p2s_progress = tqdm(total=len(trial_list), iterable=trial_list, desc="Running Pose2Sim", unit="Trial")


    csv_failed_p2s = os.path.join(root_logs, "failed_p2s.csv")
    if os.path.exists(csv_failed_p2s):
        df_failed_p2s = pd.read_csv(csv_failed_p2s, sep=';')
    else:
        df_failed_p2s = pd.DataFrame(columns=["date", "time", "id_p", "id_s", "id_t", "identifier", "error"])

    for i, trial in enumerate(trial_list):
        if args.verbose >= 1:
            p2s_progress.set_description(f"Running Pose2Sim for: {trial.identifier}")

        if not selected(trial, args, patients=False):
            p2s_progress.update(1)
            continue

        # Get Pose method from settings dataframe
        pose = df_settings.loc[
            df_settings["setting_id"] == int(re.search("\d+", trial.id_s).group()), "pose_estimation"].values[0]
        filt = df_settings.loc[
            df_settings["setting_id"] == int(re.search("\d+", trial.id_s).group()), "filtered_2d_keypoints"].values[0]

        if len(trial.used_cams) == 1:
            move_metrabs3d_trc(trial, filt, root_HPE)
            trial.P2S_done = True

        else:
            if args.only_single_cam_trials:
                p2s_progress.update(1)
                continue

            # check if trial is in failed trials
            if trial.identifier in df_failed_p2s["identifier"].values:
                if args.verbose >= 2:  # Print the last time, the Trial failed in P2S before
                    when_date = df_failed_p2s.loc[df_failed_p2s["identifier"] == trial.identifier, "date"].values[-1]
                    when_time = df_failed_p2s.loc[df_failed_p2s["identifier"] == trial.identifier, "time"].values[-1]
                    because = df_failed_p2s.loc[df_failed_p2s["identifier"] == trial.identifier, "error"].values[-1]
                    print(f"Triangulation for {trial.identifier} failed before.\n"
                          f"It failed on {when_date} at {when_time} with error:\n "
                          f"{because}.")

                p2s_progress.update(1)
                continue

            trial.HPE_done = iDrinkLog.all_2d_HPE_done(trial, root_HPE, pose)

            if trial.HPE_done:

                trial.P2S_done = iDrinkLog.files_exist(os.path.join(trial.dir_trial, 'pose-3d'), '.trc',
                                                       args.verbose)

                if trial.P2S_done and not args.run_again:
                    if args.verbose >= 2:
                        print(f"Pose2Sim for {trial.identifier} already done.")
                    continue
                else:
                    if args.verbose >= 1:
                        print(f"Trial: {trial.identifier} \t Posemode: {pose}")
                    try:
                        iDrinkUtilities.unpack_zip_to_trial(trial, pose, filt, root_val)
                        if 'metrabs' in pose:
                            trial.config_dict['personAssociation']['single_person']['tracked_keypoint'] = 'thor'
                        try:
                            trial.run_pose2sim(only_triangulation=False)
                        except:
                            trial.config_dict['triangulation']['reproj_error_threshold_triangulation'] = 40
                            trial.config_dict['triangulation']['interp_if_gap_smaller_than'] = 100
                            trial.save_configuration()
                            trial.run_pose2sim(only_triangulation=False, do_sync=False)

                            """trial.config_dict['triangulation']['reproj_error_threshold_triangulation'] = 200
                            trial.config_dict['triangulation']['interp_if_gap_smaller_than'] = 50
                            trial.config_dict['triangulation']['handle_LR_swap'] = False
                            trial.save_configuration()
                            trial.run_pose2sim(only_triangulation=False, do_sync=False)
                            
                            
                            
                            
                            
                            """

                        trial.P2S_done = True

                    except Exception as e:
                        if args.verbose >= 2:
                            print(f"Pose2Sim for {trial.identifier} failed with error: {e}")

                        iDrinkLog.log_error(args, trial, e, 'Pose2Sim', pose, log_val_errors)

                        curr_time = time.localtime()
                        df_temp = pd.Series({"date": time.strftime("%d/%m/%Y", curr_time),
                                             "time": time.strftime("%H:%M:%S", curr_time),
                                             "id_p": trial.id_p, "id_s": trial.id_s, "id_t": trial.id_t,
                                             "identifier": trial.identifier,
                                             "error": str(e.args)})

                        df_failed_p2s = pd.concat([df_failed_p2s, df_temp], ignore_index=True)

                        trial.P2S_done = False
                        p2s_progress.update(1)

                iDrinkUtilities.del_json_from_trial(trial, False)

            else:
                print(f"Pose Estimation for {trial.identifier} not done yet. Please repeat Pose Estimation Mode")

        p2s_progress.update(1)

    if all([trial.P2SPose_done for trial in trial_list]) is False:
        print("Pose2Sim could not be completed for all Trials. Please Check whether Pose Estimaiton is fully done.\n"
              "If not, please run Pose Estimation first. And then repeat Pose2Sim Mode.")

    p2s_progress.close()
    ctx.df_trials = df_trials = iDrinkLog.update_trial_csv(trial_list, log_val_trials)

def write_txt_with_timestamps(directory, phase, identifier):
    """
    Writes a .txt file with the timestamp for debugging purposes.

    if the file already exists it appends the line to the file.



    :param directory:  string with the directory where the file should be saved
    :param phase: string with the phase of the pipeline
    :param identifier: identifier of trial
    :return:
    """

    curr_time = time.localtime()
    line = f"{identifier}\t {time.strftime('%d/%m/%Y %H:%M:%S', curr_time)}\t Phase: {phase}"

    if "last_trial.txt" in os.listdir(directory):
        with open(os.path.join(directory, "last_trial.txt"), 'a') as file:
            file.write(f'\n{line}')
    else:
        with open(os.path.join(directory, "last_trial.txt"), 'w') as file:
            file.write(line)

def run_opensim(ctx, trial_list):
    args = ctx.args
    df_settings = ctx.df_settings
    df_trials = ctx.df_trials
    log_val_errors = ctx.log_val_errors
    log_val_trials = ctx.log_val_trials
    root_logs = ctx.root_logs
    root_val = ctx.root_val



    if args.verbose >= 1:
        opensim_progress = tqdm(total=len(trial_list), iterable=trial_list, desc="Running Opensim", unit="Trial")
    for i, trial in enumerate(trial_list):
        opensim_progress.set_description(f"Running Opensim for: {trial.identifier}")
        """if trial.identifier != 'S003_P15_T079':
            continue"""

        if args.verbose >= 1:
            opensim_progress.update(1)
        pose = df_settings.loc[
            df_settings["setting_id"] == int(re.search("\d+", trial.id_s).group()), "pose_estimation"].values[0]

        if not selected(trial, args):
            opensim_progress.update(1)
            continue

        if args.only_single_cam_trials and len(trial.used_cams) > 1:
            opensim_progress.update(1)
            continue

        joint_kin_exist = iDrinkLog.files_exist(os.path.join(trial.dir_trial, 'movement_analysis', 'ik_tool'), '.csv')
        chest_pos_exist = iDrinkLog.files_exist(
            os.path.join(trial.dir_trial, 'movement_analysis', 'kin_opensim_analyzetool'), 'OutputsVec3.sto')

        trial.OS_done = joint_kin_exist and chest_pos_exist

        write_txt_with_timestamps(root_val, 'Opensim', trial.identifier)

        if trial.OS_done:
            if args.verbose >= 2:
                print(f"Opensim for {trial.identifier} already done.")
            continue
        else:
            try:
                trial.prepare_opensim()
                iDrinkOpenSim.open_sim_pipeline(trial, os.path.join(root_logs, 'opensim'), args.verbose)
                """
                mmpose: Coco17_UpperBody
                pose2sim: Coco17_UpperBody
                metrabs_multi: bml_movi_87
                """

                trial.OS_done = True
            except Exception as e:
                if args.verbose >= 2:
                    print(f"Error in openSim\n"
                          f"Trial: {trial.identifier}\n"
                          f"{e}")

                iDrinkLog.log_error(args, trial, e, 'Opensim', pose, log_val_errors)

                trial.OS_done = False

            iDrinkUtilities.del_geometry_from_trial(trial, verbose=args.verbose)

    if args.verbose >= 1:
        opensim_progress.close()
    ctx.df_trials = df_trials = iDrinkLog.update_trial_csv(trial_list, log_val_trials)

def run_murphy_measures(ctx, trial_list):
    args = ctx.args
    log_val_errors = ctx.log_val_errors
    path_csv_murphy_measures = ctx.path_csv_murphy_measures
    path_csv_murphy_timestamps = ctx.path_csv_murphy_timestamps
    root_data = ctx.root_data

    if args.verbose >= 1:
        murphy_progress = tqdm(total=len(trial_list), iterable=trial_list, desc="Running Murphy Measuresfor: ",
                               unit="Trial")

    for trial in trial_list:
        if args.verbose >= 1:
            murphy_progress.set_description(f"Running Murphy Measures for: {trial.identifier}")
            murphy_progress.update(1)


        # check that .sto files exist
        mov_files = glob.glob(os.path.join(trial.dir_trial, 'movement_analysis', 'ik_tool', f'*.csv'))

        chest_files = glob.glob(
            os.path.join(trial.dir_trial, 'movement_analysis', 'kin_opensim_analyzetool', f'*OutputsVec3.sto'))

        if not mov_files or not chest_files:
            if args.verbose >= 1:
                murphy_progress.set_description(f"No Mov data for for: {trial.identifier}")
            continue


        try:
            path_preprocessed = os.path.join(root_data, 'preprocessed_data', '01_murphy_out')

            iDrinkMurphyMeasures.MurphyMeasures(trial=trial, trial_id=trial.identifier,
                                                csv_timestamps=path_csv_murphy_timestamps,
                                                csv_measures=path_csv_murphy_measures,
                                                write_mov_data=True,
                                                path_mov_data=path_preprocessed,
                                                filt_fps=60,
                                                filt_cutoff_vel=4, filt_cutoff_pos=4,
                                                filt_order_pos=4, filt_order_vel=6,
                                                )
        except Exception as e:
            if args.verbose >= 2:
                print(f"Error in murphy_measures\n"
                      f"Trial: {trial.identifier}\n"
                      f"{e}")

            iDrinkLog.log_error(args, trial, e, 'Murphy', '', log_val_errors)

    if args.verbose >= 1:
        murphy_progress.close()


def run_statistics(ctx):
    args = ctx.args
    path_csv_murphy_measures = ctx.path_csv_murphy_measures
    root_data = ctx.root_data
    root_stat = ctx.root_stat
    root_val = ctx.root_val
    df_timestamps = pd.read_csv(os.path.join(root_stat, '02_categorical', 'murphy_timestamps.csv'), sep=';')

    # iDrinkStatisticalAnalysis.preprocess_timeseries(dir_root=root_val, detect_outliers=False, verbose=args.verbose)
    iDrinkStatisticalAnalysis.get_omc_mmc_error_old(dir_root=root_val, df_timestamps=df_timestamps, verbose=args.verbose)

    corrections = ['fixed', 'dynamic']

    dir_processed = os.path.join(root_data, 'preprocessed_data')
    dir_results = os.path.join(root_stat, '01_continuous', '01_results')

    for correct in corrections:
        preprocess_timeseries(root_val,
                                downsample=True, drop_last_rows=False, detect_outliers= det_outliers,
                                joint_vel_thresh=thresh_elbowVelocity, hand_vel_thresh=hand_vel_thresh, correct=correct, fancy_offset=False,
                                verbose=1, plot_debug=False, print_able=False, empty_dst=True, debug=debug, debug_c=50)
        dir_src = '02_fully_preprocessed' if correct == 'fixed' else '03_fully_preprocessed_dynamic'
        dir_src = os.path.join(root_data, 'preprocessed_data', dir_src)
        normalize_data(dir_src=dir_src, dynamic = True if correct == 'dynamic' else False, verbose=1)

    get_error_timeseries(dir_processed = dir_processed, dir_results = dir_results, empty_dst=True, verbose=1, debug=debug)
    get_error_mean_rmse(dir_results, overwrite_csvs=True, verbose=1)
    get_rom_rmse(dir_results, overwrite_csvs=True, verbose=1)
    get_timeseries_correlations(dir_processed, dir_results, overwrite_csvs=False, verbose=1)
    #iDrinkStatisticalAnalysis.get_multiple_correlations(dir_processed, dir_results, verbose=1) #TODO: Implement this function
    #iDrinkStatisticalAnalysis.get_omc_mmc_error_old(root_val, path_csv_murphy_timestamps, correct=correct, verbose=1)

    runs_statistics_discrete(path_csv_murphy_measures, root_stat, make_plots=True,
                             thresh_PeakVelocity_mms=None, thresh_elbowVelocity=None)
    runs_statistics_discrete(path_csv_murphy_measures, root_stat, make_plots=True,
                             thresh_PeakVelocity_mms=hand_vel_thresh, thresh_elbowVelocity=thresh_elbowVelocity)
    # generate Plots

STEPS = {"calibration": run_calibrations,
         "hpe": run_HPE,
         "triangulation": run_triangulation,
         "opensim": run_opensim,
         "murphy_measures": run_murphy_measures}


def run_mode(ctx, no_calib=False):
    """
    Runs the pipeline for the stages given in ctx.args.mode.

    :param ctx: PipelineContext
    :param no_calib: Skip the calibration that pose_estimation and pose2sim would otherwise run first.
    """
    args = ctx.args

    # First create list of trials to iterate through
    trial_list = create_trial_objects(ctx)
    trial_list.sort(key=lambda x: x.id_s, reverse=False)

    if type(args.mode) == str:
        args.mode = [args.mode]

    for mode in args.mode:
        for step in steps_for_mode(mode, args.poseback, no_calib):
            if step == "statistics":
                run_statistics(ctx)
            elif step == "calibration":
                ctx.df_trials = run_calibrations(ctx, trial_list)
            else:
                STEPS[step](ctx, trial_list)


def main(argv=None):
    args = parser.parse_args(argv)

    if args.DEBUG or sys.gettrace() is not None:
        print("Debug Mode is activated\n")

    ctx = setup(args.config)
    ctx.args = prepare_args(args, ctx.paths)

    if args.mode is None:
        parser.error("No stage given. Use --stages, e.g. --stages murphy_measures")

    print("Starting with Mode: ", args.mode)
    if args.dry_run:
        trial_list = create_trial_objects(ctx)
        print(f"Dry run: stages {args.mode} would process {len(trial_list)} trials:")
        for trial in trial_list:
            print("  ", trial.identifier)
        return

    run_mode(ctx, no_calib=False)


if __name__ == '__main__':
    main()
