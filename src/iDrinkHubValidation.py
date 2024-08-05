import glob
import os
import sys
import time
import re

import argparse
import pandas as pd

from iDrink import iDrinkTrial

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
                    help='Method for Pose Estimation: "openpose", "mmpose", "pose2sim"')
parser.add_argument('--trial_id', metavar='t_id', type=str, default=None,
                    help='Trial ID if only one single trial should be processed')
parser.add_argument('--patient_id', metavar='p_id', type=str, default=None,
                    help='Patient ID if only one single patient should be processed')
parser.add_argument('--DEBUG', action='store_true', default=False,
                    help='Debug mode')

root_MMC = r"C:\iDrink\Test Folder structures"  # Root directory of all MMC-Data --> Videos and Openpose json files
root_OMC = r"C:\iDrink\OMC_data_newStruct"  # Root directory of all OMC-Data --> trc of trials.
root_data = r"C:\iDrink\validation_root"  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
default_dir = os.path.join(root_data, "default_files")  # Default Files for the iDrink Validation
df_settings = pd.read_csv(os.path.join(root_data, "validation_settings.csv"), sep=';')


def create_trial_objects(mode):
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


    :param mode:
    :return:
    """
    def trials_from_video():
        """
        Creates the trial objects for the Pose Estimation Pipeline.

        :return:
        """


        if sys.gettrace() is not None:  # If Debugger is in Use, limit settings to first 10
            n_settings = 10
        else:
            n_settings = df_settings["setting_id"].max()

        for setting_id in range(1, n_settings+1):
            s_id = f"S{setting_id:03d}"
            cam_setting = df_settings[df_settings["setting_id"] == setting_id]["cam_setting"][0]

            for p in os.listdir(root_MMC):

                p_dir = os.path.join(root_MMC, p)
                # Get the patient_id
                p_id = re.search(r'(P\d+)', p_dir).group(1)

                # Get all video files of patient
                video_files = glob.glob(os.path.join(p_dir, "**", "*.mp4"), recursive=True)

                videos_used = []
                cams_used = []

                affected_dict = {}
                side_dict = {}

                trials_df = pd.DataFrame(columns=["trial_id", "affected", "side", "cam_setting", "cams_used", "videos_used"])

                for video_file in video_files:
                    # Extract trial ID and format it
                    trial_number = int(os.path.basename(video_file).split('_')[0].replace('trial', ''))
                    t_id = f'T{trial_number:03d}'

                    if t_id not in trials_df["trial_id"].values:
                        # Add new row to dataframe only containing the trial_id
                        new_row = pd.Series({"trial_id": t_id, "cams_used": [], "videos_used": []})
                        trials_df = pd.concat([trials_df, new_row.to_frame().T], ignore_index=True)

                    try:
                        affected = re.search(r'unaffected', video_file).group(0)
                    except:
                        affected = 'affected'

                    side = re.search(r'R|L', video_file).group(0)
                    cam = re.search(r'(cam\d+)', video_file).group(0)



                    # Check whether Cam is used for Setting
                    cams_tuple = eval(df_settings[df_settings["setting_id"] == setting_id]["cams"][0])  # Get the tuple of cams for the setting

                    if int(re.search(r'\d+', cam).group()) not in cams_tuple:
                        continue
                    else:
                        temp = trials_df[trials_df["trial_id"] == t_id]["cams_used"]
                        temp.append(int(re.search(r'\d+', cam).group()))
                        trials_df[trials_df["trial_id"] == t_id]["cams_used"] = temp

                        tempt = trials_df[trials_df["trial_id"] == t_id]["videos_used"]
                        tempt.append(video_file)
                        trials_df[trials_df["trial_id"] == t_id]["videos_used"] = tempt



                    trials_df[trials_df["trial_id"] == t_id]["affected"] = affected
                    trials_df[trials_df["trial_id"] == t_id]["side"] = side
                    trials_df[trials_df["trial_id"] == t_id]["cam_setting"] = cam_setting


                print("Videos used for setting: ", setting_id, videos_used)
                # Create the trial object
                trial = iDrinkTrial.Trial(id_p=p_id, id_t=t_id, affected=affected, measured_side=side)

                # Run the pipeline
                trial.run_pipeline(mode)

    def trials_from_json():
        pass
    def trials_from_p2s():
        pass

    def trials_from_opensim():
        pass

    def trials_from_murphy():
        pass



    match mode:
        case "pose_estimation":
            print("creating trial objects for Pose Estimation")
            trials_from_video()

        case "pose2sim":
            print("creating trial objects for Pose2Sim")

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



def run_mode(mode):
    """
    Runs the pipeline for given mode.

    :param mode:
    :return:
    """

    match args.mode:
        case "pose_estimation":  # Runs only the Pose Estimation
            print("Pose Estimaton Method: ", args.poseback)

        case "pose2sim":  # Runs only the Pose2Sim
            print("Johann, take this out")

        case "opensim":  # Runs only Opensim
            print("Johann, take this out")

        case "murphy_measures":  # runs only the calculation of murphy measures
            print("Johann, take this out")

        case "statistics":  # runs only the statistic script
            print("Johann, take this out")

        case "full":  # runs the full pipeline
            print("Johann, take this out")

        case _:  # If no mode is given
            print("Invalid Mode was given. Please specify a valid mode.")
            sys.exit(1)



if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()

    if args.DEBUG or sys.gettrace() is not None:
        print("Debug Mode is activated\n"
              "Starting debugging script.")
        mode = 'pose_estimation'

        create_trial_objects(mode)




    if args.mode is not None:
        print("Starting with Mode: ", args.mode)
        run_mode(args.mode)
    else:
        print("No Mode was given. Please specify a mode.")
        sys.exit(1)