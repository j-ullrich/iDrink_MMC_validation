import glob
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup
from openpyxl import Workbook


def cam_as_ray_calibration():
    """Not Yet Implemented"""

    return

def delta_calibration_val(curr_trial, path_error_csv, verbose=1):
    """
    This function runs the calibration for the session.
    It looks for existing calibration recordings and generates a Calib.toml in the calibration directory
    based on these videos.
    This Function is targeted for use in the iDrink GUI.

    self can be Session or Trial Object
    """

    # Check if calibration file already exists
    calib_file = os.path.join(curr_trial.dir_calib, f'Calib_{curr_trial.id_s}_{curr_trial.id_p}.toml')
    if os.path.isfile(calib_file):
        if verbose >= 2:
            print(f"Calibration file {calib_file} already exists.")
        curr_trial.calib = calib_file
        curr_trial.calib_done = True
        return

    # prepare Log of Calibration errors
    if os.path.isfile(path_error_csv):
        df_error = pd.read_csv(path_error_csv, sep=';')
    else:
        df_error = pd.DataFrame(columns=["date", "time", "s_id", "p_id",  "cam_used", "error"])

    # Find all video files in the calibration folder
    cams = curr_trial.used_cams

    formats = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    patterns = [os.path.join(curr_trial.dir_calib_videos, f) for f in formats]
    video_files = []
    for pattern in patterns:
        video_files.extend(glob.glob(pattern))

    # Find all video files in the calibration folder
    cam_names= []
    pattern = "".join([f"{i}" for i in cams])
    pattern = re.compile(f'cam[{pattern}]').pattern
    for file_name in video_files:
        match = re.search(pattern, file_name)
        if match:
            cam_names.append(match.group())

    if verbose >= 2:
        print(cam_names)

    # run calibration for each configuration 1.save toml  2. save reporjection error in a new excel where all configurations are listed
    # board = CharucoBoard(squaresX=7, squaresY=5, square_length=115, marker_length=92, marker_bits=4, dict_size=250, manually_verify=False)  # here, in mm but any unit works
    board = CharucoBoard(squaresX=7, squaresY=5, square_length=115 / 1000, marker_length=92 / 1000, marker_bits=4,
                         dict_size=250, manually_verify=False)  # here, in mm but any unit works
    # run calibration for each configuration 1.save toml  2. save reporjection error in a new excel where all configurations are listed
    # Perform calibration for each valid configuration
    cgroup = CameraGroup.from_names(cam_names, fisheye=False)
    # Perform calibration
    videos = [[video] for video in video_files]
    error, all_rows = cgroup.calibrate_videos(videos, board)

    new_row = pd.Series({"date": time.strftime("%d.%m.%Y"), "time": time.strftime("%H:%M:%S"), "s_id": curr_trial.id_s, "p_id": curr_trial.id_p, "cam_used": cam_names, "error": error})

    df_error = pd.concat([df_error, new_row.to_frame().T], ignore_index=True)

    # Save the camera group configuration to a TOML file named after the configuration
    curr_trial.calib = calib_file
    cgroup.dump(curr_trial.calib)

    curr_trial.calib_done = True

    df_error.to_csv(path_error_csv, sep=';', index=False)

    if verbose >= 1:
        print(f"Calibration for Session {curr_trial.id_s} done and saved to {curr_trial.calib}.")

def delta_full_calibration_val(curr_trial, path_error_csv, verbose=1):
    """
    This function runs the calibration for the session.
    It looks for existing calibration recordings and generates a Calib.toml in the calibration directory
    based on these videos.
    This Function is targeted for use in the iDrink GUI.

    self can be Session or Trial Object
    """

    # Check if calibration file already exists
    calib_file = os.path.join(curr_trial.dir_calib_videos, f'Calib_full_{curr_trial.id_p}.toml')
    if os.path.isfile(calib_file):
        if verbose >= 2:
            print(f"Calibration file {os.path.basename(calib_file)} already exists.")
        return

    # prepare Log of Calibration errors
    if os.path.isfile(path_error_csv):
        df_error = pd.read_csv(path_error_csv, sep=';')
    else:
        df_error = pd.DataFrame(columns=["p_id",  "cam_used", "error"])

    # Find all video files in the calibration folder
    cams = curr_trial.used_cams

    formats = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    patterns = [os.path.join(curr_trial.dir_calib_videos, f) for f in formats]
    video_files = []
    for pattern in patterns:
        video_files.extend(glob.glob(pattern))


    # Find all video files in the calibration folder
    cam_names= []
    for file_name in video_files:
        match = re.search(r'cam\d+', file_name)
        if match:
            cam_names.append(match.group())

    if verbose >= 2:
        print(cam_names)

    # run calibration for each configuration 1.save toml  2. save reporjection error in a new excel where all configurations are listed
    # board = CharucoBoard(squaresX=7, squaresY=5, square_length=115, marker_length=92, marker_bits=4, dict_size=250, manually_verify=False)  # here, in mm but any unit works
    board = CharucoBoard(squaresX=7, squaresY=5, square_length=115 / 1000, marker_length=92 / 1000, marker_bits=4,
                         dict_size=250, manually_verify=False)  # here, in mm but any unit works
    # run calibration for each configuration 1.save toml  2. save reporjection error in a new excel where all configurations are listed
    # Perform calibration for each valid configuration
    cgroup = CameraGroup.from_names(cam_names, fisheye=False)
    # Perform calibration
    videos = [[video] for video in video_files]
    error, all_rows = cgroup.calibrate_videos(videos, board)

    new_row = pd.Series({"p_id": curr_trial.id_p, "cam_used": cam_names, "error": error})
    df_error = pd.concat([df_error, new_row.to_frame().T], ignore_index=True)

    # Save the camera group configuration to a TOML file named after the configuration
    cgroup.dump(calib_file)
    df_error.to_csv(path_error_csv, sep=';', index=False)

    if verbose >= 1:
        print(f"Full Calibration for Particiant {curr_trial.id_p} done and saved to {calib_file}.")



def calibrate_vids_in_directory(directory, verbose=1):


    """Calibrates videos in given directory."""

    p_id = re.search("P\d+", directory).group()
    calib_file = os.path.join(directory, f'{p_id}_calibration.toml')

    formats = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    patterns = [os.path.join(directory, f) for f in formats]
    video_files = []
    for pattern in patterns:
        video_files.extend(glob.glob(pattern))


    # Find all video files in the calibration folder
    cam_names= []
    for file_name in video_files:
        match = re.search(r'cam\d+', file_name)
        if match:
            cam_names.append(match.group())

    # run calibration for each configuration 1.save toml  2. save reporjection error in a new excel where all configurations are listed
    # board = CharucoBoard(squaresX=7, squaresY=5, square_length=115, marker_length=92, marker_bits=4, dict_size=250, manually_verify=False)  # here, in mm but any unit works
    board = CharucoBoard(squaresX=7, squaresY=5, square_length=115 / 1000, marker_length=92 / 1000, marker_bits=4,
                         dict_size=250, manually_verify=False)  # here, in mm but any unit works
    # run calibration for each configuration 1.save toml  2. save reporjection error in a new excel where all configurations are listed
    # Perform calibration for each valid configuration
    cgroup = CameraGroup.from_names(cam_names, fisheye=False)
    # Perform calibration
    videos = [[video] for video in video_files]
    error, all_rows = cgroup.calibrate_videos(videos, board)

    # Save the camera group configuration to a TOML file named after the configuration
    cgroup.dump(calib_file)


if __name__ == '__main__':

    #calibrate_vids_in_directory(r"I:\Delta\data_newStruc\P13\01_Measurement\04_Video\05_Calib_before")
    d = []

    for i in range(7, 253):
        p_id = f"P{i:02d}"
        d.append(rf"I:\Delta\data_newStruc\{p_id}\01_Measurement\04_Video\05_Calib_before")

    """d = [r"I:\Delta\data_newStruc\P10\01_Measurement\04_Video\05_Calib_before",
         r"I:\Delta\data_newStruc\P11\01_Measurement\04_Video\05_Calib_before"]"""

    for directory in d:
        if os.path.isdir(directory):
            calibrate_vids_in_directory(directory)

    pass