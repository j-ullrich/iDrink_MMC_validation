import os
"""
Class for Session Objects

This Class is meant to store all values necessary for session creation and running the calibration. 
It must not contain any information necessary for running a trial analysis. Any Information that is needed for a trial 
needs to be transfered to the trial object when creating it.
This class is meant to create a new session and the corresponding folders.


Trial objects should contain the following informations:
- id_s: Session ID

Paths to directories:
- dir_root: Directory containing default, reference, session data and global settings file
- dir_default: directory containing all default files
- dir_ref: Directory containing reference files
- dir_calib: Directory containing calibration files and videos
"""

class Session:
    def __init__(self, id_s, dir_root, chosen_calib=None, choose_first_calib=True):
        self.dir_root = dir_root
        self.dir_default = os.path.realpath(os.path.join(self.dir_root, 'Default-Files'))
        self.dir_ref = os.path.realpath(os.path.join(self.dir_root, 'Reference_Data'))

        self.id_s = id_s
        self.dir_session = os.path.realpath(os.path.join(self.dir_root, 'Session Data', self.id_s))

        self.dir_calib = os.path.realpath(os.path.join(self.dir_session, f'{self.id_s}_calibration'))
        self.dir_calib_videos = os.path.realpath(os.path.join(self.dir_calib, f'{self.id_s}_calibration_videos'))
        self.dir_calib_files = os.path.realpath(os.path.join(self.dir_calib, f'{self.id_s}_calibration_files'))
        self.chosen_calib = chosen_calib
        self.choose_first_calib = choose_first_calib
        self.used_calibration = None
        self.calib_file = None  # Path to Calibration file of session

        self.cams_chosen = None
        self.rec_resolution = (1920, 1080)
        self.frame_rate = 60
        self.clean_video = True

    def create_session(self):
        """
        This function creates the session folder and the corresponding subfolders.
        """
        # Create the session folder
        os.makedirs(self.dir_session, exist_ok=True)
        # Create the calibration folder
        os.makedirs(self.dir_calib, exist_ok=True)
        os.makedirs(self.dir_calib_videos, exist_ok=True)
        os.makedirs(self.dir_calib_files, exist_ok=True)

    def record_calibration_videos(self, cams_chosen, send_frames):
        """
        This function records videos for each camera and saves them to the calibration videos folder.
        """

        from .iDrinkVisualInput import record_videos
        prefix = f'{self.id_s}_calibration_'
        video_files, _, _ = record_videos(cams_chosen=cams_chosen, resolution=self.rec_resolution,
                                       out_dir=self.dir_calib_videos, record_duration=60,
                                       filename_prefix=prefix, file_fps=self.frame_rate, send_frames=send_frames)

        return video_files

    def calibrate(self):
        """
        This function runs the calibration for the session.

        It looks for existing calibration recordings and generates a Calib.toml in the calibration directory
        based on these videos.

        This Function is targeted for use in the iDrink GUI.
        """
        import shutil
        import glob
        import os
        import re

        import pandas as pd
        import numpy as np
        from aniposelib.boards import CharucoBoard
        from aniposelib.cameras import CameraGroup
        from openpyxl import Workbook


        # Find all video files in the calibration folder
        formats = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        patterns = [os.path.join(self.dir_calib_videos, f) for f in formats]
        video_files = []
        for pattern in patterns:
            video_files.extend(glob.glob(pattern))

        # Find all video files in the calibration folder
        pattern = r"cam\d+"
        cam_names = []
        # Find cam_names that are used in the recordings and for calibration.
        for file_name in video_files:
            # Search for the pattern in each file name
            match = re.search(pattern, file_name)
            if match:
                # If a match is found, extract the matched string and add it to the cam_names list
                cam_names.append(match.group())
        print(cam_names)


        # List of available camera names (update this list based on your available cameras)
        available_cameras = cam_names



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
        self.calib_file = os.path.join(self.dir_calib, f'Calib_{self.id_s}.toml')
        cgroup.dump(self.calib_file)
        print(f"Calibration for Session {self.id_s} done and saved to {self.calib_file}.")

        # TODO: Write global Workbook to save reprojection errors for all sessions, participants and trials in one file.

        """# Initialize a workbook to save reprojection error
        # TODO; Create global Workbook for all session, with corresponding reprojection errors. -> Database for all sessions, participants and Trials.
        wb = Workbook()
        ws = wb.active
        ws.append(["Recording Session", "Reprojection Error"])  # Header row
        
        # Save the calibration error for the current configuration in the workbook
        ws.append([self.id_s, error])

        # Save the workbook with reprojection errors
        excel_filename = 'calibration_reprojection_errors.xlsx'
        wb.save(excel_filename)
        print(f"All reprojection errors have been saved to {excel_filename}.")"""

    def move_calibration_files(self, empty_dir_calib=True):
        """
        This function copies one of the calibration files to the calibration folder. So that it will be found by pose2sim

        This unction is not Neccessary for Trial Analyis while recording. It might be usefull for Batch processing when multiple calibrated camera settings exist for one sessio.
        """
        import shutil

        #Empty the calibration directory
        if empty_dir_calib:
            for file in os.listdir(self.dir_calib):
                if file.endswith(".toml"):
                    os.remove(os.path.join(self.dir_calib, file))
        if self.chosen_calib is None and not self.choose_first_calib:
            raise ValueError("No calibration file chosen and choose_first is set to False")

        # Move one file from the calibration files to the calibration folder
        if self.choose_first_calib:
            for file in os.listdir(self.dir_calib_files):
                if file.endswith(".toml"):
                    shutil.copy(os.path.join(self.dir_calib_files, file), self.dir_calib)
                    break
                else:
                    for file in os.listdir(self.dir_calib_files):
                        if file.endswith(f"{self.chosen_calib}.toml"):
                            shutil.copy(os.path.join(self.dir_calib_files, file), self.dir_calib)
                            break
                        else:
                            raise FileNotFoundError(f"No calibration file {self.chosen_calib}.toml found.")

        self.used_calibration = file

        # TODO Add funciton where User cann choose calibration file based on list of possible settings.

    def new_participant(self, id_p, assessement="", record_duration=60*10, pose_model="Coco18_UpperBody"):
        """
        This function creates a new participant object for the session.
        The ID of the participant is the PID and has to be given by caller.
        """


        from .iDrinkParticipant import Participant

        new_p = Participant(id_s=self.id_s, id_p=id_p, dir_root=self.dir_root, dir_session=self.dir_session,
                            used_calib=self.used_calibration, assessement=assessement,
                            rec_resolution=self.rec_resolution, frame_rate=self.frame_rate, clean_video=self.clean_video,
                            record_duration=record_duration, pose_model=pose_model, cams_chosen = self.cams_chosen)
        new_p.create_participant()

        return new_p




