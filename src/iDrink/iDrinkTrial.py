import glob
import os
import re

import numpy as np
import pandas as pd

"""
Class for Trial objects.

Trial objects should contain the following informations:
- id_s: Session ID
- id_p: Patient ID (PID)
- id_t: Trial ID

- date_time: Date and Time of Recording e.g. 240430_1430

- path_config: Path to Config.toml

Information for Therapists after Recording:pip
- skipped: "True" / "False" - flag that tells whether step / trial was skipped
- skip_reason: Explanation why the trial was skipped
- Score: Score assessed by therapist

Paths to directories:
- dir_trial: Trial directory
- dir_default: directory containing all default files
- dir_ref: Directory containing reference files

Information about Recording
- Frame Rate: 30 or 60 FPS

About the used model
- used framework: e.g. MMPose, OpenPose etc.
- pose_model: e.g. B25, B25_UPPER_EARS


For Movement Analysis
- measured_side: "left" / "right"
- Affected: "True" / "False" --> tells whether the measured side is the affected side of the patient

- ang_joint_kin: angular joint kinematics
- ep_kin_movement_time: endpoint kinematics - movement Time
- ep_kin_movement_units: endpoint kinematics - movement units
- ep_kin_peak_velocities: endpoint kinematics - peak velocity

- path_murphy_measures: path to directory containing the murphy measure data
- path_ang_joint_kin: path to angular joint kinematics
- path_ep_kin_movement_time: path to endpoint kinematics - movement Time
- path_ep_kin_movement_units: path to endpoint kinematics - movement units
- path_ep_kin_peak_velocities: path to endpoint kinematics - peak velocity

For Opensim
- opensim_model: Path to the .osim file
- opensim_model_scaled: Path to the scaled .osim file

- opensim_scaling: path to the scaling .xml
- opensim_inverse_kinematics. path to the inverse kinematics .xml file
- opensim_analyze: Path to Analyze Tool .xml

- opensim_marker: relative path to .trc file
- opensim_marker_filtered: relative path to filtered .trc file
- opensim_marker_scaling: path to filtered scaling .trc files
- opensim_motion: relative path to .mot file

- opensim_dir_analyze_results: Path to the Analyze Results directory

- opensim_scaling_time_range: time range for scaling (string) e.g. "0.017 0.167"
- opensim_IK_time_range: time range for the Inverse Kinematics (string) e.g. "0 6.65"
- opensim_ana_init_t: Start time of for analyze tool (string) e.g. "0"
- opensim_ana_final_t: end time of for analyze tool (string) e.g. "6.65"

"""


class Trial:
    def __init__(self, identifier=None, id_s=None, id_p=None, id_t=None, assessement="", task="",
                 dir_root=None, dir_default=None, dir_reference=None, dir_session=None, dir_calib=None,
                 dir_calib_videos=None, dir_calib_files=None, dir_participant=None, dir_trial=None,
                 date_time=None, path_config=None, skip=False, skip_reason=None, affected=False,
                 frame_rate=60, rec_resolution=(1920, 1080), clean_video=True, n_frames=0,
                 used_cams=None, video_files=None, path_calib_videos=None, json_list=None,

                 config_dict=None, path_calib=None, calib=None, used_framework=None, pose_model=None, measured_side=None,

                 stabilize_hip=True, correct_skeleton=False, chosen_components=None, show_confidence_intervall=False,
                 use_analyze_tool=False, bod_kin_p2s=False, use_torso=True, is_reference=False):

        """These variables are only for the validation. They can be deleted for deployment."""

        # Booleans for the different steps in the pipeline
        self.HPE_done = False
        self.MMPose_done = False
        self.P2SPose_done = False
        self.Metrabs_multi_done = False
        self.Metrabs_single_done = False
        self.HPE_done = False
        self.calib_done = False
        self.P2S_done = False
        self.OS_done = False
        self.MM_done = False
        self.stat_done = False
        self.cam_setting = None

        """Variables for Deployment"""
        # Trial Info
        self.id_s = id_s
        self.id_p = id_p
        self.id_t = id_t
        self.identifier = identifier
        self.date_time = date_time

        # Directories and Paths
        self.dir_trial = dir_trial
        self.dir_participant = dir_participant
        self.dir_session = dir_session

        self.dir_root = dir_root
        self.dir_default = dir_default
        self.dir_reference = dir_reference
        self.dir_calib = dir_calib
        self.dir_calib_videos = dir_calib_videos
        self.dir_calib_files = dir_calib_files
        self.path_reference = None



        # Clinical info for review
        self.assessement = assessement  # Name of the assessement e.g. Drinking Task, iArat etc.
        self.task = task  # The executed Task
        self.measured_side = measured_side
        self.affected = affected
        self.is_reference = is_reference

        # Configurations, Calibrations, Settings
        self.path_config = path_config
        self.config_dict = config_dict
        self.path_calib = path_calib
        self.calib = calib

        # Data preparation and filtering
        self.stabilize_hip = stabilize_hip

        # Recording
        self.used_cams = used_cams  # List of used Cameras
        self.dir_recordings = os.path.realpath(os.path.join(dir_trial, "videos", "recordings"))
        self.dir_rec_pose = os.path.realpath(os.path.join(dir_trial, "videos", "pose"))
        self.dir_rec_blurred = os.path.realpath(os.path.join(dir_trial, "videos", "blurred"))
        self.frame_rate = frame_rate
        self.rec_resolution = rec_resolution
        self.clean_video = clean_video
        self.n_frames = n_frames

        self.video_files = video_files  # List of Recordings if not in dir_recordings (Move them to dir_recordings before running the pipeline)
        self.path_calib_videos = path_calib_videos  # List of Calibration Videos if not in dir_calib_videos (Move them to dir_calib_videos before running the pipeline)

        self.json_list = json_list


        # Visual Output
        self.render_out = os.path.join(dir_trial, "videos", 'Render_out')

        # Plots
        self.filteredplots = True # Data filtered before plotted
        self.chosen_components = chosen_components
        self.show_confidence_intervall = show_confidence_intervall


        # Pose Estimation
        self.PE_dim = 2  # Dimension of Pose Estimation
        self.used_framework = '' # openpose, mmpose, pose2sim
        self.pose_model = pose_model
        self.write_pose_videos = False

        # Movement Analysis
        # Settings
        self.use_analyze_tool = use_analyze_tool
        self.bod_kin_p2s = bod_kin_p2s
        self.use_torso = use_torso
        self.filenename_appendix = ''
        self.correct_skeleton = correct_skeleton

        # Setting for Phase Detection
        self.use_dist_handface = False
        self.use_acceleration = False
        self.use_joint_vel = False
        self.extended = True # Decide whether extended Phase detection should be used (7 or 5 phases)

        # Thresholds TODO: User should be able to set the threshold
        self.phase_thresh_vel = 0.05  # Fraction of peak that needs to be passed for next Phase
        self.phase_thresh_pos = 0.05  # Fraction of peak that needs to be passed for next Phase

        # Filtering
        self.butterworth_cutoff = 10
        self.butterworth_order = 5

        # Smoothing TODO: Let User choose, which data to smoothen
        self.smooth_velocity = True
        self.smooth_distance = True
        self.smooth_trunk_displacement = False

        self.smoothing_divisor_vel = 4
        self.smoothing_divisor_dist = 4
        self.smoothing_divisor_trunk_displacement = 4
        self.reference_frame_trunk_displacement = 0

        # Directories
        # Dir containing all fies and subfolders for movement analysis
        self.dir_movement_analysis = os.path.realpath(os.path.join(self.dir_trial, "movement_analysis"))
        # Dir containing files for murphy measures
        self.dir_murphy_measures = os.path.realpath(os.path.join(self.dir_movement_analysis, "murphy_measures"))
        # Directory containing Kinematics based on raw .trc files
        self.dir_kin_trc = os.path.realpath(os.path.join(self.dir_movement_analysis, "kin_trc"))
        # Directory containing kinematics based on inverse Kinematics from P2S Function
        self.dir_kin_p2s = os.path.realpath(os.path.join(self.dir_movement_analysis, "kin_p2s"))
        # Dir containing kinematics based on the inverse kinematics tool from OpenSim
        self.dir_kin_ik_tool = os.path.realpath(os.path.join(self.dir_movement_analysis, "ik_tool"))
        # Dir containing kinematics calculated by OpenSim Analyzer Tool
        self.dir_anatool_results = os.path.realpath(os.path.join(self.dir_movement_analysis, "kin_opensim_analyzetool"))

        # Paths to Movement files
        self.path_opensim_ik = None

        # Opensim Analyze Tool
        self.path_opensim_ana_pos = None
        self.path_opensim_ana_vel = None
        self.path_opensim_ana_acc = None
        self.path_opensim_ana_ang_pos = None
        self.path_opensim_ana_ang_vel = None
        self.path_opensim_ana_ang_acc = None

        # Pose2Sim Inverse Kinematics
        self.path_p2s_ik_pos = None
        self.path_p2s_ik_vel = None
        self.path_p2s_ik_acc = None

        # Pose2Sim .trc files
        self.path_trc_pos = None
        self.path_trc_vel = None
        self.path_trc_acc = None

        # Movement Data
        # OpenSim Inverse Kinematics Tool
        self.opensim_ik = None
        self.opensim_ik_ang_pos = None
        self.opensim_ik_ang_vel = None
        self.opensim_ik_ang_acc = None

        # Opensim Analyze Tool
        self.opensim_ana_pos = None
        self.opensim_ana_vel = None
        self.opensim_ana_acc = None
        self.opensim_ana_ang_pos = None
        self.opensim_ana_ang_vel = None
        self.opensim_ana_ang_acc = None

        # Pose2Sim Inverse Kinematics
        self.p2s_ik_pos = None
        self.p2s_ik_vel = None
        self.p2s_ik_acc = None

        # Pose2Sim .trc files
        self.trc_pos = None
        self.trc_vel = None
        self.trc_acc = None

        # Movement Data used for analysis
        # Marker/Keypoint Position, Velocity and Acceleration
        self.marker_pos = None
        self.marker_vel = None
        self.marker_acc = None
        self.marker_source = None

        # joint Position, Velocity and Acceleration
        self.joint_pos = None
        self.joint_vel = None
        self.joint_acc = None
        self.joint_source = None

        # Murphy Measures - File Paths
        self.path_ang_joint_kin = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ang_joint_kin.csv"))
        self.path_ep_kin_movement_time = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_movement_time.csv"))
        self.path_ep_kin_movement_units = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_movement_units.csv"))
        self.path_ep_kin_peak_velocities = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_peak_velocities.csv"))
        self.path_mov_phases_timeframe = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_mov_phases_timeframe.csv"))

        # Murphy Measures - Data
        self.mov_phases_timeframe = None
        self.ang_joint_kin = None
        self.ep_kin_movement_time = None
        self.ep_kin_movement_units = None
        self.ep_kin_peak_velocities = None

        # OpenSim files and paths
        self.opensim_model = None
        self.opensim_model_scaled = None
        self.opensim_scaling = None
        self.opensim_inverse_kinematics = None
        self.opensim_analyze = None
        self.opensim_marker = None
        self.opensim_marker_filtered = None
        self.opensim_marker_scaling = None
        self.opensim_motion = None
        self.opensim_scaling_time_range = None
        self.opensim_IK_time_range = None
        self.opensim_ana_init_t = None
        self.opensim_ana_final_t = None
        self.opensim_dir_analyze_results = self.get_opensim_path(
            os.path.join(self.dir_movement_analysis, "kin_opensim_analyzetool"))

    def __str__(self):
        return f"Trial ID: {self.id_t}, Patient ID: {self.id_p}, Date: {self.date_time}"

    def make_settings(self):
        """
        This Function needs to run before the analysis. It defines all the settings for the data analysis of the trial.
        """

    def create_trial(self, for_omc=False):
        """
        This function creates all the folders and their subfolders.
        """
        import shutil
        from .iDrinkSetup import write_default_configuration
        """Create empty Folder structure"""
        dirs = [
            self.dir_murphy_measures,
            self.dir_anatool_results,
            self.dir_kin_trc,
            self.dir_kin_p2s,
            self.dir_kin_ik_tool,
            self.dir_recordings,
            self.dir_rec_blurred,
            self.render_out,
            self.dir_calib,
            os.path.realpath(os.path.join(self.dir_trial, "pose-3d")),
        ]
        if for_omc:
            dirs = [
                self.dir_murphy_measures,
                self.dir_anatool_results,
                self.dir_kin_trc,
                self.dir_kin_p2s,
                self.dir_kin_ik_tool,
                self.dir_recordings,
                self.dir_rec_blurred,
                os.path.realpath(os.path.join(self.dir_trial, "pose")),
                os.path.realpath(os.path.join(self.dir_trial, "pose-3d")),
                os.path.realpath(os.path.join(self.dir_trial, "pose-associated")),
            ]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

        """Place empty config file"""

        empty_file = os.path.join(self.dir_default, "Config_empty.toml")
        if self.path_config is None:
            self.path_config = os.path.realpath(os.path.join(self.dir_trial, f"{self.identifier}_Config.toml"))

        write_default_configuration(empty_file, self.path_config)

    def update_score(self, new_score):
        self.score = new_score

    def mark_as_skipped(self, reason):
        self.skipped = True
        self.skip_reason = reason

    def ids_from_dir_trial(self):
        """
        This Funtion uses the directories to infer the Ids and the Identifier of the trial.

        Then it sets fixed paths and directory names in the object, that use the IDs or the Identifier.

        This Function is mainly used for batch processing Data, that is in the correct Folder structure,
        but doesn't have any Trial files created for them. It should ensure a uniform folder structure for recorded
        and batch processed Sessions.
        """

        import re
        self.dir_participant = os.path.realpath(os.path.join(self.dir_trial, '..'))
        self.dir_session = os.path.realpath(os.path.join(self.dir_participant, '..'))

        self.id_s = re.search(r'(S\d{8})-(\d{6})', os.path.basename(self.dir_session)).group()
        self.id_p = re.search(r'(P\d+)[^\\]*$', os.path.basename(self.dir_participant)).group(1)
        self.id_t = re.search(r'(T\d+)[^\\]*$', os.path.basename(self.dir_trial)).group(1)

        self.identifier = f"{self.id_s}_{self.id_p}_{self.id_t}"

        self.dir_calib = os.path.realpath(os.path.join(self.dir_session, f'{self.id_s}_Calibration'))
        self.dir_calib_videos = os.path.realpath(os.path.join(self.dir_calib, f'{self.id_s}_Calibration_videos'))
        self.dir_calib_files = os.path.realpath(os.path.join(self.dir_calib, f'{self.id_s}_Calibration_files'))

        self.path_ang_joint_kin = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ang_joint_kin.csv"))
        self.path_ep_kin_movement_time = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_movement_time.csv"))
        self.path_ep_kin_movement_units = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_movement_units.csv"))
        self.path_ep_kin_peak_velocities = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_peak_velocities.csv"))

    def set_filename_appendix(self):
        """
        Sets the appendix for filenames based on the settings of the trial.

        Do not change any Paths or the identifier itself.
        :return:
        """

        appendix = ""
        if self.use_analyze_tool:
            appendix += "_AnalyzeTool"
        elif self.bod_kin_p2s:
            appendix += "_P2S"
        else:
            appendix += "_TRC"

        if self.use_acceleration:
            appendix += "_Acc"

        if self.use_dist_handface:
            if self.use_torso:
                appendix += "_Distancetorso"
            else:
                appendix += "_Distanceface"

        self.filenename_appendix = appendix

        self.path_ang_joint_kin = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}{self.filenename_appendix}_ang_joint_kin.csv"))
        self.path_ep_kin_movement_time = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}{self.filenename_appendix}_ep_kin_movement_time.csv"))
        self.path_ep_kin_movement_units = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}{self.filenename_appendix}_ep_kin_movement_units.csv"))
        self.path_ep_kin_peak_velocities = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}{self.filenename_appendix}_ep_kin_peak_velocities.csv"))
        self.path_mov_phases_timeframe = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}{self.filenename_appendix}_mov_phases_timeframe.csv"))

    def check_if_affected(self):
        recnames = glob.glob(os.path.join(self.dir_recordings, "*unaffected*"))

        if recnames:
            self.affected = False
        else:
            self.affected = True

        return self.affected




    def load_configuration(self, load_default=False):
        from .iDrinkSetup import write_default_configuration

        def get_config():
            with open(self.path_config, 'r') as file:
                # Assuming TOML format, import the required module if necessary
                import toml
                self.config_dict = toml.load(file)


        try:
            if not load_default:
                get_config()
            else:
                empty_file = os.path.join(self.dir_default, "Config_empty.toml")
                self.path_config = os.path.realpath(os.path.join(self.dir_trial, f"{self.identifier}_Config.toml"))

                write_default_configuration(empty_file, self.path_config)
                get_config()

        except FileNotFoundError as e:
            print(f"Configuration file not found: {e}")
            print(f"Loading default configuration file.")

            empty_file = os.path.join(self.dir_default, "Config_empty.toml")
            self.path_config = os.path.realpath(os.path.join(self.dir_trial, f"{self.identifier}_Config.toml"))

            write_default_configuration(empty_file, self.path_config)
            get_config()
        except Exception as e:
            print(f"Another error occurred: {e}")
            print(f"Loading default configuration file.")

            empty_file = os.path.join(self.dir_default, "Config_empty.toml")
            self.path_config = os.path.realpath(os.path.join(self.dir_trial, f"{self.identifier}_Config.toml"))

            write_default_configuration(empty_file, self.path_config)
            get_config()

        return self.config_dict

    def save_configuration(self):
        """Saves config_digt to path in self.path_config"""
        import toml

        with open(self.path_config, 'w') as file:

            toml.dump(self.config_dict, file)
        return self.config_dict

    def load_calibration(self):
        with open(self.path_calib, 'r') as file:
            # Assuming TOML format, import the required module if necessary
            import toml
            self.calib = toml.load(file)
        return self.calib


    def batch_calibration(self, run_anyway=False):
        """
        Runs an old verison of the calibration. It can be still used in DEBUGGING

        # TODO: Remove this function when not needed anymore

        :return:
        """
        from .iDrinkCalibration import delta_calibration
        dir_session = os.path.realpath(os.path.join(self.dir_trial, '..', '..'))

        pattern = os.path.join(dir_session, f'{self.id_s}_calib*')
        dir_calib = sorted(glob.glob(pattern))[0]
        dir_calibration_video = os.path.realpath(os.path.join(dir_calib, "calib_video_files"))

        if run_anyway:
            delta_calibration(self)
            self.path_calib = glob.glob(os.path.join(dir_calib, '*calib*.toml'))[0]
            self.load_calibration()  # load Calibration-file
        else:
            try:
                # Try to load the first .toml in alphanumerical order
                self.path_calib = sorted(glob.glob(os.path.join(dir_calib, '*calib*.toml')))[0]
                self.load_calibration()  # load Calibration-file
                print(f"Calibration-file found: {self.calib}. Calibration will be skipped")
            except IndexError:
                # IndexError occurs when the list is empty, i.e. no file was found.
                print(f"no .toml Calibration-file found in {dir_calib}. start calibration.")
                delta_calibration(self)
                self.path_calib = glob.glob(os.path.join(dir_calib, '*calib*.toml'))[0]
                self.load_calibration()  # load Calibration-file
            except Exception as e:
                # Catch any other exceptions
                raise Exception(f"An error occurred: {e}")

    def run_analysis_Pipeline(self):
        """
        Runs all function from pose_estimation until the full analysis with the murphy measures is done.
        """

        from .iDrinkAnalytics import calculate_keypoint_vel_acc, calculate_bodykin_vel_acc, run_full_analysis
        from .iDrinkOpenSim import open_sim_pipeline

        """Pose Estimation"""
        from .iDrinkPoseEstimation import pose_estimation_2d
        try:
            pose_estimation_2d(self, writevideofiles=self.write_pose_videos)
        except Exception as e:
            print(f"An error occurred during the Pose Estimation: {e}")

        """Run Pose2Sim"""
        try:
            self.load_configuration()
            self.run_pose2sim()
        except Exception as e:
            print(f"An error occurred during the Pose2Sim Pipeline: {e}")

        """Prepare and Run OpenSim"""
        try:
            self.prepare_opensim()
            open_sim_pipeline(self)
        except Exception as e:
            print(f"An error occurred during the OpenSim Pipeline: {e}")


        """Run Data Analysis Pipeline"""
        try:
            calculate_keypoint_vel_acc(curr_trial=self)
            calculate_bodykin_vel_acc(curr_trial=self)
            run_full_analysis(curr_trial=self)
        except Exception as e:
            print(f"An error occurred during the analysis: {e}")

        print(f"Trial {self.identifier} has been processed.")


    def get_time_range(self, path_trc_file, start_time=0, frame_range=[], as_string=False):
        """
        Input:
            - p2S_file: path to .trc file
            - start_time: default = 0
            - frame_range: [starting frame, ending frame] default = []
        Output:
            - time_range formatted for OpenSim .xml files.
        Gets time range based on Pose2Sim .trc file.
        """
        from trc import TRCData
        path_trc_file = os.path.realpath(os.path.join(self.dir_trial, path_trc_file))

        trc = TRCData()
        trc.load(filename=path_trc_file)
        time = np.array(trc["Time"])
        # If Frame Range is given, the time range is written using the frame range.
        if frame_range:
            frame = np.array(trc["Frame#"])
            if frame_range[0] == 0:
                frame_range[0] = 1
            if frame_range[1] > max(frame):
                frame_range[1] = max(frame)
            start_time = time[np.where(frame == frame_range[0])[0][0]]
            final_time = time[np.where(frame == frame_range[1])[0][0]]
            if as_string:
                return f"{start_time} {final_time}"
        # Otherwise, it will end at the end of the recording
        else:
            final_time = max(time)
            if as_string:
                return f"{start_time} {final_time}"
        # if as_sting is False, return list with start and end time
        return [start_time, final_time]

    def get_opensim_path(self, path_in):

        # TODO: Make this part more general. At the moment it is a bit too focused on OpenSim

        if os.name == 'posix':
            slash = '/'
        elif os.name == 'nt':
            slash = '\\'

        if self.dir_trial in path_in:
            filepath = path_in.split(self.dir_trial + slash, 1)[1]
        else:
            return "dir_trial not in path"

        return filepath

    def get_filename(self):
        # Setup Appendix for Filename if needed
        ref = ""
        aff = ""
        if self.is_reference:
            ref = "_reference"
        if self.affected:
            aff = "_affected"
        filename = f"{self.identifier}{self.filenename_appendix}{aff}{ref}"

        return filename

    def find_file(self, directory, extension, flag=None):
        import glob
        if flag is not None:
            pattern = os.path.join(directory, f"{self.id_s}_*{self.id_p}_*{self.id_t}*{flag}*{extension}")
        else:
            pattern = os.path.join(directory, f"{self.id_s}_*{self.id_p}_*{self.id_t}*{extension}")

        try:
            filepath = glob.glob(pattern)[0]
        except Exception as e:
            print(e,"\n")
            print(f"File not found in {directory} with pattern {pattern}\n")

        return filepath

    def prepare_opensim(self):
        import shutil

        # Copy Geometry from default to trial folder
        dir_geom = os.path.realpath(os.path.join(self.dir_default, "Geometry"))
        new_dir_geom = os.path.realpath(os.path.join(self.dir_trial, "Geometry"))

        shutil.copytree(dir_geom, new_dir_geom, dirs_exist_ok=True)


        self.opensim_model = os.path.join(self.dir_default, f"iDrink_{self.pose_model}.osim")
        self.opensim_model_scaled = os.path.join(self.dir_trial, f"{self.identifier}_Scaled_{self.pose_model}.osim")

        self.opensim_scaling = os.path.join(self.dir_trial, f"Scaling_Setup_iDrink_{self.pose_model}.xml")
        self.opensim_inverse_kinematics = os.path.join(self.dir_trial, f"IK_Setup_iDrink_{self.pose_model}.xml")
        self.opensim_analyze = os.path.join(self.dir_trial, f"AT_Setup.xml")

        self.opensim_marker = self.get_opensim_path(self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc"))
        self.opensim_marker_filtered = self.get_opensim_path(
            self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc", flag="filt"))
        self.opensim_motion = os.path.splitext(
            self.get_opensim_path(self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc", flag="filt")))[
                                  0] + ".mot"

        """self.opensim_scaling_time_range = self.get_time_range(path_trc_file=self.opensim_marker_filtered,
                                                              frame_range=[0, 1], as_string=True)"""
        self.opensim_scaling_time_range = self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=True) # Changed to use the whole recording for scaling might mitigate errors in the scaling process
        self.opensim_IK_time_range = self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=True)
        self.opensim_ana_init_t = str(
            self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=False)[0])
        self.opensim_ana_final_t = str(
            self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=False)[1])



    def run_pose2sim(self, only_triangulation=True, do_sync=False):
        """Run Pose2Sim Pipeline"""
        from Pose2Sim import Pose2Sim
        # os.chdir(self.dir_trial)
        self.config_dict.get("project").update({"project_dir": self.dir_trial})
        self.config_dict['pose']['pose_framework'] = self.used_framework
        self.config_dict['pose']['pose_model'] = self.pose_model

        self.save_configuration()




        """self.config_dict['triangulation']['reproj_error_threshold_triangulation'] = 200
        self.config_dict['triangulation']['interp_if_gap_smaller_than'] = 400"""

        Pose2Sim.calibration(config=self.config_dict)

        # Pose Estimation not used for iDrink validation. It is done seperatly.
        """if self.used_framework == "Pose2Sim":
            # Change the config_dict so that
            if self.pose_model == "Coco17_UpperBody":
                self.config_dict['pose']['pose_model'] = 'COCO_17'
            Pose2Sim.poseEstimation(config=self.config_dict)
            
            self.config_dict['pose']['pose_model'] = self.pose_model"""
        if not only_triangulation:
            if do_sync:
                Pose2Sim.synchronization(config=self.config_dict)

            Pose2Sim.personAssociation(config=self.config_dict)

        Pose2Sim.triangulation(config=self.config_dict)
        Pose2Sim.filtering(config=self.config_dict)
        if self.pose_model in ['BODY_25', 'BODY_25B']:
            print("Model supported for Marker augmentation.\n"
                  "Starting augmentation:")
            Pose2Sim.markerAugmentation(config=self.config_dict)
        else:
            print('Marker augmentation is only supported with OpenPose BODY_25 and BODY_25B models.\n'
                  'Augmentation will be skipped.')


    def save_all_mov_paths(self):
        """
        Read Files based on Opensim Paths and load position, velocity and acceleration data of interest.

        The data is placed in a DataFrame.

        input:
            - Paths from config file
        output:
            - dict_mov: DataFrame containing movement Data
        """
        self.path_opensim_ik = self.find_file(self.dir_kin_ik_tool, ".csv")

        # Retrieve paths and data using the output of the OpenSim Analyze Tool (.sto files)
        # marker position
        self.path_opensim_ana_pos = self.find_file(self.dir_anatool_results, ".sto", flag="BodyKinematics_pos")
        self.path_opensim_ana_vel = self.find_file(self.dir_anatool_results, ".sto", flag="BodyKinematics_vel")
        self.path_opensim_ana_acc = self.find_file(self.dir_anatool_results, ".sto", flag="BodyKinematics_acc")
        self.path_opensim_ana_ang_pos = self.find_file(self.dir_anatool_results, ".sto", flag="Kinematics_q")
        self.path_opensim_ana_ang_vel = self.find_file(self.dir_anatool_results, ".sto", flag="Kinematics_u")
        self.path_opensim_ana_ang_acc = self.find_file(self.dir_anatool_results, ".sto", flag="Kinematics_dudt")

        # Retrieve path and data for marker position and velocity using the Body Kinematics calculated by Pose2Sim Inverse Kinematics
        self.path_p2s_ik_pos = self.find_file(self.dir_kin_p2s, ".csv", flag="p2s_pos")
        self.path_p2s_ik_vel = self.find_file(self.dir_kin_p2s, ".csv", flag="p2s_vel")
        self.path_p2s_ik_acc = self.find_file(self.dir_kin_p2s, ".csv", flag="p2s_acc")

        # Retrieve path and data for marker position and velocity using the keypoints
        self.path_trc_pos = os.path.join(self.dir_trial, self.opensim_marker_filtered)
        self.path_trc_vel = self.find_file(self.dir_kin_trc, ".csv", flag="keypoint_vel")
        self.path_trc_acc = self.find_file(self.dir_kin_trc, ".csv", flag="keypoint_acc")

    def save_all_mov_data(self):
        """
        Read Files based on Opensim Paths and load position, velocity and acceleration data of interest.

        The data is placed in a DataFrame.

        input:
            - Paths from config file
        output:
            - dict_mov: DataFrame containing movement Data
        """
        from .iDrinkOpenSim import read_opensim_file
        from .iDrinkAnalytics import get_keypoint_positions, get_measured_side

        # Read output from Opensim Inverese Kinematics from .csv into DataFrame
        self.opensim_ik = pd.read_csv(self.path_opensim_ik)

        # Retrieve paths and data using the output of the OpenSim Analyze Tool (.sto files)
        self.opensim_ana_pos = read_opensim_file(self.path_opensim_ana_pos)
        self.opensim_ana_vel = read_opensim_file(self.path_opensim_ana_vel)
        self.opensim_ana_acc = read_opensim_file(self.path_opensim_ana_acc)
        self.opensim_ana_ang_pos = read_opensim_file(self.path_opensim_ana_ang_pos)
        self.opensim_ana_ang_vel = read_opensim_file(self.path_opensim_ana_ang_vel)
        self.opensim_ana_ang_acc = read_opensim_file(self.path_opensim_ana_ang_acc)

        # Retrieve path and data for marker position and velocity using the Body Kinematics calculated by Pose2Sim Inverse Kinematics
        self.p2s_ik_pos = pd.read_csv(self.path_p2s_ik_pos)
        self.p2s_ik_vel = pd.read_csv(self.path_p2s_ik_vel)
        self.p2s_ik_acc = pd.read_csv(self.path_p2s_ik_acc)

        # Retrieve path and data for marker position and velocity using the keypoints
        self.trc_pos = get_keypoint_positions(self.path_trc_pos)
        self.trc_vel = pd.read_csv(self.path_trc_vel)
        self.trc_acc = pd.read_csv(self.path_trc_acc)

        # Retrieve Joint information using the output of the Inverse Kinematics Tool
        # TODO: If we actually want to use it like that, we might need to smoothen it a bit. Especially the acceleration seem unstable in the plot.
        self.opensim_ik_ang_pos = self.opensim_ik.drop(columns="time")
        self.opensim_ik_ang_vel = pd.DataFrame(data=np.gradient(self.opensim_ik_ang_pos.values, axis=0),
                                               columns=self.opensim_ik_ang_pos.columns)
        self.opensim_ik_ang_acc = pd.DataFrame(data=np.gradient(self.opensim_ik_ang_vel.values, axis=0),
                                               columns=self.opensim_ik_ang_vel.columns)

        # TODO: Side determination might be done somewhere else. Or done differently
        # Determine the measured side and save to object
        if not self.measured_side:
            self.measured_side = get_measured_side(self.opensim_ik)

    def get_mov_data_for_analysis(self):
        """
        Read Files based on Opensim Paths and load position, velocity and acceleration data of interest.

        The data is placed in a DataFrame.

        input:
            - Paths from config file
        output:
            - dict_mov: DataFrame containing movement Data
        """
        from .iDrinkOpenSim import read_opensim_file
        from .iDrinkAnalytics import get_keypoint_positions, get_measured_side

        if self.use_analyze_tool is None or self.bod_kin_p2s is None:
            print("Please determine settings for movement analysis.")
            return False  # Tell calling function that no data was loaded

        # Read output from Opensim Inverese Kinematics from .csv into DataFrame
        self.opensim_ik = pd.read_csv(self.path_opensim_ik)

        if self.use_analyze_tool:
            # Retrieve paths and data using the output of the OpenSim Analyze Tool (.sto files)
            _, self.marker_pos = read_opensim_file(self.path_opensim_ana_pos)
            _, self.marker_vel = read_opensim_file(self.path_opensim_ana_vel)
            _, self.marker_acc = read_opensim_file(self.path_opensim_ana_acc)
            _, self.joint_pos = read_opensim_file(self.path_opensim_ana_ang_pos)
            _, self.joint_vel = read_opensim_file(self.path_opensim_ana_ang_vel)
            _, self.joint_acc = read_opensim_file(self.path_opensim_ana_ang_acc)
            self.marker_source = "anatool"
            self.joint_source = "anatool"
        else:

            if self.bod_kin_p2s:
                # Retrieve path and data for marker position and velocity using the Body Kinematics calculated by Pose2Sim Inverse Kinematics
                self.marker_pos = pd.read_csv(self.path_p2s_ik_pos)
                self.marker_vel = pd.read_csv(self.path_p2s_ik_vel)
                self.marker_acc = pd.read_csv(self.path_p2s_ik_acc)
                self.marker_source = "p2s"
            else: # use trc files
                # Retrieve path and data for marker position and velocity using the keypoints
                self.marker_pos = get_keypoint_positions(self.path_trc_pos)
                self.marker_vel = pd.read_csv(self.path_trc_vel)
                self.marker_acc = pd.read_csv(self.path_trc_acc)
                self.marker_source = "trc"

            # Retrieve Joint information using the output of the Inverse Kinematics Tool
            # TODO: If we actually want to use it like that, we might need to smoothen it a bit. Especially the acceleration seem unstable in the plot.
            self.joint_pos = self.opensim_ik.drop(columns="time")
            self.joint_vel = pd.DataFrame(data=np.gradient(self.opensim_ik_ang_pos.values, axis=0),
                                          columns=self.opensim_ik_ang_pos.columns)
            self.joint_acc = pd.DataFrame(data=np.gradient(self.opensim_ik_ang_vel.values, axis=0),
                                          columns=self.opensim_ik_ang_vel.columns)
            self.joint_source = "invkintool"

        # TODO: Side determination might be done somewhere else. Or done differently
        # Determine the measured side and save to object
        if not self.measured_side:
            self.measured_side = get_measured_side(self.opensim_ik)

        return True  # Tell calling function that data was loaded successfully



