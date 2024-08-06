"""
This file contains all Functions needed to Setup files and functions for the iPose Pipeline
"""

import glob
import os
import re
import string


def write_default_configuration(path_empty, path_default):
    """
    Input:
        - config_empty:     Path to an empty config containing no settings or variables.
                            It contains only comments for manual changes.


    Write the default config file for the Pose2Sim Pipeline of iDrink.

    The config should incorporate all settings needed for Pose2Sim, while all other settings are stored in the trial-objects / -files.

    - OpenPose
    - Pose2Sim
    - OpenSim
    - Plotting
    - Blender render
    - File handling
    - any other stuff that might be needed
    """
    from .iDrinkUtilities import write_to_config

    # TODO: In GUI, let User choose some of the settings. Which is to be decided.

    """Set Variables"""
    """Category: Project"""
    dict_project = {
        "multi_person": False,
        "frame_rate": 60,  # fps
        "frame_range": [],  # [start, end], [] for all frames
        "exclude_from_batch": []
        # List of trials to be excluded from batch analysis, ["dir_participant\dir_trial", 'etc'].
    }

    """Category: calibration"""
    dict_calibration_convert_qualisys = {
        "binning_factor": 1  # Usually 1, except when filming in 540p where it usually is 2
    }
    "Category: calibration.convert.optitrack"
    # See readme for instructions
    "Category: calibration.convert.vicon"
    # No parameter needed
    "Category: calibration.convert.opencap"
    # No parameter needed
    "Category: calibration.convert.easymocap"
    # No parameter needed
    "Category: calibration.convert.biocv"
    # No parameter needed
    "Category: calibration.convert.anipose"
    # No parameter needed
    "Category: calibration.convert.freemocap"
    # No parameter needed

    """Category: calibration.convert"""
    dict_calibration_convert = {
        "convert_from": 'anipose',
        # 'qualisys', 'optitrack', vicon', 'opencap', 'easymocap', 'biocv', 'anipose', or 'freemocap'
        "qualisys": dict_calibration_convert_qualisys
    }

    """Category: calibration.calculate.intrinsics"""
    dict_calibration_calculate_intrinsics = {
        "overwrite_intrinsics": False,  # overwrite (or not) if they have already been calculated?
        "show_detection_intrinsics": False,  # true or false (lowercase)
        "intrinsics_extension": "mp4",  # true or false (lowercase)
        "extract_every_N_sec": 1,  # if video, extract frames every N seconds (can be <1 )
        "intrinsics_corners_nb": [4, 7],
        "intrinsics_square_size": 60  # mm
    }
    dict_calibration_calculate_extrinsics_board = {
        "show_reprojection_error": True,  # true or false (lowercase)
        "extrinsics_extension": "png",  # any video or image extension
        "extrinsics_corners_nb": [4, 7],  # [H,W] rather than [w,h]
        "extrinsics_square_size": 60,  # mm [h,w] if square is actually a rectangle
    }
    dict_calibration_calculate_extrinsics_scene = {
        "show_reprojection_error": True,  # true or false (lowercase)
        "extrinsics_extension": "png",  # any video or image extension
        # list of 3D coordinates to be manually labelled on images. Can also be a 2 dimensional plane.
        # in m -> unlike for intrinsics, NOT in mm!
        "object_coords_3d": [[-2.0, 0.3, 0.0],
                             [-2.0, 0.0, 0.0],
                             [-2.0, 0.0, 0.05],
                             [-2.0, -0.3, 0.0],
                             [0.0, 0.3, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.05],
                             [0.0, -0.3, 0.0]],
    }
    dict_calibration_calculate_extrinsics_keypoints = {
        """Not yet implemented"""
    }
    dict_calibration_calculate_extrinsics = {
        "calculate_extrinsics": True,  # true or false (lowercase)
        "extrinsics_method": "scene",  # 'board', 'scene', 'keypoints'
        "moving_cameras": False,  # Not implemented yet
        "board": dict_calibration_calculate_extrinsics_board,
        "scene": dict_calibration_calculate_extrinsics_scene,
        "keypoints": dict_calibration_calculate_extrinsics_keypoints
    }
    dict_calibration_calculate = {
        "intrinsics": dict_calibration_calculate_intrinsics,
        "extrinsics": dict_calibration_calculate_extrinsics
    }

    dict_calibration = {
        "calibration_type": "convert",  # 'convert' or 'calculate'
        "convert": dict_calibration_convert,
        "calculate": dict_calibration_calculate
    }

    """Category: pose"""
    dict_pose = {
        "vid_img_extension": "mp4",  # any video or image extension
        "pose_framework": "openpose",  # 'openpose', 'mediapipe', 'alphapose', 'deeplabcut', 'MMpose'
        "pose_model": "Coco18_UpperBody",  # With openpose: BODY_25B, BODY_25, BODY_135, COCO, MPII
        # With mediapipe: BLAZEPOSE.
        # With alphapose: HALPE_26, HALPE_68, HALPE_136, COCO_133.
        # With deeplabcut: CUSTOM. See example at the end of the file.
        # Custom Openpose: B25_UPPER_BODY, B25_UPPER_Ears, B25_UPPER
        # Custom MMPose: Coco18_UpperBody

        # What follows has not been implemented yet
        "overwrite_pose": False,
        "openpose_path": "",  # only checked if OpenPose is used
        "mode": 'performance', # 'lightweight', 'balanced', 'performance'
        "det_frequency": 1, # Run person detection only every N frames, and inbetween track previously detected bounding boxes (keypoint detection is still run on all frames).
                            # Equal to or greater than 1, can be as high as you want in simple uncrowded cases. Much faster, but might be less accurate.
        "tracking": False,   # Gives consistent person ID across frames. Slightly slower but might facilitate synchronization if other people are in the background
        "display_detection": False,  # set true to display the detection in real time
        "overwrite_pose": False, # set to false if you don't want to recalculate pose estimation when it has already been done
        "save_video": 'to_video', # 'to_video' or 'to_images', 'none', or ['to_video', 'to_images']
        "output_format": 'openpose', # 'openpose', 'mmpose', 'deeplabcut', 'none' or a list of them # /!\ only 'openpose' is supported for now
    }

    """Category: synchronization"""
    dict_synchronization = {
        "display_sync_plots": True,  # true or false (lowercase)
        "keypoints_to_consider": ['RWrist'],  # 'all' if all points should be considered, for example if the participant did not perform any particicular sharp movement. In this case, the capture needs to be 5-10 seconds long at least
        # ['RWrist', 'RElbow'] list of keypoint names if you want to specify the keypoints to consider.
        "approx_time_maxspeed": 'auto',  # 'auto' if you want to consider the whole capture (default, slower if long sequences)
        # [10.0, 2.0, 8.0, 11.0] list of times in seconds, one value per camera if you want to specify the approximate time of a clear vertical event by one person standing alone in the scene
        "time_range_around_maxspeed": 2.0,  # Search for best correlation in the range [approx_time_maxspeed - time_range_around_maxspeed, approx_time_maxspeed  + time_range_around_maxspeed]
        "likelihood_threshold": 0.4,  # Keypoints whose likelihood is below likelihood_threshold are filtered out
        "filter_cutoff": 6,  # time series are smoothed to get coherent time-lagged correlation
        "filter_order": 4
    }

    """Category: personAssociation"""
    dict_person_association_single_person = {
        "reproj_error_threshold_association": 20,
        "tracked_keypoint": "Nose"
    }
    dict_person_association_multi_person = {
        "reconstruction_error_threshold": 0.1, # 0.1 = 10 cm
        "min_affinity": 0.2 # affinity below which a correspondence is ignored
    }
    dict_person_association = {
        "likelihood_threshold_association": 0.3,
        "single_person": dict_person_association_single_person,
        "multi_person": dict_person_association_multi_person
    }

    """Category: triangulation"""
    dict_triangulation = {
        "reorder_trc": False,
        "reproj_error_threshold_triangulation": 15,  # px
        "likelihood_threshold_triangulation": 0.3,
        "min_cameras_for_triangulation": 2,
        "interpolation": "cubic",  # linear, slinear, quadratic, cubic, or none
        "interp_if_gap_smaller_than": 10,  # do not interpolate bigger gaps
        "show_interp_indices": True,
        # true or false (lowercase). For each keypoint, return the frames that need to be interpolated
        "handle_LR_swap": False,
        # Better if few cameras (eg less than 4) with risk of limb swapping (eg camera facing sagittal plane), otherwise slightly less accurate and slower
        "undistort_points": False,
        # Better if distorted image (parallel lines curvy on the edge or at least one param > 10^-2), but unnecessary (and slightly slower) if distortions are low
        "make_c3d": False  # save triangulated data in c3d format in addition to trc # Coming soon!
    }

    """Category: filtering"""
    dict_filtering_butterworth = {
        "order": 4,
        "cut_off_frequency": 6  # Hz
    }

    dict_filtering_kalman = {
        "trust_ratio": 100,  #: measurement_trust/process_trust ~= process_noise/measurement_noise
        "smooth": True  # should be true, unless you need real-time filtering
    }

    dict_filtering_butterworth_on_speed = {
        "order": 4,
        "cut_off_frequency": 10  # Hz
    }

    dict_filtering_gaussian = {
        "sigma_kernel": 2  # px
    }

    dict_filtering_LOESS = {
        "nb_values_used": 30  #: fraction of data used * nb frames
    }

    dict_filtering_median = {
        "kernel_size": 9
    }

    dict_filtering = {
        "type": "butterworth",  # butterworth, kalman, gaussian, LOESS, median, butterworth_on_speed
        "display_figures": False,  # true or false (lowercase)
        "make_c3d": False,
        "butterworth": dict_filtering_butterworth,
        "kalman": dict_filtering_kalman,
        "butterworth_on_speed": dict_filtering_butterworth_on_speed,
        "gaussian": dict_filtering_gaussian,
        "LOESS": dict_filtering_LOESS,
        "median": dict_filtering_median
    }

    """Category: markerAugmentation"""
    dict_marker_augmentation = {
        "participant_height": 1.7,  # m
        "participant_mass": 70,  # kg
        "make_c3d": False
    }

    """Category: opensim"""
    dict_opensim = {
        "static_trial": [],
        "opensim_bin_path": [],
        "opensim_model": [],
        "opensim_model_scaled": [],
        "opensim_scaling": [],
        "opensim_inverse_kinematics": [],
        "opensim_marker": [],
        "opensim_marker_filtered": [],
        "opensim_motion": [],
        "opensim_time_range": [],
        "opensim_analyze": [],
        "opensim_marker_scaling": [],
        "opensim_dir_analyze_results": [],
        "measurements": [],
        "opensim_scaling_time_range": [],
        "opensim_IK_time_range": [],
        "opensim_ana_init_t": [],
        "opensim_ana_final_t": []
    }

    categories = {
        "project": dict_project,
        "calibration": dict_calibration,
        "pose": dict_pose,
        "synchronization": dict_synchronization,
        "personAssociation": dict_person_association,
        "triangulation": dict_triangulation,
        "filtering": dict_filtering,
        "markerAugmentation": dict_marker_augmentation,
        "opensim": dict_opensim,
    }

    write_to_config(path_empty, path_default, categories)


def prep_participant(dir_session, id_session, id_participant):
    """
    NOT IMPLEMENTED
    """
    if not os.path.exists(dir_session):
        print(f"Participant folder not found. Create new participant folder with ID {id_participant}")
        prep_participant(dir_session, id_session, id_participant)

    return


def prep_next_trial(dir_session, dir_participant, id_session, id_participant):
    """
    Looks if paths for corresponding session ID and participant ID exist.
    On Session Level, it checks, whether Calibration folder already exists.

    If necessary it creates the participant folder and then the folder for the trial with its subfolders
    The User can Add a name to the Trial, but doesn't need to. The name is added as appendix to the foldername

    If not, it reminds the user to calibrate before recording the participant.
    And asks whether to execute the calibration and create the corresponding folder structures.
    """
    dir_calib = os.path.realpath(
        os.path.join(dir_session, f"{re.match(r'(S00)', os.path.basename(dir_session)).group(1)}_calibration"))
    if not os.path.exists(dir_calib):
        print(f"No Calibration folder found for Session {id_session}.\n"
              f"Please execute Calibration before you record any participants.")
        do_calibration = input("Do you want to start with the calibration? (y/n)")
        while True:
            if do_calibration == "y":
                # run_calibration() TODO: Implement
                break
            elif do_calibration == "n":
                break
            else:
                do_calibration = input("Answer invalid. (y/n)")

    if not os.path.exists(dir_participant):
        print(f"Participant folder not found. Create new participant folder with ID {id_participant}")
        prep_participant(dir_session, id_session, id_participant)

    """Start with Trial creation"""
    # Look through Participant Folder and search highest Trial ID.
    subfolder_list = [folder for folder in glob.glob(os.path.join(dir_participant, '*/'))]
    trial_id_regex = re.compile(r"T\d+")
    trial_ids = []
    for folder in subfolder_list:
        # Suche nach Trial-ID in jedem Pfad
        match = trial_id_regex.search(folder)
        if match:
            trial_ids.append(match.group())

    print("Found Trial-IDs:", trial_ids)
    highest_id = max(trial_ids, key=lambda id: int(id[1:]), default=None)

    if highest_id is not None:
        # Increment ID and write New ID.
        new_id = int(highest_id[1:]) + 1
        num_length = len(highest_id[1:])
        id_trial = f"T{new_id:0{num_length}d}"
    else:
        id_trial = "T00"

    T_identifier = f"{id_session}_{id_participant}_{id_trial}"

    def get_trial_folder_name():
        # Ask user if Appendix is wanted
        add_appendix = input("Do you want to add an appendix to the trial-folder? (y/n)")
        while True:
            if add_appendix == "y":
                trial_appendix = input("Please enter appendix.")
                valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
                if all(c in valid_chars for c in trial_appendix):
                    trial_appendix = f"_{trial_appendix}"
                    break
                else:
                    invalid_chars = set(trial_appendix) - set(valid_chars)
                    print(f"Invalid characters: {invalid_chars}")
            elif add_appendix == "n":
                trial_appendix = ""
                break
            else:
                add_appendix = input("Answer invalid. (y/n)")

        return f"{T_identifier}{trial_appendix}"

    trial_folder_name = get_trial_folder_name()

    while True:
        dir_trial = os.path.realpath(
            os.path.join(dir_participant, trial_folder_name))
        if os.path.exists(dir_trial):
            print(f"{trial_folder_name} already taken. Please change name")
            trial_folder_name = get_trial_folder_name()
        else:
            os.makedirs(dir_trial, exist_ok=True)
            break

    """Create empty Folder structure"""
    dirnames = {
        "Analyze Results": ["murphy_measures",
                            "kin_opensim_analyzetool",
                            "kin_trc",
                            "kin_keypoint"],
        "pose": [],
        "pose-3d": [],
        "pose_associated": [],
        "videos": ["recordings",
                   "videos_blurred", ]
    }

    for folder in dirnames.keys():
        os.makedirs(os.path.join(dir_trial, folder))
        for subfolder in dirnames[folder]:
            os.makedirs(os.path.join(dir_trial, folder, subfolder))

    """Place empty config file"""

    empty_file = os.path.join(dir_default, "Config_empty.toml")
    config_path = os.path.realpath(os.path.join(dir_trial, f"{T_identifier}_Config.toml"))

    write_default_configuration(empty_file, config_path)
