import argparse
import os
import re
import sys
import gc

import cv2
import numpy as np
import pandas as pd
import simplepyutils as spu
import toml
import torch
import torchvision.io
from tqdm import tqdm
from trc import TRCData

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), 'metrabs_pytorch')))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), 'posepile')))
import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
import posepile.joint_info
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config

from metrabs_common import filter_df, df_to_trc, get_column_names, add_to_dataframe
from metrabs_pt_models import load_multiperson_model, load_crop_model

parser = argparse.ArgumentParser(description='Metrabs 2D Pose Estimation for iDrink using Pytorch')
parser.add_argument('--identifier', metavar='id', type=str, help='Identifier for the trial')
parser.add_argument('--video_file', metavar='dvi', type=str,
                    help='Path to video-file for pose estimation')
parser.add_argument('--dir_trial', metavar='dt', type=str,
                    help='Path to Trial directory')
parser.add_argument('--calib_file', metavar='c', type=str,
                    help='Path to calibration file')
parser.add_argument('--skeleton', metavar='skel', type=str, default='coco_19',
                    help='Skeleton to use for pose estimation, Default: coco_19')
parser.add_argument('--model_path', metavar='m', type=str,
                    default=os.path.join(os.getcwd(), 'metrabs_models'),
                    help=f'Path to the model to use for pose estimation. \n'
                         f'Default: {os.path.join(os.getcwd(), "metrabs_models")}')
parser.add_argument('--verbose', metavar='v', type=int, default=1, help='Verbose Mode (0, 1, 2), Default: 1')
parser.add_argument('--DEBUG', metavar='d', type=bool, default=False, help='Debug Mode, Default: False')


def metrabs_pose_estimation_3d_val(video_file, calib_file, dir_out_video, dir_out_trc, model_path, identifier,
                               skeleton='bml_movi_87', verbose=1, DEBUG=False):

    get_config(os.path.realpath(os.path.join(model_path, 'config.yaml')))
    multiperson_model_pt = load_multiperson_model(model_path).cuda()

    joint_names = multiperson_model_pt.per_skeleton_joint_names[skeleton]
    joint_edges = multiperson_model_pt.per_skeleton_joint_edges[skeleton].cpu().numpy()

    calib = toml.load(calib_file)

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_out_video):
        os.makedirs(dir_out_video)

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_out_video):
        os.makedirs(dir_out_video)

    ##################################################
    #############  OPENING THE VIDEO  ################
    # For a video file
    cap = cv2.VideoCapture(video_file)
    # Check if file is opened correctly
    if not cap.isOpened():
        print("Could not open file")
        exit()
    # get intrinsics from calib file
    cam = re.search(r"cam\d*", video_file).group()
    intrinsic_matrix = None
    distortions = None
    for key in calib.keys():
        if calib.get(key).get("name") == cam:
            intrinsic_matrix = calib.get(key).get("matrix")
            distortions = calib.get(key).get("distortions")
    print(f"Current Video: {os.path.basename(video_file)}")
    # Initializing variables for the loop
    frame_idx = 0


    # Prepare DataFrame
    df = pd.DataFrame(columns=get_column_names(joint_names))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_markers = len(joint_names)

    if verbose >= 2:
        print(f"Number of frames: {n_frames}")
        print(f"FPS: {fps}")
        print(f"Number of markers: {n_markers}")
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if verbose >=1:
        progress = tqdm(total=tot_frames, desc=f"Processing {identifier} - {os.path.basename(video_file)}", position=0, leave=True, unit='frame')
    with torch.inference_mode(), torch.device('cuda'):
        frames_in, _, _ = torchvision.io.read_video(video_file, output_format='TCHW', pts_unit='sec')
        for frame_idx, frame in enumerate(frames_in):
            pred = multiperson_model_pt.detect_poses(frame, skeleton=skeleton, detector_threshold=0.01,
                                                     suppress_implausible_poses=False, max_detections=1,
                                                     intrinsic_matrix=torch.FloatTensor(intrinsic_matrix),
                                                     distortion_coeffs=torch.FloatTensor(distortions), num_aug=2)
            # Save detection's parameters
            bboxes = pred['boxes'].cpu().numpy()
            pose_result_3d = pred['poses3d'].cpu().numpy()

            df = add_to_dataframe(df, pose_result_3d)

            frame_idx += 1
            if verbose >= 1:
                progress.update(1)
        # Release the VideoCapture object and close progressbar

        del frames_in
        gc.collect()

        if verbose >= 1:
            progress.close()

    if verbose >= 2:
        print(f'Call filtering function')
    df_filt = filter_df(df, fps, verbose)

    if not os.path.exists(dir_out_trc):
        os.makedirs(dir_out_trc)

    trc_file_filt = os.path.join(dir_out_trc,
                                 f"{identifier}_{os.path.basename(video_file).split('.mp4')[0]}_0-{frame_idx}_filt_iDrinkbutter.trc")
    trc_file_unfilt = os.path.join(dir_out_trc,
                                   f"{identifier}_{os.path.basename(video_file).split('.mp4')[0]}_0-{frame_idx}_iDrink.trc")

    df_to_trc(df_filt, trc_file_filt, identifier, fps, n_frames, n_markers)
    df_to_trc(df, trc_file_unfilt, identifier, fps, n_frames, n_markers)

    if verbose >= 2:
        print(f'3D Pose Estimation done and .trc files saved to {dir_out_trc}')


def metrabs_pose_estimation_3d(video_file, calib_file, dir_out_video, dir_out_trc, multiperson_model_pt, identifier,
                               skeleton='bml_movi_87', verbose=1, DEBUG=False):

    joint_names = multiperson_model_pt.per_skeleton_joint_names[skeleton]
    joint_edges = multiperson_model_pt.per_skeleton_joint_edges[skeleton].cpu().numpy()

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_out_video):
        os.makedirs(dir_out_video)

    calib = toml.load(calib_file)

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_out_video):
        os.makedirs(dir_out_video)

    ##################################################
    #############  OPENING THE VIDEO  ################
    # For a video file
    cap = cv2.VideoCapture(video_file)
    # Check if file is opened correctly
    if not cap.isOpened():
        print("Could not open file")
        exit()
    # get intrinsics from calib file
    cam = re.search(r"cam\d*", os.path.basename(video_file)).group()
    intrinsic_matrix = None
    distortions = None
    for key in calib.keys():
        if calib.get(key).get("name") == cam:
            intrinsic_matrix = calib.get(key).get("matrix")
            distortions = calib.get(key).get("distortions")
    print(f"Current Video: {os.path.basename(video_file)}")
    # Initializing variables for the loop
    frame_idx = 0


    # Prepare DataFrame
    df = pd.DataFrame(columns=get_column_names(joint_names))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_markers = len(joint_names)

    if verbose >= 2:
        print(f"Number of frames: {n_frames}")
        print(f"FPS: {fps}")
        print(f"Number of markers: {n_markers}")
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if verbose >=1:
        progress = tqdm(total=tot_frames, desc=f"Processing {os.path.basename(video_file)}", position=0, leave=True)
    with torch.inference_mode(), torch.device('cuda'):
        frames_in, _, _ = torchvision.io.read_video(video_file, output_format='TCHW')
        for frame_idx, frame in enumerate(frames_in):
            pred = multiperson_model_pt.detect_poses(frame, skeleton=skeleton, detector_threshold=0.01,
                                                     suppress_implausible_poses=False, max_detections=1,
                                                     intrinsic_matrix=torch.FloatTensor(intrinsic_matrix),
                                                     distortion_coeffs=torch.FloatTensor(distortions), num_aug=2)
            # Save detection's parameters
            bboxes = pred['boxes'].cpu().numpy()
            pose_result_3d = pred['poses3d'].cpu().numpy()

            df = add_to_dataframe(df, pose_result_3d)

            frame_idx += 1
            if verbose >= 1:
                progress.update(1)
        # Release the VideoCapture object and close progressbar

        del frames_in
        gc.collect()

        if verbose >= 1:
            progress.close()

    if verbose >= 2:
        print(f'Call filtering function')
    df_filt = filter_df(df, fps, verbose)

    if not os.path.exists(dir_out_trc):
        os.makedirs(dir_out_trc)

    trc_file_filt = os.path.join(dir_out_trc,
                                 f"{os.path.basename(video_file).split('.mp4')[0]}_0-{frame_idx}_filt_iDrinkbutter.trc")
    trc_file_unfilt = os.path.join(dir_out_trc,
                                   f"{os.path.basename(video_file).split('.mp4')[0]}_0-{frame_idx}_unfilt_iDrink.trc")

    df_to_trc(df_filt, trc_file_filt, identifier, fps, n_frames, n_markers)
    df_to_trc(df, trc_file_unfilt, identifier, fps, n_frames, n_markers)

    if verbose >= 2:
        print(f'3D Pose Estimation done and .trc files saved to {dir_out_trc}')


if __name__ == '__main__':
    args = parser.parse_args()

    if sys.gettrace() is not None or args.DEBUG:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

        if os.name == 'posix':  # if running on WSL
            args.identifier = "S20240501-115510_P01_T01"
            args.model_path = "/mnt/c/iDrink/metrabs_models/pytorch/metrabs_eff2l_384px_800k_28ds_pytorch"
            args.dir_trial = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T07"
            args.video_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T44/videos/recordings/cam1_trial_44_R_affected.mp4"
            args.calib_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_Calibration/Calib_S20240501-115510.toml"
            args.skeleton = 'bml_movi_87'
            args.DEBUG = True
            args.filter_2d = False
            args.verbose = 1

        else:
            args.identifier = "S20240501-115510_P01_T01"
            args.model_path = r"C:\iDrink\metrabs_models\pytorch\metrabs_eff2l_384px_800k_28ds_pytorch"
            args.dir_trial = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T07"
            args.video_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T07\videos\recordings\cam2_trial_07_L_unaffected.mp4"
            args.calib_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_Calibration\Calib_S20240501-115510.toml"
            args.skeleton = 'bml_movi_87'
            args.DEBUG = True
            args.filter_2d = False
            args.verbose = 1

    get_config(os.path.realpath(os.path.join(args.model_path, 'config.yaml')))
    multiperson_model_pt = load_multiperson_model(args.model_path).cuda()

    dir_out_video = os.path.realpath(os.path.join(args.dir_trial, 'videos', 'pose'))
    dir_out_trc = os.path.realpath(os.path.join(args.dir_trial, 'pose-3d'))

    metrabs_pose_estimation_3d(args.video_file, args.calib_file, dir_out_video, dir_out_trc, multiperson_model_pt,
                               args.identifier, args.skeleton, args.verbose, args.DEBUG)
    pass
