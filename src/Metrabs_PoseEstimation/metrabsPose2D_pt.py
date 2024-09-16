import argparse
import json
import os
import re
import sys
import gc

import cv2
import numpy as np
import simplepyutils as spu
import toml
import torch
import torchvision.io
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), 'metrabs_pytorch')))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), 'posepile')))
import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
import posepile.joint_info
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config

parser = argparse.ArgumentParser(description='Metrabs 2D Pose Estimation for iDrink using Pytorch')
parser.add_argument('--identifier', metavar='id', type=str, help='Identifier for the trial')
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
parser.add_argument('--DEBUG', metavar='d', type=bool, default=False, help='Debug Mode, Default: False')


def load_multiperson_model(model_path):
    model_pytorch = load_crop_model(model_path)
    skeleton_infos = spu.load_pickle(f'{model_path}/skeleton_infos.pkl')
    joint_transform_matrix = np.load(f'{model_path}/joint_transform_matrix.npy')

    with torch.device('cuda'):
        return multiperson_model.Pose3dEstimator(
            model_pytorch.cuda(), skeleton_infos, joint_transform_matrix)


def load_crop_model(model_path):
    cfg = get_config()
    ji_np = np.load(f'{model_path}/joint_info.npz')
    ji = posepile.joint_info.JointInfo(ji_np['joint_names'], ji_np['joint_edges'])
    backbone_raw = getattr(effnet_pt, f'efficientnet_v2_{cfg.efficientnet_size}')()
    preproc_layer = effnet_pt.PreprocLayer()
    backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
    model = metrabs_pt.Metrabs(backbone, ji)
    model.eval()

    inp = torch.zeros((1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32)
    intr = torch.eye(3, dtype=torch.float32)[np.newaxis]

    model((inp, intr))
    model.load_state_dict(torch.load(f'{model_path}/ckpt.pt'))
    return model


def pose_data_to_json(pose_data_samples):
    """
    Write 2D Keypoints to Json File

    Args:
        pose_data_samples: List of PoseData Objects
        data_source: 'mmpose' or 'metrabs'

    Thanks to Loïc Kreienbühl
    """
    json_data = {}
    json_data["people"] = []

    json_data = {}
    json_data["people"] = []
    person_id = -1
    cat_id = 1
    score = 0.8  # Assume good certainty for all keypoints
    # TODO: Check for amount of People in pose_data_samples. There seems to be an error --> ca. 85 People with the same estimated poses

    for pose_data in pose_data_samples:
        keypoints = pose_data
        keypoints_with_score = []
        for i in range(keypoints.shape[0]):
            keypoints_with_score.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), score])

        json_data["people"].append({
            'person_id': person_id,
            'pose_keypoints_2d': keypoints_with_score,
        })
        person_id += 1

    return json_data


def json_out(pred, id, json_dir, video):
    json_name = os.path.join(json_dir, f"{os.path.basename(video).split('.mp4')[0]}_{id:06d}.json")
    json_file = open(json_name, "w")
    json.dump(pose_data_to_json(pred), json_file, indent=6)
    id += 1
def filter_2d_pose_data(curr_trial, json_dir, json_dir_filt, filter='butter', verbose=1):
    """
    This Function loads all json files in a directory, filters the data and saves replaces the json files with the filtered data in json files.

    A mean, butterworth and median filter can be applied.

    Currently, the median filter is applied to mitigate issues for edge cases.


    List all json files in directory, extract The name without the framenumber,
    create Pandas Dataframe with each row being a json file, and the columns being the keypoints.
    Filter the data and then save them in json files."""

    import pandas as pd
    from scipy.signal import medfilt
    sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'iDrink')))
    from iDrinkAnalytics import use_butterworth_filter, smooth_timeseries

    json_files = [filename for filename in os.listdir(json_dir) if filename.endswith('.json')]
    array_created = False

    # Load Data from json Files
    for frame_id, json_file in enumerate(json_files):
        with open(os.path.join(json_dir, json_file)) as f:
            data = json.load(f)
            if not array_created:
                array_created = True

                arr_data = np.zeros((len(data['people']), len(json_files), len(data['people'][0]['pose_keypoints_2d'])))  # [person_id][Frame_id][Keypoint_id]

            for i in range(len(data['people'])):
                arr_data[i, frame_id] = data['people'][i]['pose_keypoints_2d']

    # Filter the Data
    for person_id in range(np.shape(arr_data)[0]):
        for keypoint_id in range(np.shape(arr_data)[2]):

            match filter:
                case 'mean':
                    arr_data[person_id, :, keypoint_id] = smooth_timeseries(curr_trial, arr_data[person_id, :, keypoint_id])

                case 'butter':
                    arr_data[person_id, :, keypoint_id] = use_butterworth_filter(curr_trial=curr_trial, data=arr_data[person_id, :, keypoint_id],
                                                                         cutoff=curr_trial.butterworth_cutoff,
                                                                         fs=curr_trial.frame_rate,
                                                                         order=curr_trial.butterworth_order)
                case 'median':
                    arr_data[person_id, :, keypoint_id] = medfilt(arr_data[person_id, :, keypoint_id], kernel_size=3)
                case _:
                    arr_data[person_id, :, keypoint_id] = medfilt(arr_data[person_id, :, keypoint_id], kernel_size=3)



    # Save the data
    if verbose >= 1:
        progress = tqdm(total=len(json_files), desc=f"Writing filtered json-files in {os.path.basename(json_dir)}", position=0, leave=True)
    for frame_id, json_file in enumerate(json_files):
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)

        for i in range(len(data['people'])):
            data['people'][i]['pose_keypoints_2d'] = arr_data[i, frame_id].tolist()

        with open(os.path.join(json_dir_filt, json_file), 'w') as f:
            json.dump(data, f, indent=6)
        if verbose >= 1:
            progress.update(1)
    if verbose >= 1:
        progress.close()


def metrabs_pose_estimation_2d_val(curr_trial, video_files, calib_file, model_path, identifier, root_val,
                               skeleton='bml_movi_87', verbose=1, DEBUG=False):
    get_config(os.path.realpath(os.path.join(model_path, 'config.yaml')))
    multiperson_model_pt = load_multiperson_model(model_path).cuda()

    joint_names = multiperson_model_pt.per_skeleton_joint_names[skeleton]
    joint_edges = multiperson_model_pt.per_skeleton_joint_edges[skeleton].cpu().numpy()

    calib = toml.load(calib_file)

    # Iterate over video_files
    for video in video_files:

        used_cam = re.search(r'(cam\d+)', video).group(0)
        json_dir_filt = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "02_filtered",
                                                     f"{curr_trial.id_p}", f"{curr_trial.id_p}_{used_cam}", "metrabs",
                                                     f"{os.path.basename(video).split('.mp4')[0]}_json"))
        json_dir_unfilt = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "01_unfiltered",
                                                        f"{curr_trial.id_p}", f"{curr_trial.id_p}_{used_cam}", "metrabs",
                                                        f"{os.path.basename(video).split('.mp4')[0]}_json"))
        out_video = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "01_unfiltered",
                                                         f"{curr_trial.id_p}", f"{curr_trial.id_p}_{used_cam}",
                                                         "metrabs"))

        for d in [json_dir_filt, json_dir_unfilt, out_video]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)


        ##################################################
        #############  OPENING THE VIDEO  ################
        # For a video file
        cap = cv2.VideoCapture(video)

        # Check if file is opened correctly
        if not cap.isOpened():
            print(f"Could not open file: {video}")
            exit()



        # get intrinsics from calib file
        cam = re.search(r"cam\d*", video).group()
        intrinsic_matrix = None
        distortions = None

        for key in calib.keys():
            if calib.get(key).get("name") == cam:
                intrinsic_matrix = calib.get(key).get("matrix")
                distortions = calib.get(key).get("distortions")

        # Initializing variables for the loop
        frame_idx = 0
        buffer = []
        BUFFER_SIZE = 27

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        with torch.inference_mode(), torch.device('cuda'):
            frames_in, _, _ = torchvision.io.read_video(video, output_format='TCHW')

            if verbose >= 1:
                progress = tqdm(total=tot_frames, desc=f"Trial: {curr_trial.identifier} - {cam} - Video: {os.path.basename(video)}", position=0, leave=True)
            for frame_idx, frame in enumerate(frames_in):
                """pred = multiperson_model_pt.detect_poses(frame, skeleton=skeleton,
                                                         intrinsic_matrix=torch.FloatTensor(intrinsic_matrix),
                                                         distortion_coeffs=torch.FloatTensor(distortions))"""
                pred = multiperson_model_pt.detect_poses(frame, skeleton=skeleton, detector_threshold=0.01,
                                                         suppress_implausible_poses=False, max_detections=1,
                                                         intrinsic_matrix=torch.FloatTensor(intrinsic_matrix),
                                                         distortion_coeffs=torch.FloatTensor(distortions), num_aug=2)

                # Save detection's parameters
                bboxes = pred['boxes'].cpu().numpy()
                pose_result_2d = pred['poses2d'].cpu().numpy()

                ################## JSON Output #################
                # Add track id (useful for multiperson tracking)
                json_out(pose_result_2d, frame_idx, json_dir_unfilt, video)


                frame_idx += 1
                if verbose >= 1:
                    progress.update(1)

            if verbose >= 1:
                progress.close()

            if verbose >= 2:
                print(f"Garbage collection for {video}")

            del frames_in
            gc.collect()

            filter_2d_pose_data(curr_trial, json_dir_unfilt, json_dir_filt)

    del multiperson_model_pt
    gc.collect()

def metrabs_pose_estimation_2d(dir_video, calib_file, dir_out_video, dir_out_json, multiperson_model_pt, identifier,
                               skeleton='bml_movi_87', DEBUG=False):

    joint_names = multiperson_model_pt.per_skeleton_joint_names[skeleton]
    joint_edges = multiperson_model_pt.per_skeleton_joint_edges[skeleton].cpu().numpy()

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_out_video):
        os.makedirs(dir_out_video)

    calib = toml.load(calib_file)

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_out_video):
        os.makedirs(dir_out_video)

    calib = toml.load(calib_file)

    # Path to the first image/video file
    video_files = [filename for filename in os.listdir(dir_video) if
                   filename.endswith('.mp4') or filename.endswith('.mov') or filename.endswith('.avi')]

    for video_name in video_files:
        filepath = os.path.realpath(os.path.join(dir_video, video_name))

        ##################################################
        #############  OPENING THE VIDEO  ################
        # For a video file
        cap = cv2.VideoCapture(filepath)

        # Check if file is opened correctly
        if not cap.isOpened():
            print("Could not open file")
            exit()

        # Prepare Jsonwriterprocess
        json_dir = os.path.join(dir_out_json, f"{os.path.basename(video_name).split('.mp4')[0]}_json")

        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # get intrinsics from calib file
        cam = re.search(r"cam\d*", video_name).group()
        intrinsic_matrix = None
        distortions = None

        for key in calib.keys():
            if calib.get(key).get("name") == cam:
                intrinsic_matrix = calib.get(key).get("matrix")
                distortions = calib.get(key).get("distortions")

        print(f"Current Video: {video_name}")

        # Initializing variables for the loop
        frame_idx = 0
        buffer = []
        BUFFER_SIZE = 27

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=tot_frames, desc=f"Processing {video_name}", position=0, leave=True)

        with torch.inference_mode(), torch.device('cuda'):
            frames_in, _, _ = torchvision.io.read_video(filepath, output_format='TCHW')

            for frame_idx, frame in enumerate(frames_in):
                """pred = multiperson_model_pt.detect_poses(frame, skeleton=skeleton,
                                                         intrinsic_matrix=torch.FloatTensor(intrinsic_matrix),
                                                         distortion_coeffs=torch.FloatTensor(distortions))"""
                pred = multiperson_model_pt.detect_poses(frame, skeleton=skeleton, detector_threshold=0.01,
                                                         suppress_implausible_poses=False, max_detections=1,
                                                         intrinsic_matrix=torch.FloatTensor(intrinsic_matrix),
                                                         distortion_coeffs=torch.FloatTensor(distortions), num_aug=2)



                # Save detection's parameters
                bboxes = pred['boxes'].cpu().numpy()
                pose_result_2d = pred['poses2d'].cpu().numpy()

                ################## JSON Output #################
                # Add track id (useful for multiperson tracking)
                json_out(pose_result_2d, frame_idx, json_dir, video_name)

                frame_idx += 1
                progress.update(1)

            # Release the VideoCapture object and close progressbar
            cap.release()
            progress.close()


if __name__ == '__main__':
    args = parser.parse_args()

    if sys.gettrace() is not None or args.DEBUG:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

        if os.name == 'posix':  # if running on WSL
            args.identifier = "S20240501-115510_P01_T01"
            args.model_path = "/mnt/c/iDrink/metrabs_models/pytorch/metrabs_eff2l_384px_800k_28ds_pytorch"
            args.dir_trial = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_P07/S20240501-115510_P07_T07"
            args.calib_file = "/mnt/c/iDrink/Session Data/S20240501-115510/S20240501-115510_Calibration/Calib_S20240501-115510.toml"
            args.skeleton = 'bml_movi_87'
            args.filter_2d = False

        else:
            args.identifier = "S20240501-115510_P01_T01"
            args.model_path = r"C:\iDrink\metrabs_models\pytorch\metrabs_eff2l_384px_800k_28ds_pytorch"
            args.dir_trial = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_P07\S20240501-115510_P07_T07"
            args.calib_file = r"C:\iDrink\Session Data\S20240501-115510\S20240501-115510_Calibration\Calib_S20240501-115510.toml"
            args.skeleton = 'bml_movi_87'
            args.filter_2d = False

    get_config(os.path.realpath(os.path.join(args.model_path, 'config.yaml')))
    multiperson_model_pt = load_multiperson_model(args.model_path).cuda()

    dir_video = os.path.realpath(os.path.join(args.dir_trial, 'videos', 'recordings'))
    dir_out_video = os.path.realpath(os.path.join(args.dir_trial, 'videos', 'pose'))
    dir_out_json = os.path.realpath(os.path.join(args.dir_trial, 'pose'))



    metrabs_pose_estimation_2d(dir_video, args.calib_file, dir_out_video, dir_out_json, multiperson_model_pt,
                               args.identifier, args.skeleton, args.DEBUG)

    os.chdir(old_wd)


    pass
