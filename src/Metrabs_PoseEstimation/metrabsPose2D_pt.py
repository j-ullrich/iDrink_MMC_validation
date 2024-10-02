import argparse
import json
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

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "iDrink")))
from iDrinkUtilities import pack_as_zip

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


def filter_df(df_unfiltered, fs, verbose, normcutoff=False):
    """
    Use a butterworth-filter on the 3D-keypoints in the DF and return the filtered DF.

    :param df: DataFrame with 3D coordinates
    :param fs: Sampling frequency

    :return: DataFrame with filtered 3D coordinates

    """
    from scipy.signal import butter, sosfiltfilt

    df_filtered = pd.DataFrame(columns=df_unfiltered.columns)

    cutoff = 5
    order = 2  # Desired order 5. Because of filtfilt, half of that needs to be given. --> filtfilt doubles the order

    nyquist = 0.5 * fs

    if cutoff >= nyquist:
        if verbose >= 1:
            print(f"Warning: Cutoff frequency {cutoff} is higher than Nyquist frequency {nyquist}.")
            print("Filtering with Nyquist frequency.")
        cutoff = int(nyquist - 1)

    if normcutoff:
        cutoff = cutoff / nyquist
    else:
        sos = butter(order, cutoff, btype="low", analog=False, output="sos", fs=fs)

    if verbose >= 2:
        print(f"Filtering 3D keypoints:\n"
              f"Filter: Butterworth\n"
              f"Order: {order}\n"
              f"Sampling frequency: {fs} Hz\n"
              f"cutoff frequency of {cutoff} Hz")

    if verbose >= 1:
        progress = tqdm(total=len(df_unfiltered.columns), desc="Filtering 3D keypoints", position=0, leave=True)

    for column in df_unfiltered.columns:
        data = np.array(df_unfiltered[column].tolist())
        df_filtered[column] = sosfiltfilt(sos, data, axis=0).tolist()

        if verbose >= 1:
            progress.update(1)

    if verbose >= 1:
        progress.close()

    return df_filtered


def df_to_trc(df, trc_file, identifier, fps, n_frames, n_markers, verbose=1):
    """
    Converts the DataFrame to a .trc file according to nomenclature used by Pose2Sim.

    Writes to trc file to given path.

    :param df: DataFrame with 3D coordinates
    """
    trc = TRCData()

    if verbose >= 1:
        print(f"Writing .trc file {os.path.basename(trc_file)}")

    trc['PathFileType'] = '4'
    trc['DataFormat'] = '(X/Y/Z)'
    trc['FileName'] = f'{identifier}.trc'
    trc['DataRate'] = fps
    trc['CameraRate'] = fps
    trc['NumFrames'] = n_frames
    trc['NumMarkers'] = n_markers
    trc['Units'] = 'm'
    trc['OrigDataRate'] = fps
    trc['OrigDataStartFrame'] = 0
    trc['OrigNumFrames'] = n_frames
    trc['Frame#'] = [i for i in range(n_frames)]
    trc['Time'] = [round(i / fps, 3) for i in range(n_frames)]
    trc['Markers'] = df.columns.tolist()

    for column in df.columns:
        trc[column] = df[column].tolist()

    if verbose >= 2:
        print(f'columns added to {os.path.basename(trc_file)}')

    for column in tqdm(df.columns, desc="Writing .trc file", position=0, leave=True):
        trc[column] = df[column].tolist()

    for i in range(n_frames):
        trc[i] = [trc['Time'][i], df.iloc[i, :].tolist()]

    if verbose >= 2:
        print(f'Timestamps added to {os.path.basename(trc_file)}')

    trc.save(trc_file)
    if verbose >= 1:
        print(f"Saved .trc file {os.path.basename(trc_file)}")


def get_column_names(joint_names, verbose=1):
    """
    Uses the list of joint names to create a list of column names for the DataFrame

    e.g. neck --> neck_x, neck_y, neck_z

    Input: joint_names: List of joint names
    Output: List of column names for x, y, and z axis
    """
    columns = []
    """
    # Old version might be used if x, y, z coordinates need to be split up in DataFrame
    for joint in joint_names:
        for axis in ['x', 'y', 'z']:
            columns.append(f"{joint}_{axis}")"""

    for joint in joint_names:
        columns.append(joint)

    if verbose >= 2:
        print(f'Columns created: {columns}')

    return columns


def add_to_dataframe(df, pose_result_3d):
    """
    Add 3D  Keypoints to DataFrame

    Returns DataFrame with added 3D keypoints
    """

    temp = []
    for i in range(pose_result_3d.shape[1]):
        temp.append([x / 1000 for x in pose_result_3d[0][i].tolist()])

    df.loc[len(df)] = temp

    return df


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

def plot_results(image, pred, joint_names, joint_edges, show=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Rectangle
    import matplotlib
    matplotlib.use('TkAgg')

    fig = plt.figure(figsize=(40, 20.8))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
    for x, y, w, h, c in pred['boxes'].cpu().numpy():
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -75)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(2000, 5000)
    poses3d = pred['poses3d'].cpu().numpy()
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, pred['poses2d'].cpu().numpy()):
        for i_start, i_end in joint_edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)
    if show:
        fig.show()
    fig.clear()
    matplotlib.pyplot.close()

def plot_results_2d(image, pred, joint_names, joint_edges, show=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Rectangle
    import matplotlib
    matplotlib.use('TkAgg')

    fig = plt.figure(figsize=(40, 20.8))
    image_ax = fig.add_subplot(1, 1, 1)
    image_ax.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
    for x, y, w, h, c in pred['boxes'].cpu().numpy():
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    for pose2d in pred['poses2d'].cpu().numpy():
        for i_start, i_end in joint_edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)

    if show:
        fig.show()

    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image_array


def metrabs_pose_estimation_2d_val(curr_trial, video_files, calib_file, model_path, identifier, root_val,
                               skeleton='bml_movi_87', write_video=False, verbose=1, DEBUG=False):
    get_config(os.path.realpath(os.path.join(model_path, 'config.yaml')))
    multiperson_model_pt = load_multiperson_model(model_path).cuda()

    joint_names = multiperson_model_pt.per_skeleton_joint_names[skeleton]
    joint_edges = multiperson_model_pt.per_skeleton_joint_edges[skeleton].cpu().numpy()

    # Create DataFrame for 3D Poses
    df = pd.DataFrame(columns=get_column_names(joint_names))

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



        writer = None

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
            frames_in, _, vid_meta = torchvision.io.read_video(video, output_format='TCHW')

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
                pose_result_3d = pred['poses3d'].cpu().numpy()
                ################## JSON Output #################
                # Add track id (useful for multiperson tracking)

                json_out(pose_result_2d, frame_idx, json_dir_unfilt, video)
                df = add_to_dataframe(df, pose_result_3d)

                # Visualize Pose

                if write_video:
                    frame_out = plot_results_2d(frame, pred, joint_names, joint_edges)

                    if writer is None:
                        size = (frame_out.shape[1], frame_out.shape[0])
                        fps = vid_meta['video_fps']
                        vid_out = os.path.join(out_video, f"{os.path.basename(video).split('.mp4')[0]}.avi")
                        writer = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

                        writer.write(frame_out)
                    else:
                        writer.write(frame_out)

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

            """Write trc for 3D Pose Estimation"""
            dir_out_trc_unfilt = os.path.realpath(
                os.path.join(root_val, "02_pose_estimation", "01_unfiltered", f"{curr_trial.id_p}",
                             f"{curr_trial.id_p}_{used_cam}", "metrabs", "single-cam"))
            dir_out_trc_filt = os.path.realpath(
                os.path.join(root_val, "02_pose_estimation", "02_filtered", f"{curr_trial.id_p}",
                             f"{curr_trial.id_p}_{used_cam}", "metrabs", "single-cam"))

            for dir_out_trc in [dir_out_trc_unfilt, dir_out_trc_filt]:
                if not os.path.exists(dir_out_trc):
                    os.makedirs(dir_out_trc)

            trc_file_filt = os.path.join(dir_out_trc_filt,
                                         f"{cam}_{curr_trial.id_p}_{curr_trial.id_t}_{os.path.basename(video).split('.mp4')[0]}_0-{frame_idx}_filt_iDrinkbutter.trc")
            trc_file_unfilt = os.path.join(dir_out_trc_unfilt,
                                           f"{identifier}_{os.path.basename(video).split('.mp4')[0]}_0-{frame_idx}_iDrink.trc")


            if verbose >= 2:
                print(f'Call filtering function')
            df_filt = filter_df(df, fps, verbose)



            df_to_trc(df_filt, trc_file_filt, identifier, fps, n_frames, n_markers)
            df_to_trc(df, trc_file_unfilt, identifier, fps, n_frames, n_markers)

            if verbose >= 2:
                print(f'3D Pose Estimation done and .trc files saved to {dir_out_trc}')

    pack_as_zip(json_dir_unfilt)
    pack_as_zip(json_dir_filt)

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
