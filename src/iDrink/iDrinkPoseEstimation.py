import json
import os
import importlib.resources as pkg_resources
import time
import re
import glob
from tqdm import tqdm
import sys

import cv2
# Solve the memory leak issue
import matplotlib
import numpy as np
import torch
from mmdet.apis import DetInferencer
from mmpose.apis import convert_keypoint_definition, init_model, inference_topdown, \
    inference_pose_lifter_model
from mmpose.visualization import Pose3dLocalVisualizer, PoseLocalVisualizer


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from iDrink.iDrinkUtilities import pack_as_zip

import queue
import threading

import itertools

matplotlib.use('Agg')


def get_all_trial_vids(trial):
    """
    takes one of the videos of the trial. Gets all available cameras indebentedly from setting and returns all videos of the trial as list.
    """
    import os
    import re
    import glob

    cam_dir = os.path.realpath(os.path.join(trial.video_files[0], '..', '..'))

    # get trial as str from video name e.g. 'trial_1'
    t_str = re.search(r'trial_\d+', trial.video_files[0]).group(0)

    return glob.glob(os.path.join(cam_dir, "**", f"{t_str}_*.mp4"), recursive=True)

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
    from .iDrinkAnalytics import use_butterworth_filter, smooth_timeseries

    json_files = [filename for filename in os.listdir(json_dir) if filename.endswith('.json')]
    array_created = False

    if not os.path.exists(json_dir_filt):
        os.makedirs(json_dir_filt, exist_ok=True)

    # Load Data from json Files
    if verbose >= 1:
        progress = tqdm(total=len(json_files), desc=f"Loading json-files in {os.path.basename(json_dir)}", position=0, leave=True)
    for frame_id, json_file in enumerate(json_files):


        with open(os.path.join(json_dir, json_file)) as f:
            if verbose >= 1:
                progress.set_description(f"Loading: {f}")
            data = json.load(f)
            if not array_created:
                arr_data = np.zeros((len(data['people']), len(json_files), len(data['people'][0]['pose_keypoints_2d'])))  # [person_id][Frame_id][Keypoint_id]
                array_created = True

            for i in range(len(data['people'])):
                try:
                    arr_data[i, frame_id] = data['people'][i]['pose_keypoints_2d']
                except Exception as e:
                    print(f"Error in {os.path.basename(__file__)}.{filter_2d_pose_data.__name__} while loading data from json {json_file}\n"
                          f"{e}")
                    print(f"Data: {data['people'][i]['pose_keypoints_2d']}")
        if verbose >= 1:
            progress.update(1)

    if verbose >= 1:
        progress.close()


    # Filter the Data
    for person_id in range(np.shape(arr_data)[0]):
        for keypoint_id in range(np.shape(arr_data)[2]):

            match filter:
                case 'mean':
                    arr_data[person_id, :, keypoint_id] = smooth_timeseries(curr_trial, arr_data[person_id, :, keypoint_id])

                case 'butter':
                    arr_data[person_id, :, keypoint_id] = use_butterworth_filter(data=arr_data[person_id, :, keypoint_id],
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
            try: # TODO: Add function to find most probable person if there are too many --> Which is the cause of this error in p2s HPE
                data['people'][i]['pose_keypoints_2d'] = arr_data[i, frame_id].tolist()
            except Exception as e:
                print(f"Error in {os.path.basename(__file__)}.{filter_2d_pose_data.__name__} while writing {json_file}\n "
                      f"{e}\n"
                      f"Data: {data['people'][i]['pose_keypoints_2d']}")

        with open(os.path.join(json_dir_filt, json_file), 'w') as f:
            json.dump(data, f, indent=6)
        if verbose >= 1:
            progress.update(1)
    if verbose >= 1:
        progress.close()

def filt_p2s_pose(trial, root_val, verbose=1):
    """
    Creates folder for filtered jsons then filters the 2D Pose data and saves it in the new folder.

    :param trial:
    :param root_val:
    :param verbose:
    :return:
    """
    # Get  json directories for each cam of trial and save in list
    id_t = f"trial_{int(trial.id_t.split('T')[1])}"

    # Create destination directories for filtered jsons
    unfilt_dir = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "01_unfiltered"))
    filt_dir = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "02_filtered"))

    cam_dirs = glob.glob(os.path.realpath(os.path.join(unfilt_dir, trial.id_p, f"{trial.id_p}_cam*")))
    cams = [re.search(r'cam\d+', cam).group(0) for cam in cam_dirs]

    if verbose >= 2:
        print(f"Filtering Pose Data for {trial.id_p}_{trial.id_t}\n"
              f"Cameras:\t{cams}\n")


    # iterate over cameras
    for cam in cams:
        json_dir_unfilt = glob.glob(os.path.realpath(os.path.join(unfilt_dir, f"{trial.id_p}",
                                              f"{trial.id_p}_{cam}", "pose2sim", rf"{id_t}_*_json")))[0]

        json_dir_filt = os.path.realpath(os.path.join(filt_dir, os.path.relpath(json_dir_unfilt, unfilt_dir)))

        if verbose >= 2:
            print("Source:\t", json_dir_unfilt, "\n")
            print("Destination:\t", json_dir_filt, "\n")

        # filter the 2D Pose data
        filter_2d_pose_data(trial, json_dir_unfilt, json_dir_filt, filter='butter', verbose=verbose)

        pack_as_zip(json_dir_unfilt)
        pack_as_zip(json_dir_filt)



def pose_data_to_json(pose_data_samples):
    """
    Write 2D Keypoints to Json File

    Thanks to Loïc Kreienbühl
    """
    json_data = {}
    json_data["people"] = []

    for pose_data in pose_data_samples:
        for instance in pose_data.pred_instances:
            keypoints = instance.keypoints.reshape(-1, 2).tolist()  # Reshape keypoints to pairs of (x, y)
            keypoint_scores = instance.keypoint_scores.flatten().tolist()
            # Interleave keypoints and keypoint_scores
            keypoints_with_scores = []
            for (x, y), score in zip(keypoints, keypoint_scores):
                keypoints_with_scores.extend([x, y, score])
            # Convert numpy.float32 to float
            keypoints_with_scores = [float(x) for x in keypoints_with_scores]
            score = float(instance.bbox_scores[0])  # Assuming there's only one score per instance
            json_data["people"].append({
                "image_id": pose_data.img_path,  # Assuming img_path is the image name
                "category_id": 1,  # Assuming category_id is always 1 for person
                "person_id": [-1],
                "pose_keypoints_2d": keypoints_with_scores,
                "score": score,
            })

    # return json.dumps(json_data, indent=4)  # Convert the list of dictionaries into a JSON string
    return json_data

def validation_pose_estimation_2d(curr_trial, root_val, video_files=None, writevideofiles=False, filter_2d=False, DEBUG=False):
    from multiprocessing import Process, Queue
    from threading import Thread

    q_in = queue.Queue()
    q_out = queue.Queue()
    q_json = queue.Queue()

    def frame_loader(cap, q_in, start_event):
        while True:
            ret, frame = cap.read()
            q_in.put((ret, frame))

            start_event.set()
            if not ret:
                break

    def frame_writer(q_out, output_video_detection, output_video_pose, stop_event):
        while True:

            if q_out.empty():
                if stop_event.is_set():
                    return
                time.sleep(0.5)
            else:
                dict = q_out.get()

                output_video_detection.write(dict['detection'])
                output_video_pose.write(dict['pose'])
                q_out.task_done()

    def json_out(q_json, json_dir, video, stop_event):
        json_id = 0
        while True:
            if q_json.empty():
                if stop_event.is_set():
                    return
                time.sleep(0.5)
            else:
                data = q_json.get()

                json_name = os.path.join(json_dir, f"{os.path.basename(video).split('.mp4')[0]}_{json_id:06d}.json")
                json_file = open(json_name, "w")
                json.dump(pose_data_to_json(data), json_file, indent=6)
                json_id += 1
                q_json.task_done()


    # Path to the first image/video file
    if video_files is None:
        video_files = curr_trial.video_files

    ##########################################
    #############  DET MODEL  ################
    # Initialize the detection inferencer
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    inferencer_detection = DetInferencer('rtmdet_tiny_8xb32-300e_coco', device=device, show_progress=False)

    # Class names (coco2017 dataset)
    label_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                   'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone',
                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear',
                   'hair drier', 'toothbrush']

    #############  POSE2D MODEL  ################

    #rel_model_cfg = r".mim\configs\body_2d_keypoint\topdown_heatmap\coco\td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
    rel_model_cfg = r".mim\configs\body_2d_keypoint\topdown_heatmap\coco\td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py"
    model_cfg = pkg_resources.files('mmpose').joinpath(rel_model_cfg).__str__()

    #ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192-e1ebdd6f_20220913.pth'
    ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth'

    # # Initialize the pose inferencer
    # inferencer_pose = MMPoseInferencer('human', device=device)

    # Initialize the pose inferencer
    model_pose = init_model(model_cfg, ckpt, device=device)

    # Adjust settings of the model
    model_pose.test_cfg.flip_test = False



    # build the visualizer
    visualizer_2d = PoseLocalVisualizer()
    # set skeleton, colormap and joint connection rule
    visualizer_2d.set_dataset_meta(model_pose.dataset_meta)
    for video in video_files:

        ##################################################
        #############  OPENING THE VIDEO  ################
        # For a video file
        cap = cv2.VideoCapture(video)

        # Check if file is opened correctly
        if not cap.isOpened():
            print("Could not open file")
            exit()

        # Prepare Output Paths for json and videos.
        used_cam = re.search(r'(cam\d+)', video).group(0)
        json_dir_filt = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "02_filtered",
                                                     f"{curr_trial.id_p}", f"{curr_trial.id_p}_{used_cam}", "mmpose",
                                                     f"{os.path.basename(video).split('.mp4')[0]}_json"))
        out_video_filt = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "02_filtered",
                                                      f"{curr_trial.id_p}", f"{curr_trial.id_p}_{used_cam}", "mmpose"))

        json_dir_unfilt = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "01_unfiltered",
                                                        f"{curr_trial.id_p}", f"{curr_trial.id_p}_{used_cam}", "mmpose",
                                                        f"{os.path.basename(video).split('.mp4')[0]}_json"))
        out_video_unfilt = os.path.realpath(os.path.join(root_val, "02_pose_estimation", "01_unfiltered",
                                                         f"{curr_trial.id_p}", f"{curr_trial.id_p}_{used_cam}",
                                                         "mmpose"))

        for d in [json_dir_filt, out_video_filt, json_dir_unfilt, out_video_unfilt]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

        # Prepare Threads
        stop_event = threading.Event()
        start_event = threading.Event()

        p1 = Thread(target=frame_loader, args=(cap, q_in, start_event))
        if writevideofiles:
            # Prepare the output videos
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            size = (frame_width, frame_height)

            framerate = cap.get(cv2.CAP_PROP_FPS)
            curr_trial.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_video_detection = cv2.VideoWriter(
                os.path.join(out_video_unfilt, f"{os.path.basename(video).split('.mp4')[0]}_detection.avi"),
                cv2.VideoWriter_fourcc(*'MJPG'), framerate, size)
            output_video_pose = cv2.VideoWriter(
                os.path.join(out_video_unfilt, f"{os.path.basename(video).split('.mp4')[0]}_pose.avi"),
                cv2.VideoWriter_fourcc(*'MJPG'),
                framerate, size)

            # Create Thread for writing the frames
            p2 = Thread(target=frame_writer, args=(q_out, output_video_detection, output_video_pose, stop_event))

        p3 = Thread(target=json_out, args=(q_json, json_dir_unfilt, video, stop_event))

        p1.start()
        if writevideofiles:
            p2.start()
        p3.start()

        # Initializing variables for the loop
        frame_idx = 0
        buffer = []
        BUFFER_SIZE = 27

        cam = re.search(r'(cam\d+)', video).group(0)
        progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), unit='Frame', desc=f"Trial: {curr_trial.identifier} - {cam} - Video: {os.path.basename(video)}")

        while True:
            # Read frame from the webcam
            if not start_event:
                start_event.wait()

            (ret, frame) = q_in.get()

            # If frame is read correctly ret is True
            if not ret:
                break

            # Stop setting for development
            if DEBUG:
                if frame_idx == 30:
                    break

            ##############################################
            ################## DETECTION #################
            # Perform inference on the frame
            detection = inferencer_detection(frame, return_vis=True)

            # Save detection's parameters
            bboxes = detection['predictions'][0]['bboxes']
            scores = detection['predictions'][0]['scores']
            labels = detection['predictions'][0]['labels']

            # get the bbox coordinate of the most probable human
            max_score = 0
            max_score_idx = 0
            for i in range(len(bboxes)):
                if scores[i] > max_score and label_names[labels[i]] == 'person':
                    max_score = scores[i]
                    max_score_idx = i

            ############################################
            ################## 2D POSE #################
            pose_result_2d = inference_topdown(model_pose, frame, bboxes=[bboxes[max_score_idx]], bbox_format='xyxy')

            if writevideofiles:
                ################## VISUALIZATION DETECTION #################
                output_frame_detection = detection['visualization'][0]

                ################## VISUALIZATION 2D #################
                output_frame_pose = visualizer_2d.add_datasample('visualization', frame, draw_gt=False, data_sample=pose_result_2d[0],
                                                                 show=False)

                q_out.put({'detection': output_frame_detection,
                           'pose': output_frame_pose})


            ################## JSON Output #################
            # Add track id (useful for multiperson tracking)
            if not hasattr(pose_result_2d[0], 'track_id'):
                setattr(pose_result_2d[0], 'track_id', '0')

            q_json.put(pose_result_2d)

            ################# BUFFER ######################
            # Fill buffer with the first results so that the buffer starts full
            if frame_idx == 0:
                buffer = [pose_result_2d] * BUFFER_SIZE
            # Buffer for 27/81/243 frames

            # Append at the end
            # while len(buffer) >= BUFFER_SIZE:
            #     buffer.pop(0)
            # buffer.append(pose_result_2d)

            # Append at the start
            while len(buffer) >= BUFFER_SIZE:
                buffer.pop(-1)
            buffer.insert(0, pose_result_2d)

            frame_idx += 1
            progress_bar.update(1)
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the VideoCapture object and close display window
        stop_event.set()
        cap.release()
        progress_bar.close()

        p1.join()
        if writevideofiles:
            p2.join()
        p3.join()
        q_in.queue.clear()
        q_out.queue.clear()
        q_json.queue.clear()
        if writevideofiles:
            output_video_detection.release()
            output_video_pose.release()
        cv2.destroyAllWindows()

        # TODO: Use unfiltered json to filter and copy into the filtered folder. --> Only need to estimate Pose once and not twice

        filter_2d_pose_data(curr_trial, json_dir_unfilt, json_dir_filt)


        pack_as_zip(json_dir_unfilt)
        pack_as_zip(json_dir_filt)


def main():
    db_path_raw = r"C:\iDelta\Session Data\S240501-1155\S240501-1155_P07\S240501-1155_P07_T01\videos\recordings"
    db_path_processed = os.path.join(r"C:\iDelta\Session Data\S240501-1155\S240501-1155_P07\S240501-1155_P07_T01",
                                     "pose", "videos")
    db_path_result = r"C:\iDelta\Pose Tests\result"

    PID = '8888888'

    # Define the directory path based on your db_path_processed and PID
    db_path_processed = db_path_processed.format(PID)
    # Check if the directory exists, if not create it
    if not os.path.exists(db_path_processed):
        os.makedirs(db_path_processed)
    # Path to the first image/video file
    path = db_path_raw.format(PID)
    video_files = [filename for filename in os.listdir(path) if
                   filename.endswith('.mp4') or filename.endswith('.mov') or filename.endswith('.avi')]
    first_video_file = video_files[0] if video_files else ''
    filepath = os.path.realpath(os.path.join(path, first_video_file))

    ##########################################
    #############  DET MODEL  ################
    # Initialize the detection inferencer
    inferencer_detection = DetInferencer('rtmdet_tiny_8xb32-300e_coco', device='cpu')

    # Class names (coco2017 dataset)
    label_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                   'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone',
                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear',
                   'hair drier', 'toothbrush']

    #############  POSE2D MODEL  ################
    model_cfg = r"C:\Users\johan\OneDrive\Dokumente\Studium\LLUI_Praktikum\05 iDrink\mmpose\configs\body_2d_keypoint\topdown_heatmap\coco\td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
    ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192-e1ebdd6f_20220913.pth'
    device = 'cpu'

    # # Initialize the pose inferencer
    # inferencer_pose = MMPoseInferencer('human', device='cpu')

    # Initialize the pose inferencer
    model_pose = init_model(model_cfg, ckpt, device=device)
    # build the visualizer
    visualizer_2d = PoseLocalVisualizer()
    # set skeleton, colormap and joint connection rule
    visualizer_2d.set_dataset_meta(model_pose.dataset_meta)

    ############## LIFT2D3D MODEL  ################
    model_cfg = r"C:\Users\johan\OneDrive\Dokumente\Studium\LLUI_Praktikum\05 iDrink\mmpose\configs\body_3d_keypoint\video_pose_lift\h36m\video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m.py"

    ckpt = 'https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth'
    device = 'cpu'

    # Initialize the 3D pose inferencer
    model_3D = init_model(model_cfg, ckpt, device=device)
    # build the visualizer
    visualizer_3d = Pose3dLocalVisualizer()
    # set skeleton, colormap and joint connection rule
    visualizer_3d.set_dataset_meta(model_3D.dataset_meta)

    ##################################################
    #############  OPENING THE VIDEO  ################
    # For a video file
    cap = cv2.VideoCapture(filepath)

    # Or, for a webcam
    # cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Could not open webcam/file")
        exit()

    # Prepare the output video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    framerate = cap.get(cv2.CAP_PROP_FPS)

    # for 2D video, it has to be twice the width, cause original + predictions are side by side
    size_2D = (2 * frame_width, frame_height)
    # for 3D video
    size_3D = (3840, 2160)

    # Prepare the output videos
    output_video_detection = cv2.VideoWriter(filepath.format(PID) + 'detection.avi',
                                             cv2.VideoWriter_fourcc(*'MJPG'), framerate, size)
    output_video_pose = cv2.VideoWriter(filepath.format(PID) + 'pose.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                                        framerate, size_2D)
    output_video_3D = cv2.VideoWriter(filepath.format(PID) + '3D.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                                      framerate, size_3D)

    # Initializing variables for the loop
    frame_idx = 0
    keypoints_3d = []
    buffer = []
    BUFFER_SIZE = 27

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        # Stop setting for development
        if frame_idx == 10:
            break
        # Show progress
        print(frame_idx)

        # bgr to rgb
        frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ##############################################
        ################## DETECTION #################
        # Perform inference on the frame
        detection = inferencer_detection(frame, return_vis=True)

        # Save detection's parameters
        bboxes = detection['predictions'][0]['bboxes']
        scores = detection['predictions'][0]['scores']
        labels = detection['predictions'][0]['labels']

        # get the bbox coordinate of the most probable human
        max_score = 0
        max_score_idx = 0
        for i in range(len(bboxes)):
            if scores[i] > max_score and label_names[labels[i]] == 'person':
                max_score = scores[i]
                max_score_idx = i
        # y1, x1, y2, x2 = bboxes[max_score_idx]

        # Cropped image
        # cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]

        ################## VISUALIZATION DETECTION #################
        output_frame_detection = detection['visualization'][0]
        # cv2.imshow('preview', output_frame_detection)
        output_video_detection.write(output_frame_detection)

        ############################################
        ################## 2D POSE #################
        pose_result_2d = inference_topdown(model_pose, frame, bboxes=[bboxes[max_score_idx]], bbox_format='xyxy')

        # Add track id (useful for multiperson tracking)
        if not hasattr(pose_result_2d[0], 'track_id'):
            setattr(pose_result_2d[0], 'track_id', '0')

        ################## VISUALIZATION 2D #################
        output_frame_pose = visualizer_2d.add_datasample('visualization', frame, data_sample=pose_result_2d[0],
                                                         show=False)
        # cv2.imshow('preview', output_frame_pose)
        output_video_pose.write(output_frame_pose)

        ################## CONVERT KEYPOINTS 2D to 3D #################
        # Convert keypoints for pose lifting

        keypoints = np.array(pose_result_2d[0].pred_instances.keypoints)
        keypoints_converted = convert_keypoint_definition(keypoints, 'coco', 'h36m')
        pose_result_2d[0].pred_instances.keypoints = keypoints_converted

        json_dir = os.path.join(r"C:\iDelta\Session Data\S240501-1155\S240501-1155_P07\S240501-1155_P07_T01\pose",
                                f"{os.path.basename(first_video_file).split('.mp4')[0]}")
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        json_name = os.path.join(json_dir, f"{os.path.basename(first_video_file).split('.mp4')[0]}{frame_idx}.json")
        json_file = open(json_name, "w")
        json.dump(pose_data_to_json(pose_result_2d), json_file, indent=6)

        ################# BUFFER ######################
        # Fill buffer with the first results so that the buffer starts full
        if frame_idx == 0:
            buffer = [pose_result_2d] * BUFFER_SIZE
        # Buffer for 27/81/243 frames

        # Append at the end
        # while len(buffer) >= BUFFER_SIZE:
        #     buffer.pop(0)
        # buffer.append(pose_result_2d)

        # Append at the start
        while len(buffer) >= BUFFER_SIZE:
            buffer.pop(-1)
        buffer.insert(0, pose_result_2d)

        ####################################################
        ################## 3D POSE LIFTING #################
        pose_result_3d = inference_pose_lifter_model(model_3D, buffer, image_size=size, norm_pose_2d=False)

        # Store the 3d poses in a list
        keypoints_3d.append(pose_result_3d[0].pred_instances.keypoints[0])

        # inverse y and z in 3d keypoints
        temp_y = keypoints_3d[frame_idx][:, 2].copy()
        temp_z = keypoints_3d[frame_idx][:, 1].copy()
        temp_x = keypoints_3d[frame_idx][:, 0].copy()
        keypoints_3d[frame_idx][:, 0] = -temp_x
        keypoints_3d[frame_idx][:, 1] = -temp_y
        keypoints_3d[frame_idx][:, 2] = -temp_z
        pose_result_3d[0].pred_instances.keypoints[0] = keypoints_3d[frame_idx]

        ################## VISUALIZATION 3D #################
        output_frame_3D = visualizer_3d.add_datasample('visualization', frame, data_sample=pose_result_3d[0],
                                                       det_data_sample=pose_result_2d[0], show=False, draw_gt=False,
                                                       convert_keypoint=False, show_kpt_idx=True)
        output_frame_3D = cv2.resize(output_frame_3D, size_3D)
        # cv2.imshow('preview', output_frame_3D)
        output_video_3D.write(output_frame_3D)

        frame_idx += 1
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close display window
    cap.release()
    output_video_detection.release()
    output_video_pose.release()
    output_video_3D.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
