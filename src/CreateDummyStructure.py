import os
import json
import cv2
import numpy as np

from tqdm import tqdm

root = r"C:\iDrink\Test_folder_structures"

n_p = 10  # Number of Participants
n_cams = 10  # Number of Cameras
n_trials = 10  # Number of Trials
n_frames = 100  # Number of Frames per Trial

l_1 = ["01_measurement"]
l_2 = ["01_IMU", "02_Kinect", "03_Mocap", "04_Video", "05_Open_Pose"]
l_3_vid = ["01_Uncut", "02_Led_Video", "03_Cut", "04_Cut_ordered", "06_Calib_before"]
l_4_vid = ["drinking"]
l3_op = ["drinking"]


"""Create dummy folder structure for all participants"""
n_elements = n_p * (len(l_1) + len(l_2) + len(l_3_vid) + len(l_4_vid)) + (n_p * n_cams * n_trials) * (1+n_frames)

progress = tqdm(total=n_elements, desc="Creating dummy folder structure")

for p in range(1, n_p + 1):
    for l1 in l_1:
        for l2 in l_2:
            if l2 == "04_Video":
                for l3 in l_3_vid:
                    if l3 == "03_Cut":
                        for l4 in l_4_vid:
                            for cam in range(1, n_cams + 1):
                                folder = os.path.join(root, f"P{p:02d}", l1, l2, l3, l4, f"cam{cam}")
                                os.makedirs(folder, exist_ok=True)
                                for trial in range(n_trials):
                                    # Video properties
                                    frame_width = 640
                                    frame_height = 480
                                    fps = 30
                                    num_frames = 100
                                    aff = "R_unaffected"
                                    if trial > n_trials/2:
                                        aff = "L_affected"
                                    output_file = os.path.join(folder, f'trial{trial}_{aff}_dummy_video.mp4')

                                    # Define the codec and create VideoWriter object
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
                                    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

                                    for i in range(num_frames):
                                        # Create a dummy frame (a simple gradient image)
                                        frame = np.zeros((frame_width, frame_height, 3), dtype=np.uint8)
                                        cv2.putText(frame, f'Frame {i+1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                        out.write(frame)
                                    # Release everything if job is finished
                                    out.release()
                                    cv2.destroyAllWindows()
                                    progress.update(1)
                    else:
                        os.makedirs(os.path.join(root, f"P{p:02d}", l1, l2, l3), exist_ok=True)
                        progress.update(1)
            elif l2 == "05_Open_Pose":
                for l3 in l3_op:
                    for cam in range(1, n_cams + 1):
                        for trial in range(1, n_trials + 1):
                            aff = "R_unaffected"
                            if trial > n_trials / 2:
                                aff = "L_affected"

                            folder = os.path.join(root, f"P{p:02d}", l1, l2, l3, f"cam{cam}", f"trial{trial}_{aff}[...]_json")
                            os.makedirs(folder, exist_ok=True)
                            progress.update(1)
                            for frame in range(n_frames):
                                dummy_json_data = {"data": f"Dummy data for frame {frame}"}
                                json_file_path = os.path.join(folder, f"trial{trial}_{aff}_{frame:12d}_keypoints.json")
                                progress.update(1)

            else:
                os.makedirs(os.path.join(root, f"P{p:02d}", l1, l2), exist_ok=True)
                progress.update(1)