"""Helpers shared by the Metrabs wrappers (filtering of the keypoints and writing of .trc files)."""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from trc import TRCData

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
