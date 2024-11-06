import sys
import os
from os import remove

from tqdm import tqdm
import pandas as pd
import numpy as np
from trc import TRCData
import platform

import glob

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from iDrink import iDrinkUtilities



"""
Look for all .trc files of unfiltered keypoints, load them into an df, filter them and overwrite existing filtered trc files.

"""


def use_butterworth_filter(data, cutoff, fs, order=4, normcutoff=False, verbose = 2):
    """
    Input:
        - data: The data to be filtered
        - cutoff: The cutoff frequency of the filter
        - fs: The sampling frequency
        - order: The order of the filter

    Output:
        - filtered_data: The filtered data
    """
    from scipy.signal import butter, sosfiltfilt

    order = int(order / 2)  # Order is "doubled" again by using filter 2 times

    nyquist = 0.5 * fs

    if cutoff >= nyquist:
        if verbose >= 2:
            print(f"Warning: Cutoff frequency {cutoff} is higher than Nyquist frequency {nyquist}.")
            print("Filtering with Nyquist frequency.")
        cutoff = nyquist - 1

    if normcutoff:
        cutoff = cutoff / nyquist

    sos = butter(order, cutoff, btype="low", analog=False, output="sos", fs=fs)

    filtered_data = sosfiltfilt(sos, data, axis=0)

    return filtered_data

def filter_df(df_unfiltered, fs, cutoff, order,  verbose=1, normcutoff=False):
    """
    Use a butterworth-filter on the 3D-keypoints in the DF and return the filtered DF.

    :param df: DataFrame with 3D coordinates
    :param fs: Sampling frequency

    :return: DataFrame with filtered 3D coordinates

    """
    from scipy.signal import butter, sosfiltfilt

    df_filtered = pd.DataFrame(columns=df_unfiltered.columns)

    if verbose >= 1:
        progress = tqdm(total=len(df_unfiltered.columns), desc="Filtering 3D keypoints", position=0, leave=True)

    for column in df_unfiltered.columns:
        data = np.array(df_unfiltered[column].tolist())
        df_filtered[column] = use_butterworth_filter(data, cutoff, order = order, fs=fs, normcutoff=normcutoff).tolist()

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

    for column in tqdm(df.columns, desc="Writing .trc file", position=0, leave=True, disable=verbose < 1):
        trc[column] = df[column].tolist()

    for i in range(n_frames):
        trc[i] = [trc['Time'][i], df.iloc[i, :].tolist()]

    if verbose >= 2:
        print(f'Timestamps added to {os.path.basename(trc_file)}')

    trc.save(trc_file)
    if verbose >= 1:
        print(f"Saved .trc file {os.path.basename(trc_file)}")

def trc_to_df(trc_file, verbose=1):
    """
    Loads a .trc file and returns the data in a DaaFrame.
    :param trc_file:
    :param verbose:
    :return:
    """
    pass

    trc = TRCData()
    trc.load(trc_file)

    df = pd.DataFrame(columns=trc['Markers'])

    for column in trc['Markers']:
        df[column] = trc[column]

    return df


def read_opensim_file(file_path):
    """
    Reads an opensim file (.mot, .sto) and returns metadata as 2D-list and Data as pandas Dataframe.

    input:
        Path to file

    output:
        Metadata: List
        Data: pandas Dataframe
    """
    # Read Metadata and end at "endheader"
    metadata = []
    with open(file_path, 'r') as file:
        for row in file:
            metadata.append(row.split('\n')[0])
            if "endheader" in row.strip().lower():
                break

        file.close()

    # Read the rest of the file into a DataFrame
    df = pd.read_csv(file_path, skiprows=len(metadata), sep="\t")
    return metadata, df

def remove_duplicates_unfiltered(trc_files_unfiltered):
    """
    Gets list of paths to unfiltered trc files.

    If multiple trc files have the same idP and id_t, only the last one is kept.

    The s_id is removed from the filename.

    """

    trc_files = glob.glob(os.path.join(trc_files_unfiltered, '*.trc'))
    if not trc_files:
        print(f"No .trc files found in {trc_files_unfiltered}")
        return

    trc_files = sorted(trc_files, key=lambda x: x.split('_')[-1])

    list_ids = []
    for trc_file in trc_files:



        dirname = os.path.dirname(trc_file)
        filename = os.path.basename(trc_file)

        split_id = 0
        if 'S' in filename.split('_')[split_id]:
            id_s = filename.split('_')[split_id]
            split_id += 1
        else:
            id_s = None

        id_p = filename.split('_')[split_id]
        id_t = filename.split('_')[split_id+1]
        ids = (id_p, id_t)
        if ids in list_ids:
            pass
            os.remove(trc_file)
        else:
            list_ids.append((id_p, id_t))
            if id_s is not None:
                filename = filename.split(f'{id_s}_')[1]
            #rename file
            os.rename(trc_file, os.path.join(dirname, filename))





def get_unfiltered_trc_files(root_hpe, fps, cutoff, order,  remove_old_files=False, verbose=1):

    dir_unfilt = os.path.join(root_hpe, '01_unfiltered')
    dir_filt = os.path.join(root_hpe, '02_filtered')

    idx_p = sorted(os.listdir(dir_unfilt))

    pass

    for id_p in idx_p:
        dir_p_unfilt = os.path.join(dir_unfilt, id_p)
        dir_p_filt = os.path.join(dir_filt, id_p)

        cams = sorted(dirname.split('_')[1] for dirname in os.listdir(dir_p_unfilt))

        for cam in cams:
            dir_cam_unfilt = os.path.join(dir_p_unfilt, f'{id_p}_{cam}')
            dir_cam_filt = os.path.join(dir_p_filt, f'{id_p}_{cam}')

            dir_single_cam_unfilt = os.path.join(dir_cam_unfilt, 'metrabs', 'single-cam')
            dir_single_cam_filt = os.path.join(dir_cam_filt, 'metrabs', 'single-cam')


            if os.path.isdir(dir_single_cam_unfilt):
                remove_duplicates_unfiltered(dir_single_cam_unfilt)
                trc_filenames_unfilt = sorted([f for f in os.listdir(dir_single_cam_unfilt) if f.endswith('.trc')])



            else:
                print(f"Directory {dir_single_cam_unfilt} does not exist.")
                continue

            prgs = tqdm(total=len(trc_filenames_unfilt), desc=f"Processing {id_p}_{cam}", position=0, leave=True)

            for trc_filename_unfilt in trc_filenames_unfilt:
                prgs.set_description(f"Processing {id_p}_{cam} - {trc_filename_unfilt}")

                id_t = trc_filename_unfilt.split('_')[1]
                while 'T' not in id_t:
                    i = 0
                    id_t = trc_filename_unfilt.split('_')[i]
                    i += 1

                    if '.trc' in id_t:
                        raise ValueError(f"Could not find id_t in {trc_filename_unfilt}")

                    return

                trc_filename_filt = os.path.join(dir_single_cam_filt, trc_filename_unfilt.replace('_iDrink', '_iDrinkbutter_filt'))

                trc_path_filt = os.path.join(dir_single_cam_filt, trc_filename_filt)
                trc_path_unfilt = os.path.join(dir_single_cam_unfilt, trc_filename_unfilt)

                if remove_old_files:
                    old_files = glob.glob(os.path.join(dir_single_cam_filt, f'*{id_p}*{id_t}*'))
                    for file in old_files:
                        os.remove(file)

                df_unfilt = trc_to_df(trc_path_unfilt)

                df_filt = filter_df(df_unfilt, fs=fps, cutoff=cutoff, order=order, verbose=0)

                df_to_trc(df_filt, trc_path_filt, identifier=trc_filename_filt, fps=fps, n_frames=len(df_filt), n_markers=len(df_filt.columns), verbose=0)

                prgs.update(1)

            prgs.close()





if __name__ == '__main__':
    """Set Root Paths for Processing"""
    drive = iDrinkUtilities.get_drivepath()

    root_iDrink = os.path.join(drive, 'iDrink')  # Root directory of all iDrink Data
    root_MMC = os.path.join(root_iDrink, "Delta",
                            "data_newStruc")  # Root directory of all MMC-Data --> Videos and Openpose json files
    root_val = os.path.join(root_iDrink,
                            "validation_root")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
    default_dir = os.path.join(root_val, "01_default_files")  # Default Files for the iDrink Validation
    root_HPE = os.path.join(root_val, "02_pose_estimation")  # Root directory of all Pose Estimation Data

    fps = 60
    cutoff = 5
    order = 4

    get_unfiltered_trc_files(root_HPE, fps=fps, cutoff=cutoff, order=order, remove_old_files=True, verbose=1)
