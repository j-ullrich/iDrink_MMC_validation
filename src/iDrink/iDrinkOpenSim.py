import logging
import logging.handlers
import sys
import os
import string
import glob
from tqdm import tqdm

import numpy as np
import opensim
import pandas as pd
from trc import TRCData


def edit_xml(xml_file, xml_keys, new_values, target_file=None, filename_appendix=None):
    from xml.etree import ElementTree

    """
    by default, this function edits an existing file. 
    If target Directory is given, a new xml_file is created.
    A new Appendix to the new filename can be given. e.g. The Session/Participant ID

    Input:
        xml_file: path to xml_file to edit
        xml_keys: list of keys in file to be edited
        new_values: the new values
            ith value corresponds to ith key

        Possible changes to function: 
        Change so Input changes to:

        table: pandas Dataframe containing:
            - the path to the default xml_files
            - the path for new and edited xml_files
            - key that will be changed
            - new values
    """

    tree = ElementTree.parse(xml_file)
    root = tree.getroot()

    # Find the element to be edited
    for i in range(len(xml_keys)):
        for key in root.iter(xml_keys[i]):
            # print(f"changing {key.text} to {new_value}")
            key.text = new_values[i]  # Change the value of the element

    # check create_new_file
    if target_file is not None:  # if target file is given, create new file
        # check whether destination_folder exists
        target_dir = os.path.dirname(target_file)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if filename_appendix is not None:
            basename, extension = os.path.splitext(os.path.basename(xml_file))
            filename = f"{basename}_{filename_appendix}{extension}"
            target_file = os.path.join(target_dir, filename)

        try:
            with open(target_file, "wb") as f:
                tree.write(f)
        except Exception as e:
            print(f"Error: {e}")

    else:  # edit existing file
        try:
            tree.write(xml_file)
        except Exception as e:
            print(f"Error: {e}")


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

def get_joint_velocity_acceleration(csv_pos, dir_out=None, filter_pos=True, verbose=1):
    """
    Calculates velocity and acceleration for joint positions.

    if dir_out is not given, the csv files will be written into the same folder as the input file.

    :param csv_pos:
    :param dir_out:
    :return:
    """
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from iDrinkAnalytics import use_butterworth_filter

    if dir_out is None:
        dir_out = os.path.dirname(csv_pos)

    # Read Data
    df_pos = pd.read_csv(csv_pos)
    list_columns = df_pos.columns.tolist()

    df_vel = pd.DataFrame(columns=list_columns)
    df_vel["time"] = df_pos["time"]

    df_acc = df_vel.copy(deep=True)

    if verbose >= 1:
        progressbar = tqdm(total=len(df_pos.columns[2:]), desc="Calculating Joint Velocity and Acceleration", unit="Component")

    for component in df_pos.columns[2:]:
        # Calculate gradient for each component and write it to DataFrame

        if filter_pos:
            comp_array = use_butterworth_filter(np.array(df_pos[component]), cutoff=10, fs=100, order=6, normcutoff=False)
        else:
            comp_array = np.array(df_pos[component])

        comp_array = np.gradient(comp_array)

        # write to Dataframe
        df_vel[component] = comp_array
        df_acc[component] = np.gradient(comp_array)

        if verbose >= 1:
            progressbar.update(1)

    if verbose >= 1:
        progressbar.close()

    # Save to .csv
    df_vel.to_csv(os.path.join(dir_out, os.path.basename(csv_pos).replace("pos", "vel")), index=False)
    df_acc.to_csv(os.path.join(dir_out, os.path.basename(csv_pos).replace("pos", "acc")), index=False)

    if verbose >= 1:
        print(f"Joint Velocity and Acceleration calculated and saved to:\n"
              f"{os.path.join(dir_out, os.path.basename(csv_pos).replace('pos', 'vel'))}\n"
              f"{os.path.join(dir_out, os.path.basename(csv_pos).replace('pos', 'acc'))}\n")

def mot_to_csv(curr_trial=None, path_mot=None, path_dst=None, verbose=1):
    """
    This function saves the angles calculated by the Inverse Kinematics Tool to a .csv file.

    there are two options:

    1. trial is given -> source and destination paths are retrieved front object.
    2. path_mot and path_dst are given -> source and destination paths are given as arguments.

    If all are given, priority lies on the path_mot and path_dst arguments.

    :param curr_trial: iDrinkTrial Object
    :param path_mot: path to the .mot file
    :param dir_dst: path to the destination folder

    """

    mode1 = False if curr_trial is None else True
    mode2 = False if path_mot is None or path_dst is None else True

    if not mode1 and not mode2:
        raise ValueError(f"Error in iDrinkOpenSim.ik_tool_to_csv\n"
                         f"Either curr_trial or path_mot and path_dst must be given.\n"
                         f"Current values are:\n"
                         f"curr_trial: {curr_trial}\t path_mot: {path_mot}\t path_dst: {path_dst}")

    # Get Filename and path
    if path_dst is None:

        dir_out = curr_trial.dir_kin_ik_tool
        # Make sure, Folder exists
        if not os.path.exists(dir_out):
            os.makedirs(dir_out, exist_ok=True)

        path_dst = os.path.realpath(os.path.join(dir_out, f"{curr_trial.identifier}_Kinematics_pos.csv"))



    # Read data and save to .csv
    if path_mot is None:
        path_src = os.path.join(curr_trial.dir_trial, curr_trial.opensim_motion)

    _, dat_measured = read_opensim_file(path_mot)
    dat_measured.to_csv(path_dst, index=False)

    get_joint_velocity_acceleration(csv_pos=path_dst, dir_out=os.path.dirname(path_dst), filter_pos=False, verbose=verbose)

    if verbose >= 1:
        print(f"Data from: \t{path_mot}\n"
              f"saved to: \t{path_dst}\n")


def open_sim_pipeline(curr_trial, log_dir = None, verbose=1):
    """
    This function loads an opensim Model and runs the Scale- and Inverse Kinematics Tool on it.
    the .mot file will be saved to the determined folder.
    Then it creates the table for the measured data and returns it to tha calling instance.

    Input:
        dict_path:  Dictionary containing the following paths:
                        - model:        OpenSim Model
                        - scaling:      .xml for scaling
                        - invkin:       .xml for Inverse Kinematics
                        - modelscaled:  scaled model for Inverse Kinematics
                        - motfile:      Motion file for Inverse Kinematics
                        - marker:       marker files
                        - filmark:      filtered marker files
    """

    seq_name = os.path.basename(curr_trial.dir_trial)

    logging.info("\n\n---------------------------------------------------------------------")
    logging.info(f"Running OpenSim for {seq_name}, for all frames.")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {curr_trial.dir_trial}")

    def correct_skeleton_orientation(opensim_motion, realign=True, center=True):
        """
        Loads an openSim Motion File and changes Alignment and position in Coordinate-System.

        This Function is probably not needed anymore. FOr now, I will keep it

        if realign True:
            Tilt, List and Rotation are set to 0

        if center True:
            X, Z axes are set to 0
            Y axis is set to 1

        """

        metadata, data = read_opensim_file(opensim_motion)

        # Process Data
        if realign:
            data[["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx"]] = 0

        if center:
            data[["pelvis_tx", "pelvis_ty", "pelvis_tz"]] = [0, 1, 0]

        with open(opensim_motion, 'w') as file:
            # Write Metadata to file
            for row in metadata:
                file.write(row)

            # Add the DatFrame data to the file
            data.to_csv(file, sep='\t', index=False)

            file.close()

    def stabilize_hip_movement(curr_trial, path_trc, verbose = 1):
        """
        Take and trc file and smooth the movement of CHip, RHip, and LHip.

        I used different methods. Sliding Window mean / median, Splines, Interpolation.

        The most stable result gave putting the mean on all values of CHip, RHip, and LHip.
        """

        trc = TRCData()

        trc.load(filename=path_trc)

        if curr_trial.pose_model == "OMC":
            comp={"hip_L": -1,
                  "hip_R": -1,}
        elif curr_trial.pose_model == "bml_movi_87":
            comp = {"mhip": -1,
                    "rhip": -1,
                    "lhip": -1,
                    "pelv": -1}
        else:
            comp = {"CHip": -1,
                    "RHip": -1,
                    "LHip": -1}
        hips = {}

        for c in comp.keys():
            if c in trc.keys():
                comp[c] = list(trc.keys()).index(c) - list(trc.keys()).index("Time") - 1
                print(f"{c} found at index {comp[c]}")

                hips[c] = np.array(trc[c])
                hips[c][:, ] = np.nanmean(hips[c], axis=0, keepdims=True)

            else:
                print(f"{c} not found in .trc File.")
        if verbose >=1:
            progress = tqdm(total=len(trc['Frame#']), desc=f"{curr_trial.identifier} - Stabilizing Hip Movement", unit="Frame")
        for frame in trc['Frame#']:
            for c in comp.keys():
                if comp[c] > -1:
                    trc[frame][1][comp[c]] = hips[c].tolist()[frame - 1]
            if verbose >= 1:
                progress.update(1)

        if verbose >= 1:
            progress.close()

        trc.save(path_trc)

    try:
        """get paths to default files"""
        scaling_default = os.path.join(curr_trial.dir_default, f"Scaling_Setup_iDrink_{curr_trial.pose_model}.xml")
        invkin_default = os.path.join(curr_trial.dir_default, f"IK_Setup_iDrink_{curr_trial.pose_model}.xml")
        analyze_default = os.path.join(curr_trial.dir_default, f"AT_Setup.xml")

        """edit xml files for scaling and Inverse Kinematics"""
        edit_xml(scaling_default, ["model_file", "output_model_file", "marker_file", "time_range"],
                 [curr_trial.opensim_model, curr_trial.opensim_model_scaled, curr_trial.opensim_marker_filtered,
                  curr_trial.opensim_scaling_time_range], target_file=curr_trial.opensim_scaling)
        edit_xml(invkin_default, ["marker_file", "output_motion_file", "model_file", "time_range"],
                 [curr_trial.opensim_marker_filtered, curr_trial.opensim_motion, os.path.join(curr_trial.dir_trial, curr_trial.opensim_model_scaled),
                  curr_trial.opensim_IK_time_range], target_file=curr_trial.opensim_inverse_kinematics) # TODO: Check path settings difference between MMC and OMC. For OMC, Scaling needed relativa paths, IK needed absolute paths
        edit_xml(analyze_default, ["results_directory", "model_file", "coordinates_file", "initial_time", "final_time"],
                 [curr_trial.opensim_dir_analyze_results, curr_trial.opensim_model_scaled, curr_trial.opensim_motion,
                  curr_trial.opensim_ana_init_t, curr_trial.opensim_ana_final_t],
                 target_file=curr_trial.opensim_analyze)

    except FileNotFoundError as e:
        print("Either Scaling or Inverse Kinematics Setup file not found. Check default folder.\n")
        print(e)

    if curr_trial.stabilize_hip:
        stabilize_hip_movement(curr_trial, os.path.join(curr_trial.dir_trial, curr_trial.opensim_marker_filtered), verbose)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)


    # Set Logfile for Opensim
    opensim.Logger.removeFileSink()
    opensim.Logger.addFileSink(os.path.join(log_dir, f'{curr_trial.identifier}_opensim.log'))


    model = opensim.Model(curr_trial.opensim_model)
    model.initSystem()
    scaleTool = opensim.ScaleTool(curr_trial.opensim_scaling)
    ikTool = opensim.InverseKinematicsTool(curr_trial.opensim_inverse_kinematics)
    # Necessary for Table Processing
    ikTool.set_marker_file(curr_trial.opensim_marker_filtered)
    ikTool.set_output_motion_file(curr_trial.opensim_motion)
    ikTool.set_results_directory(os.path.join(curr_trial.dir_trial, f'pose-3d'))
    # Run Scaling and Invkin Tools
    scaleTool.run()
    ikTool.run()

    kinematic_analysis = opensim.Kinematics(curr_trial.opensim_analyze)
    if not os.path.exists(os.path.join(curr_trial.dir_trial, curr_trial.opensim_dir_analyze_results)):
        os.makedirs(os.path.join(curr_trial.dir_trial, curr_trial.opensim_dir_analyze_results))
    # Create analyze tool
    analyzetool = opensim.AnalyzeTool(curr_trial.opensim_analyze, True)
    analyzetool.setName(curr_trial.identifier)
    # Set Model file and motion file paths
    model_relpath = curr_trial.get_opensim_path(curr_trial.opensim_model_scaled)
    analyzetool.setModelFilename(model_relpath)
    analyzetool.setCoordinatesFileName(curr_trial.opensim_motion)
    analyzetool.run()

    # Add paths of Analyzertool Output to trial Object
    curr_trial.path_opensim_ana_pos = glob.glob(os.path.join(curr_trial.dir_anatool_results, r"*BodyKinematics_pos*"))[0]
    curr_trial.path_opensim_ana_vel = glob.glob(os.path.join(curr_trial.dir_anatool_results, r"*BodyKinematics_vel*"))[0]
    curr_trial.path_opensim_ana_acc = glob.glob(os.path.join(curr_trial.dir_anatool_results, r"*BodyKinematics_acc*"))[0]
    curr_trial.path_opensim_ana_ang_pos = glob.glob(os.path.join(curr_trial.dir_anatool_results, r"*Kinematics_q*"))[0]
    curr_trial.path_opensim_ana_ang_vel = glob.glob(os.path.join(curr_trial.dir_anatool_results, r"*Kinematics_u*"))[0]
    curr_trial.path_opensim_ana_ang_acc = glob.glob(os.path.join(curr_trial.dir_anatool_results, r"*Kinematics_dudt*"))[0]

    # Save the angles calculated by the Inverse Kinematics Tool to a .csv file
    mot_to_csv(curr_trial)



if __name__ == '__main__':

    mot_omc = r"I:\iDrink\validation_root\03_data\OMC\S15133\S15133_P07\S15133_P07_T043\pose-3d\S15133_P07_T043_R_affected.mot"
    omc_dir_dst = r"I:\iDrink\validation_root\03_data\OMC\S15133\S15133_P07\S15133_P07_T043\movement_analysis\ik_tool"
    path_omc_pos = r"I:\iDrink\validation_root\03_data\OMC\S15133\S15133_P07\S15133_P07_T043\movement_analysis\ik_tool\S15133_P07_T043_Kinematics_pos.csv"
    path_omc_vel = r"I:\iDrink\validation_root\03_data\OMC\S15133\S15133_P07\S15133_P07_T043\movement_analysis\ik_tool\S15133_P07_T043_Kinematics_vel.csv"
    path_omc_acc = r"I:\iDrink\validation_root\03_data\OMC\S15133\S15133_P07\S15133_P07_T043\movement_analysis\ik_tool\S15133_P07_T043_Kinematics_acc.csv"

    mot_mmc = r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\pose-3d\S003_P07_T043_0-928_filt_butterworth.mot"
    mmc_dir_dst = r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\ik_tool"
    path_mmc_pos = r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\ik_tool\S003_P07_T043_Kinematics_pos.csv"
    path_mmc_vel = r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\ik_tool\S003_P07_T043_Kinematics_vel.csv"
    path_mmc_acc = r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\ik_tool\S003_P07_T043_Kinematics_acc.csv"

    mot_to_csv(path_mot=mot_omc, path_dst=path_omc_pos)
    get_joint_velocity_acceleration(csv_pos=path_omc_pos, dir_out=omc_dir_dst)

    mot_to_csv(path_mot=mot_mmc, path_dst=path_mmc_pos)
    get_joint_velocity_acceleration(csv_pos=path_mmc_pos, dir_out=mmc_dir_dst)
