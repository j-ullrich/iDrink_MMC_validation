"""
We need for Output:

- trial_id
- (phasecheck)
- (valid)
- side - L / R
- condition - affected / unaffected

- Total Movement Time
- Peak Velocity - Endeffector
- Peak Velocity - Elbow Angle
- Time to peak Velocity - seconds
- Time to first peak Velocity - seconds
- Time to peak Velocity - relative
- Time to first peak Velocity - relative
- Number of MovementUnits - Smoothness
- trunk Displacement - MM
- Shoulder Flexion Reaching - Max Flexion
- max Elbow Extension
- max shoulder Abduction

We get as Input:
path to csv containing the following columns:

- trial_id
- side - L / R
- condition - affected / unaffected

- ReachingStart
- ForwardStart
- DrinkingStart
- BackStart
- ReturningStart
- RestStart
- TotalMovementTime

"""
import os
import re

import pandas as pd
import numpy as np
from fuzzywuzzy import process

from importlib_metadata import metadata

import iDrinkTrial


class MurphyMeasures:
    """
    This class is used to calculate the Murphy measures and write them in a .csv file.

    There are two ways how we can get the murphy measures:

    1. We give the Murphy-object paths to files and folders.
        - bodypart positions, velocities and accelerations
        - joint positions, velocities and accelerations
        - trial_dir

    2. We give the object a trial_object

    """
    def __init__(self, path_csv=None,
                 # For mode 1
                 trial = None, root_data = None, source = 'opensim_analyze',
                 # For mode 2
                 path_bodyparts_pos = None, path_bodyparts_vel = None, path_bodyparts_acc=None,
                 path_joint_pos = None, path_joint_vel = None, path_joint_acc = None
                 ):

        self.trial = trial
        self.root_data = root_data
        self.dir_trial = None

        """Settings"""
        self.source = source

        self.path_bodyparts_pos = path_bodyparts_pos
        self.path_bodyparts_vel = path_bodyparts_vel
        self.path_bodyparts_acc = path_bodyparts_acc

        self.path_joint_pos = path_joint_pos
        self.path_joint_vel = path_joint_vel
        self.path_joint_acc = path_joint_acc

        self.path_csv = path_csv
        self.df = None

        """np.arrays to calculate measures"""
        self.time = None
        self.hand_vel = None
        self.elbow_vel = None
        self.trunk_pos = None

        """Contents of the .csv file"""
        self.trial_id = None
        self.side = None
        self.condition = None
        self.ReachingStart = None
        self.ForwardStart = None
        self.DrinkingStart = None
        self.BackStart = None
        self.ReturningStart = None
        self.RestStart = None
        self.TotalMovementTime = None

        self.PeakVelocity_mms = None
        self.elbowVelocity = None
        self.tTopeakV_s = None
        self.tToFirstpeakV_s = None
        self.tTopeakV_rel = None
        self.tToFirstpeakV_rel = None
        self.NumberMovementUnits = None
        self.InterjointCoordination = None
        self.trunkDisplacementMM = None
        self.trunkDisplacementDEG = None
        self.ShoulerFlexionReaching = None
        self.ElbowExtension = None
        self.shoulderAbduction = None

        if self.path_csv is not None:
            self.df = self.read_csv(self.path_csv)
            self.get_data(self.df)

        # if paths  are given, directly calculate the measures
        if trial is not None:
            self.trial_id = trial.trial_id
            self.get_paths()

        if (    self.path_bodyparts_pos is not None
            and self.path_bodyparts_vel is not None
            and self.path_bodyparts_acc is not None
            and self.path_joint_pos is not None
            and self.path_joint_vel is not None
            and self.path_joint_acc is not None):

            self.read_files()
            self.get_measures()
            self.write_measures()


    def get_paths_from_root(self):

        s_id = self.trial_id.split('_')[0]
        p_id = self.trial_id.split('_')[1]
        t_id = self.trial_id.split('_')[2]

        s_num = re.search(r'\d+', s_id).group()

        path_analyzetool = os.path.join(self.root_data, f"setting_{s_num}",
                                        f'{s_id}', f'{s_id}_{p_id}', f'{s_id}_{p_id}_{t_id}', 'movement_analysis',
                                        'kin_opensim_analyzetool')
        path_p2s = os.path.join(self.root_data, f"setting_{s_num}",
                                        f'{s_id}', f'{s_id}_{p_id}', f'{s_id}_{p_id}_{t_id}', 'movement_analysis',
                                        'kin_p2s')


        match self.source:
            case 'opensim_analyze': # Use bodypart and joint values from analyzetool
                self.path_bodyparts_pos = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_BodyKinematics_pos_global.sto')
                self.path_bodyparts_vel = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_BodyKinematics_vel_global.sto')
                self.path_bodyparts_acc = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_BodyKinematics_acc_global.sto')
                self.path_joint_pos = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_Kinematics_q.sto')
                self.path_joint_vel = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_Kinematics_u.sto')
                self.path_joint_acc = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_Kinematics_dudt.sto')

            case 'pose2sim': # Use bodyparts from pose2sim and joint angles from analyzetool
                self.path_bodyparts_pos = os.path.join(path_p2s, f'{s_id}_{p_id}_{t_id}_Body_kin_p2s_pos.csv')
                self.path_bodyparts_vel = os.path.join(path_p2s, f'{s_id}_{p_id}_{t_id}_Body_kin_p2s_vel.csv')
                self.path_bodyparts_acc = os.path.join(path_p2s, f'{s_id}_{p_id}_{t_id}_Body_kin_p2s_acc.csv')
                self.path_joint_pos = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_Kinematics_q.sto')
                self.path_joint_vel = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_Kinematics_u.sto')
                self.path_joint_acc = os.path.join(path_analyzetool, f'{s_id}_{p_id}_{t_id}_Kinematics_dudt.sto')

            case _:
                raise ValueError(f"Error in iDrinkMurphyMeasures.get_paths_from_root: {self.source} invalid.")


    def get_paths_from_trial(self):
        """
        get all paths from trial object.
        :return:
        """

        self.dir_trial = self.trial.dir_trial

        match self.source:
            case 'opensim_analyze': # Use bodypart and joint values from analyzetool
                self.path_bodyparts_pos = self.trial.path_opensim_ana_pos
                self.path_bodyparts_vel = self.trial.path_opensim_ana_vel
                self.path_bodyparts_acc = self.trial.path_opensim_ana_acc
                self.path_joint_pos = self.trial.path_opensim_ana_ang_pos
                self.path_joint_vel = self.trial.path_opensim_ana_ang_vel
                self.path_joint_acc = self.trial.path_opensim_ana_ang_acc

            case 'pose2sim': # Use bodyparts from pose2sim and joint angles from analyzetool
                self.path_bodyparts_pos = self.trial.path_p2s_ik_pos
                self.path_bodyparts_vel = self.trial.path_p2s_ik_vel
                self.path_bodyparts_acc = self.trial.path_p2s_ik_acc
                self.path_joint_pos = self.trial.path_opensim_ana_ang_pos
                self.path_joint_vel = self.trial.path_opensim_ana_ang_vel
                self.path_joint_acc = self.trial.path_opensim_ana_ang_acc

            case _:
                raise ValueError(f"Error in iDrinkMurphyMeasures.get_paths_from_trial: {self.source} invalid.")


    def get_paths(self, ):
        """
        Get paths for bodyparts and joints based on trial object.

        To all this function, the trial_id must be known.
        """
        if self.trial_id is None:
            print("No trial_id given.\n"
                  "Murphy object needs a trial_id to retrieve data from DataFrame.")
            return

        if self.root_data is None:
            print("No root_data given.")
            return

        if self.trial is not None:
            self.get_paths_from_trial()
        elif all([self.trial_id, self.root_data] is not None):
            self.get_paths_from_root()


        pass


    def get_measures(self, ):
        """
        This function calculates the Murphy measures from the given data.
        """
        pass


    def write_measures(self, ):
        """
        This function adds the calculated measures to the .csv file.
        """
        pass

    @staticmethod
    def standardize_data(df, verbose=1):
        """
        gets a DataFrame containing data of joints or endeffector positions.

        It checks for the type of data and then renames the columns to a standardized set for later functions.

        metadata either contains a list or a string ('Speeds', 'Coordinates', 'Accelerations')

        :param verbose:
        :param metadata:
        :param df:
        :return: Datatype, DataFrame
        """

        def standardize_columns(columns_old, columns_stand, verbose=1):
            """
            This function takes a list of columns and a list of standardized names and renames the columns to the standardized names.

            :param verbose:
            :param columns_old:
            :param columns_stand:
            :return: columns_new
            """

            columns_old = [col.lower() for col in columns_old]  # make all columns lowercase
            columns_old = [col.replace(" ", "") for col in columns_old]  # Get rid of all whitespaces
            if any('rot' in col for col in columns_old):  # Check if 'rot' is contained in any of the columns
                columns_old = [col.replace('rot', 'o') for col in columns_old]  # Replace 'rot' with 'ox'
            if '#times' in columns_old:  # Check if '#times' is in the columns and rename to 'time'"""
                columns_old[columns_old.index('#times')] = 'time'

            # Safety check for columns that are not in the standardized list
            columns_new = []
            for col_old in columns_old:
                if col_old not in columns_stand:
                    # Finde element in columns_stand that is most similar to col_old
                    if verbose >= 2:
                        print(f"old: {col_old}\tnew: {process.extractOne(col_old, columns_stand)}")
                    columns_new.append(process.extractOne(col_old, columns_stand)[
                                           0])  # Look for the most similar element in columns_stand
                else:
                    columns_new.append(col_old)

            return columns_new

        stand_bodypart = ['time', 'pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_ox', 'pelvis_oy', 'pelvis_oz',
                          'sacrum_x', 'sacrum_y', 'sacrum_z', 'sacrum_ox', 'sacrum_oy', 'sacrum_oz',
                          'femur_r_x', 'femur_r_y', 'femur_r_z', 'femur_r_ox', 'femur_r_oy', 'femur_r_oz',
                          'patella_r_x', 'patella_r_y', 'patella_r_z', 'patella_r_ox', 'patella_r_oy', 'patella_r_oz',
                          'tibia_r_x', 'tibia_r_y', 'tibia_r_z', 'tibia_r_ox', 'tibia_r_oy', 'tibia_r_oz',
                          'talus_r_x', 'talus_r_y', 'talus_r_z', 'talus_r_ox', 'talus_r_oy', 'talus_r_oz',
                          'calcn_r_x', 'calcn_r_y', 'calcn_r_z', 'calcn_r_ox', 'calcn_r_oy', 'calcn_r_oz',
                          'toes_r_x', 'toes_r_y', 'toes_r_z', 'toes_r_ox', 'toes_r_oy', 'toes_r_oz',
                          'femur_l_x', 'femur_l_y', 'femur_l_z', 'femur_l_ox', 'femur_l_oy', 'femur_l_oz',
                          'patella_l_x', 'patella_l_y', 'patella_l_z', 'patella_l_ox', 'patella_l_oy', 'patella_l_oz',
                          'tibia_l_x', 'tibia_l_y', 'tibia_l_z', 'tibia_l_ox', 'tibia_l_oy', 'tibia_l_oz',
                          'talus_l_x', 'talus_l_y', 'talus_l_z', 'talus_l_ox', 'talus_l_oy', 'talus_l_oz',
                          'calcn_l_x', 'calcn_l_y', 'calcn_l_z', 'calcn_l_ox', 'calcn_l_oy', 'calcn_l_oz',
                          'toes_l_x', 'toes_l_y', 'toes_l_z', 'toes_l_ox', 'toes_l_oy', 'toes_l_oz',
                          'lumbar5_x', 'lumbar5_y', 'lumbar5_z', 'lumbar5_ox', 'lumbar5_oy', 'lumbar5_oz',
                          'lumbar4_x', 'lumbar4_y', 'lumbar4_z', 'lumbar4_ox', 'lumbar4_oy', 'lumbar4_oz',
                          'lumbar3_x', 'lumbar3_y', 'lumbar3_z', 'lumbar3_ox', 'lumbar3_oy', 'lumbar3_oz',
                          'lumbar2_x', 'lumbar2_y', 'lumbar2_z', 'lumbar2_ox', 'lumbar2_oy', 'lumbar2_oz',
                          'lumbar1_x', 'lumbar1_y', 'lumbar1_z', 'lumbar1_ox', 'lumbar1_oy', 'lumbar1_oz',
                          'torso_x', 'torso_y', 'torso_z', 'torso_ox', 'torso_oy', 'torso_oz',
                          'head_x', 'head_y', 'head_z', 'head_ox', 'head_oy', 'head_oz',
                          'abdomen_x', 'abdomen_y', 'abdomen_z', 'abdomen_ox', 'abdomen_oy', 'abdomen_oz',
                          'humerus_r_x', 'humerus_r_y', 'humerus_r_z', 'humerus_r_ox', 'humerus_r_oy', 'humerus_r_oz',
                          'ulna_r_x', 'ulna_r_y', 'ulna_r_z', 'ulna_r_ox', 'ulna_r_oy', 'ulna_r_oz',
                          'radius_r_x', 'radius_r_y', 'radius_r_z', 'radius_r_ox', 'radius_r_oy', 'radius_r_oz',
                          'hand_r_x', 'hand_r_y', 'hand_r_z', 'hand_r_ox', 'hand_r_oy', 'hand_r_oz',
                          'humerus_l_x', 'humerus_l_y', 'humerus_l_z', 'humerus_l_ox', 'humerus_l_oy', 'humerus_l_oz',
                          'ulna_l_x', 'ulna_l_y', 'ulna_l_z', 'ulna_l_ox', 'ulna_l_oy', 'ulna_l_oz',
                          'radius_l_x', 'radius_l_y', 'radius_l_z', 'radius_l_ox', 'radius_l_oy', 'radius_l_oz',
                          'hand_l_x', 'hand_l_y', 'hand_l_z', 'hand_l_ox', 'hand_l_oy', 'hand_l_oz',
                          'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z']
        stand_joints = ['time', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                        'knee_angle_r', 'knee_angle_r_beta', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
                        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                        'knee_angle_l', 'knee_angle_l_beta', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
                        'L5_S1_Flex_Ext', 'L5_S1_Lat_Bending', 'L5_S1_axial_rotation', 'L4_L5_Flex_Ext',
                        'L4_L5_Lat_Bending',
                        'L4_L5_axial_rotation', 'L3_L4_Flex_Ext', 'L3_L4_Lat_Bending', 'L3_L4_axial_rotation',
                        'L2_L3_Flex_Ext', 'L2_L3_Lat_Bending', 'L2_L3_axial_rotation', 'L1_L2_Flex_Ext',
                        'L1_L2_Lat_Bending',
                        'L1_L2_axial_rotation', 'L1_T12_Flex_Ext', 'L1_T12_Lat_Bending', 'L1_T12_axial_rotation',
                        'Abs_r3', 'Abs_r2', 'Abs_r1', 'Abs_t1', 'Abs_t2', 'neck_flexion', 'neck_bending',
                        'neck_rotation',
                        'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r',
                        'wrist_dev_r',
                        'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l',
                        'wrist_dev_l']

        if 'elbow_flex_l' in df.columns:

            if verbose >= 1:
                print("Standardizing: \tJoint Data.")

            df.columns = standardize_columns(df.columns, stand_joints, verbose)

        elif any(i in df.columns for i in [' hand_l_x', 'hand_l_x', 'hand_l_X']):
            if verbose >= 1:
                print("Standardizing:\tEndeffector Data")

            df.columns = standardize_columns(df.columns, stand_bodypart, verbose)

        else:
            raise ValueError("Error in iDrinkStatisticalAnalysis.standardize_data\n"
                             "Neither 'elbow_flex_l' nor 'hand_l_x' are in Data.\n"
                             "Please check the data and try again.")

        return df


    def read_file(self, file_path):
        """
        Reads a file and returns meta_dat and a DataFrame
        :param path:
        :return:
        """
        """
        Reads an opensim file (.mot, .sto) and returns meta_dat as 2D-list and Data as pandas Dataframe.

        input:
            Path to file

        output:
            Metadata: List
            Data: pandas Dataframe
        """
        # Read Metadata and end at "endheader"
        if os.path.splitext(file_path)[1] == '.sto':
            meta_dat = []
            with open(file_path, 'r') as file:
                for row in file:
                    meta_dat.append(row.split('\n')[0])
                    if "endheader" in row.strip().lower():
                        break
                file.close()

            # Read the rest of the file into a DataFrame
            df = pd.read_csv(file_path, skiprows=len(meta_dat), sep="\t")
        elif  os.path.splitext(file_path)[1] == '.csv':
            meta_dat = None
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Error in iDrinkMurphyMeasures.read_file: {file_path} \n"
                             f"invalid file format {os.path.splitext(file_path)}.")

        df = self.standardize_data(df)
        return meta_dat, df


    def read_files(self):
        """
        Reads all files and creates numpy arrays for calculation of murphy measures.

        - time
        - hand_vel
        - elbow_vel
        - trunk_pos

        :return:
        """
        def reduce_axes(data):
            """
            This function gets absolute velocity based on velocity in 3 Dimensions
            """

            return np.sqrt(np.sum(np.array([axis**2 for axis in data]), axis=0))

        _, df = self.read_file(self.path_bodyparts_vel)

        self.time = df['time'].values

        # hand_{self.side.lower()}_x, hand_{self.side.lower()}_y, hand_{self.side.lower()}_z
        hand_vel = [df[f'hand_{self.side.lower()}_{axis}'].values for axis in ['x', 'y', 'z']]
        self.hand_vel = reduce_axes(hand_vel)

        _, df = self.read_file(self.path_joint_vel)
        self.elbow_vel = df[f'elbow_flex_{self.side.lower()}'].values

        _, df = self.read_file(self.path_bodyparts_pos)
        # TODO: Decide what to use for trunk displacement
        trunk_pos = [df[f'head_{axis}'].values for axis in ['x', 'y', 'z']]  # For now, we use head position
        self.trunk_pos = reduce_axes(trunk_pos)



    def get_data(self, df, verbose=1):
        """
        Sync the attributes with the DataFrame.
        """
        columns_of_interest = ["ReachingStart", "ForwardStart", "DrinkingStart", "BackStart",
                               "ReturningStart", "RestStart", "TotalMovementTime"]

        if self.trial_id is None:
            print("No trial_id given.\n"
                  "Murphy object needs a trial_id to retrieve data from DataFrame.")
            return

        self.side = self.trial.measured_side
        self.condition = self.trial.affected

        for column in columns_of_interest:
            self.__setattr__(column, df.loc[df['trial_id'] == self.trial_id, column].values[0])


    def read_csv(self, path_csv):
        """
        This function reads the .csv file containing the data. and calls the get_data function.
        It also returns the DataFrame from the .csv file.
        """
        self.df = pd.read_csv(path_csv)

        return self.df





if __name__ == '__main__':

    """
    For now, the main is only for debugging the script with a hardcoded participant.
    """
    # TODO: Add argparse so file could be used standalone to get the measures when .csv file and directory is given.
    if os.name == 'posix':  # Running on Linux
        drive = '/media/devteam-dart/Extreme SSD'
        root_iDrink = os.path.join(drive, 'iDrink')  # Root directory of all iDrink Data
    else:
        path_phases = r"I:\DIZH_data\P01\OMC\P01_trialVectors.csv"
        dir_trials = r"I:\iDrink\validation_root\03_data\OMC\S15133\S15133_P01" # Directory containing folders of P01




    pd.DataFrame(columns = ['trial', 'side', 'condition',
                            'ReachingStart', 'ForwardStart', 'DrinkingStart',  'BackStart', 'ReturningStart', 'RestStart',
                            'TotalMovementTime', 'PeakVelocity_mms',  'elbowVelocity',
                            'tTopeakV_s', 'tToFirstpeakV_s', 'tTopeakV_rel', 'tToFirstpeakV_rel',
                            'NumberMovementUnits', 'InterjointCoordination',
                            'trunkDisplacementMM', 'trunkDisplacementDEG',
                            'ShoulerFlexionReaching', 'ElbowExtension', 'shoulderAbduction'
                            ]
                 )