"""
We compare each MMC-Trial Objects time series and Murphy Measures to the same metrics of the OMC-trials.

"""
import os
import sys
import shutil
import glob
import re
from tqdm import tqdm
from fuzzywuzzy import process

from trc import TRCData

import pandas as pd
import numpy as np
import scipy as sp

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from iDrinkOpenSim import read_opensim_file

def runs_statistics_discrete(df_mmc, df_omc, root_stat):
    """
    Takess Murphy Measures of MMC and OMC and compares them. Then plots the results and saves data and plots in the Statistics Folder.
    :param df_mmc:
    :param df_omc:
    :return:
    """

    pass


def run_statistics_continuous(df_mmc_pos, df_mmc_vel, df_omc_pos, df_omc_vel, isjoint, root_stat):
    """
    Gets two DataFrames form MMC and OMC. Once the Position of the Bodyparts and then their Velocity.

    This function works with endeffector Pos/Vel and Joint Pos/Vel.

    It then plots the

    :param df_mmc_pos:
    :param df_mmc_vel:
    :param df_omc_pos:
    :param df_omc_vel:
    :param isjoint:
    :param root_stat:
    :return:
    """

    pass


def standardize_data(df, metadata=None, verbose=1):
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
                columns_new.append(process.extractOne(col_old, columns_stand)[0])  # Look for the most similar element in columns_stand
            else:
                columns_new.append(col_old)

        return columns_new

    stand_rawkps =  [] # List containing the standardized names of the raw keypoints
    stand_bodypart =  ['time','pelvis_x','pelvis_y','pelvis_z','pelvis_ox','pelvis_oy','pelvis_oz',
                       'sacrum_x','sacrum_y','sacrum_z','sacrum_ox','sacrum_oy','sacrum_oz',
                       'femur_r_x','femur_r_y','femur_r_z','femur_r_ox','femur_r_oy','femur_r_oz',
                       'patella_r_x','patella_r_y','patella_r_z','patella_r_ox','patella_r_oy','patella_r_oz',
                       'tibia_r_x','tibia_r_y','tibia_r_z','tibia_r_ox','tibia_r_oy','tibia_r_oz',
                       'talus_r_x','talus_r_y','talus_r_z','talus_r_ox','talus_r_oy','talus_r_oz',
                       'calcn_r_x','calcn_r_y','calcn_r_z','calcn_r_ox','calcn_r_oy','calcn_r_oz',
                       'toes_r_x','toes_r_y','toes_r_z','toes_r_ox','toes_r_oy','toes_r_oz',
                       'femur_l_x','femur_l_y','femur_l_z','femur_l_ox','femur_l_oy','femur_l_oz',
                       'patella_l_x','patella_l_y','patella_l_z','patella_l_ox','patella_l_oy','patella_l_oz',
                       'tibia_l_x','tibia_l_y','tibia_l_z','tibia_l_ox','tibia_l_oy','tibia_l_oz',
                       'talus_l_x','talus_l_y','talus_l_z','talus_l_ox','talus_l_oy','talus_l_oz',
                       'calcn_l_x','calcn_l_y','calcn_l_z','calcn_l_ox','calcn_l_oy','calcn_l_oz',
                       'toes_l_x','toes_l_y','toes_l_z','toes_l_ox','toes_l_oy','toes_l_oz',
                       'lumbar5_x','lumbar5_y','lumbar5_z','lumbar5_ox','lumbar5_oy','lumbar5_oz',
                       'lumbar4_x','lumbar4_y','lumbar4_z','lumbar4_ox','lumbar4_oy','lumbar4_oz',
                       'lumbar3_x','lumbar3_y','lumbar3_z','lumbar3_ox','lumbar3_oy','lumbar3_oz',
                       'lumbar2_x','lumbar2_y','lumbar2_z','lumbar2_ox','lumbar2_oy','lumbar2_oz',
                       'lumbar1_x','lumbar1_y','lumbar1_z','lumbar1_ox','lumbar1_oy','lumbar1_oz',
                       'torso_x','torso_y','torso_z','torso_ox','torso_oy','torso_oz',
                       'head_x','head_y','head_z','head_ox','head_oy','head_oz',
                       'abdomen_x','abdomen_y','abdomen_z','abdomen_ox','abdomen_oy','abdomen_oz',
                       'humerus_r_x','humerus_r_y','humerus_r_z','humerus_r_ox','humerus_r_oy','humerus_r_oz',
                       'ulna_r_x','ulna_r_y','ulna_r_z','ulna_r_ox','ulna_r_oy','ulna_r_oz',
                       'radius_r_x','radius_r_y','radius_r_z','radius_r_ox','radius_r_oy','radius_r_oz',
                       'hand_r_x','hand_r_y','hand_r_z','hand_r_ox','hand_r_oy','hand_r_oz',
                       'humerus_l_x','humerus_l_y','humerus_l_z','humerus_l_ox','humerus_l_oy','humerus_l_oz',
                       'ulna_l_x','ulna_l_y','ulna_l_z','ulna_l_ox','ulna_l_oy','ulna_l_oz',
                       'radius_l_x','radius_l_y','radius_l_z','radius_l_ox','radius_l_oy','radius_l_oz',
                       'hand_l_x','hand_l_y','hand_l_z','hand_l_ox','hand_l_oy','hand_l_oz',
                       'center_of_mass_x','center_of_mass_y','center_of_mass_z']
    stand_joints = []


    # List containing the standardized names of the bodypart values
    stand_joint =  [] # List containing the standardized names of the joints

    kinematicState = ""

    if metadata is not None:
        if 'Coordinates' in metadata:
            if verbose >= 2:
                print("Data is Position Data")

            kinematicState = 'pos'

        if 'Speeds' in metadata:
            if verbose >= 2:
                print("Data is Speed Data")

            kinematicState = 'vel'

        if 'Accelerations' in metadata:
            if verbose >= 2:
                print("Data is Acceleration Data")

            kinematicState = 'acc'

        pass

    columns = df.columns
    dat_type = ""
    if 'elbow_flex_l' in df.columns:

        if verbose >=1:
            print("Joint Data already standardized\n"
                  "No further Standardization needed.")

        """Change columns to standardized names"""
        df.columns = stand_joint


    elif any( i in df.columns for i in [' hand_l_x', 'hand_l_x', 'hand_l_X']):
        if verbose >=1:
            print("Standardizing:\tEndeffector Data")
            df.columns = standardize_columns(df.columns, stand_bodypart, verbose)

    else:
        raise ValueError("Error in iDrinkStatisticalAnalysis.standardize_data\n"
                         "Neither 'elbow_flex_l' nor 'hand_l_x' are in Data.\n"
                         "Please check the data and try again.")

    return kinematicState, df




if __name__ == '__main__':
    # this part is for Development and Debugging


    if sys.gettrace() is not None:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

    joint_kin_openism= {'pos': r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\kin_opensim_analyzetool\S003_P07_T043_Kinematics_q.sto",
                'vel': r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\kin_opensim_analyzetool\S003_P07_T043_Kinematics_u.sto",
                'acc': r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\kin_opensim_analyzetool\S003_P07_T043_Kinematics_dudt.sto"}
    body_kin_opensim = {'pos': r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\kin_opensim_analyzetool\S003_P07_T043_BodyKinematics_pos_global.sto",
                    'vel': r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\kin_opensim_analyzetool\S003_P07_T043_BodyKinematics_vel_global.sto",
                    'acc': r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\kin_opensim_analyzetool\S003_P07_T043_BodyKinematics_acc_global.sto"}

    # Body Kin position by pose2sim
    body_kin_p2s = {'pos': r"I:\iDrink\validation_root\03_data\setting_003\P07\S003\S003_P07\S003_P07_T043\movement_analysis\kin_p2s\S003_P07_T043_affected_Body_kin_p2s_pos.csv"
                }

    files = {'joint_kin_openism' : joint_kin_openism,
             'body_kin_opensim' : body_kin_opensim,
             'body_kin_p2s' : body_kin_p2s}

    file = files.get('joint_kin_openism').get('pos')

    continuous = True


    metadata = None
    if os.path.splitext(file)[1] == '.csv':
        df_mmc_pos = pd.read_csv(file)
        if 'pos' in file:
            metadata = 'Coordinates'
    else:
        metadata, df_mmc_pos = read_opensim_file(file)

    kinematicState, df_mmc_pos = standardize_data(df_mmc_pos, metadata)

    if continuous:
        run_statistics_continuous(df_mmc_pos, df_mmc_vel, df_omc_pos, df_omc_vel, isjoint, root_stat)
