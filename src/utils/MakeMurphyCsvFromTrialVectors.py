import os
import sys


import pandas as pd
import numpy as np

from tqdm import tqdm

def run():
    # Define the root directory of the iDrink Data
    drives=['C:', 'D:', 'E:', 'F:', 'I:']
    if os.name=='posix':  # Running on Linux
        drive = '/media/devteam-dart/Extreme SSD'
    else:
        drive = drives[3] + '\\'

    root_iDrink = os.path.join(drive, 'iDrink')  # Root directory of all iDrink Data
    root_MMC = os.path.join(root_iDrink, "Delta", "data_newStruc")  # Root directory of all MMC-Data --> Videos and Openpose json files
    root_OMC = os.path.join(root_iDrink, "OMC_data_newStruct")  # Root directory of all OMC-Data --> trc of trials.
    root_val = os.path.join(root_iDrink, "validation_root")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
    default_dir = os.path.join(root_val, "01_default_files")  # Default Files for the iDrink Validation
    root_HPE = os.path.join(root_val, "02_pose_estimation")  # Root directory of all Pose Estimation Data
    root_data = os.path.join(root_val, "03_data")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
    root_stat = os.path.join(root_val, '04_Statistics')
    root_logs = os.path.join(root_val, "05_logs")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.

    root_vector = os.path.join(root_iDrink, 'DIZH_data')

    path_csv_murphy_timestamps = os.path.join(root_stat, '02_categorical', 'murphy_timestamps.csv')
    path_csv_murphy_measures = os.path.join(root_stat, '02_categorical', 'murphy_measures.csv')

    if not os.path.exists(os.path.join(root_stat, '02_categorical')):
        os.makedirs(os.path.join(root_stat, '02_categorical'), exist_ok=True)

    p_list = ['P01', 'P04', 'P05', 'P07', 'P08', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P17', 'P19', 'P24', 'P25',
              'P27', 'P28', 'P30', 'P31', 'P34']

    df_timestamps = pd.DataFrame(columns=['id_p', 'id_t', 'valid', 'side', 'condition', 'ReachingStart',
           'ForwardStart', 'DrinkingStart', 'BackStart', 'ReturningStart',
           'RestStart', 'TotalMovementTime'])
    prog = tqdm(total=len(p_list), desc=f'Processing participants', unit='p_id')

    for id_p in p_list:

        csv_in = os.path.join(root_vector, f'{id_p}_trialVectors.csv')

        df = pd.read_csv(csv_in)


        for t in range(len(df)):

            id_t = f"T{df['trial'][t]:03d}"
            new_row = pd.DataFrame([{
                'id_p': id_p,
                'id_t': id_t,
                'valid': df['valid'][t],
                'side': df['side'][t],
                'condition': df['condition'][t],
                'ReachingStart': df['ReachingStart'][t],
                'ForwardStart': df['ForwardStart'][t],
                'DrinkingStart': df['DrinkingStart'][t],
                'BackStart': df['BackStart'][t],
                'ReturningStart': df['ReturningStart'][t],
                'RestStart': df['RestStart'][t],
                'TotalMovementTime': df['TotalMovementTime'][t]
            }])

            df_timestamps = pd.concat([df_timestamps, new_row], ignore_index=True)
        prog.update(1)

    prog.close()


    df_timestamps.to_csv(path_csv_murphy_timestamps, index=False, sep=';')

if __name__ == '__main__':
    run()