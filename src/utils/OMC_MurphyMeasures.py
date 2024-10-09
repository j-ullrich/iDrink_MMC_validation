import glob
import os
import sys
import time
import re
import shutil
import subprocess

from tqdm import tqdm

import argparse
import pandas as pd

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from iDrink import iDrinkTrial, iDrinkOpenSim, iDrinkUtilities, iDrinkLog, iDrinkMurphyMeasures

if sys.gettrace() is not None:
    print("Debug Mode is activated\n"
          "Starting debugging script.")

"""Set Root Paths for Processing"""
drives = ['C:', 'D:', 'E:', 'F:', 'G:', 'I:']
if os.name == 'posix':  # Running on Linux
    drive = '/media/devteam-dart/Extreme SSD'
else:
    drive = drives[5]

root_iDrink = os.path.join(drive, 'iDrink')
root_val = os.path.join(root_iDrink, "validation_root")
root_stat = os.path.join(root_val, '04_Statistics')
root_data = os.path.join(root_val, "03_data")
root_omc = os.path.join(root_val, '03_data', 'OMC', 'S15133')

csv_timestamps = os.path.join(root_stat, '02_categorical', 'murphy_timestamps.csv')
csv_measures =os.path.join(root_stat, '02_categorical', 'murphy_measures.csv')

def run(verbose=1):


    p_list = sorted([p.split("_")[1] for p in os.listdir(root_omc)])

    id_s = "S15133"

    total = 0
    for id_p in p_list:
        p_dir = os.path.join(root_omc, f"{id_s}_{id_p}")
        t_list = sorted([t.split("_")[2] for t in os.listdir(p_dir)])
        total += len(t_list)

    progress = None

    for id_p in p_list:

        p_dir = os.path.join(root_omc, f"{id_s}_{id_p}")

        t_list = sorted([t.split("_")[2] for t in os.listdir(p_dir)])

        if verbose >= 1:
            if progress is None:
                progress = tqdm(total=total, desc=f"Processing")
        for id_t in t_list:

            if verbose >= 1:
                progress.set_description(f"Processing {id_p}_{id_t}")

            identifier = f"{id_s}_{id_p}_{id_t}"

            t_dir = os.path.join(p_dir, f"{id_s}_{id_p}_{id_t}")

            dir_bodykin = os.path.join(t_dir, 'movement_analysis', 'kin_opensim_analyzetool')
            dir_jointkin = os.path.join(t_dir, 'movement_analysis', 'ik_tool')

            path_bodyparts_pos = os.path.join(dir_bodykin, f'{identifier}_BodyKinematics_pos_global.sto')
            path_bodyparts_vel = os.path.join(dir_bodykin, f'{identifier}_BodyKinematics_vel_global.sto')
            path_bodyparts_acc = os.path.join(dir_bodykin, f'{identifier}_BodyKinematics_acc_global.sto')

            path_trunk_pos = os.path.join(dir_bodykin, f'{identifier}_OutputsVec3.sto')

            path_joint_pos = os.path.join(dir_jointkin, f'{identifier}_Kinematics_pos.csv')
            path_joint_vel = os.path.join(dir_jointkin, f'{identifier}_Kinematics_vel.csv')
            path_joint_acc = os.path.join(dir_jointkin, f'{identifier}_Kinematics_acc.csv')

            try:
                iDrinkMurphyMeasures.MurphyMeasures(trial_id=identifier, csv_timestamps=csv_timestamps, csv_measures=csv_measures, verbose=0,
                                                    path_bodyparts_pos=path_bodyparts_pos,
                                                    path_bodyparts_vel=path_bodyparts_vel,
                                                    path_bodyparts_acc=path_bodyparts_acc,
                                                    path_trunk_pos=path_trunk_pos,
                                                    path_joint_pos=path_joint_pos,
                                                    path_joint_vel=path_joint_vel,
                                                    path_joint_acc=path_joint_acc
                                                    )
            except Exception as e:
                print(f"Error: {e}")

            if verbose >= 1:
                progress.update(1)

    if verbose >= 1:
        progress.close()




if __name__ == "__main__":
    run()
    pass




