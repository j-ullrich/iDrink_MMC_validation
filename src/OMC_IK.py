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

from iDrink import iDrinkTrial, iDrinkOpenSim, iDrinkUtilities, iDrinkLog


def prepare_opensim(self, filterflag="filt"):
    self.opensim_model = os.path.join(self.dir_default, f"iDrink_{self.pose_model}.osim")
    #self.opensim_model_scaled = os.path.join(self.dir_trial, f"Scaled_{self.pose_model}.osim")
    self.opensim_model_scaled = f"Scaled_{self.pose_model}.osim"

    self.opensim_scaling = os.path.join(self.dir_trial, f"Scaling_Setup_iDrink_{self.pose_model}.xml")
    self.opensim_inverse_kinematics = os.path.join(self.dir_trial, f"IK_Setup_iDrink_{self.pose_model}.xml")
    self.opensim_analyze = os.path.join(self.dir_trial, f"AT_Setup.xml")

    self.opensim_marker = self.get_opensim_path(self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc"))
    self.opensim_marker_filtered = self.get_opensim_path(
        self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc", flag=filterflag))
    self.opensim_motion = os.path.splitext(
        self.get_opensim_path(self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc", flag=filterflag)))[
                              0] + ".mot"

    self.opensim_scaling_time_range = self.get_time_range(path_trc_file=self.opensim_marker_filtered,
                                                          frame_range=[0, 10], as_string=True)
    self.opensim_IK_time_range = self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=True)
    self.opensim_ana_init_t = str(
        self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=False)[0])
    self.opensim_ana_final_t = str(
        self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=False)[1])


"""Set Root Paths for Processing"""
drives=['C:', 'D:', 'E:', 'I:']
if os.name=='posix':  # Running on Linux
    drive = '/media/devteam-dart/Extreme SSD'
else:
    drive = drives[3]

root_iDrink = os.path.join(drive, 'iDrink')  # Root directory of all iDrink Data
root_OMC = os.path.join(root_iDrink, "OMC_data_newStruct", "Data")  # Root directory of all OMC-Data --> trc of trials.
root_val = os.path.join(root_iDrink, "validation_root")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
root_dat_out = os.path.join(root_val, "03_data", "OMC")  # Root directory of all the data for the validation
default_dir = os.path.join(root_val, "01_default_files")
root_logs = os.path.join(root_val, "05_logs")
csv_path = os.path.join(root_logs, "OMC_Opensim_log.csv")

DEBUG = False
verbose = 1

p_list = os.listdir(root_OMC)

id_s = "S15133"  # O:15 M:13 C:3
trial_list = []

df_log = pd.DataFrame(columns=["Date", "Time", "identifier", "status", "exception"])

if DEBUG:
    p_list = ['P07', 'P08', 'P10', 'P11']  # Temporary

if verbose >=1:
    print(f"p_list: \n"
            f"{p_list}")

for p_id in p_list:

    trc_dir = os.path.realpath(os.path.join(root_OMC, p_id, "trc"))
    trc_files = glob.glob(os.path.join(trc_dir, "*.trc"))

    if DEBUG:
        unaffected_trials = glob.glob(os.path.join(trc_dir, "*unaffected*.trc"))
        affected_trials = [f for f in glob.glob(os.path.join(trc_dir, "*affected*.trc")) if "unaffected" not in f]
        trc_files = unaffected_trials[:3] + affected_trials[:3]

    if verbose >= 1:
        print(f"trc_files for {p_id}: \n"
              f"{trc_files}")
    for trc_file in trc_files:
        id_t = re.search("\d+", os.path.basename(trc_file)).group()
        id_t = f"T{int(id_t):03d}"
        identifier = f"{id_s}_{p_id}_{id_t}"

        dir_s = os.path.realpath(os.path.join(root_dat_out, id_s))
        dir_p = os.path.realpath(os.path.join(dir_s, f"{id_s}_{p_id}"))
        dir_t = os.path.realpath(os.path.join(dir_p, identifier))



        trial = iDrinkTrial.Trial(id_s=id_s, id_p=p_id, id_t=id_t, identifier=identifier,
                                  dir_session=dir_s, dir_participant=dir_p, dir_trial=dir_t,
                                  dir_default=default_dir, pose_model='OMC')

        trial.create_trial(for_omc=True)
        trial.load_configuration()

        trial_done = iDrinkLog.files_exist(os.path.join(dir_t, 'pose-3d'), '.mot', verbose=1)
        if trial_done:
            print(f"Skipping {identifier} as it is already done.")
            df_log = df_log.append({"Date": time.strftime("%d.%m.%Y"), "Time": time.strftime("%H:%M:%S"), "identifier": identifier, "status": "Already Done", "exception": ""}, ignore_index=True)
            continue

        # copy trc file to pose-3d folder
        dir_pose3d = os.path.realpath(os.path.join(dir_t, "pose-3d"))
        trc_nameparts = os.path.basename(trc_file).split('_')

        new_filename = f"{trial.identifier}_{trc_nameparts[-2]}_{trc_nameparts[-1]}"
        shutil.copy2(trc_file, os.path.join(dir_pose3d, new_filename))

        prepare_opensim(trial, filterflag=None)

        trial_list.append(trial)

        try:
            iDrinkOpenSim.open_sim_pipeline(trial, os.path.join(root_logs, 'opensim'))
            df_log = df_log.append({"Date": time.strftime("%d.%m.%Y"), "Time": time.strftime("%H:%M:%S"), "identifier": identifier, "status": "success", "exception": ""}, ignore_index=True)
        except Exception as e:
            print(e)
            df_log = df_log.append({"Date": time.strftime("%d.%m.%Y"), "Time": time.strftime("%H:%M:%S"), "identifier": identifier, "status": "failed", "exception": str(e)}, ignore_index=True)
            pass

        df_log.to_csv(csv_path, index=False)

    df_log.to_csv(csv_path, index=False)

