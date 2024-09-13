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

from iDrink import iDrinkTrial, iDrinkOpenSim, iDrinkUtilities


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

root_OMC = r"C:\iDrink\OMC_data_newStruct\Data"
roo_dat_out = r"C:\iDrink\validation_root\03_data\OMC"
root_val = r"C:\iDrink\validation_root"
default_dir = os.path.join(root_val, "01_default_files")
csv_path = os.path.join(root_val, "OMC_Opensim_log.csv")

p_list = os.listdir(root_OMC)

id_s = "S15133"  # O:15 M:13 C:3
trial_list = []

df_log = pd.DataFrame(columns=["identifier", "status", "exception"])

p_list = ['P07', 'P08', 'P10', 'P11']  # Temporary

for p_id in p_list:

    trc_dir = os.path.realpath(os.path.join(root_OMC, p_id, "trc"))
    trc_files = glob.glob(os.path.join(trc_dir, "*.trc"))

    # TODO: Take following out
    unaffected_trials = glob.glob(os.path.join(trc_dir, "*unaffected*.trc"))
    affected_trials = [f for f in glob.glob(os.path.join(trc_dir, "*affected*.trc")) if "unaffected" not in f]
    trc_files = unaffected_trials[:3] + affected_trials[:3]

    for trc_file in trc_files:
        id_t = re.search("\d+", os.path.basename(trc_file)).group()
        id_t = f"T{int(id_t):03d}"
        identifier = f"{id_s}_{p_id}_{id_t}"

        dir_s = os.path.realpath(os.path.join(roo_dat_out, id_s))
        dir_p = os.path.realpath(os.path.join(dir_s, f"{id_s}_{p_id}"))
        dir_t = os.path.realpath(os.path.join(dir_p, identifier))



        trial = iDrinkTrial.Trial(id_s=id_s, id_p=p_id, id_t=id_t, identifier=identifier,
                                  dir_session=dir_s, dir_participant=dir_p, dir_trial=dir_t,
                                  dir_default=default_dir, pose_model='OMC')

        trial.create_trial(for_omc=True)
        trial.load_configuration()

        # copy trc file to pose-3d folder
        dir_pose3d = os.path.realpath(os.path.join(dir_t, "pose-3d"))
        trc_nameparts = os.path.basename(trc_file).split('_')

        new_filename = f"{trial.identifier}_{trc_nameparts[-2]}_{trc_nameparts[-1]}"
        shutil.copy2(trc_file, os.path.join(dir_pose3d, new_filename))

        prepare_opensim(trial, filterflag=None)

        trial_list.append(trial)

        try:
            iDrinkOpenSim.open_sim_pipeline(trial)
            df_log = df_log.append({"identifier": identifier, "status": "success", "exception": ""}, ignore_index=True)
        except Exception as e:
            print(e)
            df_log = df_log.append({"identifier": identifier, "status": "failed", "exception": str(e)}, ignore_index=True)
            pass

        df_log.to_csv(csv_path, index=False)

