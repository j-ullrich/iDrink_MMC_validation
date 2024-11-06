import glob
import os
import sys
import time
import re
import shutil
from operator import index
import platform

from tqdm import tqdm

import pandas as pd

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from iDrink import iDrinkTrial, iDrinkOpenSim, iDrinkUtilities, iDrinkLog

def prepare_opensim(self, filterflag="filt"):
    # Copy Geometry from default to trial folder
    dir_geom = os.path.realpath(os.path.join(self.dir_default, "Geometry"))
    new_dir_geom = os.path.realpath(os.path.join(self.dir_trial, "Geometry"))

    shutil.copytree(dir_geom, new_dir_geom, dirs_exist_ok=True)

    self.opensim_model = os.path.join(self.dir_default, f"iDrink_{self.pose_model}.osim")
    #self.opensim_model_scaled = os.path.join(self.dir_trial, f"Scaled_{self.pose_model}.osim")
    self.opensim_model_scaled = f"{self.identifier}_Scaled_{self.pose_model}.osim"

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
                                                          frame_range=[0, 20], as_string=True)
    self.opensim_IK_time_range = self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=True)
    self.opensim_ana_init_t = str(
        self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=False)[0])
    self.opensim_ana_final_t = str(
        self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=False)[1])


drive = iDrinkUtilities.get_drivepath()
root_iDrink = os.path.join(drive, 'iDrink')  # Root directory of all iDrink Data
root_OMC = os.path.join(root_iDrink, "OMC_data_newStruct", "Data")  # Root directory of all OMC-Data --> trc of trials.
root_val = os.path.join(root_iDrink, "validation_root")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
root_dat_out = os.path.join(root_val, "03_data", "OMC")  # Root directory of all the data for the validation
default_dir = os.path.join(root_val, "01_default_files")
root_logs = os.path.join(root_val, "05_logs")
csv_path = os.path.join(root_logs, "OMC_Opensim_log.csv")

def run_opensim_OMC(stabilize_hip=True):
    DEBUG = False
    verbose = 1
    p_list = os.listdir(root_OMC)

    p_new_struc = [
        "P07",
        "P08",
        "P10",
        "P11",
        "P12",
        "P13",
        "P14",
        "P15",
        "P16",
        "P17",
        "P18",
        "P19",
        "P20",
        "P21",
        "P22",
        "P241",
        "P242",
        "P251",
        "P252",
    ]


    id_s = "S15133"  # O:15 M:13 C:3
    trial_list = []
    if os.path.isfile(csv_path):
        df_log = pd.read_csv(csv_path)
    else:
        df_log = pd.DataFrame(columns=["Date", "Time", "identifier", "status", "exception"])

    if DEBUG:
        p_list = ['P07', 'P08', 'P10', 'P11']  # Temporary

    match platform.uname().node:
        case 'DESKTOP-N3R93K5':
            dict_p_list = {1: ["P13", "P15"],
                           2: ["P19", "P241", "P242"],
                           3: ["P251", "P252"]}
            p_list_full = ["P13", "P15", "P19", "P241", "P242", "P251", "P252"]
        case 'DESKTOP-0GLASVD':
            dict_p_list = {1: ["P07", "P08"],
                            2: ["P10", "P11"],
                            3: ["P12"]}
            p_list_full = ["P07", "P08", "P10", "P11", "P12"]
        case _:  # Default case
            dict_p_list = {1: [],
                           2: [],
                           3: []}
            p_list_full = []

    p_list = dict_p_list[2]

    if verbose >=2:
        print(f"p_list: \n"
                f"{p_list}")

    for p_id in p_list:

        trc_dir = os.path.realpath(os.path.join(root_OMC, p_id, "trc"))
        trc_files = glob.glob(os.path.join(trc_dir, "*.trc"))

        """if DEBUG:
            unaffected_trials = glob.glob(os.path.join(trc_dir, "*unaffected*.trc"))
            affected_trials = [f for f in glob.glob(os.path.join(trc_dir, "*affected*.trc")) if "unaffected" not in f]
            trc_files = unaffected_trials[:3] + affected_trials[:3]"""

        if verbose >= 2:
            print(f"trc_files for {p_id}: \n"
                  f"{trc_files}")
        for trc_file in sorted(trc_files):
            id_t = re.search("\d+", os.path.basename(trc_file)).group()
            id_t = f"T{int(id_t):03d}"
            identifier = f"{id_s}_{p_id}_{id_t}"

            dir_s = os.path.realpath(os.path.join(root_dat_out, id_s))
            dir_p = os.path.realpath(os.path.join(dir_s, f"{id_s}_{p_id}"))
            dir_t = os.path.realpath(os.path.join(dir_p, identifier))

            trial = iDrinkTrial.Trial(id_s=id_s, id_p=p_id, id_t=id_t, identifier=identifier,
                                      dir_session=dir_s, dir_participant=dir_p, dir_trial=dir_t,
                                      dir_default=default_dir, pose_model='OMC')
            trial.stabilize_hip = stabilize_hip

            trial.create_trial(for_omc=True)
            trial.load_configuration()

            #trial_done = iDrinkLog.files_exist(trial.dir_kin_ik_tool, '.csv', verbose=1)

            joint_kin_exist = iDrinkLog.files_exist(os.path.join(trial.dir_trial, 'movement_analysis', 'ik_tool'),
                                                    '.csv')
            chest_pos_exist = iDrinkLog.files_exist(
                os.path.join(trial.dir_trial, 'movement_analysis', 'kin_opensim_analyzetool'), 'OutputsVec3.sto')

            trial_done = joint_kin_exist and chest_pos_exist

            if trial_done:
                print(f"Skipping {identifier} as it is already done.")
                new_row = pd.DataFrame({"Date": time.strftime("%d.%m.%Y"),
                                                  "Time": time.strftime("%H:%M:%S"),
                                                  "identifier": identifier,
                                                  "status": "Already Done",
                                                  "exception": ""}, index=[len(df_log)])
                df_log = pd.concat([df_log, new_row], ignore_index=True)
                iDrinkUtilities.del_geometry_from_trial(trial)
                continue

            # copy trc file to pose-3d folder
            dir_pose3d = os.path.realpath(os.path.join(dir_t, "pose-3d"))
            trc_nameparts = os.path.basename(trc_file).split('_')

            new_filename = f"{trial.identifier}_{trc_nameparts[-2]}_{trc_nameparts[-1]}"
            shutil.copy2(trc_file, os.path.join(dir_pose3d, new_filename))

            trial_list.append(trial)

            try:
                prepare_opensim(trial, filterflag=None)
                iDrinkOpenSim.open_sim_pipeline(trial, os.path.join(root_logs, 'opensim'))

                new_row = pd.DataFrame({"Date": time.strftime("%d.%m.%Y"),
                                        "Time": time.strftime("%H:%M:%S"),
                                        "identifier": identifier,
                                        "status": "success",
                                        "exception": ""}, index=[len(df_log)])
                df_log = pd.concat([df_log, new_row], ignore_index=True)

            except Exception as e:
                print(e)
                time.sleep(3)
                new_row = pd.DataFrame({"Date": time.strftime("%d.%m.%Y"),
                                        "Time": time.strftime("%H:%M:%S"),
                                        "identifier": identifier,
                                        "status": "failed",
                                        "exception": str(e)}, index=[len(df_log)])
                df_log = pd.concat([df_log, new_row], ignore_index=True)

            iDrinkUtilities.del_geometry_from_trial(trial)
            df_log.to_csv(csv_path, index=False)

        df_log.to_csv(csv_path, index=False)

def get_mot_based_vel_acc(root_omc, verbose=1):

    id_s = "S15133"

    dir_s = os.path.realpath(os.path.join(root_omc, id_s))

    p_list = sorted([p.split("_")[1] for p in os.listdir(dir_s)])
    progress_bar = None

    total = 0
    for id_p in p_list:
        p_dir = os.path.join(root_omc, f"{id_s}_{id_p}")
        t_list = sorted([t.split("_")[2] for t in os.listdir(p_dir)])
        total += len(t_list)

    for id_p in p_list:

        p_dir = os.path.join(dir_s, f"{id_s}_{id_p}")

        t_list = sorted([t.split("_")[2] for t in os.listdir(p_dir)])

        if verbose >= 1:
            if progress_bar is None:
                progress_bar = tqdm(total=total, desc="Processing Trials", unit="trials")
        for t_id in t_list:
            if verbose >= 1:
                progress_bar.set_description(f"Processing {id_p}_{t_id}")

            identifier = f"{id_s}_{id_p}_{t_id}"
            t_dir = os.path.join(p_dir, f"{id_s}_{id_p}_{t_id}")

            dir_dst = os.path.join(t_dir, "movement_analysis", "ik_tool")

            mot_file = glob.glob(os.path.join(t_dir, 'pose-3d', "*.mot"))[0]
            path_dst = os.path.realpath(os.path.join(dir_dst, f"{identifier}_Kinematics_pos.csv"))

            iDrinkOpenSim.mot_to_csv(path_mot=mot_file, path_dst=path_dst, verbose=0)

            if verbose >= 1:
                progress_bar.update(1)

    if verbose >= 1:
        progress_bar.close()


if __name__ == "__main__":

    run_opensim_OMC()

    #get_mot_based_vel_acc(root_omc=root_dat_out, verbose=1)
