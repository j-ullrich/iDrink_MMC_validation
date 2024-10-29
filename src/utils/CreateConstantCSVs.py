import pandas as pd
import os
import numpy as np


"""Set Root Paths for Processing"""
drives=['C:', 'D:', 'E:', 'I:']
if os.name=='posix':  # Running on Linux
    drive = '/media/devteam-dart/Extreme SSD'
else:
    drive = drives[1] + '\\'

root_iDrink = os.path.join(drive, 'iDrink')  # Root directory of all iDrink Data
root_MMC = os.path.join(root_iDrink, "Delta", "data_newStruc")  # Root directory of all MMC-Data --> Videos and Openpose json files
root_OMC = os.path.join(root_iDrink, "OMC_data_newStruct")  # Root directory of all OMC-Data --> trc of trials.
root_val = os.path.join(root_iDrink, "validation_root")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
default_dir = os.path.join(root_val, "01_default_files")  # Default Files for the iDrink Validation
root_HPE = os.path.join(root_val, "02_pose_estimation")  # Root directory of all Pose Estimation Data
root_data = os.path.join(root_val, "03_data")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
root_stat = os.path.join(root_val, '04_Statistics')
root_logs = os.path.join(root_val, "05_logs")  # Root directory of all iDrink Data for the validation --> Contains all the files necessary for Pose2Sim and Opensim and their Output.
metrabs_models_dir = os.path.join(root_val, "06_metrabs_models")  # Directory containing the Metrabs Models

def createCAD_murphy():
    """
    Create .csv file containing the  Clinically Acceptable Difference (CAD) for all Murphy Measures.

    Calculated as in Marwens Master Thesis.

    "For example, in case of the drinking task, Kwakkel et al [25] explain that the CAD values corresponding to 6 graduations in the ARAT scores after 3 months of stroke represent 15% of the kinematic measures studied."

    Based on the following papers:

    - Murphy et al. 2011: "Kinematic Variables Quantifying Upper-Extremity Performance After Stroke During Reaching and Drinking From a Glass"
    - Kwackel et al. 2019: "Standardized measurement of quality of upper limb movement after stroke:  Consensus-based core recommendations from the Second Stroke Recovery and Rehabilitation Roundtable

    :return:
    """
    """dat = np.array([6.49, 0.83, 11.4, 3.1])
    df_means_per_group = pd.DataFrame(columns=['healthy_mean', 'healthy_std',
                                               'stroke_whole_mean', 'stroke_whole_std',
                                               'stroke_mild_mean', 'stroke_mild_std',
                                               'stroke_moderate_mean', 'stroke_moderate_std'],
                                      index = ['mov_time', 'peak_V', 'peak_V_elb', 't_to_PV', 't_first_PV', 't_PV_rel',
                                               't_first_PV_rel', 'n_mov_units', 'interj_coord', 'trunk_disp',
                                               'arm_flex_reach', 'elb_ext', 'arm_abd', 'arm_flex_drink'])"""


    data = {
        'healthy_mean': [6.49, 616, 121.8, 0.47, 0.43, 46.0, 42.5, 2.3, 0.96, 26.7, 45.6, 53.7, 30.1, 51.7],
        'healthy_std': [0.83, 93.8, 25.3, 0.08, 0.07, 6.9, 6.9, 0.3, 0.02, 16.8, 5.1, 7.8, 10.1, 5.3],
        'stroke_whole_mean': [11.4, 431, 64.9, 0.70, 0.46, 38.4, 27.1, 8.4, 0.82, 77.2, 44.5, 64.1, 47.6, 54.3],
        'stroke_whole_std': [3.1, 82.7, 20.5, 0.19, 0.12, 8.6, 12.2, 4.2, 0.35, 48.6, 7.2, 11.5, 14.9, 10.9],
        'stroke_mild_mean': [9.30, 471, 78.0, 0.61, 0.42, 39.5, 33.0, 5.4, 0.95, 50.1, 41.7, 60.5, 37.2, 46.6],
        'stroke_mild_std': [1.68, 87.7, 19.3, 0.16, 0.14, 8.7, 9.9, 2.1, 0.02, 22.9, 6.2, 10.4, 5.3, 4.9],
        'stroke_moderate_mean': [13.3, 395, 53.3, 0.79, 0.50, 37.5, 21.8, 11.1, 0.69, 101.7, 47.1, 67.2, 57.1, 61.3],
        'stroke_moderate_std': [2.9, 62.0, 13.6, 0.17, 0.17, 8.1, 11.9, 3.6, 0.46, 53.4, 7.4, 11.9, 14.5, 10.1]
    }

    index = ['mov_time', 'peak_V', 'peak_V_elb', 't_to_PV', 't_first_PV', 't_PV_rel',
             't_first_PV_rel', 'n_mov_units', 'interj_coord', 'trunk_disp',
             'arm_flex_reach', 'elb_ext', 'arm_abd', 'arm_flex_drink']

    df = pd.DataFrame(data, index=index)

    df_cad = pd.DataFrame(columns = index, index = ['CAD']).rename_axis('Measure', axis=1)

    for i in index:

        min_mean = 100000
        min_std = 100000
        max_mean = -100000
        max_std = -100000

        for col_mean, col_std in zip(['healthy_mean', 'stroke_whole_mean', 'stroke_mild_mean', 'stroke_moderate_mean'],
                                        ['healthy_std', 'stroke_whole_std', 'stroke_mild_std', 'stroke_moderate_std']):

            min_mean = min(min_mean, df.loc[i, col_mean])
            min_std = min(min_std, df.loc[i, col_std])
            max_mean = max(max_mean, df.loc[i, col_mean])
            max_std = max(max_std, df.loc[i, col_std])

        low_bound = min_mean - 1.96 * min_std
        high_bound = max_mean + 1.96 * max_std

        cad = 0.15 * (high_bound - low_bound)

        df_cad.loc['CAD', i] = round(cad, 2)

    os.path.exists(os.path.join(root_stat, '02_categorical')) or os.makedirs(os.path.join(root_stat, '02_categorical'))

    df_cad.to_csv(os.path.join(root_stat, '02_categorical', ' clinically_acceptable_difference.csv'))

    pass





if __name__ == "__main__":

    createCAD_murphy()