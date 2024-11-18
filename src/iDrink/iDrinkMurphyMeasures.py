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
import sys

import pandas as pd
import numpy as np
import scipy
from fuzzywuzzy import process

from importlib_metadata import metadata
from keras.src.utils.file_utils import exists
from sympy.logic.algorithms.dpll import find_pure_symbol

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from iDrink import iDrinkTrial


class MurphyMeasures:
    """
    This class is used to calculate the Murphy measures and write them in a .csv file.

    There are two ways how we can get the murphy measures:

    1. We give the Murphy-object paths to files and folders.
        - bodypart positions, velocities and accelerations
        - joint positions, velocities and accelerations
        - trial_dir

    2. we give the object the trial_id and the root path containing all the session data

    3. We give the object a trial_object

    """
    def __init__(self, trial_id = None, csv_timestamps=None, csv_measures=None,
                 filt_fps=None, filt_cutoff_vel=None, filt_cutoff_pos=None, filt_order_pos=None, filt_order_vel=None,
                 verbose=0, write_mov_data=False,
                 path_mov_data=None,
                 # For mode 1
                 trial = None, root_data = None,
                 # For mode 3
                 path_bodyparts_pos = None, path_bodyparts_vel = None, path_bodyparts_acc=None, path_trunk_pos=None,
                 path_joint_pos = None, path_joint_vel = None, path_joint_acc = None
                 ):

        self.trial = trial
        self.root_data = root_data
        self.dir_trial = None

        self.verbose = verbose

        # Filtering when reading files
        self.filt_fps = filt_fps
        self.filt_cutoff_vel = filt_cutoff_vel
        self.filt_cutoff_pos = filt_cutoff_pos
        self.filt_order_pos = filt_order_pos
        self.filt_order_vel = filt_order_vel


        """Settings"""
        self.write_mov_data = write_mov_data
        self.path_mov_data = path_mov_data

        self.path_bodyparts_pos = path_bodyparts_pos
        self.path_bodyparts_vel = path_bodyparts_vel
        self.path_bodyparts_acc = path_bodyparts_acc
        self.path_trunk_pos = path_trunk_pos

        self.path_joint_pos = path_joint_pos
        self.path_joint_vel = path_joint_vel
        self.path_joint_acc = path_joint_acc

        self.csv_timestamps = csv_timestamps
        self.csv_measures = csv_measures
        self.df_timestamps = None
        self.df_measures = None

        """data to calculate measures"""
        self.time = None
        self.phases = None

        self.hand_vel = None
        self.elbow_vel = None

        self.elbow_flex_pos = None
        self.shoulder_flex_pos = None
        self.shoulder_abduction_pos = None

        self.trunk_displacement = None
        self.trunk_ang = None

        """Contents of the .csv file"""
        self.trial_id = trial_id
        self.valid = None
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
        self.ShoulderFlexionReaching = None
        self.ElbowExtension = None
        self.shoulderAbduction = None

        """Other useful variables"""
        self.id_s = self.trial_id.split('_')[0]
        self.id_p = self.trial_id.split('_')[1]
        self.id_t = self.trial_id.split('_')[2]
        self.identifier = self.trial_id


        if self.csv_measures is not None and os.path.isfile(self.csv_measures):
            self.df_measures = pd.read_csv(self.csv_measures, sep=';')
        else:
            self.df_measures = pd.DataFrame(columns = ['identifier', 'id_p', 'id_t', 'valid', 'side', 'condition',
                                              'ReachingStart','ForwardStart', 'DrinkingStart', 'BackStart',
                                              'ReturningStart','RestStart', 'TotalMovementTime',
                                              'PeakVelocity_mms',  'elbowVelocity', 'tTopeakV_s',
                                              'tToFirstpeakV_s', 'tTopeakV_rel', 'tToFirstpeakV_rel',
                                              'NumberMovementUnits', 'InterjointCoordination', 'trunkDisplacementMM',
                                              'trunkDisplacementDEG','ShoulderFlexionReaching', 'ElbowExtension',
                                              'shoulderAbduction']
                                   )

        if self.csv_timestamps is not None and os.path.isfile(self.csv_timestamps):
            self.df_timestamps = pd.read_csv(self.csv_timestamps, sep=';')
            self.get_data(self.df_timestamps)

        # if paths  are given, directly calculate the measures
        if trial is not None:
            self.id_t = trial.id_t
            self.id_p = trial.id_p
            self.id_s = trial.id_s
            self.identifier  = trial.identifier

        if (    self.path_bodyparts_pos is None
            and self.path_bodyparts_vel is None
            and self.path_bodyparts_acc is None
            and self.path_joint_pos is None
            and self.path_joint_vel is None
            and self.path_joint_acc is None):

            self.get_paths()



        if self.valid:
            self.read_files()
            self.get_measures()
            self.write_measures()
        elif self.verbose >= 2:
            print(f"Skipping {self.identifier} due to invalid data.")

    def use_butterworth_filter(self, data, cutoff, fs, order=4, normcutoff=False):
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
            if self.verbose >=2:
                print(f"Warning: Cutoff frequency {cutoff} is higher than Nyquist frequency {nyquist}.")
                print("Filtering with Nyquist frequency.")
            cutoff = nyquist - 1

        if normcutoff:
            cutoff = cutoff / nyquist

        sos = butter(order, cutoff, btype="low", analog=False, output="sos", fs=fs)

        filtered_data = sosfiltfilt(sos, data)

        return filtered_data


    def get_paths_from_root(self):

        s_num = re.search(r'\d+', self.id_s).group()

        if 'OMC' in self.root_data:
            root_body_kin = os.path.join(self.root_data, f'{self.id_s}', f'{self.id_s}_{self.id_p}',
                                            f'{self.trial_id}', 'movement_analysis', 'kin_opensim_analyzetool')
            root_joint_kin = os.path.join(self.root_data, f'{self.id_s}', f'{self.id_s}_{self.id_p}',
                                            f'{self.trial_id}', 'movement_analysis', 'ik_tool')
        else:
            root_body_kin = os.path.join(self.root_data, f"setting_{s_num}",
                                            f'{self.id_s}', f'{self.id_s}_{self.id_p}', f'{self.trial_id}', 'movement_analysis',
                                            'kin_opensim_analyzetool')
            root_joint_kin = os.path.join(self.root_data, f"setting_{s_num}",
                                            f'{self.id_s}', f'{self.id_s}_{self.id_p}', f'{self.trial_id}', 'movement_analysis',
                                            'ik_tool')

        self.path_bodyparts_pos = os.path.join(root_body_kin, f'{self.trial_id}_BodyKinematics_pos_global.sto')
        self.path_bodyparts_vel = os.path.join(root_body_kin, f'{self.trial_id}_BodyKinematics_vel_global.sto')
        self.path_bodyparts_acc = os.path.join(root_body_kin, f'{self.trial_id}_BodyKinematics_acc_global.sto')
        self.path_trunk_pos = os.path.join(root_body_kin, f'{self.trial_id}_OutputsVec3.sto')

        self.path_joint_pos = os.path.join(root_joint_kin, f'{self.trial_id}_Kinematics_pos.csv')
        self.path_joint_vel = os.path.join(root_joint_kin, f'{self.trial_id}_Kinematics_vel.csv')
        self.path_joint_acc = os.path.join(root_joint_kin, f'{self.trial_id}_Kinematics_acc.csv')


    def get_paths_from_trial(self):
        """
        get all paths from trial object.
        :return:
        """

        self.dir_trial = self.trial.dir_trial

        dir_movement_analysis = os.path.join(self.dir_trial, 'movement_analysis')

        root_body_kin = os.path.join(dir_movement_analysis, 'kin_opensim_analyzetool')
        root_joint_kin = os.path.join(dir_movement_analysis, 'ik_tool')

        self.path_bodyparts_pos = os.path.join(root_body_kin, f'{self.trial.identifier}_BodyKinematics_pos_global.sto')
        self.path_bodyparts_vel = os.path.join(root_body_kin, f'{self.trial.identifier}_BodyKinematics_vel_global.sto')
        self.path_bodyparts_acc = os.path.join(root_body_kin, f'{self.trial.identifier}_BodyKinematics_acc_global.sto')
        self.path_trunk_pos = os.path.join(root_body_kin, f'{self.trial.identifier}_OutputsVec3.sto')

        self.path_joint_pos = os.path.join(root_joint_kin, f'{self.trial.identifier}_Kinematics_pos.csv')
        self.path_joint_vel = os.path.join(root_joint_kin, f'{self.trial.identifier}_Kinematics_vel.csv')
        self.path_joint_acc = os.path.join(root_joint_kin, f'{self.trial.identifier}_Kinematics_acc.csv')


    def get_paths(self, ):
        """
        Get paths for bodyparts and joints based on trial object.

        To all this function, the trial_id must be known.
        """
        if self.trial_id is None:
            print("No trial_id given.\n"
                  "Murphy object needs a trial_id to retrieve data from DataFrame.")
            return

        if self.trial is not None:
            self.get_paths_from_trial()
            return

        if self.root_data is None:
            raise ValueError("No root_data given.")

        elif all([self.trial_id, self.root_data]):
            self.get_paths_from_root()

    def get_phase_ids(self, phase_1, phase_2=None):
        """
        Get the ids of the start and end of a phase.

        if phase 2 is given, the ids are calculated from the start of phase 1 to the end of phase 2.

        :param phase_1:
        :param phase_2:
        :return:
        """

        t_start = self.phases[phase_1][0]
        t_end = self.phases[phase_1 if phase_2 is None else phase_2][1]

        id_start = np.where(self.time == self.time.flat[np.abs(self.time - t_start).argmin()])[0][0]
        id_end = np.where(self.time == self.time.flat[np.abs(self.time - t_end).argmin()])[0][0]

        return id_start, id_end

    def get_movement_units(self, vel_data):
        """
        Calculate the amount of Movement Units.

        Input:
            - vel_data: Array for unit-calculation

        Output:
            - mov_units_sum
            - mov_units_mean
            - df_mov_units

        Movement Units are calculated for::
            - reaching
            - forward transportation
            - back transportation
            - returning

        This measure is explained by Murphy as follows:
            One movement unit is defined as a difference between a local minimum and next maximum velocity value that
            exceeds the amplitude limit of 0.020 m/s, and the time between two subsequent peaks has to be at least 0.150 s.
            The minimum value for drinking task is 4, at least one unit per movement phase. Those peaks reflect
            repetitive acceleration and deceleration during reaching and correspond to movement smoothness and efficiency.


        Here it is calculated as follows:
        We move along the phase and look at local maxima and minima.
        If they exceed the thresholds awr by Murphy we count them as a movement unit.
        """

        df_mov_units = pd.DataFrame(
            columns=["phase", "n_movement_units", "values", "loc_maxima_ids", "loc_minima_ids"], )
        time = 0.15  # s, min temporal distance between two peaks
        min_v = 0.02  # m/s, min height difference between min and next peak

        fps = round(len(self.time)/self.time[-1], 2)
        length = int(np.ceil(time * fps))

        for curr_phase in list(self.phases.keys()):
            id_start, id_end = self.get_phase_ids(curr_phase)

            min_v_rel = min_v + min(vel_data)

            peaks_max, _ = scipy.signal.find_peaks(vel_data[id_start:id_end], distance=length)
            peaks_min, _ = scipy.signal.find_peaks(-vel_data[id_start:id_end], distance=length)
            n_units = 0
            values = []
            max_indices = []
            min_indices = []
            for max_peak in peaks_max:
                min_peaks = peaks_min[peaks_min < max_peak]
                if len(min_peaks) >= 1:
                    min_peak = min_peaks[-1]
                    diff_peaks = np.sqrt((vel_data[max_peak] - vel_data[min_peak]) ** 2)
                    if diff_peaks > min_v_rel:
                        values.append(vel_data[max_peak])
                        max_indices.append(vel_data[max_peak])
                        min_indices.append(vel_data[max_peak])
                        n_units += 1
            df_mov_units.loc[len(df_mov_units)] = [curr_phase, n_units, values, max_indices, min_indices]

        mov_units_sum = df_mov_units["n_movement_units"].sum()
        mov_units_mean = df_mov_units["n_movement_units"].mean()


        return mov_units_sum, mov_units_mean, df_mov_units


    @ staticmethod
    def get_interjoint_coordination(pos_joint1, pos_joint2):
        """
        Calculates the interjoint coordination of two joints using a temporal cross-correlation of zero time lag between two joints.
        A correlation coefficient of 1 indicates a high correlation and that joint movements are tightly coupled.

        For Murphy Measures, elbow and shoulder are used.

        Input:
            - pos_joint1: Positions of Joint 1, arraylike
            - pos_joint2: Positions of joint 2, arraylike
        Output:
            - corr_inter: Temporal Cross Correlation of joint 1 and 2

        code from https://github.com/cf-project-delta/Delta_3D_reconstruction/blob/main/processing/murphy_measures.py
        """

        corr_inter = np.corrcoef(pos_joint1, pos_joint2)[1, 0]

        return round(corr_inter, 4)

    def get_trunk_displacement(self, ):
        """
        Calculate the trunk displacement in mm.
        """

        id_start, _ = self.get_phase_ids("reaching")
        displacement = self.trunk_displacement[id_start:]
        max_displacement_mm = np.max(displacement)*1000

        return round(max_displacement_mm, 4)


    def get_trunk_rotation(self, ):
        """
        Calculate the trunk rotation in degrees.
        """

        id_start, _ = self.get_phase_ids("reaching")

        rotation = self.trunk_ang[id_start] - self.trunk_ang

        max_rotation_deg = np.max(rotation)

        return round(max_rotation_deg, 4)


    def get_max_shoulder_flexion_reaching(self, ):
        """
        Calculate the shoulder flexion during reaching phase.
        """
        id_start, id_end = self.get_phase_ids("reaching")

        return round(np.max(self.shoulder_flex_pos[id_start:id_end]), 4)

    def get_min_elbow_flexion(self, ):
        """
        Gets the minimum value of elbow flexion during reaching phase.

        ---> min of flexion is max of extension
        """

        id_start, id_end = self.get_phase_ids("reaching")

        return round(np.min(self.elbow_flex_pos[id_start:id_end]), 4)


    def get_max_shoulder_abd_drink_reach(self):
        """
        Calculate the shoulder abduction during drinking phase.
        """
        id_start, id_end = self.get_phase_ids("reaching")
        max_sh_abd_reach = round(np.max(self.shoulder_abduction_pos[id_start:id_end]), 4)

        id_start, id_end = self.get_phase_ids("drinking")
        max_sh_abd_drink = round(np.max(self.shoulder_abduction_pos[id_start:id_end]), 4)

        return max(max_sh_abd_reach, max_sh_abd_drink)

    def get_peaks_of_movement(self, data):

        id_start, id_end = self.get_phase_ids("reaching", "returning")

        peaks = []
        prom = max(data*0.1)
        i=0
        while not peaks:
            peaks, peaks_info = scipy.signal.find_peaks(data[id_start:id_end], prominence=prom)
            peaks = [peak + id_start for peak in peaks]
            prom = prom * 0.5

            i += 1

            if self.verbose >= 2:
                print(f"get_peaks_of_movement:\n"
                      f"Prominence: {prom}\n"
                      f"Iterations: {i}")

            if i > 10:
                raise TimeoutError("Could not find peaks in data.")

        return peaks, peaks_info


    def get_measures(self, ):
        """
        This function calculates the Murphy measures from the given data.
        """


        # TODO: Decide on prominence value
        """Peak needs to be at least 10% of max velocity"""
        """peak_ids_hand, _ = scipy.signal.find_peaks(self.hand_vel, prominence=max(self.hand_vel)*0.1)
        peak_ids_elbow, _ = scipy.signal.find_peaks(self.elbow_vel, prominence=max(self.elbow_vel)*0.1)"""

        peak_ids_hand, _ = self.get_peaks_of_movement(self.hand_vel)
        peak_ids_elbow, _ = self.get_peaks_of_movement(self.elbow_vel)

        # max velocity of hand mm/s
        if len(peak_ids_hand) == 0:
            self.PeakVelocity_mms = None
        else:
            peak_vel_hand = np.max([self.hand_vel[peak] for peak in peak_ids_hand])
            self.PeakVelocity_mms = round(peak_vel_hand, 3)

        # max elbow velocity deg/s
        if len(peak_ids_elbow) == 0:
            self.elbowVelocity = None
        else:
            peak_vel_elbow = np.max([self.elbow_vel[peak] for peak in peak_ids_elbow])
            self.elbowVelocity = round(peak_vel_elbow, 3)

        # time to peak hand velocity
        peak_id = np.where(self.hand_vel == self.hand_vel.flat[np.abs(self.hand_vel - peak_vel_hand).argmin()])[0][0]
        self.tTopeakV_s = self.time[peak_id]

        # time to first peak hand velocity
        first_peak_id = peak_ids_hand[0]
        self.tToFirstpeakV_s = self.time[peak_id]

        # time to peak hand velocity %
        self.tTopeakV_rel = (self.tTopeakV_s / self.TotalMovementTime) * 100

        # time to first peak hand velocity %
        self.tToFirstpeakV_rel = (self.tToFirstpeakV_s / self.TotalMovementTime) * 100

        self.NumberMovementUnits = self.get_movement_units(self.hand_vel)[0]

        self.InterjointCoordination = self.get_interjoint_coordination(self.elbow_flex_pos, self.shoulder_flex_pos)

        self.trunkDisplacementMM = self.get_trunk_displacement() # trunk displacement in mm

        self.trunkDisplacementDEG = self.get_trunk_rotation()  # Trunk Displacement in degrees

        self.ShoulderFlexionReaching = self.get_max_shoulder_flexion_reaching()

        self.ElbowExtension = self.get_min_elbow_flexion()

        self.shoulderAbduction = self.get_max_shoulder_abd_drink_reach()

    def write_measures(self):
        """
        This function adds the calculated measures to the .csv file.
        """
        murphy_measures = ["identifier",
                           "id_s",
                           "id_p",
                           "id_t",
                           "side",
                           "valid",
                           "ReachingStart",
                           "ForwardStart",
                           "DrinkingStart",
                           "BackStart",
                           "ReturningStart",
                           "RestStart",
                           "TotalMovementTime",
                           "condition",
                           "PeakVelocity_mms",
                           "elbowVelocity",
                           "tTopeakV_s",
                           "tToFirstpeakV_s",
                           "tTopeakV_rel",
                           "tToFirstpeakV_rel",
                           "NumberMovementUnits",
                           "InterjointCoordination",
                           "trunkDisplacementMM",
                           "trunkDisplacementDEG",
                           "ShoulderFlexionReaching",
                           "ElbowExtension",
                           "shoulderAbduction"]

        if self.identifier in list(self.df_measures['identifier']):
            for column in murphy_measures:
                self.df_measures.loc[self.df_measures['identifier'] == self.identifier, column] = self.__getattribute__(
                    column)
        else:
            # Add new row to dataframe
            new_row = pd.DataFrame([{column: self.__getattribute__(column) for column in murphy_measures}])
            self.df_measures = pd.concat([self.df_measures, new_row], ignore_index=True)


        self.df_measures.to_csv(self.csv_measures, sep=';', index=False)

        if self.verbose >= 1:
            print(f"Measures for {self.identifier} written to {self.csv_measures}")

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

        def standardize_columns(columns_old, columns_stand, verbose=0):
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

            if verbose >= 2:
                print("Standardizing: \tJoint Data.")

            df.columns = standardize_columns(df.columns, stand_joints, verbose)

        elif any(i in df.columns for i in [' hand_l_x', 'hand_l_x', 'hand_l_X']):
            if verbose >= 2:
                print("Standardizing:\tEndeffector Data")

            df.columns = standardize_columns(df.columns, stand_bodypart, verbose)

        else:
            raise ValueError("\n"
                             "Error in iDrinkStatisticalAnalysis.standardize_data\n"
                             "Neither 'elbow_flex_l' nor 'hand_l_x' are in Data.\n"
                             "Please check the data and try again.\n")

        return df


    def read_file(self, file_path, standardize=True):
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
        def get_dataframe_from_vec3(df):
            """
            Reads Dataframe retrieved from Outputreporter and returns a DataFrame with columns for each axis.
            :param df:
            :return:
            """

            new_df = pd.DataFrame()

            for column in df.columns:

                if column == 'time':
                    new_df['time'] = df['time']
                    continue

                new_column = column.split('/')[-1].split('|')[0]
                new_columns = [f'{new_column}_{axis}' for axis in ['x', 'y', 'z']]

                splitted_column = df[column].str.split(',', expand=True)

                if splitted_column.shape[1] == len(new_columns):
                    new_df[new_columns] = splitted_column.astype(np.float64)

                else:
                    for i, c in enumerate(splitted_column):
                        new_df[f'{new_column}_{i}'] = splitted_column[c]


            return new_df


        # Read Metadata and end at "endheader"
        if os.path.isfile(file_path) is False:
            raise FileNotFoundError(f"Error in iDrinkMurphyMeasures.read_file: {file_path} not found.")

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

            for meta in meta_dat:
                if "Vec3" in meta:
                    df = get_dataframe_from_vec3(df)


        elif  os.path.splitext(file_path)[1] == '.csv':
            meta_dat = None
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Error in iDrinkMurphyMeasures.read_file: {file_path} invalid file format {os.path.splitext(file_path)}.\n")
        if standardize:
            df = self.standardize_data(df, verbose= self.verbose)
        return meta_dat, df


    def read_files(self):
        """
        Reads all files and creates numpy arrays for calculation of murphy measures.

        - time
        - hand_vel
        - elbow_vel
        - trunk_pos

        if write_to_csv is True:
            Data is put into Dataframe and written to csv.

        :return:
        """

        def magnitude(data):
            """
            This function returns magnitude of a n-dimensional vector
            """

            return np.sqrt(np.sum(np.array([axis ** 2 for axis in data]), axis=0))

        # set values for filtering
        if self.filt_fps is None:
            self.filt_fps = round(len(self.time)/self.time[-1])

        if self.filt_cutoff_vel is None:
            self.filt_cutoff_vel = 6

        if self.filt_cutoff_pos is None:
            self.filt_cutoff_pos = 6

        if self.filt_order_pos is None:
            self.filt_order_pos = 4

        if self.filt_order_vel is None:
            self.filt_order_vel = 4



        _, df = self.read_file(self.path_bodyparts_vel)
        self.time = df['time'].values

        # Get hand velocity in mm/s
        hand_vel = [df[f'hand_{self.side.lower()}_{axis}'].values for axis in ['x', 'y', 'z']]
        hand_vel = magnitude(hand_vel)


        self.hand_vel = self.use_butterworth_filter(hand_vel,
                                                    cutoff=self.filt_cutoff_vel, fs=self.filt_fps,
                                                    order=self.filt_order_vel, normcutoff=False) * 1000

        # Get trunk displacement in mm
        _, df = self.read_file(self.path_trunk_pos, standardize=False)
        trunk_pos = [df[f'chest_{axis}'].values for axis in ['x', 'y', 'z']]
        trunk_pos = self.use_butterworth_filter(trunk_pos,
                                                cutoff=self.filt_cutoff_pos, fs=self.filt_fps,
                                                order=self.filt_order_pos, normcutoff=False).transpose()

        self.trunk_displacement = np.linalg.norm(trunk_pos[0] - trunk_pos, axis=1) * 1000

        _, df = self.read_file(self.path_bodyparts_pos)
        trunk_ang =[df[f'torso_{axis}'].values for axis in ['ox', 'oy', 'oz']]
        trunk_ang = magnitude(trunk_ang)
        self.trunk_ang = self.use_butterworth_filter(trunk_ang,
                                                     cutoff=self.filt_cutoff_pos, fs=self.filt_fps,
                                                     order=self.filt_order_pos, normcutoff=False)


        # get joint velocities
        _, df = self.read_file(self.path_joint_vel)
        elbow_vel = np.sqrt(df[f'elbow_flex_{self.side.lower()}'].values ** 2)
        self.elbow_vel = self.use_butterworth_filter(elbow_vel,
                                                     cutoff=self.filt_cutoff_vel, fs=self.filt_fps,
                                                     order=self.filt_order_vel, normcutoff=False)

        # Get joint Positions
        _, df = self.read_file(self.path_joint_pos)

        elbow_flex_pos = df[f'elbow_flex_{self.side.lower()}']
        self.elbow_flex_pos = self.use_butterworth_filter(elbow_flex_pos,
                                                          cutoff=self.filt_cutoff_pos, fs=self.filt_fps,
                                                          order=self.filt_order_pos, normcutoff=False)

        shoulder_flex_pos = df[f'arm_flex_{self.side.lower()}'].values
        self.shoulder_flex_pos = self.use_butterworth_filter(shoulder_flex_pos,
                                                             cutoff=self.filt_cutoff_pos, fs=self.filt_fps,
                                                             order=self.filt_order_pos, normcutoff=False)

        shoulder_abduction = -df[f'arm_add_{self.side.lower()}'].values  # Abduction is the negativ of adduction
        self.shoulder_abduction_pos = self.use_butterworth_filter(shoulder_abduction,
                                                                  cutoff=self.filt_cutoff_pos, fs=self.filt_fps,
                                                                  order=self.filt_order_pos, normcutoff=False)

        if self.write_mov_data:

            os.makedirs(os.path.dirname(self.path_mov_data), exist_ok=True)
            # make sure all arrays have the same length

            if not all(len(arr) == len(self.time) for arr in [self.hand_vel, self.elbow_vel, self.trunk_displacement,
                                                              self.trunk_ang, self.elbow_flex_pos,
                                                              self.shoulder_flex_pos, self.shoulder_abduction_pos]):

                min_len = min(len(arr) for arr in [self.hand_vel, self.elbow_vel, self.trunk_displacement,
                                                   self.trunk_ang, self.elbow_flex_pos,
                                                   self.shoulder_flex_pos, self.shoulder_abduction_pos])
                max_len = max(len(arr) for arr in [self.hand_vel, self.elbow_vel, self.trunk_displacement,
                                                    self.trunk_ang, self.elbow_flex_pos,
                                                    self.shoulder_flex_pos, self.shoulder_abduction_pos])

                if max_len-min_len < 5: # Taking out 5 frames is acceptable. If it is more, the corresponding data is not used.
                    self.time = self.time[:min_len]
                    self.hand_vel = self.hand_vel[:min_len]
                    self.elbow_vel = self.elbow_vel[:min_len]
                    self.trunk_displacement = self.trunk_displacement[:min_len]
                    self.trunk_ang = self.trunk_ang[:min_len]
                    self.elbow_flex_pos = self.elbow_flex_pos[:min_len]
                    self.shoulder_flex_pos = self.shoulder_flex_pos[:min_len]
                    self.shoulder_abduction_pos = self.shoulder_abduction_pos[:min_len]
            try:
                df = pd.DataFrame({'time': self.time,
                                   'hand_vel': self.hand_vel,
                                   'elbow_vel': self.elbow_vel,
                                   'trunk_disp': self.trunk_displacement,
                                   'trunk_ang': self.trunk_ang,
                                   'elbow_flex_pos': self.elbow_flex_pos,
                                   'shoulder_flex_pos': self.shoulder_flex_pos,
                                   'shoulder_abduction_pos': self.shoulder_abduction_pos})
                if self.path_mov_data is None:
                    self.path_mov_data = os.path.join(self.dir_trial, 'movement_analysis', 'murphy_measures')

                csv_out = os.path.join(self.path_mov_data, f'{self.identifier}_{self.condition}_{self.side.upper()}_murphymeasures.csv')

                df.to_csv(csv_out, sep=';', index=False)
            except Exception as e:
                print(f"Error in iDrinkMurphyMeasures.read_files: \n"
                      f"Error: {e}\n"
                      f"Could not write to {self.path_mov_data}")

    def get_data(self, df):
        """
        Sync the attributes with the DataFrame.
        """
        columns_of_interest = ["side", "condition", "valid", "ReachingStart", "ForwardStart", "DrinkingStart", "BackStart",
                               "ReturningStart", "RestStart", "TotalMovementTime"]

        if self.trial_id is None:
            raise ValueError("No trial_id given. Murphy object needs a trial_id to retrieve data from DataFrame.")

        try:
            for column in columns_of_interest:
                self.__setattr__(column, df.loc[df['identifier'] == self.trial_id, column].values[0])
        except:
            """
            Trial not in form of iDrink Trial_id
            csv containts only trials of single Participant and the trial number is in the form of TXXX.
            """

            if self.id_p not in df['id_p'].values:
                raise ValueError(f"Error in iDrinkMurphyMeasures.get_data: Participant {self.id_p} not in DataFrame.")

            for column in columns_of_interest:
                self.__setattr__(column, df.loc[(df['id_t'] == self.id_t) & (df['id_p'] == self.id_p), column].values[0])

        self.phases = {"reaching": [self.ReachingStart, self.ForwardStart],
                       "forward_transportation": [self.ForwardStart, self.DrinkingStart],
                       "drinking": [self.DrinkingStart, self.BackStart],
                       "back_transport": [self.BackStart, self.ReturningStart],
                       "returning": [self.ReturningStart, self.RestStart]}


    def get_df_from_murphy(self, path_csv):
        """
        Reads, sets and returns .csv already containing Murphy measures.
        """
        self.df_measures = pd.read_csv(path_csv)

        return self.df_measures

    def get_df_from_timestamps(self, ):
        """
        Get DataFrame for Murphy measures based on csv containing timestamps.
        #TODO: Check if still needed

        :return:
        """
        df_measures = pd.DataFrame(columns = ['identifier', 'id_s', 'id_p', 'id_t', 'valid', 'side', 'condition',
                                              'ReachingStart','ForwardStart', 'DrinkingStart', 'BackStart',
                                              'ReturningStart','RestStart', 'TotalMovementTime',
                                              'PeakVelocity_mms',  'elbowVelocity', 'tTopeakV_s',
                                              'tToFirstpeakV_s', 'tTopeakV_rel', 'tToFirstpeakV_rel',
                                              'NumberMovementUnits', 'InterjointCoordination', 'trunkDisplacementMM',
                                              'trunkDisplacementDEG','ShoulderFlexionReaching', 'ElbowExtension',
                                              'shoulderAbduction']
                                   )

        df_timestamps = pd.read_csv(self.csv_timestamps)

        for index, row in df_timestamps.iterrows():
            self.trial_id = row['trial_id']
            self.get_data(df_timestamps)
            self.get_paths()
            self.read_files()
            self.get_measures()
            self.write_measures()

            df_measures.loc[len(df_measures)] = [row['identifier'], row['id_s'], row['id_p'], row['id_t'], row['valid'], row['side'], row['condition'],
                                            self.ReachingStart, self.ForwardStart, self.DrinkingStart, self.BackStart,
                                            self.ReturningStart, self.RestStart, self.TotalMovementTime,
                                            self.PeakVelocity_mms, self.elbowVelocity, self.tTopeakV_s,
                                            self.tToFirstpeakV_s, self.tTopeakV_rel, self.tToFirstpeakV_rel,
                                            self.NumberMovementUnits, self.InterjointCoordination, self.trunkDisplacementMM,
                                            self.trunkDisplacementDEG, self.ShoulderFlexionReaching, self.ElbowExtension,
                                            self.shoulderAbduction]


        self.df_measures = df_measures

if __name__ == '__main__':

    """
    For now, the main is only for debugging the script with a hardcoded participant.
    """
    # TODO: Add argparse so file could be used standalone to get the measures when .csv file and directory is given.
    if os.name == 'posix':  # Running on Linux
        drive = '/media/devteam-dart/Extreme SSD'
        root_iDrink = os.path.join(drive, 'iDrink')  # Root directory of all iDrink Data
    else:
        path_phases = r"I:\iDrink\validation_root\04_statistics\02_categorical\murphy_measures.csv"  # Path to the .csv file containing murphy measures
        path_timestamps = r"I:\iDrink\validation_root\04_statistics\02_categorical\murphy_timestamps.csv" # Path to the .csv file containing timestamps
        root_data = r"I:\iDrink\validation_root\03_data"  # Root directory of all iDrink Data
        root_data_omc = r"I:\iDrink\validation_root\03_data\OMC"
        dir_trials = r"I:\iDrink\validation_root\03_data\OMC_old\S15133\S15133_P01" # Directory containing folders of P01

    path_preprocessed = os.path.join(root_data, 'preprocessed_data', '01_murphy_out',
                                     f'S15133_P01_T001_filtered.csv')
    measures = MurphyMeasures(csv_timestamps=path_timestamps, trial_id='S15133_P01_T001', root_data=root_data_omc,
                              write_mov_data=True, path_mov_data=path_preprocessed)
    measures.get_paths()
    measures.read_files()
    measures.get_measures()
    measures.write_measures()