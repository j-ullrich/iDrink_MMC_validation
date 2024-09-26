import os

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
import pandas as pd


class MurphyMeasures:
    """
    This class is used to calculate the Murphy measures and write them in a .csv file.


    """
    def __init__(self, path_csv=None):

        self.path_csv = path_csv
        self.df = None

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
            self.get_measures()
            self.write_measures()

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

    def get_data(self, df):
        """
        This function gets all possible attributes from the DataFrame
        """
        pass

    def read_csv(self, path_csv):
        """
        This function reads the .csv file containing the data. and calls the get_data function.
        It also returns the DataFrame from the .csv file.
        """
        df = pd.read_csv(path_csv)

        return df





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
        dir_trials = r"I:\iDrink\validation_root\03_data\OMC\S15133\S15133_P01" # Directory containign folders of P01




    pd.DataFrame(columns = ['trial', 'side', 'condition',
                            'ReachingStart', 'ForwardStart', 'DrinkingStart',  'BackStart', 'ReturningStart', 'RestStart',
                            'TotalMovementTime', 'PeakVelocity_mms',  'elbowVelocity',
                            'tTopeakV_s', 'tToFirstpeakV_s', 'tTopeakV_rel', 'tToFirstpeakV_rel',
                            'NumberMovementUnits', 'InterjointCoordination',
                            'trunkDisplacementMM', 'trunkDisplacementDEG',
                            'ShoulerFlexionReaching', 'ElbowExtension', 'shoulderAbduction'
                            ]
                 )