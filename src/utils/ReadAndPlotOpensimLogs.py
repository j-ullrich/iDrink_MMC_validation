import re
import os
import glob
import numpy as np
import pandas
import pandas as pd
from sympy.codegen.cnodes import static


class OpensimLogReader:
    def __init__(self, file = None, id_t=None, id_p=None):
        self.file = file
        if os.path.isfile(self.file):
            self.csv_path = self.file.replace('_opensim.log', '_opensim_inverse_kinematics_log.csv')
        else:
            self.file = None
            self.csv_path = None

        self.df = None
        self.df_mean = None

        self.debug = False

        self.id_t = id_t
        self.id_p = id_p
        self.newest_date = None

        self.lines = None

        self.thresh_mid = 0.02
        self.thresh_high = 0.04

        self.mean_total_squared_error = None
        self.mean_rmse = None
        self.mean_max_error = None
        self.worst_bodypart = None

        self.max_total_squared_error = None
        self.max_rmse = None
        self.max_max_error = None

    def set_file(self, file):
        self.file = file
        if os.path.isfile(self.file):
            self.csv_path = self.file.replace('_opensim.log', 'opensim_inverse_kinematics_log.csv')
        else:
            self.file = None

            print(f"File {file} does not exist.")

    def write_csv(self):
        self.df.to_csv(self.csv_path, index=False)

    def read_csv(self):
        if self.csv_path is None:
            if os.path.isfile(self.file):
                self.csv_path = self.file.replace('_opensim.log', 'opensim_inverse_kinematics_log.csv')
                return
            else:
                print(f"csv_path is None and File does not exist.\n"
                      f"{self.file}")
                return

        self.df = pd.read_csv(self.csv_path)

    @staticmethod
    def parse_errors(line_data):
        """
        Goes through line_data and returns the errors and the bodypart with max error.
        :param line_data:
        :return:
        """

        total_squared_error = None
        marker_rmse = None
        marker_max_error = None
        bodypart_max = None

        for data_string in line_data:

            if 'total' in data_string:
                total_squared_error = float(data_string.split('=')[1])
            if 'RMS' in data_string:
                marker_rmse = float(data_string.split('=')[1])
            if 'max' in data_string:
                value_string, bodypart = data_string.split('(')

                marker_max_error = float(value_string.split('=')[1])
                bodypart_max = bodypart.split(')')[0]

        return total_squared_error, marker_rmse, marker_max_error, bodypart_max

    def parselines(self):
        """
        Iterates thorugh lines and creates pandas Dataframe with data of Inverse Kinematics Tool and metadata.
        :param lines:
        :return:
        """

        df = pandas.DataFrame(columns=['id_p', 'id_t', 'date_time', 'time', 'frame', 'total_squared_error', 'marker_rmse', 'marker_max_error', 'bodypart_max'])
        wanted_in_line = ['Frame', 'total squared error', 'marker error:']
        frame = 0
        start = False

        for line in self.lines:

            if 'Running tool' in line:
                start = True

            if start and all([w in line for w in wanted_in_line]):

                time_frame, line_data = line.split('\t')

                date_str = time_frame.split(']')[0][1:]
                date_time = pd.to_datetime(date_str)
                time = float(time_frame.split('(t = ')[1].split('):')[0])

                total_squared_error, marker_rmse, marker_max_error, bodypart_max = self.parse_errors(line_data.split(','))


                df_newline = pd.DataFrame({'id_p': self.id_p, 'id_t': self.id_t, 'date_time': date_time, 'time': time, 'frame': frame,
                                           'total_squared_error': total_squared_error, 'marker_rmse': marker_rmse,
                                           'marker_max_error': marker_max_error, 'bodypart_max': bodypart_max}, index=[0])

                if self.debug:
                    print(df_newline)

                df = pd.concat([df, df_newline], ignore_index=True)

                frame += 1

        self.mean_total_squared_error = df['total_squared_error'].mean()
        self.mean_rmse = df['marker_rmse'].mean()
        self.mean_max_error = df['marker_max_error'].mean()
        self.worst_bodypart = df['bodypart_max'].value_counts().idxmax()

        self.max_total_squared_error = df['total_squared_error'].max()
        self.max_rmse = df['marker_rmse'].max()
        self.max_max_error = df['marker_max_error'].max()

        self.df_mean = pd.DataFrame({'id_p': self.id_p, 'id_t': self.id_t,
                                     'mean_total_squared_error': self.mean_total_squared_error,
                                     'mean_rmse': self.mean_rmse, 'mean_max_error': self.mean_max_error,
                                     'worst_bodypart': self.worst_bodypart}, index=[0])

        self.df = df


    def read(self):
        """
        reads the opensim log file and returns the error values.

        If last_date is True, the function will return teh data of the last time, Opensim has been executed.

        """
        def get_last_date(second_to_last=False):
            """
            iterates over lines and returns the last date of the log file.
            :param get_last_date:
            :return:
            """
            dates = []
            list_of_dates = []
            idx_of_dates = []

            for i, line in enumerate(lines):
                try:
                    date_str = line.split(']')[0][1:]
                    date = pd.to_datetime(date_str).date()
                    if date not in list_of_dates:
                        list_of_dates.append(date)
                        idx_of_dates.append(i)
                except:
                    continue
                dates.append(date)

            if second_to_last:
                newest_date = list_of_dates[-2]
                return newest_date, idx_of_dates[-2]

            newest_date = list_of_dates[-1]
            return newest_date, idx_of_dates[-1]


        with open(self.file, 'r') as f:
            lines = f.readlines()
            self.newest_date, first_id_of_newest_date = get_last_date()

            self.lines = lines[first_id_of_newest_date:]


def run_first_stage(dir_opensim_logs, id_s):
    """
    runs the pipeline to read all opensimlogs of a given id_s and then plot the retrieved errors.


    :param dir_opensim_logs:
    :return:
    """
    from tqdm import tqdm

    files = glob.glob(os.path.join(dir_opensim_logs, f"{id_s}*opensim.log"))
    dict_logs = {}
    prcss = tqdm(total=len(files), desc=f"Processing {id_s}", unit='files')
    for file in files:
        id_p = os.path.basename(file).split('_')[1]
        id_t = os.path.basename(file).split('_')[2]
        identifier = f"{id_s}_{id_p}_{id_t}"

        prcss.set_description(f"Processing {id_s}_{id_p}_{id_t}")
        prcss.update(1)

        if identifier != 'S15133_P12_T009':
            continue

        if id_p not in dict_logs.keys():
            dict_logs[id_p] = []

        opensim_log = OpensimLogReader(file, id_t, id_p)
        opensim_log.read()
        opensim_log.parselines()
        opensim_log.write_csv()

        dict_logs[id_p].append(opensim_log)

def run_second_stage(root_opensim_logs, id_s):
    """
    Checks that stage one has run for all files. Then it consolidates all .csv files of id_s into one big .csv File.
    """

    pass


if __name__ == '__main__':

    root_opensim_logs = r"I:\iDrink\validation_root\05_logs\opensim"
    id_s = 'S15133'

    run_first_stage(root_opensim_logs, id_s)