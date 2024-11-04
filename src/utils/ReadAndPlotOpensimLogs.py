import re
import os
import glob
from tqdm import tqdm
import numpy as np
import pandas
import pandas as pd

import plotly as py
import plotly.express as px
import plotly.graph_objects as go


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

    def write_means(self):
        self.df_mean.to_csv(self.csv_path.replace('_opensim_inverse_kinematics_log.csv', '_opensim_log_means.csv'), index=False)

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

        self.df_mean = pd.DataFrame({'id_p': self.id_p, 'id_t': self.id_t,
                                    'total_squared_error': self.df['total_squared_error'].mean(),
                                    'marker_rmse': self.df['marker_rmse'].mean(),
                                    'marker_max_error': self.df['marker_max_error'].mean(),
                                    'bodypart_max': self.df['bodypart_max'].value_counts().idxmax()}, index=[0])

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

        for line in self.lines:
            if all([w in line for w in wanted_in_line]):

                time_frame, line_data = line.split('\t')

                date_str = time_frame.split(']')[0][1:]
                date_time = pd.to_datetime(date_str)
                frame = int(time_frame.split('Frame')[1].split('(')[0])
                time = float(time_frame.split('(t = ')[1].split('):')[0])

                total_squared_error, marker_rmse, marker_max_error, bodypart_max = self.parse_errors(line_data.split(','))


                df_newline = pd.DataFrame({'id_p': self.id_p, 'id_t': self.id_t, 'date_time': date_time, 'time': time, 'frame': frame,
                                           'total_squared_error': total_squared_error, 'marker_rmse': marker_rmse,
                                           'marker_max_error': marker_max_error, 'bodypart_max': bodypart_max}, index=[0])

                if self.debug:
                    print(df_newline)

                df = pd.concat([df, df_newline], ignore_index=True)

        self.mean_total_squared_error = df['total_squared_error'].mean()
        self.mean_rmse = df['marker_rmse'].mean()
        self.mean_max_error = df['marker_max_error'].mean()
        self.worst_bodypart = df['bodypart_max'].value_counts().idxmax()

        self.max_total_squared_error = df['total_squared_error'].max()
        self.max_rmse = df['marker_rmse'].max()
        self.max_max_error = df['marker_max_error'].max()

        self.df_mean = pd.DataFrame({'id_p': self.id_p, 'id_t': self.id_t,
                                     'total_squared_error': self.mean_total_squared_error,
                                     'marker_rmse': self.mean_rmse, 'marker_max_error': self.mean_max_error,
                                     'bodypart_max': self.worst_bodypart}, index=[0])

        self.df = df

    def get_last_ik_lines(self):
        """iterates backwards over a files lines and sets self.lines to the last set of lines of the Inverse Kinematics tool."""
        wanted_in_line = ['Frame', 'total squared error', 'marker error:']

        start_id = None
        last_id = None
        started = False

        for i in range(len(self.lines)-1, -1, -1):
            line = self.lines[i]

            if last_id is None and all([w in line for w in wanted_in_line]):
                last_id = i+1

            if start_id is None and last_id is not None and 'Running tool' in line:
                start_id = i+1
                break

            if not started and 'InverseKinematicsTool completed' in line:
                started = True

        self.lines = self.lines[start_id:last_id]



    def read(self):
        """
        reads the opensim log file and returns the error values.

        If last_date is True, the function will return teh data of the last time, Opensim has been executed.

        """
        with open(self.file, 'r') as f:
            self.lines = f.readlines()

        self.get_last_ik_lines()





def run_first_stage(dir_opensim_logs, id_s):
    """
    runs the pipeline to read all opensimlogs of a given id_s and then plot the retrieved errors.


    :param dir_opensim_logs:
    :return:
    """

    p_list = ['P08', 'P10', 'P12', 'P13', 'P14', 'P15', 'P17', 'P19', 'P24', 'P25', 'P27', 'P28', 'P30', 'P31', 'P34']
    files = glob.glob(os.path.join(dir_opensim_logs, f"{id_s}*opensim.log"))
    loglist = []
    prcss = tqdm(total=len(files), desc=f"Processing {id_s}", unit='files')
    for file in files:
        id_p = os.path.basename(file).split('_')[1]
        id_t = os.path.basename(file).split('_')[2]
        identifier = f"{id_s}_{id_p}_{id_t}"

        if id_p not in p_list:
            prcss.update(1)
            continue

        prcss.set_description(f"Processing {id_s}_{id_p}_{id_t}")
        prcss.update(1)

        if glob.glob(os.path.join(dir_opensim_logs, f"{identifier}_opensim_inverse_kinematics_log.csv")):
            csv_file = glob.glob(os.path.join(dir_opensim_logs, f"{identifier}_opensim_inverse_kinematics_log.csv"))[0]
            opensim_log = OpensimLogReader(file, id_t, id_p)
            opensim_log.csv_path = csv_file
            opensim_log.read_csv()

        else:
            opensim_log = OpensimLogReader(file, id_t, id_p)
            opensim_log.read()
            opensim_log.parselines()
            opensim_log.write_csv()
            opensim_log.write_means()

        loglist.append(opensim_log)
    prcss.close()

    prcss2 = tqdm(total=4, desc=f"Processing {id_s}", unit='files')

    df_mean = pd.concat([log.df_mean for log in loglist], ignore_index=True)
    prcss2.set_description(f"Processing {id_s}: df_mean created")
    prcss2.update(1)
    df = pd.concat([log.df for log in loglist], ignore_index=True)
    prcss2.set_description(f"Processing {id_s}: df_full created")
    prcss2.update(1)

    df_mean.to_csv(os.path.join(dir_opensim_logs, f"{id_s}_opensim_inverse_kinematics_log_means.csv"), index=False)
    prcss2.set_description(f"Processing {id_s}: df_mean written")
    prcss2.update(1)
    df.to_csv(os.path.join(dir_opensim_logs, f"{id_s}_opensim_inverse_kinematics_log.csv"), index=False)
    prcss2.set_description(f"Processing {id_s}: df_full written")
    prcss2.update(1)

    print(f"Stage 1 for {id_s} completed.")
    prcss2.close()

def plot_means(dir_opensim_logs, dir_plots, id_s, showfig=False):
    df = pd.read_csv(os.path.join(dir_opensim_logs, f"{id_s}_opensim_inverse_kinematics_log_means.csv"))

    legendnames_for_column_names = {'total_squared_error': 'Total Squared Error',
                                    'marker_rmse': 'Marker RMSE',
                                    'marker_max_error': 'Max Marker Error',
                                    'bodypart_max': 'Worst Bodypart'}

    columns_to_plot = ['total_squared_error', 'marker_rmse', 'marker_max_error']
    columns_to_plot = ['marker_rmse']
    os.makedirs(os.path.join(dir_plots, 'pat_mean'), exist_ok=True)
    prog = tqdm(total=len(df['id_p'].unique())*len(columns_to_plot), desc=f"Processing {id_s}", unit='files')
    for id_p in df['id_p'].unique():


        for column in columns_to_plot:
            prog.set_description(f"Processing {id_s}_{id_p}_{column}")
            fig = px.bar(df[df['id_p'] == id_p], x='id_t', y=column, title=f"{legendnames_for_column_names[column]} for {id_p}",)
            fig.update_layout(xaxis_title=f'Trials',
                              yaxis_title=f'{legendnames_for_column_names[column]}'
                              )

            if showfig:
                fig.show()
            fig.write_html(os.path.join(dir_plots, 'pat_mean', f"{id_p}_{column}_bar_mean.html"))
            fig.write_image(os.path.join(dir_plots, 'pat_mean', f"{id_p}_{column}_bar_mean.png"))
            prog.update(1)

    prog.close()


    prog = tqdm(total=len(columns_to_plot), desc=f"Processing {id_s}", unit='files')
    for column in columns_to_plot:
        prog.set_description(f"Processing {id_s}_{column}")

        fig = px.box(df, x='id_p', y=column, title=f"{legendnames_for_column_names[column]} for {id_s}",)
        fig.update_layout(xaxis_title=f'Participants',
                          yaxis_title=f'{legendnames_for_column_names[column]}'
                          )

        if showfig:
            fig.show()
        fig.write_html(os.path.join(dir_plots, 'pat_mean', f"{id_s}_{column}_box_means.html"))
        fig.write_image(os.path.join(dir_plots, 'pat_mean', f"{id_s}_{column}_box_means.png"))
        prog.update(1)

    prog.close()





    pass

def consolidate_csvs(dir_opensim_logs, id_s):
    """
    Checks that stage one has run for all files. Then it consolidates all .csv files of id_s into one big .csv File.
    """

    n_files = len(glob.glob(os.path.join(dir_opensim_logs, f"{id_s}*opensim.log")))
    n_csvs = len(glob.glob(os.path.join(dir_opensim_logs, f"{id_s}*opensim_inverse_kinematics_log.csv")))

    if n_files != n_csvs:
        print(f"Stage one has not run for all files of {id_s}.")
        run_first_stage(dir_opensim_logs, id_s)
    pass

    #Consolidate all csv files into one big csv file for each id_p

    files = glob.glob(os.path.join(dir_opensim_logs, f"{id_s}*opensim_inverse_kinematics_log.csv"))

    idx_p = set([os.path.basename(file).split('_')[1] for file in files])
    prcss = tqdm(total=len(idx_p), desc=f"Processing {id_s}", unit='files')

    for id_p in idx_p:
        prcss.set_description(f"Processing {id_s}_{id_p}")
        prcss.update(1)
        files_p = [file for file in files if id_p in file]

        df = pd.concat([pd.read_csv(file) for file in files_p], ignore_index=True)

        df.to_csv(os.path.join(dir_opensim_logs, f"{id_s}_{id_p}_opensim_inverse_kinematics_patient_log.csv"), index=False)

    prcss.close()

def plot_individual_trial(dir_opensim_logs, dir_plots, id_s, showfig=False):
    """
    Checks that stage two is completed for all id_ps in orginial files.

    Then it creates boxplots for each id_p and columns of the csv. files.

    :param dir_opensim_logs:
    :param id_s:
    :return:
    """


    files_stg_one = glob.glob(os.path.join(dir_opensim_logs, f"{id_s}*opensim.log"))
    idx_p_stg_one = set([os.path.basename(file).split('_')[1] for file in files_stg_one])

    files_stg_two = glob.glob(os.path.join(dir_opensim_logs, f"{id_s}*opensim_inverse_kinematics_patient_log.csv"))
    idx_p_stg_two = set([os.path.basename(file).split('_')[1] for file in files_stg_two])

    if idx_p_stg_one != idx_p_stg_two:
        print(f"Stage two has not run for all files of {id_s}.")
        consolidate_csvs(dir_opensim_logs, id_s)

    dir_plots = os.path.join(dir_plots, 'individual_trials')
    os.makedirs(dir_plots, exist_ok=True)

    files = glob.glob(os.path.join(dir_opensim_logs, f"{id_s}*opensim_inverse_kinematics_patient_log.csv"))

    legendnames_for_column_names = {'total_squared_error': 'Total Squared Error',
                                    'marker_rmse': 'Marker RMSE',
                                    'marker_max_error': 'Max Marker Error',
                                    'bodypart_max': 'Worst Bodypart'}

    columns_to_plot = ['total_squared_error', 'marker_rmse', 'marker_max_error']
    columns_to_plot = ['marker_rmse']

    for file in files:
        id_p = os.path.basename(file).split('_')[1]

        for column in columns_to_plot:
            fig = px.box(pd.read_csv(file), x='id_t', y=column, title=f"{legendnames_for_column_names[column]} for {id_p}",)
            fig.update_layout(xaxis_title=f'Trials',
                              yaxis_title=f'{legendnames_for_column_names[column]}'
                              )

            if showfig:
                fig.show()
            fig.write_html(os.path.join(dir_plots, f"{os.path.basename(file).replace('.csv', f'_{column}.html')}"))






if __name__ == '__main__':

    dir_opensim_logs = r"I:\iDrink\validation_root\05_logs\opensim"
    dir_plots = os.path.join(dir_opensim_logs, 'plots')
    id_s = 'S15133'

    run_first_stage(dir_opensim_logs, id_s)
    #run_second_stage(dir_opensim_logs, id_s)
    #run_third_stage(dir_opensim_logs, dir_plots, id_s, showfig=True)
    plot_means(dir_opensim_logs, dir_plots, id_s, showfig=False)