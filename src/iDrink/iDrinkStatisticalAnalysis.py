"""
We compare each MMC-Trial Objects time series and Murphy Measures to the same metrics of the OMC-trials.

"""
import os
import sys
import shutil
import glob
import re
from tqdm import tqdm

from trc import TRCData

import pandas as pd
import numpy as np
import scipy as sp






def runs_statistics_discrete(df_mmc, df_omc, root_stat):
    """
    Takess Murphy Measures of MMC and OMC and compares them. Then plots the results and saves data and plots in the Statistics Folder.
    :param df_mmc:
    :param df_omc:
    :return:
    """

    pass


def run_statistics_continuous(df_mmc_pos, df_mmc_vel, df_omc_pos, df_omc_vel, isjoint, root_stat):
    """
    Gets two DataFrames form MMC and OMC. Once the Position of the Bodyparts and then their Velocity.

    This function works with endeffector Pos/Vel and Joint Pos/Vel.

    It then plots the

    :param df_mmc_pos:
    :param df_mmc_vel:
    :param df_omc_pos:
    :param df_omc_vel:
    :param isjoint:
    :param root_stat:
    :return:
    """

    pass


def standardize_data(df):
    """
    gets a DataFrame containing data of joints or endeffector positions.

    It checks for the type of data and then renames the columns to a standardized set for later functions.

    :param df:
    :return:
    """