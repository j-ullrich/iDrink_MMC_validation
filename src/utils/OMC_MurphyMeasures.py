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
from iDrink import iDrinkTrial, iDrinkOpenSim, iDrinkUtilities, iDrinkLog

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

pass

p_list = sorted([p.split("_")[1] for p in os.listdir(root_omc)])

id_s = "S15133"

for id_p in p_list:

    p_dir = os.path.join(root_omc, f"{id_s}_{id_p}")

    t_list = sorted([t.split("_")[2] for t in os.listdir(p_dir)])

    for t_id in t_list:

        identifier = f"{id_s}_{id_p}_{t_id}"
        t_dir = os.path.join(p_dir, f"{id_s}_{id_p}_{t_id}")

        print(t_dir)








