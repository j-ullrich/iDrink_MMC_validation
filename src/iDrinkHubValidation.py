import glob
import os
import sys
import time





"""
This File is the starting Point of the iDrink Validation.


It creates all trial objects and then runs the pipeline for each trial and setting.

The pipeline is as follows:

- Pose Estimation
- Pose2Sim
- Opensim
- iDrink Analytics --> Calculation of Murphy Measures

The possible settings are:

- 29 different Camera Setups
- 2 Calibration Methods
- 3 HPE Methods (Openpose, MMPose and Pose2Sim Pose Estimation)
- 2 sets of 2D Keypoints (filtered and unfiltered)
- 4 ways to calculate the Murphy Measures (Analyzer only, Analyzer and keypoints, Opensim Invkin Tool and Analyzer, and Opensim invkin Tool and keypoints)

In Total we have 29 * 2 * 3 * 2 * 4 = 1392 different settings

When creating the trial objects, a csv file is created, which contains all the settings for each trial.
"""