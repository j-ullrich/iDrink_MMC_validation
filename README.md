# Master Thesis Repository - Implementation and validation of an upper limb movement quality assessement tool.

This Repository contains the code and resources relevant to the master thesis of Johann Ullrich in 2024. Furthermore, unused code written as part of the project can be found in:

* [First Analysis Repository (Discontinued)](https://github.com/DART-Lab-LLUI/iDrink_DataAnalysis)
* [iDrink prototype](https://github.com/DART-Lab-LLUI/iDrink_GUI)

The project implements a pipeline that reconstructs 3D-keypoints from several video recordings and calculates the inverse kinematics and measures for movement quality.

## Table of Contents
- [Installation](#installation)
- [Usage](#Usage)
 
## Installation

1. Create a conda environment from one of the exported files (`iDrink_env.yml`, `env1.yaml`, `env2.yml`; these are UTF-16 exports and the authoritative one is yet to be consolidated).
2. Set the data root in `src/path.txt` (`key;value` format).
3. Model weights (`*.pt`, e.g. `yolov8m.pt`) are no longer tracked in git; download them separately and place them in `src/` and `src/Metrabs_PoseEstimation/`.

## Usage

The entry point is `src/iDrinkHubValidation.py`. Run it from `src/`. The stages are: calibration, pose estimation, triangulation, OpenSim, Murphy measures and statistics.
Note: the stage, patients and pose backend are currently set in the `__main__` block of that file rather than on the command line.
