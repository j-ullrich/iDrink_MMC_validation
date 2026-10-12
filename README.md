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
2. Configure the data location: set `IDRINK_ROOT` (and optionally `IDRINK_MMC`, `IDRINK_OMC`), or copy `src/config.example.toml` to `src/config.toml` and edit it. The legacy `src/path.txt` is still read as a fallback.
3. Model weights (`*.pt`, e.g. `yolov8m.pt`) are not tracked in git; download them separately and place them in `src/` and `src/Metrabs_PoseEstimation/`.

## Usage

Run from `src/`:

```
python iDrinkHubValidation.py --stages murphy_measures --poseback metrabs_multi --patients P07,P08 --settings S027,S028
python iDrinkHubValidation.py --stages opensim --only-single-cam --dry-run   # only list the trials
```

Stages: `calibration, pose_estimation, pose2sim, opensim, murphy_measures, statistics, full` (comma separated).
Patients and settings default to the `[run]` section of `config.toml`, otherwise all trials are processed. See `--help` for all options.

Tests: `pytest tests`
