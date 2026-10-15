# Master Thesis Repository - Implementation and validation of an upper limb movement quality assessement tool.

This Repository contains the code and resources relevant to the master thesis of Johann Ullrich in 2024. Furthermore, unused code written as part of the project can be found in:

* [First Analysis Repository (Discontinued)](https://github.com/DART-Lab-LLUI/iDrink_DataAnalysis)
* [iDrink prototype](https://github.com/DART-Lab-LLUI/iDrink_GUI)

The project implements a pipeline that reconstructs 3D-keypoints from several video recordings and calculates the inverse kinematics and measures for movement quality.

## Table of Contents
- [Installation](#installation)
- [Usage](#Usage)
 
## Installation

1. Create the conda environment: `conda env create -f environment.yml` (installs the package in editable mode and provides the `idrink` command). It assumes an NVIDIA GPU with CUDA 12.1; the comment at the top of the file says what to change for a CPU-only machine. The old exports of the author's machine are in `legacy_envs/`.
2. Configure the data location: set `IDRINK_ROOT` (and optionally `IDRINK_MMC`, `IDRINK_OMC`), or copy `src/config.example.toml` to `src/config.toml` and edit it. The legacy `src/path.txt` is still read as a fallback.
3. Download the Metrabs model: `python scripts/download_weights.py` (see `--help`). The YOLOv8 person detector (`yolov8m.pt`) is fetched by `ultralytics` on first use, nothing to do.

## Usage

Run `idrink` (or `python iDrinkHubValidation.py` from `src/`):

```
idrink --stages murphy_measures --poseback metrabs_multi --patients P07,P08 --settings S027,S028
idrink --stages opensim --only-single-cam --dry-run   # only list the trials
```

`--jobs N` runs the `opensim` and `murphy_measures` stages in N worker processes (default 1 = everything in the main process, the tested path). Trials that already have a Murphy measures row are skipped unless `--run-again` is given.

Stages: `calibration, pose_estimation, pose2sim, opensim, murphy_measures, statistics, full` (comma separated).
Patients and settings default to the `[run]` section of `config.toml`, otherwise all trials are processed. See `--help` for all options.

Tests: `pytest tests`

Continuous integration (`.github/workflows/ci.yml`) runs the tests with the heavy dependencies stubbed. Known issue: the `statistics` stage of the hub refers to names it never defines and fails until fixed.
