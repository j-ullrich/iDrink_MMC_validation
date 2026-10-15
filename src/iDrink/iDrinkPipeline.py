"""
Light-weight building blocks of the validation pipeline.

Deliberately free of heavy imports (pandas, torch, ...), so it can be imported and tested anywhere.
"""
from dataclasses import dataclass
from typing import Any

# Stage names accepted by --stages, in the order "full" runs them.
STAGES = ["calibration", "pose_estimation", "pose2sim", "opensim", "murphy_measures", "statistics", "full"]

_PATH_ATTRS = ("root_iDrink", "root_MMC", "root_OMC", "root_val", "default_dir", "root_HPE", "root_data",
               "root_stat", "root_logs", "metrabs_models_dir", "log_val_settings", "log_val_trials",
               "log_val_errors", "log_calib", "log_calib_full")


@dataclass
class PipelineContext:
    """State shared by the pipeline stages. Replaces the module level globals of iDrinkHubValidation."""
    paths: Any  # iDrinkConfig.Paths
    args: Any  # argparse.Namespace
    df_settings: Any  # pandas.DataFrame, one row per setting
    df_trials: Any = None  # pandas.DataFrame, one row per trial. None if no trial csv exists yet

    def __getattr__(self, name):
        # Only called when normal lookup fails: expose the paths as str, like the old os.path.join based globals.
        if name in _PATH_ATTRS:
            return str(getattr(self.paths, name))
        raise AttributeError(name)

    @property
    def path_csv_murphy_timestamps(self):
        return str(self.paths.root_stat / '02_categorical' / 'murphy_timestamps.csv')

    @property
    def path_csv_murphy_measures(self):
        return str(self.paths.root_stat / '02_categorical' / 'murphy_measures.csv')


def _as_list(value):
    return [value] if isinstance(value, str) else value


def selected(trial, args, patients=True):
    """
    True if the trial passes the --identifier / --patients / --settings selection.

    :param patients: set False for stages that never filtered by patient (Pose2Sim).
    """
    if args.single_identifier and trial.identifier != args.single_identifier:
        return False
    if patients and args.single_patient and trial.id_p not in _as_list(args.single_patient):
        return False
    if args.single_setting and trial.id_s not in _as_list(args.single_setting):
        return False
    return True


def steps_for_mode(mode, poseback=(), no_calib=False):
    """
    Names of the steps that run for a stage, in order.

    Pose estimation needs calibration only for metrabs_multi, Pose2Sim always, unless no_calib is set.
    """
    if mode == "calibration":
        return ["calibration"]
    if mode == "pose_estimation":
        calib = ["calibration"] if ("metrabs_multi" in poseback and not no_calib) else []
        return calib + ["hpe"]
    if mode == "pose2sim":
        return ([] if no_calib else ["calibration"]) + ["triangulation"]
    if mode == "opensim":
        return ["opensim"]
    if mode == "murphy_measures":
        return ["murphy_measures"]
    if mode == "statistics":
        return ["statistics"]
    if mode == "full":
        return ["calibration", "hpe", "triangulation", "opensim", "murphy_measures", "statistics"]
    raise ValueError("Invalid Mode was given. Please specify a valid mode.")
