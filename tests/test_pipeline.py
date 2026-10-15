import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from iDrink import iDrinkConfig as cfg
from iDrink.iDrinkPipeline import PipelineContext, selected, steps_for_mode


def trial(id_p="P01", id_s="S001", identifier="S001_P01_T001"):
    return SimpleNamespace(id_p=id_p, id_s=id_s, identifier=identifier)


def args(**kw):
    base = dict(single_identifier=False, single_patient=False, single_setting=False)
    return Namespace(**{**base, **kw})


def test_selected_no_filter():
    assert selected(trial(), args())


def test_selected_patient_and_setting_accept_str_or_list():
    assert selected(trial(), args(single_patient="P01"))
    assert not selected(trial(), args(single_patient=["P02"]))
    assert selected(trial(), args(single_setting=["S001", "S002"]))
    assert not selected(trial(), args(single_setting="S002"))


def test_selected_identifier():
    assert not selected(trial(), args(single_identifier="other"))
    assert selected(trial(), args(single_identifier="S001_P01_T001"))


def test_selected_patients_ignored_for_pose2sim():
    assert selected(trial(), args(single_patient=["P02"]), patients=False)


def test_steps_for_mode():
    assert steps_for_mode("full") == ["calibration", "hpe", "triangulation", "opensim", "murphy_measures", "statistics"]
    assert steps_for_mode("pose_estimation", ["mmpose"]) == ["hpe"]
    assert steps_for_mode("pose_estimation", ["metrabs_multi"]) == ["calibration", "hpe"]
    assert steps_for_mode("pose_estimation", ["metrabs_multi"], no_calib=True) == ["hpe"]
    assert steps_for_mode("pose2sim") == ["calibration", "triangulation"]
    with pytest.raises(ValueError):
        steps_for_mode("nope")


def test_context_exposes_paths_as_str(tmp_path):
    p = cfg.load_config(tmp_path / "x.toml", env={"IDRINK_ROOT": str(tmp_path / "iDrink")})
    ctx = PipelineContext(paths=p, args=args(), df_settings=None)
    assert ctx.root_val == str(tmp_path / "iDrink" / "validation_root")
    assert ctx.path_csv_murphy_measures.endswith("murphy_measures.csv")
    with pytest.raises(AttributeError):
        ctx.nonsense
