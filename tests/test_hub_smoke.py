"""Smoke tests for iDrinkHubValidation with the heavy dependencies stubbed out."""
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest

pd = pytest.importorskip("pandas")

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

HEAVY = ["tqdm", "Pose2Sim", "iDrink.iDrinkTrial", "iDrink.iDrinkPoseEstimation", "iDrink.iDrinkLog",
         "iDrink.iDrinkOpenSim", "iDrink.iDrinkUtilities", "iDrink.iDrinkMurphyMeasures",
         "iDrink.iDrinkStatisticalAnalysis", "iDrink.iDrinkCalibration"]


class FakeTqdm:
    """Minimal tqdm: iterates like the wrapped iterable."""

    def __init__(self, iterable=None, **kw):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, *a, **k):
        pass


@pytest.fixture
def hub(monkeypatch):
    """Import the hub with stubs for everything that needs torch, opensim, cv2, ..."""
    for name in HEAVY:
        stub = mock.MagicMock(name=name)
        monkeypatch.setitem(sys.modules, name, stub)
        if name.startswith("iDrink."):
            monkeypatch.setattr(sys.modules["iDrink"] if "iDrink" in sys.modules else __import__("iDrink"),
                                name.split(".")[1], stub, raising=False)
    sys.modules["Pose2Sim"].Pose2Sim = mock.MagicMock()
    sys.modules["tqdm"].tqdm = FakeTqdm
    sys.modules.pop("iDrinkHubValidation", None)
    import iDrinkHubValidation
    yield iDrinkHubValidation
    sys.modules.pop("iDrinkHubValidation", None)


def make_ctx(hub, tmp_path, **args):
    from iDrink import iDrinkConfig
    paths = iDrinkConfig.load_config(tmp_path / "none.toml", env={"IDRINK_ROOT": str(tmp_path / "iDrink")})
    base = dict(verbose=0, poseback=["mmpose"], mode=["full"], single_identifier=False, single_patient=False,
                single_setting=False, only_single_cam_trials=False, run_again=False)
    return hub.PipelineContext(paths=paths, args=Namespace(**{**base, **args}), df_settings=pd.DataFrame(),
                               df_trials=None)


def test_import_has_no_side_effects(hub, tmp_path):
    assert not (tmp_path / "iDrink").exists()


def test_help_exits_cleanly(hub):
    with pytest.raises(SystemExit) as e:
        hub.parser.parse_args(["--help"])
    assert e.value.code == 0


def test_prepare_args(hub, tmp_path):
    ctx = make_ctx(hub, tmp_path)
    a = hub.parser.parse_args(["--stages", "opensim,murphy_measures", "--patients", "P01,P02", "--poseback", "mmpose,pose2sim"])
    a = hub.prepare_args(a, ctx.paths)
    assert a.mode == ["opensim", "murphy_measures"]
    assert a.single_patient == ["P01", "P02"] and a.single_setting is False
    assert a.poseback == ["mmpose", "pose2sim"]
    with pytest.raises(SystemExit):
        hub.prepare_args(hub.parser.parse_args(["--stages", "bogus"]), ctx.paths)


def test_setup_creates_dirs_and_loads_csvs(hub, tmp_path, monkeypatch):
    monkeypatch.setenv("IDRINK_ROOT", str(tmp_path / "iDrink"))
    monkeypatch.setattr(hub.iDrinkConfig, "SRC_DIR", tmp_path)  # ignore a local path.txt / config.toml
    logs = tmp_path / "iDrink" / "validation_root" / "05_logs"
    logs.mkdir(parents=True)
    (logs / "validation_settings.csv").write_text("setting_id;cam_setting\n1;a\n")
    ctx = hub.setup(tmp_path / "none.toml")
    assert ctx.df_trials is None
    assert list(ctx.df_settings["setting_id"]) == [1]
    assert (tmp_path / "iDrink" / "validation_root" / "03_data").is_dir()


class FakeTrial:
    def __init__(self, i):
        self.identifier, self.id_p, self.id_t, self.id_s = f"S001_P01_T{i}", "P01", f"T{i}", "S001"
        self.MMPose_done = False
        self.HPE_done = False


def test_run_HPE_skips_done_runs_rest_and_logs_errors(hub, tmp_path):
    ctx = make_ctx(hub, tmp_path)
    ctx.df_trials = pd.DataFrame({"id_t": ["T1", "T2", "T3"], "id_p": ["P01"] * 3,
                                  "MMPose_done": [True, False, False], "n_frames": [100, 0, 0]})
    trials = [FakeTrial(1), FakeTrial(2), FakeTrial(3)]
    done_ids = {"S001_P01_T1"}
    hub.iDrinkLog.does_HPE_zip_exist.side_effect = lambda t, root, posebacks: t.identifier in done_ids
    hub.iDrinkLog.update_trial_csv.return_value = "df"

    ran = []

    def backend(ctx_, trial):
        if trial.identifier.endswith("T3"):
            raise RuntimeError("boom")
        ran.append(trial.identifier)
        trial.MMPose_done = True

    entry = hub.HPE_BACKENDS["mmpose"][:4] + (backend,)
    with mock.patch.dict(hub.HPE_BACKENDS, {"mmpose": entry}):
        hub.run_HPE(ctx, trials)

    assert ran == ["S001_P01_T2"]
    assert trials[0].MMPose_done is True      # taken over from the zip check
    assert trials[0].n_frames == 100          # frame count taken over from the trials csv
    assert trials[1].MMPose_done is True
    assert trials[2].MMPose_done is False     # failed
    assert all(t.pose_model == "Coco17_UpperBody" for t in trials)
    hub.iDrinkLog.log_error.assert_called_once()
    assert hub.iDrinkLog.log_error.call_args.args[4] == "mmpose"
    assert ctx.df_trials == "df"


def test_run_HPE_unknown_backend_exits(hub, tmp_path):
    ctx = make_ctx(hub, tmp_path, poseback=["nope"])
    with pytest.raises(SystemExit):
        hub.run_HPE(ctx, [])


def test_run_mode_full_runs_steps_in_order(hub, tmp_path):
    ctx = make_ctx(hub, tmp_path, mode=["full"])
    calls = []
    trials = [FakeTrial(2), FakeTrial(1)]
    with mock.patch.object(hub, "create_trial_objects", return_value=trials), \
            mock.patch.object(hub, "run_statistics", lambda c: calls.append("statistics")), \
            mock.patch.object(hub, "run_calibrations", lambda c, t: calls.append("calibration") or "df_cal"), \
            mock.patch.dict(hub.STEPS, {k: (lambda k: lambda c, t: calls.append(k))(k)
                                        for k in ["hpe", "triangulation", "opensim", "murphy_measures"]}):
        # STEPS holds the original functions, patch the ones run_mode looks up
        hub.run_mode(ctx)
    assert calls == ["calibration", "hpe", "triangulation", "opensim", "murphy_measures", "statistics"]
    assert ctx.df_trials == "df_cal"


def opensim_trial(i, tmp_path):
    t = FakeTrial(i)
    t.dir_trial, t.used_cams, t.OS_done = str(tmp_path / f"t{i}"), [1], False
    return t


def test_run_opensim_serial_runs_only_missing_and_logs_failures(hub, tmp_path):
    ctx = make_ctx(hub, tmp_path)
    hub.iDrinkConfig.ensure_dirs(ctx.paths)
    ctx.df_settings = pd.DataFrame({"setting_id": [1], "pose_estimation": ["mmpose"]})
    trials = [opensim_trial(i, tmp_path) for i in (1, 2, 3)]
    # trial 1 already has its outputs
    hub.iDrinkLog.files_exist.side_effect = lambda d, ext, *a: "t1" in d
    hub.iDrinkLog.update_trial_csv.return_value = "df"

    def fake_run(job):
        trial = job[0]
        if trial.identifier.endswith("T3"):
            raise RuntimeError("opensim failed")
        return trial

    hub.iDrinkOpenSim.run_opensim_trial = fake_run
    hub.run_opensim(ctx, trials)

    assert [t.OS_done for t in trials] == [True, True, False]
    hub.iDrinkLog.log_error.assert_called_once()
    assert hub.iDrinkLog.log_error.call_args.args[3] == "Opensim"
    assert ctx.df_trials == "df"


def test_run_murphy_skips_rows_already_in_csv_unless_run_again(hub, tmp_path):
    ctx = make_ctx(hub, tmp_path)
    csv = Path(ctx.path_csv_murphy_measures)
    csv.parent.mkdir(parents=True)
    pd.DataFrame({"identifier": ["S001_P01_T1"]}).to_csv(csv, sep=";", index=False)
    trials = []
    for i in (1, 2):
        t = opensim_trial(i, tmp_path)
        d = Path(t.dir_trial) / "movement_analysis"
        (d / "ik_tool").mkdir(parents=True)
        (d / "kin_opensim_analyzetool").mkdir(parents=True)
        (d / "ik_tool" / "x.csv").write_text("a")
        (d / "kin_opensim_analyzetool" / "xOutputsVec3.sto").write_text("a")
        trials.append(t)

    hub.run_murphy_measures(ctx, trials)
    ran = [c.kwargs["trial_id"] for c in hub.iDrinkMurphyMeasures.MurphyMeasures.call_args_list]
    assert ran == ["S001_P01_T2"]

    hub.iDrinkMurphyMeasures.MurphyMeasures.reset_mock()
    ctx.args.run_again = True
    hub.run_murphy_measures(ctx, trials)
    assert len(hub.iDrinkMurphyMeasures.MurphyMeasures.call_args_list) == 2


def test_jobs_flag_defaults_to_serial(hub):
    assert hub.parser.parse_args([]).jobs == 1
    assert hub.parser.parse_args(["--jobs", "4"]).jobs == 4
