import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pd = pytest.importorskip("pandas")
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def iDrinkLog(monkeypatch):
    try:
        import tqdm  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "tqdm", mock.MagicMock())
    monkeypatch.delitem(sys.modules, "iDrink.iDrinkLog", raising=False)
    from iDrink import iDrinkLog
    return iDrinkLog


def old_update_trial_csv(iDrinkLog, trial_list, csv_path, columns_to_add=None):
    """Reference: the original row by row implementation."""
    df_old = pd.read_csv(csv_path, sep=';')
    df = pd.DataFrame(columns=df_old.columns)
    for trial in trial_list:
        df = pd.concat([df, iDrinkLog.get_new_row(trial, df.columns).to_frame().T], axis=0, ignore_index=True)
        if columns_to_add is not None:
            df = pd.concat([df, iDrinkLog.get_new_row(trial, columns_to_add).to_frame().T], axis=0,
                           ignore_index=True)
    return df


def make_trials(n):
    return [SimpleNamespace(identifier=f"S001_P01_T{i}", id_p="P01", used_cams=[1, 2], MMPose_done=bool(i % 2),
                            n_frames=i * 10, extra=f"x{i}") for i in range(n)]


@pytest.mark.parametrize("n", [0, 1, 7])
@pytest.mark.parametrize("extra", [None, ["extra"]])
def test_update_trial_csv_matches_reference(iDrinkLog, tmp_path, n, extra):
    csv = tmp_path / "trials.csv"
    pd.DataFrame({"identifier": ["a"], "id_p": ["P"], "used_cams": ["[]"], "MMPose_done": [False],
                  "n_frames": [0]}).to_csv(csv, sep=';', index=False)
    trials = make_trials(n)
    expected = old_update_trial_csv(iDrinkLog, trials, str(csv), extra)
    got = iDrinkLog.update_trial_csv(trials, str(csv), columns_to_add=extra, verbose=0)
    # The extra column may be a string dtype instead of object depending on the pandas version. The csv is what counts.
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    assert got.to_csv(sep=';', index=False) == expected.to_csv(sep=';', index=False)
    assert (tmp_path / "trials_safety.csv").exists()
    assert pd.read_csv(csv, sep=';').shape[0] == expected.shape[0]


def test_model_cache_loads_once_and_evicts(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(empty_cache=lambda: None)
    torch.device = lambda d: mock.MagicMock()
    stubs = {n: mock.MagicMock() for n in ["simplepyutils", "metrabs_pytorch", "metrabs_pytorch.backbones",
                                           "metrabs_pytorch.backbones.efficientnet", "metrabs_pytorch.models",
                                           "metrabs_pytorch.models.metrabs", "posepile", "posepile.joint_info",
                                           "metrabs_pytorch.multiperson", "metrabs_pytorch.multiperson.multiperson_model",
                                           "metrabs_pytorch.util"]}
    for k, v in {"torch": torch, **stubs}.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.syspath_prepend(str(SRC / "Metrabs_PoseEstimation"))
    monkeypatch.delitem(sys.modules, "metrabs_pt_models", raising=False)
    import metrabs_pt_models as m

    loads = []

    class FakeModel:
        def cuda(self):
            return self

    monkeypatch.setattr(m, "load_multiperson_model", lambda p: loads.append(p) or FakeModel())
    a = m.get_model("/models/a")
    assert m.get_model("/models/a") is a
    assert loads == ["/models/a"]
    b = m.get_model("/models/b")
    assert b is not a and len(m._model_cache) == 1
    assert loads == ["/models/a", "/models/b"]


@pytest.mark.skipif(sys.platform == "win32", reason="posix only check")
@pytest.mark.parametrize("bad", ["E:\\iDrink", "C:/iDrink", "some\\thing"])
def test_windows_path_rejected_on_posix(tmp_path, bad):
    from iDrink import iDrinkConfig as cfg
    with pytest.raises(ValueError):
        cfg.load_config(tmp_path / "none.toml", env={"IDRINK_ROOT": bad})
