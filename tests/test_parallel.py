import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pd = pytest.importorskip("pandas")
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from iDrink.iDrinkParallel import run_parallel  # noqa: E402


def square(x):
    if x == 3:
        raise ValueError("three")
    return x * x


def pid(_):
    return os.getpid()


def test_serial_keeps_order_and_captures_errors():
    res = run_parallel(square, [1, 2, 3, 4], jobs=1)
    assert [r.item for r in res] == [1, 2, 3, 4]
    assert [r.value for r in res if r.ok] == [1, 4, 16]
    assert isinstance(res[2].error, ValueError) and not res[2].ok
    assert not getattr(res[2].error, "__notes__", None)  # serial keeps the real traceback instead


def test_parallel_matches_serial_and_runs_in_other_processes():
    items = list(range(8))
    serial = run_parallel(square, items, jobs=1)
    par = run_parallel(square, items, jobs=2)
    assert [(r.item, r.ok, r.value) for r in par] == [(r.item, r.ok, r.value) for r in serial]
    assert str(par[3].error) == "three"
    assert any("Traceback" in n for n in par[3].error.__notes__)
    assert set(r.value for r in run_parallel(pid, items, jobs=2)) != {os.getpid()}


def test_progress_called_per_item():
    calls = []
    run_parallel(square, [1, 2], jobs=1, progress=lambda: calls.append(1))
    run_parallel(square, [1, 2], jobs=2, progress=lambda: calls.append(1))
    assert len(calls) == 4


def test_unpicklable_items_give_clear_error():
    with pytest.raises(RuntimeError, match="--jobs 1"):
        run_parallel(square, [lambda: 1, lambda: 2], jobs=2)


def test_log_error_keeps_worker_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "tqdm", sys.modules.get("tqdm") or mock.MagicMock())
    from iDrink import iDrinkLog
    err = run_parallel(square, [3, 4], jobs=2)[0].error
    csv = tmp_path / "errors.csv"
    iDrinkLog.log_error(Namespace(), SimpleNamespace(identifier="S001_P01_T001", used_cams=[1]), err, "Opensim", "p",
                        str(csv))
    text = (tmp_path / "errors_tracebacks.log").read_text()
    assert "ValueError" in text and "square" in text


def test_merge_measures_updates_and_appends(tmp_path, monkeypatch):
    for name in ["opensim", "scipy", "scipy.signal", "scipy.stats", "plotly", "sympy", "sympy.logic",
                 "sympy.logic.algorithms", "sympy.logic.algorithms.dpll", "tqdm", "fuzzywuzzy", "fuzzywuzzy.fuzz", "scipy.optimize",
                 "fuzzywuzzy.process", "matplotlib", "matplotlib.pyplot", "statsmodels", "statsmodels.api",
                 "openpyxl", "toml", "cv2", "trc", "plotly.express", "plotly.graph_objects", "plotly.io"]:
        try:
            __import__(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    monkeypatch.delitem(sys.modules, "iDrink.iDrinkMurphyMeasures", raising=False)
    monkeypatch.setitem(sys.modules, "iDrink.iDrinkTrial", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "iDrink.iDrinkStatisticalAnalysis", mock.MagicMock())
    from iDrink import iDrinkMurphyMeasures as mm

    csv = tmp_path / "murphy.csv"
    row = lambda ident, v: {c: (ident if c == "identifier" else v) for c in mm.col_measures}  # noqa: E731
    mm.merge_measures(str(csv), [row("A", 1), row("B", 2)])
    df = mm.merge_measures(str(csv), [row("B", 20), row("C", 3)])
    assert list(df["identifier"]) == ["A", "B", "C"]
    assert df.loc[df["identifier"] == "B", "PeakVelocity_mms"].iloc[0] == 20
    assert pd.read_csv(csv, sep=";").shape[0] == 3
