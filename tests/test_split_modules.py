"""The statistics/plots modules were split into packages. The old module names must still expose every function."""
import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OPTIONAL = ["tqdm", "scipy", "scipy.stats", "scipy.signal", "plotly", "plotly.io", "plotly.express",
            "plotly.graph_objects", "plotly.subplots", "statsmodels", "statsmodels.api", "opensim", "matplotlib",
            "matplotlib.pyplot", "cv2", "toml", "openpyxl", "aniposelib", "fuzzywuzzy", "importlib_metadata",
            "sklearn", "seaborn"]

STATS = ["delete_existing_files", "get_murphy_corrrelation", "get_mmc_omc_difference", "runs_statistics_discrete",
         "omc_trial_present", "get_omc_dirs", "get_omc_mmc_error_old", "get_error_timeseries", "get_error_mean_rmse",
         "get_rom_rmse_old", "get_rom_rmse", "get_timeseries_correlations", "get_multiple_correlations",
         "preprocess_timeseries", "get_offset_unfancy", "normalize_data", "murphy_measures"]
PLOTS = ["plot_murphy_blandaltman", "plot_murphy_blandaltman_old", "plot_murphy_error_rmse_box_bar_plot",
         "plot_timeseries_blandaltman_scale_location", "plot_timeseries_boxplot_error_rmse",
         "plot_timeseries_barplot_error_rmse", "plot_timeseries_averaged", "plot_measured_vs_errors",
         "plot_timeseries_RMSE", "plot_two_timeseries", "calibration_boxplot",
         "generate_plots_grouped_different_settings", "generate_plots_for_timeseries", "write_plottable_identifier",
         "get_plottable_timeseries_kinematics"]


@pytest.fixture
def stubbed(monkeypatch):
    for name in OPTIONAL:
        try:
            importlib.import_module(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, mock.MagicMock(name=name))
    for name in [m for m in sys.modules if m.startswith("iDrink.")]:
        monkeypatch.delitem(sys.modules, name)
    yield


@pytest.mark.parametrize("module,names", [("iDrink.iDrinkStatisticalAnalysis", STATS),
                                           ("iDrink.iDrinkValPlots", PLOTS)])
def test_old_module_exposes_all_names(stubbed, module, names):
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing


def test_plot_function_that_reads_global_verbose_can_import(stubbed):
    from iDrink.plots import timeseries
    assert timeseries.verbose == 1
