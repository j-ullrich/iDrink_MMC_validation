import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Never pick up a developer's local src/config.toml or src/path.txt."""
    from iDrink import iDrinkConfig
    monkeypatch.setattr(iDrinkConfig, "SRC_DIR", tmp_path / "no_src")
    for var in ("IDRINK_ROOT", "IDRINK_MMC", "IDRINK_OMC", "IDRINK_CONFIG"):
        monkeypatch.delenv(var, raising=False)
