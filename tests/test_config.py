import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from iDrink import iDrinkConfig as cfg


def test_env_overrides_file(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('[paths]\nroot_iDrink = "/from/file"\n[run]\npatients = ["P01"]\n')
    p = cfg.load_config(toml, env={"IDRINK_ROOT": "/from/env"})
    assert p.root_iDrink == Path("/from/env")
    assert p.patients == ("P01",)
    assert p.root_MMC == Path("/from/env/Delta/data_newStruc")
    assert p.log_val_trials == Path("/from/env/validation_root/05_logs/validation_trials.csv")


def test_missing_root_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "SRC_DIR", tmp_path)  # ignore a local legacy path.txt
    with pytest.raises(FileNotFoundError):
        cfg.load_config(tmp_path / "nope.toml", env={})


def test_ensure_dirs_creates_tree(tmp_path):
    p = cfg.load_config(tmp_path / "nope.toml", env={"IDRINK_ROOT": str(tmp_path / "iDrink")})
    cfg.ensure_dirs(p)
    assert p.root_logs.is_dir() and p.root_stat.is_dir()


def test_rebase_path_windows_to_posix():
    out = cfg.rebase_path(r"E:\iDrink\validation_root\03_data\x", "/home/u/iDrink")
    assert out == "/home/u/iDrink/validation_root/03_data/x"


def test_rebase_path_without_marker_unchanged():
    assert cfg.rebase_path("/some/other/path", "/r") == "/some/other/path"
