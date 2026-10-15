import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("download_weights",
                                              Path(__file__).resolve().parents[1] / "scripts" / "download_weights.py")
dw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dw)


def make_archive(path, files=dw.REQUIRED_FILES, prefix=dw.MODEL_NAME):
    with tarfile.open(path, "w:gz") as tar:
        for name in files:
            data = b"x"
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_downloads_and_extracts(tmp_path):
    url = make_archive(tmp_path / "m.tar.gz").as_uri()
    model_dir = dw.fetch_model(tmp_path / "models", url)
    assert model_dir == tmp_path / "models" / "pytorch" / dw.MODEL_NAME
    assert dw.missing_files(model_dir) == []


def test_second_run_is_skipped(tmp_path, monkeypatch):
    url = make_archive(tmp_path / "m.tar.gz").as_uri()
    dw.fetch_model(tmp_path / "models", url)
    monkeypatch.setattr(dw, "download", lambda *a: pytest.fail("downloaded again"))
    dw.fetch_model(tmp_path / "models", url)


def test_dry_run_downloads_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(dw, "download", lambda *a: pytest.fail("downloaded"))
    dw.fetch_model(tmp_path / "models", "file:///nope", dry_run=True)
    assert not (tmp_path / "models").exists()


def test_incomplete_model_is_an_error(tmp_path):
    url = make_archive(tmp_path / "m.tar.gz", files=["ckpt.pt"]).as_uri()
    with pytest.raises(RuntimeError, match="incomplete"):
        dw.fetch_model(tmp_path / "models", url)


def test_archive_without_checkpoint_is_an_error(tmp_path):
    url = make_archive(tmp_path / "m.tar.gz", files=["README"]).as_uri()
    with pytest.raises(RuntimeError, match="ckpt.pt"):
        dw.fetch_model(tmp_path / "models", url)


def test_path_traversal_is_rejected(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="outside"):
        dw.safe_extract(evil, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_main_reports_failure_with_hint(tmp_path, capsys):
    code = dw.main(["--models-dir", str(tmp_path / "m"), "--url", (tmp_path / "missing.tar.gz").as_uri()])
    assert code == 1
    assert "manually" in capsys.readouterr().err
