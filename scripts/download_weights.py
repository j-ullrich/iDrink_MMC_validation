#!/usr/bin/env python3
"""
Downloads the Metrabs PyTorch model used by the pipeline.

    python scripts/download_weights.py [--models-dir DIR] [--url URL] [--dry-run]

The model is extracted to <models-dir>/pytorch/metrabs_eff2l_384px_800k_28ds_pytorch. By default <models-dir> is
<root_iDrink>/validation_root/06_metrabs_models (see src/iDrink/iDrinkConfig.py).

The YOLOv8 person detector (yolov8m.pt) needs no download here: ultralytics fetches it on first use.
Standard library only, so it also runs before the conda environment exists.
"""
import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MODEL_NAME = "metrabs_eff2l_384px_800k_28ds_pytorch"
# Link from the README of https://github.com/isarandi/metrabs
DEFAULT_URL = "https://bit.ly/metrabs_l_pt"
REQUIRED_FILES = ["ckpt.pt", "config.yaml", "joint_info.npz", "skeleton_infos.pkl", "joint_transform_matrix.npy"]


def default_models_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from iDrink import iDrinkConfig
    return iDrinkConfig.load_config().metrabs_models_dir


def missing_files(model_dir):
    return [f for f in REQUIRED_FILES if not (Path(model_dir) / f).is_file()]


def safe_extract(archive, dest):
    """Extract a tar.gz, refusing members that would land outside dest."""
    dest = Path(dest).resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Refusing to extract {member.name!r}: outside of {dest}")
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing to extract link {member.name!r}")
        tar.extractall(dest)


def download(url, target):
    with urllib.request.urlopen(url) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out)


def fetch_model(models_dir, url=DEFAULT_URL, dry_run=False):
    """Returns the model directory. Does nothing if all required files are already there."""
    model_dir = Path(models_dir) / "pytorch" / MODEL_NAME
    if not missing_files(model_dir):
        print(f"Model already present: {model_dir}")
        return model_dir

    print(f"Downloading {url}\n  -> {model_dir}")
    if dry_run:
        return model_dir

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=model_dir.parent) as tmp:
        archive = Path(tmp) / "model.tar.gz"
        download(url, archive)
        extracted = Path(tmp) / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)
        candidates = [p for p in extracted.rglob("ckpt.pt")]
        if not candidates:
            raise RuntimeError("The archive does not contain ckpt.pt. Wrong --url?")
        source = candidates[0].parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.move(str(source), str(model_dir))

    missing = missing_files(model_dir)
    if missing:
        raise RuntimeError(f"Downloaded model is incomplete, missing: {missing}")
    print(f"Done: {model_dir}")
    return model_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", help="Default: <root_iDrink>/validation_root/06_metrabs_models")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Default: {DEFAULT_URL}")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be downloaded")
    args = parser.parse_args(argv)

    models_dir = args.models_dir or default_models_dir()
    try:
        fetch_model(models_dir, args.url, args.dry_run)
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}\nDownload the model manually (see the Metrabs README), extract it so that "
              f"{Path(models_dir) / 'pytorch' / MODEL_NAME / 'ckpt.pt'} exists, or pass --url / --models-dir.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
