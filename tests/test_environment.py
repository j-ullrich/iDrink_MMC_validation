"""environment.yml must declare every third-party package the code imports."""
import ast
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# import name -> package name(s) that provide it
ALIASES = {"cv2": {"opencv-python"}, "yaml": {"pyyaml"}, "PIL": {"pillow"}, "trc": {"trc-data-reader"},
           "tensorflow_hub": {"tensorflow-hub"}, "tensorflow_io": {"tensorflow-io"}, "mpl_interactions":
           {"mpl-interactions"}, "PyQt6": {"pyqt6"}, "hydra": {"hydra-core"}, "mpl_toolkits": {"matplotlib"},
           "onnxruntime": {"onnxruntime-gpu", "onnxruntime"}}
LOCAL = {"iDrink", "Pose2Sim", "Metrabs_PoseEstimation", "iDrinkHubValidation", "metrabs_common", "metrabs_pt_models",
         "metrabs_pytorch", "posepile", "utils", "Utilities", "metrabs_tf", "spu", "tfu", "init"}
# Imports that only exist in code that is not part of the pipeline (demos, converters, unused Pose2Sim parts)
IGNORED_DIRS = ("/metrabs/", "convert_model_from_tf", "/scripts/", "/utils/")
OPTIONAL = {"poseviz", "cameralib", "barecat", "smpl", "smplx", "numba", "cvxpy", "requests", "ezc3d", "boxlib",
            "pyransac3d", "rlemasklib", "transforms3d", "lz4", "msgpack_numpy", "jpeg4py", "pycocotools", "ffmpeg",
            "h5py", "aspset510", "bvhtoolbox", "cachetools", "fastdtw", "spacepy", "tensorflow_inputs",
            "mpl_interactions_unused"}


def declared():
    env = yaml.safe_load((ROOT / "environment.yml").read_text())
    names = set()
    for dep in env["dependencies"]:
        entries = dep["pip"] if isinstance(dep, dict) else [dep]
        for e in entries:
            if e.startswith("-"):
                continue
            names.add(re.split(r"[=<>\[ ]", e, maxsplit=1)[0].lower().replace("_", "-"))
    return env, names


def imported_third_party():
    std = set(sys.stdlib_module_names)
    out = {}
    pipeline_files = [p for p in SRC.rglob("*.py")
                      if not any(d in str(p) for d in IGNORED_DIRS) and "Utilities" not in p.parts]
    for p in pipeline_files:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods = [n.module.split(".")[0]]
            else:
                continue
            for m in mods:
                if m not in std and m not in LOCAL:
                    out.setdefault(m, set()).add(p.relative_to(SRC).as_posix())
    return out


def test_valid_and_portable():
    env, names = declared()
    assert env["name"] == "idrink"
    text = (ROOT / "environment.yml").read_text()
    # no conda build strings (pkg=1.2.3=py311h...) and no Windows only packages
    assert not re.search(r"^\s*- [\w.-]+=[^=\s]+=\S+", text, re.M)
    for bad in ("vs2015_runtime", "vc=", "comtypes", "_win-64", "win_inet_pton"):
        assert bad not in text


def test_every_import_is_declared():
    _, names = declared()
    missing = {}
    for mod, files in imported_third_party().items():
        if mod in OPTIONAL:
            continue
        provided = ALIASES.get(mod, {mod.lower().replace("_", "-")})
        if not (provided & names):
            missing[mod] = sorted(files)[:3]
    assert not missing, f"imported but not in environment.yml: {missing}"
