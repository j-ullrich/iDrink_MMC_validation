"""
Central configuration for the iDrink validation pipeline.

Paths are resolved in this order (first hit wins, per key):
    1. Environment variables IDRINK_ROOT, IDRINK_MMC, IDRINK_OMC
    2. config.toml  (path given explicitly, env IDRINK_CONFIG, or <src>/config.toml)
    3. legacy path.txt in <src> (``key;value`` lines)

Only root_iDrink is mandatory. root_MMC and root_OMC default to the folders used so far below root_iDrink.
Nothing in this module touches the file system unless ensure_dirs() is called.
"""
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent

_ENV_KEYS = {'root_iDrink': 'IDRINK_ROOT', 'root_MMC': 'IDRINK_MMC', 'root_OMC': 'IDRINK_OMC'}


@dataclass(frozen=True)
class Paths:
    root_iDrink: Path
    root_MMC: Path
    root_OMC: Path
    patients: tuple = ()
    settings: tuple = ()

    @property
    def root_val(self): return self.root_iDrink / "validation_root"
    @property
    def default_dir(self): return self.root_val / "01_default_files"
    @property
    def root_HPE(self): return self.root_val / "02_pose_estimation"
    @property
    def root_data(self): return self.root_val / "03_data"
    @property
    def root_stat(self): return self.root_val / "04_Statistics"
    @property
    def root_logs(self): return self.root_val / "05_logs"
    @property
    def metrabs_models_dir(self): return self.root_val / "06_metrabs_models"
    @property
    def log_val_settings(self): return self.root_logs / "validation_settings.csv"
    @property
    def log_val_trials(self): return self.root_logs / "validation_trials.csv"
    @property
    def log_val_errors(self): return self.root_logs / "validation_errors.csv"
    @property
    def log_calib(self): return self.root_logs / "calib_errors.csv"
    @property
    def log_calib_full(self): return self.root_logs / "calib_errors_full.csv"


def _read_toml(path):
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    return {**data.get('paths', {}), **{k: v for k, v in data.items() if not isinstance(v, dict)}}, data.get('run', {})


def _read_legacy_txt(path):
    values = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, sep, value = line.partition(';')
            if not sep or key not in _ENV_KEYS:
                raise ValueError(f"Unknown line in {path}: {line}")
            values[key] = value.strip()
    return values


def load_config(config_file=None, env=None):
    """Resolve the pipeline paths. Raises FileNotFoundError if root_iDrink cannot be determined."""
    env = os.environ if env is None else env
    values, run = {}, {}

    config_file = config_file or env.get('IDRINK_CONFIG') or SRC_DIR / 'config.toml'
    if Path(config_file).is_file():
        values, run = _read_toml(config_file)
    elif (SRC_DIR / 'path.txt').is_file():
        values = _read_legacy_txt(SRC_DIR / 'path.txt')

    for key, var in _ENV_KEYS.items():
        if env.get(var):
            values[key] = env[var]

    if not values.get('root_iDrink'):
        raise FileNotFoundError("root_iDrink is not set. Set IDRINK_ROOT, or create config.toml "
                                "(see config.example.toml).")

    if os.name == 'posix':
        for key in _ENV_KEYS:
            v = values.get(key)
            if v and (re.match(r"^[A-Za-z]:[\\/]", v) or '\\' in v):
                raise ValueError(f"{key} is a Windows path ({v!r}) but this is not Windows. Set IDRINK_ROOT / "
                                 f"IDRINK_MMC / IDRINK_OMC or config.toml for this machine.")

    root = Path(values['root_iDrink'])
    return Paths(root_iDrink=root,
                 root_MMC=Path(values['root_MMC']) if values.get('root_MMC') else root / "Delta" / "data_newStruc",
                 root_OMC=Path(values['root_OMC']) if values.get('root_OMC') else root / "OMC_data_newStruct",
                 patients=tuple(run.get('patients', ())),
                 settings=tuple(run.get('settings', ())))


def ensure_dirs(paths):
    """Create the directory tree (formerly done at import time in iDrinkHubValidation)."""
    for d in [paths.root_MMC, paths.root_OMC, paths.root_val, paths.default_dir, paths.root_HPE,
              paths.root_data, paths.root_stat, paths.root_logs]:
        d.mkdir(parents=True, exist_ok=True)


def rebase_path(path, root):
    """
    Map a path stored on another machine/OS onto the local root_iDrink.

    Everything after the last 'iDrink' path segment is re-attached to root.
    Paths that do not contain such a segment are returned unchanged.
    """
    parts = str(path).replace('\\', '/').split('/')
    if 'iDrink' not in parts:
        return str(path)
    idx = len(parts) - 1 - parts[::-1].index('iDrink')
    return str(Path(root).joinpath(*parts[idx + 1:]))
