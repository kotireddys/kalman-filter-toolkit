"""Monte Carlo result caching, keyed on config hash so figure iteration
doesn't require re-simulation. Cache invalidates automatically whenever any
field of the config changes (the hash changes), never by manual bookkeeping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import BenchmarkConfig, config_hash

DEFAULT_CACHE_DIR = Path("results") / "cache"


def cache_path(config: BenchmarkConfig, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{config.name}_{config_hash(config)}.npz"


def save_stacked_results(config: BenchmarkConfig, stacked: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    path = cache_path(config, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for estimator_name, arrays in stacked.items():
        for key, value in arrays.items():
            flat[f"{estimator_name}__{key}"] = value
    np.savez_compressed(path, **flat)
    return path


def load_stacked_results(config: BenchmarkConfig, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict | None:
    path = cache_path(config, cache_dir)
    if not path.exists():
        return None
    with np.load(path) as data:
        estimator_names = sorted({key.split("__", 1)[0] for key in data.files})
        stacked = {
            name: {key.split("__", 1)[1]: data[key] for key in data.files if key.startswith(f"{name}__")}
            for name in estimator_names
        }
    return stacked
