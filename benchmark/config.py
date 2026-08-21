"""Config schema and YAML loading for the Monte Carlo consistency benchmark.

Every field has a default, so a YAML file only needs to override what it
cares about; `load_config` merges the given mapping onto these defaults
recursively, one dataclass level at a time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrajectoryConfig:
    kind: str = "benign"  # "benign" or "aggressive" -- see benchmark/trajectory.py
    duration_s: float = 20.0
    dt: float = 0.02  # IMU period; 50 Hz
    turn_rate_dps: float = 10.0  # peak commanded turn rate about each axis
    accel_amplitude: float = 1.0  # peak commanded body-frame specific force, m/s^2


@dataclass
class NoiseConfig:
    gyro_noise: float = 0.01  # rad/s, 1-sigma white noise
    accel_noise: float = 0.1  # m/s^2, 1-sigma white noise
    gyro_bias_noise: float = 1e-5  # rad/s/sqrt(s), bias random-walk PSD
    accel_bias_noise: float = 1e-4  # m/s^2/sqrt(s), bias random-walk PSD
    gyro_bias_init_std: float = 0.01  # rad/s, true bias drawn ~ N(0, this) per run
    accel_bias_init_std: float = 0.05  # m/s^2
    gnss_noise_std: float = 0.5  # m, per-axis


@dataclass
class SensorConfig:
    gnss_rate_hz: float = 5.0
    gnss_dropout_prob: float = 0.0  # iid per-sample dropout probability
    gnss_outage_windows: list = field(default_factory=list)  # list of [start_s, end_s]


@dataclass
class InitConfig:
    yaw_error_deg: float = 5.0  # true initial yaw error injected into the estimate
    position_error_std: float = 0.5  # m, true initial position error (per axis, drawn per run)
    velocity_error_std: float = 0.1  # m/s
    position_p0_std: float = 1.0  # m, filter's OWN initial uncertainty (P0 diagonal)
    velocity_p0_std: float = 0.5  # m/s
    attitude_p0_std_deg: float = 5.0
    bias_p0_std: float = 0.05  # shared units-agnostic std for gyro/accel bias P0 blocks


@dataclass
class DivergenceConfig:
    position_error_threshold_m: float = 50.0  # declared upfront, never post hoc


@dataclass
class BenchmarkConfig:
    name: str = "default"
    n_runs: int = 100
    seed: int = 0
    estimators: list = field(default_factory=lambda: ["ekf", "fej_ekf", "eskf", "inekf"])
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    init: InitConfig = field(default_factory=InitConfig)
    divergence: DivergenceConfig = field(default_factory=DivergenceConfig)


_NESTED = {
    "trajectory": TrajectoryConfig,
    "noise": NoiseConfig,
    "sensors": SensorConfig,
    "init": InitConfig,
    "divergence": DivergenceConfig,
}


def _merge_dataclass(cls, overrides: dict):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in _NESTED and f.name in overrides:
            kwargs[f.name] = _merge_dataclass(_NESTED[f.name], overrides[f.name])
        elif f.name in overrides:
            kwargs[f.name] = overrides[f.name]
    return cls(**kwargs)


def load_config(path: str | Path) -> BenchmarkConfig:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return _merge_dataclass(BenchmarkConfig, data)


def config_hash(config: BenchmarkConfig) -> str:
    """Stable hash of the full config, used as the Monte Carlo cache key."""
    canonical = json.dumps(dataclasses.asdict(config), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
