"""Sensor stream simulation: noisy IMU + intermittent GNSS position fixes.

`simulate_sensors` is called exactly once per Monte Carlo run (see
benchmark/runner.py), and the resulting `SensorStreams` object is the one
passed to every estimator for that run -- this is what makes "every
estimator runs on byte-identical sensor streams" true by construction rather
than by convention; tests/test_benchmark_harness.py asserts it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NoiseConfig, SensorConfig
from .trajectory import GroundTruthTrajectory


@dataclass
class SensorStreams:
    gyro_meas: np.ndarray  # (N, 3)
    accel_meas: np.ndarray  # (N, 3)
    true_gyro_bias: np.ndarray  # (N+1, 3), random-walk trajectory
    true_accel_bias: np.ndarray  # (N+1, 3)
    gnss_step_indices: np.ndarray  # (K,) indices into trajectory steps 0..N with a GNSS fix
    gnss_meas: np.ndarray  # (K, 3)


def _random_walk_bias(
    n_steps: int, init_std: float, walk_noise_psd: float, dt: float, rng: np.random.Generator
) -> np.ndarray:
    bias = np.zeros((n_steps + 1, 3))
    bias[0] = rng.normal(scale=init_std, size=3)
    step_std = walk_noise_psd * np.sqrt(dt)
    for k in range(n_steps):
        bias[k + 1] = bias[k] + rng.normal(scale=step_std, size=3)
    return bias


def simulate_sensors(
    traj: GroundTruthTrajectory,
    noise: NoiseConfig,
    sensors: SensorConfig,
    rng: np.random.Generator,
) -> SensorStreams:
    n_steps = traj.gyro_true.shape[0]

    true_gyro_bias = _random_walk_bias(n_steps, noise.gyro_bias_init_std, noise.gyro_bias_noise, traj.dt, rng)
    true_accel_bias = _random_walk_bias(n_steps, noise.accel_bias_init_std, noise.accel_bias_noise, traj.dt, rng)

    gyro_meas = traj.gyro_true + true_gyro_bias[:-1] + rng.normal(scale=noise.gyro_noise, size=(n_steps, 3))
    accel_meas = traj.accel_true + true_accel_bias[:-1] + rng.normal(scale=noise.accel_noise, size=(n_steps, 3))

    gnss_period_steps = max(1, int(round(1.0 / (sensors.gnss_rate_hz * traj.dt))))
    # start from the first period, not step 0: step 0 is the initialization
    # instant before any predict() call, and the runner's update loop only
    # ever checks for a GNSS match at step indices >= 1, so an index-0
    # candidate would never be consumed (a stuck pointer, silently dropping
    # every GNSS update for the whole run -- caught by
    # tests/test_benchmark_harness.py::test_gnss_updates_actually_fire).
    candidate_indices = np.arange(gnss_period_steps, n_steps + 1, gnss_period_steps)

    keep = np.ones(candidate_indices.shape[0], dtype=bool)
    if sensors.gnss_dropout_prob > 0.0:
        keep &= rng.uniform(size=candidate_indices.shape[0]) >= sensors.gnss_dropout_prob
    for start_s, end_s in sensors.gnss_outage_windows:
        times = traj.t[candidate_indices]
        keep &= ~((times >= start_s) & (times < end_s))

    gnss_step_indices = candidate_indices[keep]
    gnss_meas = traj.p[gnss_step_indices] + rng.normal(scale=noise.gnss_noise_std, size=(gnss_step_indices.shape[0], 3))

    return SensorStreams(
        gyro_meas=gyro_meas,
        accel_meas=accel_meas,
        true_gyro_bias=true_gyro_bias,
        true_accel_bias=true_accel_bias,
        gnss_step_indices=gnss_step_indices,
        gnss_meas=gnss_meas,
    )
