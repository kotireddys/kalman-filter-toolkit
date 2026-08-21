"""Ablation sweeps. Each function takes a base config, overrides one
parameter across a declared set of sweep points, runs the full Monte Carlo
harness at each point (cached, so re-plotting doesn't re-simulate), and
produces one figure -- per the deliverable's "each a config, each producing
a figure" requirement. Thresholds and swept ranges are declared here, in
code, rather than chosen post hoc after seeing results.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .cache import DEFAULT_CACHE_DIR
from .config import BenchmarkConfig
from .figures import plot_ablation_sweep
from .runner import run_monte_carlo

YAW_SWEEP_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
INIT_ERROR_SCALE = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
IMU_NOISE_SCALE = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GNSS_OUTAGE_S = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)


def _sweep(base_config: BenchmarkConfig, make_point_config, sweep_values, metric_key, cache_dir):
    results = {name: [] for name in base_config.estimators}
    for value in sweep_values:
        point_config = make_point_config(base_config, value)
        stacked = run_monte_carlo(point_config, cache_dir=cache_dir)
        for name in base_config.estimators:
            results[name].append(stacked[name][metric_key])
    return {name: np.array(values) for name, values in results.items()}  # (n_points, n_runs)


def run_yaw_sweep(base_config: BenchmarkConfig, out_dir: str | Path, cache_dir=DEFAULT_CACHE_DIR) -> Path:
    def make_point_config(cfg, yaw_deg):
        return dataclasses.replace(
            cfg, name=f"{cfg.name}_yaw{yaw_deg:g}",
            init=dataclasses.replace(cfg.init, yaw_error_deg=float(yaw_deg)),
        )

    results = _sweep(base_config, make_point_config, YAW_SWEEP_DEG, "ate_attitude_rmse_rad", cache_dir)
    return plot_ablation_sweep(
        list(YAW_SWEEP_DEG), results,
        ylabel="Attitude ATE (rad)", xlabel="Initial yaw error (deg)",
        out_path=Path(out_dir) / "ablation_yaw_sweep.png",
        title="Basin of attraction: initial yaw error",
    )


def run_init_error_sweep(base_config: BenchmarkConfig, out_dir: str | Path, cache_dir=DEFAULT_CACHE_DIR) -> Path:
    def make_point_config(cfg, scale):
        return dataclasses.replace(
            cfg, name=f"{cfg.name}_initscale{scale:g}",
            init=dataclasses.replace(
                cfg.init,
                position_error_std=cfg.init.position_error_std * scale,
                velocity_error_std=cfg.init.velocity_error_std * scale,
            ),
        )

    results = _sweep(base_config, make_point_config, INIT_ERROR_SCALE, "ate_position_rmse", cache_dir)
    return plot_ablation_sweep(
        list(INIT_ERROR_SCALE), results,
        ylabel="Position ATE (m)", xlabel="Initial position/velocity error scale (x base std)",
        out_path=Path(out_dir) / "ablation_init_error.png",
        title="Initial position/velocity error magnitude sweep",
    )


def run_imu_noise_sweep(base_config: BenchmarkConfig, out_dir: str | Path, cache_dir=DEFAULT_CACHE_DIR) -> Path:
    def make_point_config(cfg, scale):
        return dataclasses.replace(
            cfg, name=f"{cfg.name}_noisescale{scale:g}",
            noise=dataclasses.replace(
                cfg.noise,
                gyro_noise=cfg.noise.gyro_noise * scale,
                accel_noise=cfg.noise.accel_noise * scale,
            ),
        )

    results = _sweep(base_config, make_point_config, IMU_NOISE_SCALE, "ate_position_rmse", cache_dir)
    return plot_ablation_sweep(
        list(IMU_NOISE_SCALE), results,
        ylabel="Position ATE (m)", xlabel="IMU noise scale (x base std)",
        out_path=Path(out_dir) / "ablation_imu_noise.png",
        title="IMU noise scaling sweep",
    )


def run_gnss_outage_sweep(base_config: BenchmarkConfig, out_dir: str | Path, cache_dir=DEFAULT_CACHE_DIR) -> Path:
    def make_point_config(cfg, outage_s):
        mid = cfg.trajectory.duration_s / 2.0
        windows = [] if outage_s <= 0.0 else [[mid - outage_s / 2.0, mid + outage_s / 2.0]]
        return dataclasses.replace(
            cfg, name=f"{cfg.name}_outage{outage_s:g}",
            sensors=dataclasses.replace(cfg.sensors, gnss_outage_windows=windows),
        )

    results = _sweep(base_config, make_point_config, GNSS_OUTAGE_S, "ate_position_rmse", cache_dir)
    return plot_ablation_sweep(
        list(GNSS_OUTAGE_S), results,
        ylabel="Position ATE (m)", xlabel="GNSS outage window length (s)",
        out_path=Path(out_dir) / "ablation_gnss_outage.png",
        title="Extended GNSS outage sweep",
    )
