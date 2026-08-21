"""Orchestrates Monte Carlo runs: for each run, simulate ground truth and
sensors ONCE, then run every configured estimator against that SAME
`SensorStreams` object -- see benchmark/simulate.py's docstring and
tests/test_benchmark_harness.py for the byte-identical-streams guarantee
this depends on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE_DIR, load_stacked_results, save_stacked_results
from .config import BenchmarkConfig
from .estimators import ESTIMATOR_FACTORIES
from .metrics import RunTrace, compute_run_trace, stack_run_traces
from .simulate import SensorStreams, simulate_sensors
from .trajectory import GroundTruthTrajectory, generate_trajectory


@dataclass
class SingleRunResult:
    traj: GroundTruthTrajectory
    streams: SensorStreams
    traces: dict  # estimator name -> RunTrace


def _run_one_estimator(
    name: str, config: BenchmarkConfig, traj: GroundTruthTrajectory, streams: SensorStreams, init_seed: int
) -> RunTrace:
    init_rng = np.random.default_rng(init_seed)  # same seed for every estimator -> identical initial-error draws
    adapter = ESTIMATOR_FACTORIES[name](
        config.noise, config.init,
        traj.R[0], traj.v[0], traj.p[0], streams.true_gyro_bias[0], streams.true_accel_bias[0],
        init_rng,
    )

    n_steps = streams.gyro_meas.shape[0]
    R_est = np.zeros((n_steps + 1, 3, 3))
    v_est = np.zeros((n_steps + 1, 3))
    p_est = np.zeros((n_steps + 1, 3))
    bg_est = np.zeros((n_steps + 1, 3))
    ba_est = np.zeros((n_steps + 1, 3))
    P_canonical = np.zeros((n_steps + 1, 15, 15))
    R_est[0], v_est[0], p_est[0], bg_est[0], ba_est[0] = adapter.R, adapter.v, adapter.p, adapter.b_g, adapter.b_a
    P_canonical[0] = adapter.P_canonical

    predict_time_s = np.zeros(n_steps)
    gnss_innovations, gnss_S, update_time_s = [], [], []
    gnss_ptr = 0
    gnss_indices = streams.gnss_step_indices

    for k in range(n_steps):
        t0 = time.perf_counter()
        adapter.predict(streams.gyro_meas[k], streams.accel_meas[k], traj.dt)
        predict_time_s[k] = time.perf_counter() - t0

        step_idx = k + 1
        while gnss_ptr < gnss_indices.shape[0] and gnss_indices[gnss_ptr] == step_idx:
            t0 = time.perf_counter()
            innovation, S = adapter.update_gnss(streams.gnss_meas[gnss_ptr], np.eye(3) * config.noise.gnss_noise_std**2)
            update_time_s.append(time.perf_counter() - t0)
            gnss_innovations.append(innovation.copy())
            gnss_S.append(S.copy())
            gnss_ptr += 1

        R_est[step_idx], v_est[step_idx], p_est[step_idx] = adapter.R, adapter.v, adapter.p
        bg_est[step_idx], ba_est[step_idx] = adapter.b_g, adapter.b_a
        P_canonical[step_idx] = adapter.P_canonical

    return compute_run_trace(
        t=traj.t,
        R_est=R_est, v_est=v_est, p_est=p_est, bg_est=bg_est, ba_est=ba_est,
        R_true=traj.R, v_true=traj.v, p_true=traj.p,
        bg_true=streams.true_gyro_bias, ba_true=streams.true_accel_bias,
        P_canonical=P_canonical,
        gnss_innovations=gnss_innovations, gnss_S=gnss_S,
        predict_time_s=predict_time_s, update_time_s=np.array(update_time_s),
        divergence_threshold_m=config.divergence.position_error_threshold_m,
    )


def run_single(config: BenchmarkConfig, run_index: int) -> SingleRunResult:
    run_seed = config.seed + run_index
    traj = generate_trajectory(config.trajectory)
    sensor_rng = np.random.default_rng(run_seed)
    streams = simulate_sensors(traj, config.noise, config.sensors, sensor_rng)

    init_seed = config.seed + 1_000_000 + run_index  # shared across estimators, distinct from sensor stream
    traces = {name: _run_one_estimator(name, config, traj, streams, init_seed) for name in config.estimators}
    return SingleRunResult(traj=traj, streams=streams, traces=traces)


def run_monte_carlo(config: BenchmarkConfig, use_cache: bool = True, cache_dir=DEFAULT_CACHE_DIR) -> dict:
    """Runs config.n_runs Monte Carlo trials, returns per-estimator stacked
    arrays: {name: {"nees": (n_runs, n_steps), "nis_gnss": (n_runs, K), ...}}
    (see metrics.stack_run_traces for the full field list). Cached on disk
    keyed by config hash; re-simulates only if the config changed or
    use_cache=False.
    """
    if use_cache:
        cached = load_stacked_results(config, cache_dir)
        if cached is not None:
            return cached

    all_traces = {name: [] for name in config.estimators}
    for run_index in range(config.n_runs):
        result = run_single(config, run_index)
        for name in config.estimators:
            all_traces[name].append(result.traces[name])

    stacked = {name: stack_run_traces(traces, config.trajectory.dt) for name, traces in all_traces.items()}
    if use_cache:
        save_stacked_results(config, stacked, cache_dir)
    return stacked
