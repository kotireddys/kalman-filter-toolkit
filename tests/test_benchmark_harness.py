from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

import benchmark.runner as runner_module
from benchmark.config import BenchmarkConfig, TrajectoryConfig
from benchmark.metrics import GNSS_DIM, STATE_DIM
from benchmark.runner import run_monte_carlo, run_single
from benchmark.simulate import simulate_sensors


def _tiny_config(**overrides):
    return BenchmarkConfig(
        n_runs=overrides.pop("n_runs", 4),
        trajectory=TrajectoryConfig(duration_s=overrides.pop("duration_s", 2.0), dt=0.02),
        **overrides,
    )


def test_sensor_streams_simulated_exactly_once_per_run():
    """Every estimator must run on the SAME SensorStreams object -- verified
    here by counting calls: simulate_sensors must be invoked exactly once
    per run_single call, not once per estimator (which would silently give
    each estimator its own independent noise realization).
    """
    config = _tiny_config()
    with mock.patch.object(runner_module, "simulate_sensors", wraps=simulate_sensors) as spy:
        run_single(config, run_index=0)
    assert spy.call_count == 1


def test_sensor_streams_are_deterministic_given_the_same_seed():
    config = _tiny_config()
    result1 = run_single(config, run_index=0)
    result2 = run_single(config, run_index=0)
    assert np.array_equal(result1.streams.gyro_meas, result2.streams.gyro_meas)
    assert np.array_equal(result1.streams.accel_meas, result2.streams.accel_meas)
    assert np.array_equal(result1.streams.gnss_meas, result2.streams.gnss_meas)


def test_gnss_updates_actually_fire():
    """Regression test: an earlier off-by-one in simulate.py's candidate GNSS
    index list (starting at step 0, which the update loop never revisits)
    silently dropped every GNSS update for the whole run.
    """
    config = _tiny_config()
    result = run_single(config, run_index=0)
    assert result.streams.gnss_step_indices.shape[0] > 0
    for name in config.estimators:
        assert result.traces[name].nis_gnss.shape[0] > 0


def test_chi2_dof_constants_match_state_and_measurement_dimension():
    assert STATE_DIM == 15  # phi(3) + v(3) + p(3) + b_g(3) + b_a(3)
    assert GNSS_DIM == 3


def test_caching_reproduces_identical_results():
    config = _tiny_config(name="cache_test")
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        stacked_fresh = run_monte_carlo(config, use_cache=True, cache_dir=cache_dir)
        stacked_cached = run_monte_carlo(config, use_cache=True, cache_dir=cache_dir)
        for name in config.estimators:
            assert np.array_equal(stacked_fresh[name]["nees"], stacked_cached[name]["nees"])
            assert np.array_equal(stacked_fresh[name]["ate_position_rmse"], stacked_cached[name]["ate_position_rmse"])


def test_cache_invalidates_on_config_change():
    config_a = _tiny_config(name="invalidation_test", seed=1)
    config_b = _tiny_config(name="invalidation_test", seed=2)
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        stacked_a = run_monte_carlo(config_a, use_cache=True, cache_dir=cache_dir)
        stacked_b = run_monte_carlo(config_b, use_cache=True, cache_dir=cache_dir)
        assert not np.array_equal(stacked_a["ekf"]["nees"], stacked_b["ekf"]["nees"])


def test_end_to_end_small_monte_carlo_all_estimators_finite():
    config = _tiny_config(n_runs=5)
    stacked = run_monte_carlo(config, use_cache=False)
    for name in config.estimators:
        arrays = stacked[name]
        assert arrays["nees"].shape[0] == 5
        assert np.all(np.isfinite(arrays["nees"]))
        assert np.all(np.isfinite(arrays["ate_position_rmse"]))
        assert arrays["diverged"].shape == (5,)
