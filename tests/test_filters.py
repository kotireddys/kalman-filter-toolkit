from __future__ import annotations

import numpy as np

from diagnostics.covariance_health import check_covariance
from diagnostics.nis_nees import chi2_consistency_test, compute_nees, compute_nis
from filters.adaptive import SageHusaAdaptiveKF
from filters.ekf import ExtendedKalmanFilter
from filters.eskf import ErrorStateKalmanFilter
from filters.kf import KalmanFilter
from filters.sq_root import SquareRootKalmanFilter
from filters.ukf import UnscentedKalmanFilter
from noise.process_noise import pcwn_1d


def _constant_velocity_matrices(dt):
    f_1d = np.array([[1.0, dt], [0.0, 1.0]])
    q_1d = pcwn_1d(dt, sigma_a=0.2)
    f = np.block([[f_1d, np.zeros((2, 2))], [np.zeros((2, 2)), f_1d]])
    q = np.block([[q_1d, np.zeros((2, 2))], [np.zeros((2, 2)), q_1d]])
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    return f, q, h


def _simulate_constant_velocity(rng, steps, dt):
    f, q, h = _constant_velocity_matrices(dt)
    r = np.diag([4.0, 4.0])
    x_init = np.array([0.0, 1.0, 0.0, -0.25])
    x = x_init.copy()
    truth = []
    measurements = []
    for _ in range(steps):
        x = f @ x + rng.multivariate_normal(np.zeros(4), q)
        z = h @ x + rng.multivariate_normal(np.zeros(2), r)
        truth.append(x.copy())
        measurements.append(z)
    return x_init, np.asarray(truth), np.asarray(measurements), f, q, h, r


def test_kf_consistency_monte_carlo():
    rng = np.random.default_rng(42)
    runs = 50
    steps = 200
    burn_in = 50
    dt = 0.1
    nis_values = []
    nees_values = []
    run_mean_nis = []
    run_mean_nees = []
    for _ in range(runs):
        x_init, truth, measurements, f, q, h, r = _simulate_constant_velocity(rng, steps, dt)
        p0 = np.diag([25.0, 4.0, 25.0, 4.0])
        # x0 estimates the state *before* the first predict, consistent with
        # the predict-then-update loop below (predict() advances to truth[0]
        # before update() is fed measurements[0]).
        x0 = x_init + rng.multivariate_normal(np.zeros(4), p0)
        filter_ = KalmanFilter(f, h, q, r, x0=x0, P0=p0)
        run_nis = []
        run_nees = []
        for step, measurement in enumerate(measurements):
            filter_.predict()
            filter_.update(measurement)
            if step >= burn_in:
                nis = compute_nis(filter_.innovation, filter_.S)
                nees = compute_nees(truth[step], filter_.x, filter_.P)
                nis_values.append(nis)
                nees_values.append(nees)
                run_nis.append(nis)
                run_nees.append(nees)
        run_mean_nis.append(np.mean(run_nis))
        run_mean_nees.append(np.mean(run_nees))

    # Per-sample marginal chi-squared bounds hold regardless of the temporal
    # autocorrelation of NEES/NIS within a run, so this remains a valid check
    # on the pooled samples.
    nis_result = chi2_consistency_test(nis_values, dof=2)
    nees_result = chi2_consistency_test(nees_values, dof=4)
    assert nis_result["fraction_in_bounds"] > 0.8
    assert nees_result["fraction_in_bounds"] > 0.8

    # For the *average* consistency check, use the empirical spread of the
    # per-run means (runs are independent by construction) rather than a
    # chi2(N*dof)/N bound over pooled samples: NEES/NIS values within a run
    # are autocorrelated through the filter recursion, which would make that
    # naive bound far too tight and prone to false failures.
    run_mean_nis = np.asarray(run_mean_nis)
    run_mean_nees = np.asarray(run_mean_nees)
    nis_se = run_mean_nis.std(ddof=1) / np.sqrt(runs)
    nees_se = run_mean_nees.std(ddof=1) / np.sqrt(runs)
    assert abs(run_mean_nis.mean() - 2.0) < 4.0 * nis_se
    assert abs(run_mean_nees.mean() - 4.0) < 4.0 * nees_se


def test_ekf_matches_kf_on_linear_system():
    rng = np.random.default_rng(7)
    dt = 0.1
    f, q, h = _constant_velocity_matrices(dt)
    r = np.diag([2.0, 2.0])

    def f_fun(x, u):
        return f @ x

    def h_fun(x):
        return h @ x

    x0 = rng.normal(size=4)
    p0 = np.eye(4)
    measurement = np.array([1.0, -1.0])
    kf = KalmanFilter(f, h, q, r, x0=x0, P0=p0)
    ekf = ExtendedKalmanFilter(f_fun, h_fun, q, r, n=4, m=2, x0=x0, P0=p0)
    kf.predict()
    ekf.predict()
    kf.update(measurement)
    ekf.update(measurement)
    assert np.allclose(kf.x, ekf.x, atol=1e-10)
    assert np.allclose(kf.P, ekf.P, atol=1e-10)


def test_ukf_matches_kf_on_linear_system():
    dt = 0.1
    f, q, h = _constant_velocity_matrices(dt)
    r = np.diag([2.0, 2.0])

    def f_fun(x, u):
        return f @ x

    def h_fun(x):
        return h @ x

    x0 = np.array([0.5, 0.25, -0.5, 0.1])
    p0 = np.eye(4)
    measurement = np.array([0.1, -0.2])
    kf = KalmanFilter(f, h, q, r, x0=x0, P0=p0)
    ukf = UnscentedKalmanFilter(f_fun, h_fun, q, r, n=4, m=2, x0=x0, P0=p0)
    kf.predict()
    ukf.predict()
    kf.update(measurement)
    ukf.update(measurement)
    assert np.allclose(kf.x, ukf.x, atol=1e-8)
    assert np.allclose(kf.P, ukf.P, atol=1e-8)


def test_covariance_health_after_many_steps():
    dt = 0.1
    f, q, h = _constant_velocity_matrices(dt)
    r = np.diag([2.0, 2.0])
    filter_ = KalmanFilter(f, h, q, r, x0=np.zeros(4), P0=np.eye(4) * 10.0)
    for step in range(1000):
        filter_.predict()
        measurement = np.array([np.sin(step * dt), np.cos(step * dt)])
        filter_.update(measurement)
    health = check_covariance(filter_.P)
    assert health["healthy"]


def test_sqrt_filter_tracks_linear_system():
    dt = 0.1
    f, q, h = _constant_velocity_matrices(dt)
    r = np.diag([1.0, 1.0])
    filter_ = SquareRootKalmanFilter(f, h, q, r, x0=np.zeros(4), P0=np.eye(4))
    filter_.predict()
    filter_.update(np.array([1.0, 2.0]))
    assert filter_.P.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(filter_.P) > 0.0)


def test_eskf_position_update_smoke():
    filter_ = ErrorStateKalmanFilter()
    filter_.predict(np.zeros(3), np.zeros(3), 0.01)
    filter_.update_position(np.array([1.0, 2.0, 3.0]), np.eye(3))
    assert filter_.state_vector.shape == (16,)
    assert np.all(np.isfinite(filter_.state_vector))


def test_adaptive_filter_updates_noise_matrices():
    dt = 0.1
    f, q, h = _constant_velocity_matrices(dt)
    r = np.diag([2.0, 2.0])
    filter_ = SageHusaAdaptiveKF(f, h, q, r, x0=np.zeros(4), P0=np.eye(4), b=0.9)
    filter_.predict()
    filter_.update(np.array([0.5, -0.5]))
    assert filter_.Q.shape == (4, 4)
    assert filter_.R.shape == (2, 2)
