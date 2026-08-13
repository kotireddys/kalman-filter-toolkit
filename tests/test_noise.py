from __future__ import annotations

import numpy as np

from noise.allan_variance import compute_allan_deviation, fit_noise_params
from noise.measurement_noise import static_R_estimate
from noise.process_noise import pcwa_1d, pcwn_1d, psd_to_discrete_Q, van_loan


def test_allan_variance_white_noise_slope():
    rng = np.random.default_rng(123)
    data = rng.normal(scale=1.0, size=40000)
    taus, adev = compute_allan_deviation(data, fs=100.0)
    fit_slice = slice(0, min(8, len(taus)))
    slope, _intercept = np.polyfit(np.log10(taus[fit_slice]), np.log10(adev[fit_slice]), 1)
    assert -0.65 < slope < -0.35
    params = fit_noise_params(taus, adev)
    assert params["arw"] > 0.0
    assert params["bias_instability"] > 0.0
    assert params["rrw"] > 0.0


def test_pcwn_symmetry_and_positive_definite():
    q = pcwn_1d(0.1, sigma_a=0.2)
    assert np.allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) >= 0.0)


def test_pcwa_symmetry_and_positive_definite():
    q = pcwa_1d(0.1, sigma_j=0.2)
    assert np.allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) >= 0.0)


def test_van_loan_matches_euler_for_small_dt():
    dt = 1e-3
    f_c = np.array([[0.0, 1.0], [0.0, 0.0]])
    q_c = np.array([[0.0, 0.0], [0.0, 0.04]])
    f_d, q_d = van_loan(f_c, q_c, dt)
    q_euler = psd_to_discrete_Q(np.eye(2), q_c, dt)
    assert np.allclose(f_d, np.eye(2) + f_c * dt, atol=1e-6)
    assert np.allclose(q_d, q_euler, atol=1e-5)


def test_static_R_estimate_recovers_covariance():
    rng = np.random.default_rng(9)
    r_true = np.array([[2.0, 0.3], [0.3, 1.0]])
    samples = rng.multivariate_normal(np.zeros(2), r_true, size=10000)
    r_est = static_R_estimate(samples)
    assert np.allclose(r_est, r_true, atol=0.15)
