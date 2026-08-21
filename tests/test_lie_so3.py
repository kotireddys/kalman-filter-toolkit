from __future__ import annotations

import numpy as np

from filters.lie.so3 import exp_so3, left_jacobian, left_jacobian_inv, log_so3


def _random_phi(rng, max_angle):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, max_angle)
    return axis * angle


def test_exp_so3_produces_valid_rotation():
    rng = np.random.default_rng(0)
    for _ in range(200):
        phi = _random_phi(rng, 3.0)
        R = exp_so3(phi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_log_exp_round_trip_large_angles():
    rng = np.random.default_rng(1)
    # Stay clear of the theta = pi singularity (axis sign is ambiguous there);
    # 0.95*pi exercises the large-angle closed form without hitting it.
    for _ in range(500):
        phi = _random_phi(rng, 0.95 * np.pi)
        R = exp_so3(phi)
        phi_back = log_so3(R)
        assert np.allclose(exp_so3(phi_back), R, atol=1e-9)
        assert np.isclose(np.linalg.norm(phi_back), np.linalg.norm(phi), atol=1e-9)


def test_log_exp_round_trip_small_angles():
    rng = np.random.default_rng(2)
    for _ in range(200):
        phi = _random_phi(rng, 1e-5)
        R = exp_so3(phi)
        phi_back = log_so3(R)
        assert np.allclose(phi_back, phi, atol=1e-9)


def test_left_jacobian_inverse_identity():
    rng = np.random.default_rng(3)
    for _ in range(200):
        phi = _random_phi(rng, 0.95 * np.pi)
        Jl = left_jacobian(phi)
        Jl_inv = left_jacobian_inv(phi)
        assert np.allclose(Jl @ Jl_inv, np.eye(3), atol=1e-9)


def test_left_jacobian_matches_exp_derivative_identity():
    # Standard identity: exp(phi + dphi) ~= exp(phi) exp(Jr(dphi) dphi) for
    # small dphi is right-Jacobian; the left-Jacobian identity used here is
    # exp(phi + dphi) ~= exp(Jl(phi) dphi) exp(phi). Verify numerically.
    rng = np.random.default_rng(4)
    for _ in range(100):
        phi = _random_phi(rng, 2.0)
        dphi = rng.normal(scale=1e-6, size=3)
        lhs = exp_so3(phi + dphi)
        Jl = left_jacobian(phi)
        rhs = exp_so3(Jl @ dphi) @ exp_so3(phi)
        assert np.allclose(lhs, rhs, atol=1e-8)
