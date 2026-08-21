from __future__ import annotations

import numpy as np

from filters.lie.se23 import adjoint, exp, hat, inverse, log, make_state, split_state, vee


def _random_xi(rng, max_angle=0.95 * np.pi, max_translation=5.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, max_angle)
    phi = axis * angle
    nu = rng.uniform(-max_translation, max_translation, size=3)
    rho = rng.uniform(-max_translation, max_translation, size=3)
    return np.concatenate([phi, nu, rho])


def _random_state(rng):
    return exp(_random_xi(rng))


def test_hat_vee_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        xi = _random_xi(rng)
        assert np.allclose(vee(hat(xi)), xi, atol=1e-12)


def test_make_split_state_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        R = exp(_random_xi(rng))[:3, :3]
        v = rng.normal(size=3)
        p = rng.normal(size=3)
        X = make_state(R, v, p)
        R2, v2, p2 = split_state(X)
        assert np.allclose(R, R2) and np.allclose(v, v2) and np.allclose(p, p2)


def test_inverse_is_group_inverse():
    rng = np.random.default_rng(2)
    for _ in range(100):
        X = _random_state(rng)
        assert np.allclose(X @ inverse(X), np.eye(5), atol=1e-9)
        assert np.allclose(inverse(X) @ X, np.eye(5), atol=1e-9)


def test_exp_log_round_trip_large_errors():
    rng = np.random.default_rng(3)
    for _ in range(500):
        xi = _random_xi(rng)
        X = exp(xi)
        xi_back = log(X)
        assert np.allclose(exp(xi_back), X, atol=1e-8)
        # Also check the recovered tangent vector itself, not just its image
        # under exp, since exp is many-to-one only at the theta=pi boundary
        # which _random_xi stays clear of.
        assert np.allclose(xi_back, xi, atol=1e-7)


def test_exp_log_round_trip_small_errors():
    rng = np.random.default_rng(4)
    for _ in range(200):
        xi = _random_xi(rng, max_angle=1e-5, max_translation=1e-5)
        X = exp(xi)
        xi_back = log(X)
        assert np.allclose(xi_back, xi, atol=1e-9)


def test_adjoint_identity_against_direct_conjugation():
    """X exp(xi) X^-1 == exp(Ad_X xi), checked by direct matrix conjugation
    rather than assumed from the closed-form Adjoint formula alone.
    """
    rng = np.random.default_rng(5)
    for _ in range(200):
        X = _random_state(rng)
        xi = _random_xi(rng, max_angle=0.5, max_translation=1.0)
        lhs = X @ exp(xi) @ inverse(X)
        rhs = exp(adjoint(X) @ xi)
        assert np.allclose(lhs, rhs, atol=1e-8)


def test_adjoint_is_group_homomorphism_on_composition():
    rng = np.random.default_rng(6)
    for _ in range(50):
        X = _random_state(rng)
        Y = _random_state(rng)
        assert np.allclose(adjoint(X) @ adjoint(Y), adjoint(X @ Y), atol=1e-8)
