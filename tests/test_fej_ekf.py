from __future__ import annotations

import numpy as np

from diagnostics.covariance_health import check_covariance
from filters.ekf import ExtendedKalmanFilter


def _linear_f(x, u):
    return x.copy()


def _linear_h(x):
    return x[:1]


def _F_jac(x, u):
    return np.eye(x.size)


def _H_jac(x):
    H = np.zeros((1, x.size))
    H[0, 0] = 1.0
    return H


def test_x_lin_equals_x0_initially_and_tracks_its_own_trajectory():
    x0 = np.array([1.0, 2.0])
    filt = ExtendedKalmanFilter(
        _linear_f, _linear_h, Q=np.eye(2) * 0.01, R=np.eye(1) * 0.1, n=2, m=1, x0=x0, use_fej=True
    )
    assert np.allclose(filt.x_lin, x0)
    filt.predict()
    # f here is identity, so x_lin should still track it (unchanged since f=identity)
    assert np.allclose(filt.x_lin, x0)


def test_update_never_moves_x_lin():
    x0 = np.array([1.0, 2.0])
    filt = ExtendedKalmanFilter(
        _linear_f, _linear_h, Q=np.eye(2) * 0.01, R=np.eye(1) * 0.1, n=2, m=1, x0=x0, use_fej=True
    )
    filt.predict()
    x_lin_before = filt.x_lin.copy()
    filt.update(np.array([5.0]))
    assert np.allclose(filt.x_lin, x_lin_before)
    assert not np.allclose(filt.x, x0)  # running estimate DID move


def test_fej_and_standard_jacobians_agree_when_x_lin_equals_x():
    x0 = np.array([1.0, 2.0])
    filt_std = ExtendedKalmanFilter(_linear_f, _linear_h, Q=np.eye(2), R=np.eye(1), n=2, m=1, x0=x0, use_fej=False)
    filt_fej = ExtendedKalmanFilter(_linear_f, _linear_h, Q=np.eye(2), R=np.eye(1), n=2, m=1, x0=x0, use_fej=True)
    filt_std.predict()
    filt_fej.predict()
    filt_std.update(np.array([3.0]))
    filt_fej.update(np.array([3.0]))
    assert np.allclose(filt_std.x, filt_fej.x)
    assert np.allclose(filt_std.P, filt_fej.P)


def test_fej_and_standard_diverge_once_x_lin_and_x_differ():
    """A nonlinear H means the Jacobian genuinely differs when evaluated at
    x_lin (frozen) vs the current (already-updated) x -- this is the whole
    point of FEJ, so once the two diverge the resulting posteriors must too.
    """

    def h_nonlinear(x):
        return np.array([x[0] ** 2])

    def H_jac_nonlinear(x):
        return np.array([[2.0 * x[0], 0.0]])

    x0 = np.array([2.0, 0.0])
    filt_std = ExtendedKalmanFilter(
        _linear_f, h_nonlinear, Q=np.eye(2) * 0.01, R=np.eye(1) * 0.5, n=2, m=1,
        H_jac=H_jac_nonlinear, x0=x0, use_fej=False
    )
    filt_fej = ExtendedKalmanFilter(
        _linear_f, h_nonlinear, Q=np.eye(2) * 0.01, R=np.eye(1) * 0.5, n=2, m=1,
        H_jac=H_jac_nonlinear, x0=x0, use_fej=True
    )
    for z in (5.0, 6.0, 4.5):
        filt_std.predict()
        filt_fej.predict()
        filt_std.update(np.array([z]))
        filt_fej.update(np.array([z]))
    # x_lin stayed at x0 throughout (f is identity), so H was evaluated at a
    # different point than the standard filter's continuously-updated x.
    assert not np.allclose(filt_std.x, filt_fej.x)


def test_augment_state_sets_x_lin_to_init_value():
    x0 = np.array([1.0, 2.0])
    filt = ExtendedKalmanFilter(_linear_f, _linear_h, Q=np.eye(2), R=np.eye(1), n=2, m=1, x0=x0, use_fej=True)
    filt.predict()
    filt.update(np.array([3.0]))
    landmark_init = np.array([10.0, -5.0, 0.0])
    filt.augment_state(landmark_init, P_init_block=np.eye(3) * 4.0)
    assert filt.n == 5
    assert filt.x.shape == (5,)
    assert np.allclose(filt.x[2:], landmark_init)
    assert np.allclose(filt.x_lin[2:], landmark_init)
    assert filt.P.shape == (5, 5)
    assert np.allclose(filt.P[2:, 2:], np.eye(3) * 4.0)
    assert np.allclose(filt.P[:2, 2:], 0.0)


def test_augment_state_with_cross_covariance():
    x0 = np.array([1.0, 2.0])
    filt = ExtendedKalmanFilter(_linear_f, _linear_h, Q=np.eye(2), R=np.eye(1), n=2, m=1, x0=x0)
    cross = np.array([[0.5, 0.25]])  # shape (k=1, n=2)
    filt.augment_state(np.array([7.0]), P_init_block=np.eye(1) * 2.0, cross_cov=cross)
    assert filt.P.shape == (3, 3)
    assert np.allclose(filt.P[2, :2], cross[0])
    assert np.allclose(filt.P[:2, 2], cross[0])
    health = check_covariance(filt.P)
    assert health["symmetric"]


def test_marginalize_state_removes_indices():
    x0 = np.array([1.0, 2.0, 3.0])
    filt = ExtendedKalmanFilter(_linear_f, _linear_h, Q=np.eye(3), R=np.eye(1), n=3, m=1, x0=x0, use_fej=True)
    filt.marginalize_state([1])
    assert filt.n == 2
    assert np.allclose(filt.x, [1.0, 3.0])
    assert np.allclose(filt.x_lin, [1.0, 3.0])
    assert filt.P.shape == (2, 2)
