"""Adapter layer: a uniform predict/update/state/covariance interface over
the four kernels being compared (EKF, FEJ-EKF, ESKF, InEKF), all running the
same problem -- IMU dead reckoning with intermittent GNSS position fixes --
so their outputs are directly comparable.

Every adapter exposes its error-state covariance in the same canonical
15-dim order [phi(3), v(3), p(3), b_g(3), b_a(3)], reordering from whatever
native order the underlying kernel uses internally, so a single NEES
computation (benchmark/metrics.py) works uniformly across all four.
"""

from __future__ import annotations

import numpy as np

from filters.ekf import ExtendedKalmanFilter
from filters.eskf import ErrorStateKalmanFilter, quat_to_rot
from filters.inekf import InvariantEKF
from filters.lie.so3 import exp_so3, log_so3

from .config import InitConfig, NoiseConfig

GRAVITY = np.array([0.0, 0.0, -9.80665])


# ---------------------------------------------------------------------------
# Generic nonlinear INS model shared by the EKF / FEJ-EKF adapters:
# x = [phi(3), v(3), p(3), b_g(3), b_a(3)], measurement h(x) = p (GNSS).
# ---------------------------------------------------------------------------


def _f_ins(x, u):
    gyro_meas, accel_meas, dt = u
    phi, v, p, bg, ba = x[0:3], x[3:6], x[6:9], x[9:12], x[12:15]
    omega, a = gyro_meas - bg, accel_meas - ba
    R = exp_so3(phi)
    phi_new = log_so3(R @ exp_so3(omega * dt))
    v_new = v + (R @ a + GRAVITY) * dt
    p_new = p + v * dt
    return np.concatenate([phi_new, v_new, p_new, bg, ba])


def _h_gnss(x):
    return x[6:9]


def _H_jac_gnss(_x):
    H = np.zeros((3, 15))
    H[:, 6:9] = np.eye(3)
    return H


def _process_noise_ins(noise: NoiseConfig, dt: float) -> np.ndarray:
    """Simple diagonal Q for the generic INS model -- deliberately not as
    rigorously derived as InEKF's G Qc G^T construction (this model has no
    equivalent closed-form noise Jacobian ready to hand), but the right
    order of magnitude for a tuning parameter that the benchmark's own ANEES
    metric will judge the consistency of anyway.
    """
    return np.diag(
        [(noise.gyro_noise * dt) ** 2] * 3
        + [(noise.accel_noise * dt) ** 2] * 3
        + [1e-10] * 3
        + [noise.gyro_bias_noise**2 * dt] * 3
        + [noise.accel_bias_noise**2 * dt] * 3
    )


class GenericEkfAdapter:
    """Wraps ExtendedKalmanFilter with the INS model above."""

    def __init__(self, noise: NoiseConfig, init: InitConfig, R0, v0, p0, bg0, ba0, P0, use_fej: bool, rng):
        yaw_err = np.array([0.0, 0.0, np.deg2rad(init.yaw_error_deg)])
        R_est = R0 @ exp_so3(yaw_err)
        phi0 = log_so3(R_est)
        v_est = v0 + rng.normal(scale=init.velocity_error_std, size=3)
        p_est = p0 + rng.normal(scale=init.position_error_std, size=3)
        x0 = np.concatenate([phi0, v_est, p_est, bg0, ba0])
        self._noise = noise
        # F_jac intentionally omitted: falls back to ExtendedKalmanFilter's
        # own tested central-difference numerical Jacobian. An analytic
        # F_jac was tried and had a real bug in the bias-coupling block
        # (caught by checking it against finite differences before trusting
        # it -- see git history); given three such bugs already this
        # session, the numerical fallback is the safer choice here.
        self.filt = ExtendedKalmanFilter(
            _f_ins, _h_gnss, Q=np.eye(15), R=np.eye(3), n=15, m=3,
            H_jac=_H_jac_gnss, x0=x0, P0=P0, use_fej=use_fej,
        )

    def predict(self, gyro_meas, accel_meas, dt):
        self.filt.Q = _process_noise_ins(self._noise, dt)
        self.filt.predict((gyro_meas, accel_meas, dt))

    def update_gnss(self, z, R_meas):
        self.filt.R = R_meas
        self.filt.update(z)
        return self.filt.innovation, self.filt.S

    @property
    def R(self):
        return exp_so3(self.filt.x[0:3])

    @property
    def v(self):
        return self.filt.x[3:6]

    @property
    def p(self):
        return self.filt.x[6:9]

    @property
    def b_g(self):
        return self.filt.x[9:12]

    @property
    def b_a(self):
        return self.filt.x[12:15]

    @property
    def P_canonical(self):
        return self.filt.P  # already [phi, v, p, bg, ba]


class EskfAdapter:
    _CANONICAL_FROM_NATIVE = [6, 7, 8, 3, 4, 5, 0, 1, 2, 12, 13, 14, 9, 10, 11]
    # native ESKF error order: [p(0:3), v(3:6), theta(6:9), b_a(9:12), b_g(12:15)]
    # canonical order:         [phi,    v,      p,           b_g,      b_a      ]

    def __init__(self, noise: NoiseConfig, init: InitConfig, R0, v0, p0, bg0, ba0, P0_native, rng):
        yaw_err = np.array([0.0, 0.0, np.deg2rad(init.yaw_error_deg)])
        R_est = R0 @ exp_so3(yaw_err)
        q_est = _rot_to_quat(R_est)
        v_est = v0 + rng.normal(scale=init.velocity_error_std, size=3)
        p_est = p0 + rng.normal(scale=init.position_error_std, size=3)
        x0 = np.concatenate([p_est, v_est, q_est, ba0, bg0])
        self.filt = ErrorStateKalmanFilter(
            x0=x0, P0=P0_native,
            accel_noise=noise.accel_noise, gyro_noise=noise.gyro_noise,
            accel_bias_noise=noise.accel_bias_noise, gyro_bias_noise=noise.gyro_bias_noise,
        )

    def predict(self, gyro_meas, accel_meas, dt):
        self.filt.predict(accel_meas, gyro_meas, dt)

    def update_gnss(self, z, R_meas):
        self.filt.update_position(z, R_meas)
        return self.filt.innovation, self.filt.S

    @property
    def R(self):
        return quat_to_rot(self.filt.q)

    @property
    def v(self):
        return self.filt.v

    @property
    def p(self):
        return self.filt.p

    @property
    def b_g(self):
        return self.filt.b_g

    @property
    def b_a(self):
        return self.filt.b_a

    @property
    def P_canonical(self):
        idx = self._CANONICAL_FROM_NATIVE
        return self.filt.P[np.ix_(idx, idx)]


class InekfAdapter:
    def __init__(self, noise: NoiseConfig, init: InitConfig, R0, v0, p0, bg0, ba0, P0, rng):
        yaw_err = np.array([0.0, 0.0, np.deg2rad(init.yaw_error_deg)])
        R_est = R0 @ exp_so3(yaw_err)
        v_est = v0 + rng.normal(scale=init.velocity_error_std, size=3)
        p_est = p0 + rng.normal(scale=init.position_error_std, size=3)
        self.filt = InvariantEKF(
            R0=R_est, v0=v_est, p0=p_est, bg0=bg0, ba0=ba0, P0=P0,
            gyro_noise=noise.gyro_noise, accel_noise=noise.accel_noise,
            gyro_bias_noise=noise.gyro_bias_noise, accel_bias_noise=noise.accel_bias_noise,
            error_convention="left",
        )

    def predict(self, gyro_meas, accel_meas, dt):
        self.filt.predict(gyro_meas, accel_meas, dt)

    def update_gnss(self, z, R_meas):
        self.filt.update_position_world(z, R_meas)
        return self.filt.innovation, self.filt.S

    @property
    def R(self):
        return self.filt.R

    @property
    def v(self):
        return self.filt.v

    @property
    def p(self):
        return self.filt.p

    @property
    def b_g(self):
        return self.filt.b_g

    @property
    def b_a(self):
        return self.filt.b_a

    @property
    def P_canonical(self):
        return self.filt.P  # already [phi, v, p, bg, ba]


def _rot_to_quat(R):
    phi = log_so3(R)
    theta = np.linalg.norm(phi)
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = phi / theta
    return np.array([np.cos(theta / 2.0), *(axis * np.sin(theta / 2.0))])


def build_p0_canonical(init: InitConfig) -> np.ndarray:
    return np.diag(
        [np.deg2rad(init.attitude_p0_std_deg) ** 2] * 3
        + [init.velocity_p0_std**2] * 3
        + [init.position_p0_std**2] * 3
        + [init.bias_p0_std**2] * 6
    )


def build_p0_eskf_native(init: InitConfig) -> np.ndarray:
    p0 = build_p0_canonical(init)
    idx = EskfAdapter._CANONICAL_FROM_NATIVE
    native = np.zeros_like(p0)
    native[np.ix_(idx, idx)] = p0
    return native


ESTIMATOR_FACTORIES = {
    "ekf": lambda noise, init, R0, v0, p0, bg0, ba0, rng: GenericEkfAdapter(
        noise, init, R0, v0, p0, bg0, ba0, build_p0_canonical(init), use_fej=False, rng=rng
    ),
    "fej_ekf": lambda noise, init, R0, v0, p0, bg0, ba0, rng: GenericEkfAdapter(
        noise, init, R0, v0, p0, bg0, ba0, build_p0_canonical(init), use_fej=True, rng=rng
    ),
    "eskf": lambda noise, init, R0, v0, p0, bg0, ba0, rng: EskfAdapter(
        noise, init, R0, v0, p0, bg0, ba0, build_p0_eskf_native(init), rng
    ),
    "inekf": lambda noise, init, R0, v0, p0, bg0, ba0, rng: InekfAdapter(
        noise, init, R0, v0, p0, bg0, ba0, build_p0_canonical(init), rng
    ),
}
