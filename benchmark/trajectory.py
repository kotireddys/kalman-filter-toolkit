"""Ground-truth trajectory generation with configurable excitation.

Observability of attitude/bias/position directions depends on how the
vehicle moves (see filters/ekf.py's FEJ docstring and tests/test_observability.py),
so this module provides two excitation levels:

- "benign": slow, mostly single-axis (yaw) turning with gentle forward
  acceleration -- weak excitation, closer to the regime where a standard
  EKF's inconsistency is least visible.
- "aggressive": faster, multi-axis (roll+pitch+yaw) turning with larger,
  oscillating specific force -- richer excitation, where FEJ-EKF/InEKF's
  consistency advantage over a standard EKF should be most visible.

Both are integrated with one RK4 step per sample (re-orthonormalizing R via
SVD afterward), the same scheme filters/inekf.py uses for its own nominal
mean propagation, for the same reason: simple, robust, and accurate enough
at typical IMU rates without needing a hand-derived closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TrajectoryConfig

GRAVITY = np.array([0.0, 0.0, -9.80665])


@dataclass
class GroundTruthTrajectory:
    t: np.ndarray  # (N+1,) time stamps, t[0]=0
    R: np.ndarray  # (N+1, 3, 3)
    v: np.ndarray  # (N+1, 3)
    p: np.ndarray  # (N+1, 3)
    gyro_true: np.ndarray  # (N, 3) body-frame angular velocity used on step k -> k+1
    accel_true: np.ndarray  # (N, 3) body-frame specific force used on step k -> k+1
    dt: float


def _angular_velocity(t: np.ndarray, config: TrajectoryConfig) -> np.ndarray:
    peak = np.deg2rad(config.turn_rate_dps)
    if config.kind == "benign":
        wz = peak * np.sin(2.0 * np.pi * 0.05 * t)
        wx = 0.05 * peak * np.sin(2.0 * np.pi * 0.02 * t)
        wy = 0.05 * peak * np.cos(2.0 * np.pi * 0.02 * t)
    elif config.kind == "aggressive":
        wz = peak * np.sin(2.0 * np.pi * 0.2 * t)
        wx = 0.6 * peak * np.sin(2.0 * np.pi * 0.15 * t + 1.0)
        wy = 0.6 * peak * np.cos(2.0 * np.pi * 0.17 * t + 0.5)
    else:
        raise ValueError(f"unknown trajectory kind: {config.kind!r}")
    return np.stack([wx, wy, wz], axis=-1)


def _specific_force(t: np.ndarray, config: TrajectoryConfig) -> np.ndarray:
    peak = config.accel_amplitude
    if config.kind == "benign":
        ax = peak * (1.0 + 0.2 * np.sin(2.0 * np.pi * 0.03 * t))
        ay = 0.1 * peak * np.cos(2.0 * np.pi * 0.03 * t)
        az = 0.05 * peak * np.sin(2.0 * np.pi * 0.02 * t)
    elif config.kind == "aggressive":
        ax = peak * (1.0 + 0.6 * np.sin(2.0 * np.pi * 0.1 * t))
        ay = 0.5 * peak * np.cos(2.0 * np.pi * 0.13 * t)
        az = 0.3 * peak * np.sin(2.0 * np.pi * 0.11 * t + 0.7)
    else:
        raise ValueError(f"unknown trajectory kind: {config.kind!r}")
    return np.stack([ax, ay, az], axis=-1)


def generate_trajectory(config: TrajectoryConfig) -> GroundTruthTrajectory:
    n_steps = int(round(config.duration_s / config.dt))
    t_samples = np.arange(n_steps) * config.dt  # inputs held constant over [t_k, t_k+dt)
    gyro_true = _angular_velocity(t_samples, config)
    accel_true = _specific_force(t_samples, config)

    t = np.zeros(n_steps + 1)
    R = np.zeros((n_steps + 1, 3, 3))
    v = np.zeros((n_steps + 1, 3))
    p = np.zeros((n_steps + 1, 3))
    R[0] = np.eye(3)
    v[0] = np.array([1.0, 0.0, 0.0])

    dt = config.dt
    for k in range(n_steps):
        Rk, vk, pk = R[k], v[k], p[k]
        omega, a = gyro_true[k], accel_true[k]

        def deriv(Rc, vc, pc, omega=omega, a=a):
            skew_omega = np.array(
                [[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]]
            )
            return Rc @ skew_omega, Rc @ a + GRAVITY, vc

        k1 = deriv(Rk, vk, pk)
        k2 = deriv(Rk + 0.5 * dt * k1[0], vk + 0.5 * dt * k1[1], pk + 0.5 * dt * k1[2])
        k3 = deriv(Rk + 0.5 * dt * k2[0], vk + 0.5 * dt * k2[1], pk + 0.5 * dt * k2[2])
        k4 = deriv(Rk + dt * k3[0], vk + dt * k3[1], pk + dt * k3[2])
        R_new = Rk + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v_new = vk + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        p_new = pk + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

        U, _, Vt = np.linalg.svd(R_new)
        if np.linalg.det(U @ Vt) < 0.0:
            U = U.copy()
            U[:, -1] *= -1.0
        R[k + 1] = U @ Vt
        v[k + 1] = v_new
        p[k + 1] = p_new
        t[k + 1] = t[k] + dt

    return GroundTruthTrajectory(t=t, R=R, v=v, p=p, gyro_true=gyro_true, accel_true=accel_true, dt=dt)
