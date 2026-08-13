"""Error-state Kalman filter for inertial navigation."""

from __future__ import annotations

import numpy as np


def skew(vector):
    vector = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]],
        dtype=float,
    )


def quat_mult(q1, q2):
    q1 = np.asarray(q1, dtype=float).reshape(4)
    q2 = np.asarray(q2, dtype=float).reshape(4)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conj(q):
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_normalize(q):
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_to_rot(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x**2 + y**2)],
        ],
        dtype=float,
    )


def small_angle_quat(delta_theta):
    delta_theta = np.asarray(delta_theta, dtype=float).reshape(3)
    angle = np.linalg.norm(delta_theta)
    if angle < 1e-12:
        small = np.array([1.0, *(0.5 * delta_theta)], dtype=float)
        return quat_normalize(small)
    axis = delta_theta / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=float)


def _delta_quaternion(omega, dt):
    delta_theta = np.asarray(omega, dtype=float).reshape(3) * float(dt)
    angle = np.linalg.norm(delta_theta)
    if angle < 1e-12:
        return small_angle_quat(delta_theta)
    axis = delta_theta / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=float)


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _repair_covariance(matrix, min_eigenvalue=1e-12):
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


class ErrorStateKalmanFilter:
    def __init__(
        self,
        x0=None,
        P0=None,
        accel_noise=0.2,
        gyro_noise=0.02,
        accel_bias_noise=1e-3,
        gyro_bias_noise=1e-4,
        gravity=None,
    ):
        self.p = np.zeros(3) if x0 is None else np.asarray(x0, dtype=float).reshape(16)[:3]
        if x0 is None:
            self.v = np.zeros(3)
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            self.b_a = np.zeros(3)
            self.b_g = np.zeros(3)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(16)
            self.p = x0[0:3]
            self.v = x0[3:6]
            self.q = quat_normalize(x0[6:10])
            self.b_a = x0[10:13]
            self.b_g = x0[13:16]
        self.P = np.eye(15) if P0 is None else np.atleast_2d(np.asarray(P0, dtype=float))
        self.accel_noise = float(accel_noise)
        self.gyro_noise = float(gyro_noise)
        self.accel_bias_noise = float(accel_bias_noise)
        self.gyro_bias_noise = float(gyro_bias_noise)
        if gravity is None:
            self.gravity = np.array([0.0, 0.0, -9.80665], dtype=float)
        else:
            self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.x_prior = self.state_vector.copy()
        self.P_prior = self.P.copy()
        self.innovation = np.zeros(1)
        self.S = np.eye(1)
        self.K = np.zeros((15, 1))

    @property
    def state_vector(self):
        return np.concatenate([self.p, self.v, self.q, self.b_a, self.b_g])

    def _process_noise(self, dt, rotation):
        dt = float(dt)
        g_acc = (self.accel_noise**2) * np.eye(3)
        g_gyro = (self.gyro_noise**2) * np.eye(3)
        g_ba = (self.accel_bias_noise**2) * np.eye(3)
        g_bg = (self.gyro_bias_noise**2) * np.eye(3)
        G = np.zeros((15, 12), dtype=float)
        G[0:3, 0:3] = 0.5 * rotation * dt**2
        G[3:6, 0:3] = rotation * dt
        G[6:9, 3:6] = np.eye(3) * dt
        G[9:12, 6:9] = np.eye(3) * dt
        G[12:15, 9:12] = np.eye(3) * dt
        Qc = np.block(
            [
                [g_acc, np.zeros((3, 9))],
                [np.zeros((3, 3)), g_gyro, np.zeros((3, 6))],
                [np.zeros((3, 6)), g_ba, np.zeros((3, 3))],
                [np.zeros((3, 9)), g_bg],
            ]
        )
        return G @ Qc @ G.T

    def predict(self, accel, gyro, dt):
        accel = np.asarray(accel, dtype=float).reshape(3)
        gyro = np.asarray(gyro, dtype=float).reshape(3)
        dt = float(dt)

        accel_unbiased = accel - self.b_a
        gyro_unbiased = gyro - self.b_g
        rotation = quat_to_rot(self.q)
        world_accel = rotation @ accel_unbiased + self.gravity

        self.p = self.p + self.v * dt + 0.5 * world_accel * dt**2
        self.v = self.v + world_accel * dt
        self.q = quat_normalize(quat_mult(self.q, _delta_quaternion(gyro_unbiased, dt)))

        F = np.eye(15, dtype=float)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -rotation @ skew(accel_unbiased) * dt
        F[3:6, 9:12] = -rotation * dt
        F[6:9, 6:9] = np.eye(3) - skew(gyro_unbiased) * dt
        F[6:9, 12:15] = -np.eye(3) * dt
        Qd = self._process_noise(dt, rotation)
        self.P_prior = _repair_covariance(F @ self.P @ F.T + Qd)
        self.P = self.P_prior.copy()
        self.x_prior = self.state_vector.copy()
        return self.state_vector, self.P

    def update(self, innovation, H, R):
        innovation = np.asarray(innovation, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        R = np.atleast_2d(np.asarray(R, dtype=float))
        self.innovation = innovation
        self.S = _repair_covariance(H @ self.P @ H.T + R)
        self.K = np.linalg.solve(self.S, (self.P @ H.T).T).T
        delta = self.K @ innovation
        identity = np.eye(15)
        I_KH = identity - self.K @ H
        self.P = _repair_covariance(I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T)

        self.p = self.p + delta[0:3]
        self.v = self.v + delta[3:6]
        self.q = quat_normalize(quat_mult(self.q, small_angle_quat(delta[6:9])))
        self.b_a = self.b_a + delta[9:12]
        self.b_g = self.b_g + delta[12:15]

        reset = np.eye(15)
        reset[6:9, 6:9] = np.eye(3) - 0.5 * skew(delta[6:9])
        self.P = _repair_covariance(reset @ self.P @ reset.T)
        return self.state_vector, self.P

    def update_position(self, z_pos, R_pos):
        z_pos = np.asarray(z_pos, dtype=float).reshape(3)
        H = np.zeros((3, 15), dtype=float)
        H[:, 0:3] = np.eye(3)
        innovation = z_pos - self.p
        return self.update(innovation, H, R_pos)
