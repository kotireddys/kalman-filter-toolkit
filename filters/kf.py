"""Linear Kalman filter with Joseph-form covariance updates."""

from __future__ import annotations

import numpy as np


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


class KalmanFilter:
    """Discrete linear Kalman filter.

    State transition: x_k = F x_{k-1} + B u_k + w_k,  w_k ~ N(0, Q)
    Measurement:       z_k = H x_k + v_k,              v_k ~ N(0, R)
    """

    def __init__(self, F, H, Q, R, B=None, x0=None, P0=None):
        self.F = np.atleast_2d(np.asarray(F, dtype=float))
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float)) if B is not None else None

        self.n = self.F.shape[0]
        self.m = self.H.shape[0]
        self.x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float).reshape(self.n)
        self.P = np.eye(self.n) if P0 is None else np.atleast_2d(np.asarray(P0, dtype=float))

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.innovation = np.zeros(self.m)
        self.S = np.eye(self.m)
        self.K = np.zeros((self.n, self.m))

    def predict(self, u=None):
        if u is not None and self.B is not None:
            self.x_prior = self.F @ self.x + self.B @ np.asarray(u, dtype=float)
        else:
            self.x_prior = self.F @ self.x
        self.P_prior = _symmetrize(self.F @ self.P @ self.F.T + self.Q)

        self.x = self.x_prior.copy()
        self.P = self.P_prior.copy()
        return self.x, self.P

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(self.m)
        self.innovation = z - self.H @ self.x
        self.S = _symmetrize(self.H @ self.P @ self.H.T + self.R)
        self.K = np.linalg.solve(self.S, (self.P @ self.H.T).T).T

        self.x = self.x + self.K @ self.innovation

        identity = np.eye(self.n)
        I_KH = identity - self.K @ self.H
        # Joseph-form covariance update: robust to non-optimal K, always PSD.
        self.P = _symmetrize(I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T)
        return self.x, self.P

    def log_likelihood(self):
        sign, logdet = np.linalg.slogdet(self.S)
        if sign <= 0:
            return float("-inf")
        mahal = self.innovation @ np.linalg.solve(self.S, self.innovation)
        return float(-0.5 * (self.m * np.log(2.0 * np.pi) + logdet + mahal))
