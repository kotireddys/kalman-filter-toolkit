"""Adaptive Kalman filtering utilities."""

from __future__ import annotations

import numpy as np

from .kf import KalmanFilter


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _repair_covariance(matrix, min_eigenvalue=1e-12):
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


class SageHusaAdaptiveKF(KalmanFilter):
    def __init__(self, F, H, Q, R, B=None, x0=None, P0=None, b=0.95, adapt_q=True, adapt_r=True):
        super().__init__(F=F, H=H, Q=Q, R=R, B=B, x0=x0, P0=P0)
        self.b = float(b)
        self.adapt_q = bool(adapt_q)
        self.adapt_r = bool(adapt_r)
        self._P_prev = self.P.copy()

    def predict(self, u=None):
        self._P_prev = self.P.copy()
        return super().predict(u)

    def update(self, z):
        x_post, P_post = super().update(z)
        if self.adapt_r:
            r_hat = np.outer(self.innovation, self.innovation) - self.H @ self.P_prior @ self.H.T
            self.R = _repair_covariance(self.b * self.R + (1.0 - self.b) * r_hat)
            self.S = self.H @ self.P @ self.H.T + self.R
            self.S = _symmetrize(self.S)
        if self.adapt_q:
            innovation_term = self.K @ np.outer(self.innovation, self.innovation) @ self.K.T
            q_hat = P_post + innovation_term - self.F @ self._P_prev @ self.F.T
            self.Q = _repair_covariance(self.b * self.Q + (1.0 - self.b) * q_hat)
        return x_post, P_post


def mehra_adaptive_R(innovations, window_size=20, H=None, P_pred=None, min_eigenvalue=1e-12):
    """Mehra's innovation-based measurement noise estimate.

    Estimates R from the lag-0 sample autocovariance of a sliding window of
    innovations, C0 = (1/N) sum y_k y_k^T ~= H P H^T + R. If `H` and `P_pred`
    (either a single covariance or a per-sample sequence matching the window)
    are supplied, the theoretical H P H^T term is subtracted to recover R
    directly; otherwise C0 itself (an estimate of the innovation covariance
    S) is returned as a conservative fallback.
    """
    innovations = np.asarray(innovations, dtype=float)
    if innovations.ndim == 1:
        innovations = innovations[:, None]
    window = innovations[-int(window_size) :] if innovations.shape[0] >= 2 else innovations
    n = window.shape[0]
    c0 = (window.T @ window) / max(n, 1)

    if H is not None and P_pred is not None:
        H = np.atleast_2d(np.asarray(H, dtype=float))
        P_pred = np.asarray(P_pred, dtype=float)
        if P_pred.ndim == 3:
            hph = np.mean([H @ P @ H.T for P in P_pred[-n:]], axis=0)
        else:
            hph = H @ P_pred @ H.T
        c0 = c0 - hph

    return _repair_covariance(c0, min_eigenvalue=min_eigenvalue)
