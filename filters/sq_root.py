"""Square-root Kalman filter using factorized covariance propagation."""

from __future__ import annotations

import numpy as np


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _repair_covariance(matrix, min_eigenvalue=1e-12):
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _sqrt_psd(matrix):
    matrix = _repair_covariance(matrix)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(np.sqrt(eigvals))


class SquareRootKalmanFilter:
    def __init__(self, F, H, Q, R, B=None, x0=None, P0=None):
        self.F = np.atleast_2d(np.asarray(F, dtype=float))
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float)) if B is not None else None

        self.n = self.F.shape[0]
        self.m = self.H.shape[0]
        self.x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float).reshape(self.n)
        covariance = np.eye(self.n) if P0 is None else np.atleast_2d(np.asarray(P0, dtype=float))
        self.S = _sqrt_psd(covariance)
        self.x_prior = self.x.copy()
        self.P_prior = covariance.copy()
        self.innovation = np.zeros(self.m)
        self.S_meas = np.eye(self.m)
        self.K = np.zeros((self.n, self.m))

    @property
    def P(self):
        return self.S @ self.S.T

    def predict(self, u=None):
        if u is not None and self.B is not None:
            self.x_prior = self.F @ self.x + self.B @ np.asarray(u, dtype=float)
        else:
            self.x_prior = self.F @ self.x
        process_root = _sqrt_psd(self.Q)
        stacked = np.hstack((self.F @ self.S, process_root))
        _, r_factor = np.linalg.qr(stacked.T, mode="reduced")
        # r_factor.T is already a valid lower-triangular square root of
        # F P F^T + Q by construction (QR preserves stacked @ stacked.T);
        # it must NOT be run through _repair_covariance, which assumes a
        # symmetric argument and would corrupt this triangular factor.
        self.S = r_factor.T
        self.x = self.x_prior.copy()
        self.P_prior = self.P.copy()
        return self.x, self.P

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(self.m)
        self.innovation = z - self.H @ self.x
        self.S_meas = _repair_covariance(self.H @ self.P @ self.H.T + self.R)
        self.K = np.linalg.solve(self.S_meas, (self.P @ self.H.T).T).T
        self.x = self.x + self.K @ self.innovation
        identity = np.eye(self.n)
        I_KH = identity - self.K @ self.H
        posterior = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
        self.S = _sqrt_psd(_repair_covariance(posterior))
        return self.x, self.P

    def log_likelihood(self):
        sign, logdet = np.linalg.slogdet(self.S_meas)
        if sign <= 0:
            return float("-inf")
        mahal = self.innovation @ np.linalg.solve(self.S_meas, self.innovation)
        return float(-0.5 * (self.m * np.log(2.0 * np.pi) + logdet + mahal))
