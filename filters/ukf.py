"""Unscented Kalman Filter with Merwe scaled sigma points."""

from __future__ import annotations

import numpy as np


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _repair_covariance(matrix: np.ndarray, min_eigenvalue: float = 1e-12) -> np.ndarray:
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = _repair_covariance(matrix)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(np.sqrt(eigvals))


class UnscentedKalmanFilter:
    def __init__(self, f, h, Q, R, n, m, alpha=1e-3, beta=2.0, kappa=0.0, x0=None, P0=None):
        self.f = f
        self.h = h
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.n = int(n)
        self.m = int(m)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float).reshape(self.n)
        self.P = np.eye(self.n) if P0 is None else np.atleast_2d(np.asarray(P0, dtype=float))

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.innovation = np.zeros(self.m)
        self.S = np.eye(self.m)
        self.K = np.zeros((self.n, self.m))
        self._wm, self._wc = self._compute_sigma_weights()

    def _compute_sigma_weights(self):
        lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        scale = self.n + lambda_
        wm = np.full(2 * self.n + 1, 1.0 / (2.0 * scale), dtype=float)
        wc = np.full(2 * self.n + 1, 1.0 / (2.0 * scale), dtype=float)
        wm[0] = lambda_ / scale
        wc[0] = lambda_ / scale + (1.0 - self.alpha**2 + self.beta)
        return wm, wc

    def _generate_sigma_points(self, x, P):
        lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        scale = self.n + lambda_
        sqrt = _sqrt_psd(scale * P)
        sigma_points = np.zeros((2 * self.n + 1, self.n), dtype=float)
        sigma_points[0] = x
        for index in range(self.n):
            sigma_points[index + 1] = x + sqrt[:, index]
            sigma_points[self.n + index + 1] = x - sqrt[:, index]
        return sigma_points

    def predict(self, u=None):
        sigma_points = self._generate_sigma_points(self.x, self.P)
        propagated = np.array([np.asarray(self.f(point, u), dtype=float).reshape(self.n) for point in sigma_points])
        self.x_prior = np.sum(self._wm[:, None] * propagated, axis=0)
        diff = propagated - self.x_prior
        self.P_prior = np.zeros((self.n, self.n), dtype=float)
        for index in range(propagated.shape[0]):
            self.P_prior += self._wc[index] * np.outer(diff[index], diff[index])
        self.P_prior += self.Q
        self.P_prior = _repair_covariance(self.P_prior)
        self.x = self.x_prior.copy()
        self.P = self.P_prior.copy()
        return self.x, self.P

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(self.m)
        sigma_points = self._generate_sigma_points(self.x, self.P)
        propagated = np.array([np.asarray(self.h(point), dtype=float).reshape(self.m) for point in sigma_points])
        z_pred = np.sum(self._wm[:, None] * propagated, axis=0)
        dz = propagated - z_pred
        dx = sigma_points - self.x

        Pzz = np.zeros((self.m, self.m), dtype=float)
        Pxz = np.zeros((self.n, self.m), dtype=float)
        for index in range(propagated.shape[0]):
            Pzz += self._wc[index] * np.outer(dz[index], dz[index])
            Pxz += self._wc[index] * np.outer(dx[index], dz[index])
        self.S = _repair_covariance(Pzz + self.R)
        self.innovation = z - z_pred
        self.K = np.linalg.solve(self.S, Pxz.T).T
        self.x = self.x + self.K @ self.innovation
        self.P = _repair_covariance(self.P - self.K @ self.S @ self.K.T)
        return self.x, self.P

    def log_likelihood(self):
        sign, logdet = np.linalg.slogdet(self.S)
        if sign <= 0:
            return float("-inf")
        mahal = self.innovation @ np.linalg.solve(self.S, self.innovation)
        return float(-0.5 * (self.m * np.log(2.0 * np.pi) + logdet + mahal))
