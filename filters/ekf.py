"""Extended Kalman Filter with analytical or numerical Jacobians."""

from __future__ import annotations

import numpy as np


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _solve_gain(cov: np.ndarray, cross_cov: np.ndarray) -> np.ndarray:
    return np.linalg.solve(cov, cross_cov.T).T


def _numerical_jacobian(func, x: np.ndarray, eps: float, *args) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    baseline = np.asarray(func(x, *args), dtype=float).reshape(-1)
    jacobian = np.zeros((baseline.size, x.size), dtype=float)
    for index in range(x.size):
        perturbation = np.zeros_like(x)
        perturbation[index] = eps
        plus = np.asarray(func(x + perturbation, *args), dtype=float).reshape(-1)
        minus = np.asarray(func(x - perturbation, *args), dtype=float).reshape(-1)
        jacobian[:, index] = (plus - minus) / (2.0 * eps)
    return jacobian


class ExtendedKalmanFilter:
    def __init__(
        self,
        f,
        h,
        Q,
        R,
        n,
        m,
        F_jac=None,
        H_jac=None,
        x0=None,
        P0=None,
        jacobian_eps: float = 1e-7,
    ):
        self.f = f
        self.h = h
        self.F_jac = F_jac
        self.H_jac = H_jac
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.n = int(n)
        self.m = int(m)
        self.jacobian_eps = float(jacobian_eps)

        self.x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float).reshape(self.n)
        self.P = np.eye(self.n) if P0 is None else np.atleast_2d(np.asarray(P0, dtype=float))

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.innovation = np.zeros(self.m)
        self.S = np.eye(self.m)
        self.K = np.zeros((self.n, self.m))

    def _state_transition_jacobian(self, x, u):
        if self.F_jac is not None:
            return np.asarray(self.F_jac(x, u), dtype=float)
        return _numerical_jacobian(lambda state, control: self.f(state, control), x, self.jacobian_eps, u)

    def _measurement_jacobian(self, x):
        if self.H_jac is not None:
            return np.asarray(self.H_jac(x), dtype=float)
        return _numerical_jacobian(lambda state: self.h(state), x, self.jacobian_eps)

    def predict(self, u=None):
        self.x_prior = np.asarray(self.f(self.x, u), dtype=float).reshape(self.n)
        F = self._state_transition_jacobian(self.x, u)
        self.P_prior = F @ self.P @ F.T + self.Q
        self.P_prior = _symmetrize(self.P_prior)
        self.x = self.x_prior.copy()
        self.P = self.P_prior.copy()
        return self.x, self.P

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(self.m)
        H = self._measurement_jacobian(self.x)
        self.innovation = z - np.asarray(self.h(self.x), dtype=float).reshape(self.m)
        self.S = H @ self.P @ H.T + self.R
        self.S = _symmetrize(self.S)
        self.K = _solve_gain(self.S, self.P @ H.T)
        self.x = self.x + self.K @ self.innovation
        identity = np.eye(self.n)
        I_KH = identity - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
        self.P = _symmetrize(self.P)
        return self.x, self.P

    def log_likelihood(self):
        sign, logdet = np.linalg.slogdet(self.S)
        if sign <= 0:
            return float("-inf")
        mahal = self.innovation @ np.linalg.solve(self.S, self.innovation)
        return float(-0.5 * (self.m * np.log(2.0 * np.pi) + logdet + mahal))
