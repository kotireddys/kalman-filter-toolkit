"""Extended Kalman Filter with analytical or numerical Jacobians.

First-Estimates Jacobians (FEJ)
-------------------------------
`use_fej=True` turns on First-Estimates Jacobians: a linearization-point
store (`self.x_lin`), separate from the running estimate `self.x`, that both
`F_jac` and `H_jac` are evaluated at instead of `self.x`. `x_lin` is
propagated forward through the same dynamics `f` every predict() call (so it
tracks a "first-estimates trajectory"), but is *never* touched by update()
-- only the running estimate `self.x` is corrected by measurements. This is
deliberately a single flag on this class, not a separate class, so that
comparing standard EKF vs FEJ-EKF is exactly a one-variable ablation.

The failure this fixes: in a standard EKF, `F_jac`/`H_jac` are evaluated at
whatever the *current* estimate happens to be, and for any state element
that is revisited multiple times (an IMU bias touched by every predict, a
landmark touched by every sighting), that current estimate keeps moving as
updates correct it -- so different observations of the same underlying
geometry get linearized around different points. This inconsistency across
time makes the observability matrix of the *linearized* system have a
*smaller* null space than the underlying nonlinear system actually has: the
filter picks up apparent information along directions that are genuinely
unobservable in the true nonlinear problem -- most visibly global position
and yaw about gravity in an IMU+landmark navigation problem, where the true
system is invariant to a rigid rotation-about-gravity-plus-translation of
the whole robot trajectory and map, but a standard EKF's shifting
linearization point breaks that invariance in the linearized model and the
filter becomes overconfident (see tests/test_observability.py, which
assembles the local observability matrix for standard EKF, FEJ-EKF, and
InEKF over the same window and checks the recovered null-space dimension).

FEJ removes the inconsistency by using one fixed linearization point per
state element -- for a state element created via `augment_state` (e.g. a
newly-sighted landmark), that fixed point is its initialization value, since
that is its "first-ever estimate."
"""

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
        use_fej: bool = False,
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
        self.use_fej = bool(use_fej)

        self.x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float).reshape(self.n)
        self.P = np.eye(self.n) if P0 is None else np.atleast_2d(np.asarray(P0, dtype=float))
        self.x_lin = self.x.copy()

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.innovation = np.zeros(self.m)
        self.S = np.eye(self.m)
        self.K = np.zeros((self.n, self.m))

    def _state_transition_jacobian(self, x, u):
        lin_point = self.x_lin if self.use_fej else x
        if self.F_jac is not None:
            return np.asarray(self.F_jac(lin_point, u), dtype=float)
        return _numerical_jacobian(lambda state, control: self.f(state, control), lin_point, self.jacobian_eps, u)

    def _measurement_jacobian(self, x):
        lin_point = self.x_lin if self.use_fej else x
        if self.H_jac is not None:
            return np.asarray(self.H_jac(lin_point), dtype=float)
        return _numerical_jacobian(lambda state: self.h(state), lin_point, self.jacobian_eps)

    def predict(self, u=None):
        self.x_prior = np.asarray(self.f(self.x, u), dtype=float).reshape(self.n)
        F = self._state_transition_jacobian(self.x, u)
        self.P_prior = F @ self.P @ F.T + self.Q
        self.P_prior = _symmetrize(self.P_prior)
        self.x = self.x_prior.copy()
        self.P = self.P_prior.copy()
        if self.use_fej:
            # x_lin tracks its own "first-estimates" trajectory forward
            # through the same dynamics, but is never corrected by update().
            self.x_lin = np.asarray(self.f(self.x_lin, u), dtype=float).reshape(self.n)
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

    def augment_state(self, new_elements, P_init_block, cross_cov=None):
        """Grow the state by new_elements.size dimensions. `x_lin` for the
        new block is set to `new_elements` itself -- FEJ's "first-ever
        estimate is the initialization value" rule -- regardless of whether
        use_fej is currently on, so toggling the flag later stays well
        defined without re-augmenting.
        """
        new_elements = np.asarray(new_elements, dtype=float).reshape(-1)
        k = new_elements.size
        P_init_block = np.atleast_2d(np.asarray(P_init_block, dtype=float))
        if P_init_block.shape != (k, k):
            raise ValueError(f"P_init_block must be ({k},{k}), got {P_init_block.shape}")
        if cross_cov is None:
            cross_cov = np.zeros((k, self.n))
        else:
            cross_cov = np.asarray(cross_cov, dtype=float).reshape(k, self.n)

        new_P = np.zeros((self.n + k, self.n + k))
        new_P[: self.n, : self.n] = self.P
        new_P[self.n :, : self.n] = cross_cov
        new_P[: self.n, self.n :] = cross_cov.T
        new_P[self.n :, self.n :] = P_init_block

        self.x = np.concatenate([self.x, new_elements])
        self.x_lin = np.concatenate([self.x_lin, new_elements.copy()])
        self.P = _symmetrize(new_P)
        self.n = self.n + k
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.K = np.zeros((self.n, self.m))
        return self.x, self.P

    def marginalize_state(self, indices):
        """Remove the given state indices from x, x_lin, and P."""
        indices = set(int(i) for i in np.atleast_1d(np.asarray(indices, dtype=int)))
        keep = np.array([i for i in range(self.n) if i not in indices], dtype=int)
        self.x = self.x[keep]
        self.x_lin = self.x_lin[keep]
        self.P = self.P[np.ix_(keep, keep)]
        self.n = keep.size
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.K = np.zeros((self.n, self.m))
        return self.x, self.P

    def log_likelihood(self):
        sign, logdet = np.linalg.slogdet(self.S)
        if sign <= 0:
            return float("-inf")
        mahal = self.innovation @ np.linalg.solve(self.S, self.innovation)
        return float(-0.5 * (self.m * np.log(2.0 * np.pi) + logdet + mahal))
