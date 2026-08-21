"""SE_2(3) matrix Lie group utilities: exact closed-form exp, log, and
Adjoint, with small-angle Taylor fallbacks inherited from so3.py.

State convention: a group element is a 5x5 matrix

    X = [[R,  v,  p],
         [0,  1,  0],
         [0,  0,  1]]

with R in SO(3) (attitude), v in R^3 (world-frame velocity), p in R^3
(world-frame position). A tangent vector xi = (phi, nu, rho) in R^9 hats to

    xi^ = [[skew(phi),  nu,  rho],
           [0,          0,   0  ],
           [0,          0,   0  ]]

exp(xi^) = [[Exp(phi), Jl(phi) nu, Jl(phi) rho], [0,1,0], [0,0,1]] where
Exp/Jl are the SO(3) exponential and left Jacobian -- this is the standard
Barrau & Bonnabel closed form (the same left Jacobian applies to both the
velocity and position blocks, since both are "translation-like" columns
appended to the same SO(3) rotation part).
"""

from __future__ import annotations

import numpy as np

from .so3 import exp_so3, left_jacobian, left_jacobian_inv, log_so3, skew


def make_state(R: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    v = np.asarray(v, dtype=float).reshape(3)
    p = np.asarray(p, dtype=float).reshape(3)
    X = np.eye(5, dtype=float)
    X[:3, :3] = R
    X[:3, 3] = v
    X[:3, 4] = p
    return X


def split_state(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    return X[:3, :3], X[:3, 3], X[:3, 4]


def inverse(X: np.ndarray) -> np.ndarray:
    R, v, p = split_state(X)
    R_t = R.T
    X_inv = np.eye(5, dtype=float)
    X_inv[:3, :3] = R_t
    X_inv[:3, 3] = -R_t @ v
    X_inv[:3, 4] = -R_t @ p
    return X_inv


def hat(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float).reshape(9)
    phi, nu, rho = xi[0:3], xi[3:6], xi[6:9]
    Xi = np.zeros((5, 5), dtype=float)
    Xi[:3, :3] = skew(phi)
    Xi[:3, 3] = nu
    Xi[:3, 4] = rho
    return Xi


def vee(Xi: np.ndarray) -> np.ndarray:
    Xi = np.asarray(Xi, dtype=float)
    phi = np.array([Xi[2, 1], Xi[0, 2], Xi[1, 0]], dtype=float)
    nu = Xi[:3, 3]
    rho = Xi[:3, 4]
    return np.concatenate([phi, nu, rho])


def exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float).reshape(9)
    phi, nu, rho = xi[0:3], xi[3:6], xi[6:9]
    R = exp_so3(phi)
    Jl = left_jacobian(phi)
    return make_state(R, Jl @ nu, Jl @ rho)


def log(X: np.ndarray) -> np.ndarray:
    R, v, p = split_state(X)
    phi = log_so3(R)
    Jl_inv = left_jacobian_inv(phi)
    nu = Jl_inv @ v
    rho = Jl_inv @ p
    return np.concatenate([phi, nu, rho])


def adjoint(X: np.ndarray) -> np.ndarray:
    """9x9 Adjoint satisfying X exp(xi) X^-1 = exp(Ad_X xi); verified
    numerically in tests/test_lie_se23.py against direct conjugation rather
    than assumed from the paper alone.
    """
    R, v, p = split_state(X)
    Ad = np.zeros((9, 9), dtype=float)
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = skew(v) @ R
    Ad[3:6, 3:6] = R
    Ad[6:9, 0:3] = skew(p) @ R
    Ad[6:9, 6:9] = R
    return Ad
