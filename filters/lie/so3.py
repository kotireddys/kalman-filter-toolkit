"""SO(3) exponential/logarithm and left Jacobians, with small-angle Taylor
fallbacks near identity. These are the building blocks for the SE_2(3)
utilities used by the invariant EKF; kept separate because SE_2(3)'s
closed-form exp/log/Adjoint are expressed directly in terms of them.
"""

from __future__ import annotations

import numpy as np

_SMALL_ANGLE = 1e-4
_NEAR_PI = 1e-6


def skew(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]],
        dtype=float,
    )


def vee(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]], dtype=float)


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Rodrigues' formula, R = I + sin(theta)/theta [phi]_x + (1-cos theta)/theta^2 [phi]_x^2."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    skew_phi = skew(phi)
    if theta < _SMALL_ANGLE:
        # Taylor series to O(theta^4): sin(theta)/theta ~ 1 - theta^2/6,
        # (1-cos theta)/theta^2 ~ 1/2 - theta^2/24.
        a = 1.0 - theta**2 / 6.0
        b = 0.5 - theta**2 / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta**2
    return np.eye(3) + a * skew_phi + b * (skew_phi @ skew_phi)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Inverse of exp_so3. Not hardened at theta = pi exactly (axis sign is
    inherently ambiguous there); the near-pi branch below is a best-effort
    fallback, not a numerically bulletproof one, since IMU-driven navigation
    over an integration step is not expected to rotate by pi.
    """
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < _SMALL_ANGLE:
        # log(R) ~ vee(R - R^T)/2 * (1 + theta^2/6), i.e. correct the
        # leading-order vee(R - R^T)/2 term for small but non-negligible theta.
        phi = vee(R - R.T) / 2.0
        correction = 1.0 + theta**2 / 6.0
        return phi * correction

    if np.pi - theta < _NEAR_PI:
        # theta ~ pi: sin(theta) ~ 0 so the vee(R - R^T)/(2 sin theta) form is
        # singular. Extract the axis from the symmetric part instead:
        # R = 2 n n^T - I at theta = pi, so n n^T = (R + I) / 2.
        outer = (R + np.eye(3)) / 2.0
        diag = np.clip(np.diag(outer), 0.0, None)
        axis = np.sqrt(diag)
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            for j in range(3):
                if j != pivot:
                    axis[j] = outer[pivot, j] / axis[pivot]
            axis = axis / np.linalg.norm(axis)
        return axis * theta

    phi = vee(R - R.T) * (theta / (2.0 * np.sin(theta)))
    return phi


def left_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    skew_phi = skew(phi)
    if theta < _SMALL_ANGLE:
        a = 0.5 - theta**2 / 24.0
        b = 1.0 / 6.0 - theta**2 / 120.0
    else:
        a = (1.0 - np.cos(theta)) / theta**2
        b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) + a * skew_phi + b * (skew_phi @ skew_phi)


def left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    skew_phi = skew(phi)
    if theta < _SMALL_ANGLE:
        c = 1.0 / 12.0 + theta**2 / 720.0
    else:
        c = (1.0 / theta**2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * skew_phi + c * (skew_phi @ skew_phi)
