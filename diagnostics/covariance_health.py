"""Covariance matrix health checks and repair utilities."""

from __future__ import annotations

import numpy as np


def repair_covariance(P, min_eigenvalue=1e-10):
    P = np.asarray(P, dtype=float)
    P = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, float(min_eigenvalue))
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def check_covariance(P):
    P = np.asarray(P, dtype=float)
    symmetric = bool(np.allclose(P, P.T, atol=1e-10, rtol=1e-8))
    try:
        np.linalg.cholesky(0.5 * (P + P.T))
        positive_definite = True
    except np.linalg.LinAlgError:
        positive_definite = False
    condition_number = float(np.linalg.cond(P)) if P.size else float("nan")
    healthy = symmetric and positive_definite and np.isfinite(condition_number) and condition_number < 1e12
    return {
        "healthy": bool(healthy),
        "symmetric": symmetric,
        "positive_definite": positive_definite,
        "condition_number": condition_number,
    }
