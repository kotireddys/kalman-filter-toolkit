"""Shared covariance-hygiene helpers, consolidated from the per-file copies
in ukf.py, eskf.py, sq_root.py, and adaptive.py so new kernels (InEKF,
FEJ-EKF) don't grow a fifth copy. Existing filters keep their local copies
untouched to avoid churning tested code outside this task's scope.
"""

from __future__ import annotations

import numpy as np


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def repair_covariance(matrix: np.ndarray, min_eigenvalue: float = 1e-12) -> np.ndarray:
    matrix = symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def sqrt_psd(matrix: np.ndarray, min_eigenvalue: float = 1e-12) -> np.ndarray:
    matrix = repair_covariance(matrix, min_eigenvalue=min_eigenvalue)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, min_eigenvalue)
        return eigvecs @ np.diag(np.sqrt(eigvals))
