"""NIS and NEES consistency checks."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def compute_nis(innovation, S):
    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    S = np.atleast_2d(np.asarray(S, dtype=float))
    return float(innovation @ np.linalg.solve(S, innovation))


def compute_nees(x_true, x_est, P):
    x_true = np.asarray(x_true, dtype=float).reshape(-1)
    x_est = np.asarray(x_est, dtype=float).reshape(-1)
    P = np.atleast_2d(np.asarray(P, dtype=float))
    error = x_true - x_est
    return float(error @ np.linalg.solve(P, error))


def chi2_consistency_test(values, dof, alpha=0.05):
    values = np.asarray(values, dtype=float).reshape(-1)
    dof = int(dof)
    alpha = float(alpha)
    sample_count = values.size
    mean_value = float(np.mean(values))
    lower = float(chi2.ppf(alpha / 2.0, sample_count * dof) / sample_count)
    upper = float(chi2.ppf(1.0 - alpha / 2.0, sample_count * dof) / sample_count)
    sample_lower = float(chi2.ppf(alpha / 2.0, dof))
    sample_upper = float(chi2.ppf(1.0 - alpha / 2.0, dof))
    average_consistent = lower <= mean_value <= upper
    fraction_in_bounds = float(np.mean((values >= sample_lower) & (values <= sample_upper)))
    return {
        "consistent": bool(average_consistent),
        "average_stat": mean_value,
        "lower_bound": lower,
        "upper_bound": upper,
        "sample_lower_bound": sample_lower,
        "sample_upper_bound": sample_upper,
        "fraction_in_bounds": fraction_in_bounds,
    }
