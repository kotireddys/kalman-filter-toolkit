"""Innovation consistency diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def innovation_mean_test(innovations, S_avg, alpha=0.05):
    innovations = np.asarray(innovations, dtype=float)
    if innovations.ndim == 1:
        innovations = innovations[:, None]
    S_avg = np.atleast_2d(np.asarray(S_avg, dtype=float))
    sample_count = innovations.shape[0]
    mean_innovation = innovations.mean(axis=0)
    covariance_of_mean = S_avg / sample_count
    statistic = float(mean_innovation @ np.linalg.solve(covariance_of_mean, mean_innovation))
    threshold = float(chi2.ppf(1.0 - float(alpha), innovations.shape[1]))
    return {
        "consistent": bool(statistic <= threshold),
        "statistic": statistic,
        "threshold": threshold,
        "mean": mean_innovation,
    }


def innovation_whiteness_test(innovations, max_lag=25, alpha=0.05):
    innovations = np.asarray(innovations, dtype=float)
    if innovations.ndim == 1:
        innovations = innovations[:, None]
    sample_count = innovations.shape[0]
    bound = 1.0 / np.sqrt(sample_count)
    autocorrelations = []
    for lag in range(1, int(max_lag) + 1):
        lag_values = []
        for column in range(innovations.shape[1]):
            series = innovations[:, column] - np.mean(innovations[:, column])
            numerator = np.dot(series[:-lag], series[lag:])
            denominator = np.dot(series, series)
            lag_values.append(float(numerator / denominator) if denominator > 0 else 0.0)
        autocorrelations.append(lag_values)
    autocorrelations = np.asarray(autocorrelations, dtype=float)
    within_bounds = bool(np.all(np.abs(autocorrelations) <= bound))
    return {
        "consistent": within_bounds,
        "bound": bound,
        "autocorrelations": autocorrelations,
    }
