"""Overlapping Allan variance and coefficient fitting."""

from __future__ import annotations

import numpy as np


def compute_allan_variance(data, fs):
    data = np.asarray(data, dtype=float).reshape(-1)
    fs = float(fs)
    n = data.size
    if n < 4:
        raise ValueError("Allan variance requires at least four samples.")
    cumulative = np.concatenate(([0.0], np.cumsum(data)))
    max_cluster = max(1, n // 10)
    log_spaced = np.logspace(0, np.log10(max_cluster), num=min(30, max_cluster))
    cluster_sizes = np.unique(np.clip(np.round(log_spaced).astype(int), 1, max_cluster))
    taus = cluster_sizes / fs
    allan_variance = []
    for m in cluster_sizes:
        if n < 3 * m:
            continue
        cluster_means = (cumulative[m:] - cumulative[:-m]) / m
        second_difference = cluster_means[2 * m :] - 2.0 * cluster_means[m:-m] + cluster_means[: -2 * m]
        allan_variance.append(0.5 * np.mean(second_difference**2))
    valid_taus = taus[: len(allan_variance)]
    return valid_taus, np.asarray(allan_variance, dtype=float)


def compute_allan_deviation(data, fs):
    taus, allan_variance = compute_allan_variance(data, fs)
    return taus, np.sqrt(allan_variance)


def fit_noise_params(taus, adev):
    taus = np.asarray(taus, dtype=float).reshape(-1)
    adev = np.asarray(adev, dtype=float).reshape(-1)
    if taus.size != adev.size:
        raise ValueError("taus and adev must have the same length.")
    if taus.size == 0:
        raise ValueError("At least one Allan deviation point is required.")
    short_count = max(1, taus.size // 3)
    long_count = max(1, taus.size // 3)
    arw = float(np.median(adev[:short_count] * np.sqrt(taus[:short_count])))
    rrw = float(np.median(adev[-long_count:] / np.sqrt(taus[-long_count:])))
    bias_instability = float(np.min(adev) * 0.6645)
    return {"arw": arw, "bias_instability": bias_instability, "rrw": rrw}
