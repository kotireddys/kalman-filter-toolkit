"""Measurement-noise estimation and scaling utilities."""

from __future__ import annotations

import numpy as np


def static_R_estimate(measurements, known_truth=None):
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim == 1:
        measurements = measurements[:, None]
    if known_truth is None:
        residuals = measurements - measurements.mean(axis=0, keepdims=True)
    else:
        truth = np.asarray(known_truth, dtype=float)
        if truth.ndim == 1:
            truth = truth.reshape(1, -1)
        if truth.shape[0] == 1:
            truth = np.repeat(truth, measurements.shape[0], axis=0)
        residuals = measurements - truth
    covariance = np.cov(residuals, rowvar=False, bias=False)
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    return covariance


def hdop_scaled_R(R_base, hdop):
    return np.asarray(R_base, dtype=float) * float(hdop) ** 2


def heteroscedastic_R(signal_value, model, params):
    params = dict(params)
    if model == "proportional":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 1.0))
        c = float(params.get("c", 0.0))
        scale = a * abs(float(signal_value)) ** b + c
        base = np.asarray(params.get("R0", 1.0), dtype=float)
        return base * scale
    if model == "lookup":
        breakpoints = np.asarray(params["breakpoints"], dtype=float).reshape(-1)
        values = np.asarray(params["values"], dtype=float)
        query = float(signal_value)
        if values.ndim == 1:
            interpolated = np.interp(query, breakpoints, values)
            base = np.asarray(params.get("R0", 1.0), dtype=float)
            return base * interpolated
        flat = np.vstack([np.interp(query, breakpoints, values[:, column]) for column in range(values.shape[1])]).T
        return flat
    raise ValueError(f"Unknown heteroscedastic model: {model}")
