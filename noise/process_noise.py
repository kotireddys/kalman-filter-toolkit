"""Process-noise modeling and continuous-to-discrete conversion utilities."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def pcwn_1d(dt, sigma_a):
    dt = float(dt)
    sigma_a = float(sigma_a)
    q = sigma_a**2
    return q * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=float)


def pcwa_1d(dt, sigma_j):
    dt = float(dt)
    sigma_j = float(sigma_j)
    q = sigma_j**2
    return q * np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ],
        dtype=float,
    )


def van_loan(F_c, Q_c, dt):
    F_c = np.atleast_2d(np.asarray(F_c, dtype=float))
    Q_c = np.atleast_2d(np.asarray(Q_c, dtype=float))
    dt = float(dt)
    n = F_c.shape[0]
    block = np.zeros((2 * n, 2 * n), dtype=float)
    block[:n, :n] = -F_c
    block[:n, n:] = Q_c
    block[n:, n:] = F_c.T
    exp_block = expm(block * dt)
    F_d = exp_block[n:, n:].T
    Q_d = F_d @ exp_block[:n, n:]
    Q_d = 0.5 * (Q_d + Q_d.T)
    return F_d, Q_d


def psd_to_discrete_Q(G, Q_psd, dt):
    G = np.atleast_2d(np.asarray(G, dtype=float))
    Q_psd = np.atleast_2d(np.asarray(Q_psd, dtype=float))
    dt = float(dt)
    return 0.5 * ((G @ Q_psd @ G.T) * dt + (G @ Q_psd @ G.T).T * dt)
