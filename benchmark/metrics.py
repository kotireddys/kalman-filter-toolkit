"""Consistency and accuracy metrics computed from a Monte Carlo run set.

All four adapters (benchmark/estimators.py) express attitude error in the
same convention -- R_est ~= R_true @ Exp(delta_phi), i.e. a body-frame/local
perturbation -- because EKF/FEJ-EKF's rotation-vector state composes that
way, ESKF's error quaternion is right-multiplied (q_true (x) delta_q), and
InEKF's left-invariant error (used here, paired with the GNSS position
update per filters/inekf.py's own selection rule) is R_true^-1 R_est. So one
formula, `attitude_error`, is valid for NEES across all four.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diagnostics.nis_nees import chi2_consistency_test, compute_nees, compute_nis
from filters.lie.so3 import log_so3

STATE_DIM = 15  # [phi(3), v(3), p(3), b_g(3), b_a(3)]
GNSS_DIM = 3


def attitude_error(R_est: np.ndarray, R_true: np.ndarray) -> np.ndarray:
    return log_so3(R_true.T @ R_est)


@dataclass
class RunTrace:
    """Per-run, per-timestep trace for one estimator on one Monte Carlo run."""

    t: np.ndarray  # (N+1,)
    nees: np.ndarray  # (N+1,)
    attitude_err: np.ndarray  # (N+1, 3)
    velocity_err: np.ndarray  # (N+1, 3)
    position_err: np.ndarray  # (N+1, 3)
    nis_gnss: np.ndarray  # (K,) one per GNSS update actually applied
    predict_time_s: np.ndarray  # (N,)
    update_time_s: np.ndarray  # (K,)
    diverged: bool


def anees_over_time(nees_across_runs: np.ndarray, alpha: float = 0.05):
    """nees_across_runs: (n_runs, n_steps). Returns (anees, lower, upper) time
    series using chi2_consistency_test's mean-of-N-samples bound at each
    timestep -- exactly the ANEES chi^2 band, reusing the repo's existing
    diagnostics utility rather than re-deriving the chi^2 math.
    """
    n_runs, n_steps = nees_across_runs.shape
    anees = np.zeros(n_steps)
    lower = np.zeros(n_steps)
    upper = np.zeros(n_steps)
    for k in range(n_steps):
        result = chi2_consistency_test(nees_across_runs[:, k], dof=STATE_DIM, alpha=alpha)
        anees[k] = result["average_stat"]
        lower[k] = result["lower_bound"]
        upper[k] = result["upper_bound"]
    return anees, lower, upper


def anis_over_time(nis_across_runs: np.ndarray, alpha: float = 0.05):
    n_runs, n_steps = nis_across_runs.shape
    anis = np.zeros(n_steps)
    lower = np.zeros(n_steps)
    upper = np.zeros(n_steps)
    for k in range(n_steps):
        result = chi2_consistency_test(nis_across_runs[:, k], dof=GNSS_DIM, alpha=alpha)
        anis[k] = result["average_stat"]
        lower[k] = result["lower_bound"]
        upper[k] = result["upper_bound"]
    return anis, lower, upper


def ate_summary(trace: RunTrace) -> dict:
    """Absolute trajectory error: whole-run RMSE, one scalar per channel."""
    return {
        "attitude_rmse_rad": float(np.sqrt(np.mean(np.sum(trace.attitude_err**2, axis=-1)))),
        "velocity_rmse": float(np.sqrt(np.mean(np.sum(trace.velocity_err**2, axis=-1)))),
        "position_rmse": float(np.sqrt(np.mean(np.sum(trace.position_err**2, axis=-1)))),
    }


def rte_summary(trace: RunTrace, dt: float, window_s: float = 1.0) -> dict:
    """Relative trajectory error: RMSE of the error's own drift over a fixed
    window (default 1s), i.e. how fast the error itself grows locally,
    rather than its absolute magnitude.
    """
    window_steps = max(1, int(round(window_s / dt)))
    if window_steps >= trace.position_err.shape[0]:
        window_steps = trace.position_err.shape[0] - 1
    if window_steps < 1:
        return {"attitude_rmse_rad": 0.0, "velocity_rmse": 0.0, "position_rmse": 0.0}

    def _relative_rmse(err):
        delta = err[window_steps:] - err[:-window_steps]
        return float(np.sqrt(np.mean(np.sum(delta**2, axis=-1))))

    return {
        "attitude_rmse_rad": _relative_rmse(trace.attitude_err),
        "velocity_rmse": _relative_rmse(trace.velocity_err),
        "position_rmse": _relative_rmse(trace.position_err),
    }


def stack_run_traces(traces: list[RunTrace], dt: float) -> dict:
    """Stack a list of per-run RunTrace objects (all from the same config,
    hence same time base and GNSS schedule) into arrays for reporting,
    figures, and caching.
    """
    ate = [ate_summary(tr) for tr in traces]
    rte = [rte_summary(tr, dt) for tr in traces]
    return {
        "t": traces[0].t,
        "nees": np.stack([tr.nees for tr in traces]),
        "nis_gnss": np.stack([tr.nis_gnss for tr in traces]),
        "attitude_err": np.stack([tr.attitude_err for tr in traces]),
        "velocity_err": np.stack([tr.velocity_err for tr in traces]),
        "position_err": np.stack([tr.position_err for tr in traces]),
        "predict_time_s": np.stack([tr.predict_time_s for tr in traces]),
        "update_time_s": np.stack([tr.update_time_s for tr in traces]),
        "diverged": np.array([tr.diverged for tr in traces]),
        "ate_attitude_rmse_rad": np.array([a["attitude_rmse_rad"] for a in ate]),
        "ate_velocity_rmse": np.array([a["velocity_rmse"] for a in ate]),
        "ate_position_rmse": np.array([a["position_rmse"] for a in ate]),
        "rte_attitude_rmse_rad": np.array([r["attitude_rmse_rad"] for r in rte]),
        "rte_velocity_rmse": np.array([r["velocity_rmse"] for r in rte]),
        "rte_position_rmse": np.array([r["position_rmse"] for r in rte]),
    }


def compute_run_trace(
    t: np.ndarray,
    R_est: np.ndarray, v_est: np.ndarray, p_est: np.ndarray, bg_est: np.ndarray, ba_est: np.ndarray,
    R_true: np.ndarray, v_true: np.ndarray, p_true: np.ndarray, bg_true: np.ndarray, ba_true: np.ndarray,
    P_canonical: np.ndarray,
    gnss_innovations: list, gnss_S: list,
    predict_time_s: np.ndarray, update_time_s: np.ndarray,
    divergence_threshold_m: float,
) -> RunTrace:
    n_plus_1 = R_est.shape[0]
    attitude_err = np.zeros((n_plus_1, 3))
    for k in range(n_plus_1):
        attitude_err[k] = attitude_error(R_est[k], R_true[k])
    velocity_err = v_est - v_true
    position_err = p_est - p_true
    bg_err = bg_est - bg_true
    ba_err = ba_est - ba_true

    nees = np.zeros(n_plus_1)
    for k in range(n_plus_1):
        error_vec = np.concatenate([attitude_err[k], velocity_err[k], position_err[k], bg_err[k], ba_err[k]])
        nees[k] = compute_nees(error_vec, np.zeros_like(error_vec), P_canonical[k])

    nis_gnss = np.array([compute_nis(innov, S) for innov, S in zip(gnss_innovations, gnss_S, strict=True)])

    diverged = bool(np.any(np.linalg.norm(position_err, axis=-1) > divergence_threshold_m))

    return RunTrace(
        t=t, nees=nees, attitude_err=attitude_err, velocity_err=velocity_err, position_err=position_err,
        nis_gnss=nis_gnss, predict_time_s=predict_time_s, update_time_s=update_time_s, diverged=diverged,
    )
