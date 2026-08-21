"""Publication figures for the Monte Carlo consistency benchmark.

Color is assigned by estimator identity, fixed across every figure this
module produces (never re-cycled if a filter list is subset/reordered):
ekf -> blue, fej_ekf -> orange, eskf -> aqua, inekf -> yellow. The chi^2
consistency band is shared across all estimators in a given plot (its lower
and upper bounds depend only on n_runs and dof, not on which estimator is
being scored), so it is drawn once as a neutral gray region behind the
per-estimator lines rather than as four overlapping bands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import GNSS_DIM, STATE_DIM, anees_over_time, anis_over_time

ESTIMATOR_COLORS = {
    "ekf": "#2a78d6",
    "fej_ekf": "#eb6834",
    "eskf": "#1baf7a",
    "inekf": "#eda100",
}
ESTIMATOR_LABELS = {
    "ekf": "EKF",
    "fej_ekf": "FEJ-EKF",
    "eskf": "ESKF",
    "inekf": "InEKF",
}
BAND_COLOR = "#c3c2b7"


def plot_anees_vs_time(stacked: dict, out_path: str | Path, title: str = "ANEES vs. time") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    t = next(iter(stacked.values()))["t"]
    n_runs = next(iter(stacked.values()))["nees"].shape[0]
    ref_nees = next(iter(stacked.values()))["nees"]
    _anees, lower, upper = anees_over_time(ref_nees)
    ax.fill_between(t, lower, upper, color=BAND_COLOR, alpha=0.5, label=f"95% band (N={n_runs})", zorder=1)
    ax.axhline(STATE_DIM, color="#52514e", linewidth=1.0, linestyle="--", zorder=2)

    for name, arrays in stacked.items():
        anees, _, _ = anees_over_time(arrays["nees"])
        color = ESTIMATOR_COLORS.get(name, "#000000")
        label = ESTIMATOR_LABELS.get(name, name)
        ax.plot(t, anees, color=color, linewidth=2.0, label=label, zorder=3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ANEES")
    ax.set_title(title)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_anis_vs_time(stacked: dict, out_path: str | Path, title: str = "ANIS (GNSS) vs. time") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ref_nis = next(iter(stacked.values()))["nis_gnss"]
    n_runs = ref_nis.shape[0]
    t_gnss = np.arange(ref_nis.shape[1])
    _anis, lower, upper = anis_over_time(ref_nis)
    ax.fill_between(t_gnss, lower, upper, color=BAND_COLOR, alpha=0.5, label=f"95% band (N={n_runs})", zorder=1)
    ax.axhline(GNSS_DIM, color="#52514e", linewidth=1.0, linestyle="--", zorder=2)

    for name, arrays in stacked.items():
        anis, _, _ = anis_over_time(arrays["nis_gnss"])
        color = ESTIMATOR_COLORS.get(name, "#000000")
        label = ESTIMATOR_LABELS.get(name, name)
        ax.plot(t_gnss, anis, color=color, linewidth=2.0, label=label, zorder=3)

    ax.set_xlabel("GNSS update index")
    ax.set_ylabel("ANIS")
    ax.set_title(title)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_ablation_sweep(
    sweep_x, sweep_results: dict, ylabel: str, xlabel: str, out_path: str | Path, title: str
) -> Path:
    """sweep_results: {estimator_name: array of one scalar metric per sweep
    point}. Plots median with IQR band per estimator across the swept
    parameter (e.g. initial yaw error).
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, values in sweep_results.items():
        median = np.median(values, axis=-1) if values.ndim > 1 else values
        color = ESTIMATOR_COLORS.get(name, "#000000")
        label = ESTIMATOR_LABELS.get(name, name)
        if values.ndim > 1:
            q1 = np.percentile(values, 25, axis=-1)
            q3 = np.percentile(values, 75, axis=-1)
            ax.fill_between(sweep_x, q1, q3, color=color, alpha=0.2, zorder=1)
        ax.plot(sweep_x, median, color=color, linewidth=2.0, marker="o", markersize=5, label=label, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
