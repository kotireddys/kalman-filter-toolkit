"""Summary table generation (CSV + Markdown). Reports medians and IQRs
across runs -- never a mean-only figure and never a single representative
run -- per each metric's own distribution across the Monte Carlo set.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .metrics import GNSS_DIM, STATE_DIM


def _median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.median(values)), float(np.percentile(values, 25)), float(np.percentile(values, 75))


def summarize_estimator(name: str, arrays: dict) -> dict:
    nees_per_run_mean = arrays["nees"].mean(axis=1)
    nis_per_run_mean = arrays["nis_gnss"].mean(axis=1) if arrays["nis_gnss"].shape[1] > 0 else np.array([np.nan])
    runtime_all = np.concatenate([arrays["predict_time_s"].ravel(), arrays["update_time_s"].ravel()])

    nees_med, nees_q1, nees_q3 = _median_iqr(nees_per_run_mean)
    nis_med, nis_q1, nis_q3 = _median_iqr(nis_per_run_mean)
    ate_pos_med, ate_pos_q1, ate_pos_q3 = _median_iqr(arrays["ate_position_rmse"])
    ate_vel_med, ate_vel_q1, ate_vel_q3 = _median_iqr(arrays["ate_velocity_rmse"])
    ate_att_med, ate_att_q1, ate_att_q3 = _median_iqr(arrays["ate_attitude_rmse_rad"])
    rte_pos_med, rte_pos_q1, rte_pos_q3 = _median_iqr(arrays["rte_position_rmse"])

    return {
        "estimator": name,
        "nees_dof": STATE_DIM,
        "nees_mean_median": nees_med,
        "nees_mean_q1": nees_q1,
        "nees_mean_q3": nees_q3,
        "nis_dof": GNSS_DIM,
        "nis_mean_median": nis_med,
        "nis_mean_q1": nis_q1,
        "nis_mean_q3": nis_q3,
        "ate_position_rmse_median": ate_pos_med,
        "ate_position_rmse_q1": ate_pos_q1,
        "ate_position_rmse_q3": ate_pos_q3,
        "ate_velocity_rmse_median": ate_vel_med,
        "ate_velocity_rmse_q1": ate_vel_q1,
        "ate_velocity_rmse_q3": ate_vel_q3,
        "ate_attitude_rmse_rad_median": ate_att_med,
        "ate_attitude_rmse_rad_q1": ate_att_q1,
        "ate_attitude_rmse_rad_q3": ate_att_q3,
        "rte_position_rmse_median": rte_pos_med,
        "rte_position_rmse_q1": rte_pos_q1,
        "rte_position_rmse_q3": rte_pos_q3,
        "divergence_rate": float(np.mean(arrays["diverged"])),
        "runtime_mean_s": float(runtime_all.mean()),
        "runtime_p95_s": float(np.percentile(runtime_all, 95)),
        "n_runs": arrays["nees"].shape[0],
    }


def build_summary_table(stacked: dict) -> list[dict]:
    return [summarize_estimator(name, arrays) for name, arrays in stacked.items()]


def write_csv(rows: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_markdown(rows: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        ("estimator", "Estimator", "{}"),
        ("nees_mean_median", "ANEES median [Q1,Q3]", None),
        ("nis_mean_median", "ANIS median [Q1,Q3]", None),
        ("ate_position_rmse_median", "ATE pos (m)", None),
        ("ate_velocity_rmse_median", "ATE vel (m/s)", None),
        ("ate_attitude_rmse_rad_median", "ATE att (rad)", None),
        ("rte_position_rmse_median", "RTE pos (m)", None),
        ("divergence_rate", "Divergence rate", "{:.1%}"),
        ("runtime_mean_s", "Runtime mean (s)", "{:.2e}"),
        ("runtime_p95_s", "Runtime p95 (s)", "{:.2e}"),
    ]
    lines = ["| " + " | ".join(label for _, label, _ in cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        cells = []
        for key, _label, fmt in cols:
            if fmt is None:
                q1_key, q3_key = key.replace("_median", "_q1"), key.replace("_median", "_q3")
                cells.append(f"{row[key]:.3g} [{row[q1_key]:.3g}, {row[q3_key]:.3g}]")
            elif key == "estimator":
                cells.append(str(row[key]))
            else:
                cells.append(fmt.format(row[key]))
        lines.append("| " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
