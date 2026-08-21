"""Single entry point that regenerates every figure and summary table this
repo's README and docs reference, from scratch (subject to the Monte Carlo
cache -- delete results/cache/ to force full re-simulation).

Usage:
    python -m benchmark.generate_all
"""

from __future__ import annotations

from pathlib import Path

from . import ablations
from .cache import DEFAULT_CACHE_DIR
from .config import load_config
from .figures import plot_anees_vs_time, plot_anis_vs_time
from .report import build_summary_table, write_csv, write_markdown
from .runner import run_monte_carlo

CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path("results")


def _run_headline_config(name: str, cache_dir: Path) -> None:
    config = load_config(CONFIGS_DIR / f"{name}.yaml")
    out_dir = RESULTS_DIR / name
    print(f"[{name}] running {config.n_runs} Monte Carlo trials x {len(config.estimators)} estimators...")
    stacked = run_monte_carlo(config, cache_dir=cache_dir)

    plot_anees_vs_time(stacked, out_dir / "anees.png", title=f"ANEES vs. time ({name})")
    plot_anis_vs_time(stacked, out_dir / "anis.png", title=f"ANIS (GNSS) vs. time ({name})")
    rows = build_summary_table(stacked)
    write_csv(rows, out_dir / "summary.csv")
    write_markdown(rows, out_dir / "summary.md")
    print(f"[{name}] wrote figures and summary to {out_dir}/")


def main(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    _run_headline_config("benign", cache_dir)
    _run_headline_config("aggressive", cache_dir)

    ablation_out = RESULTS_DIR / "ablations"
    print("[ablations] yaw sweep...")
    ablations.run_yaw_sweep(load_config(CONFIGS_DIR / "ablation_yaw_sweep.yaml"), ablation_out, cache_dir)
    print("[ablations] initial error sweep...")
    ablations.run_init_error_sweep(load_config(CONFIGS_DIR / "ablation_init_error.yaml"), ablation_out, cache_dir)
    print("[ablations] IMU noise sweep...")
    ablations.run_imu_noise_sweep(load_config(CONFIGS_DIR / "ablation_imu_noise.yaml"), ablation_out, cache_dir)
    print("[ablations] GNSS outage sweep...")
    ablations.run_gnss_outage_sweep(load_config(CONFIGS_DIR / "ablation_gnss_outage.yaml"), ablation_out, cache_dir)
    print(f"[ablations] wrote figures to {ablation_out}/")

    print("Done.")


if __name__ == "__main__":
    main()
