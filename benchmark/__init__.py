"""Monte Carlo consistency benchmark: ANEES/NIS/ATE/RTE comparison of EKF,
FEJ-EKF, ESKF, and InEKF on identical simulated trajectories and sensor
streams. See benchmark/runner.py for the entry point and
benchmark/configs/*.yaml for example configurations.
"""

from .config import BenchmarkConfig, load_config
from .runner import run_monte_carlo, run_single

__all__ = ["BenchmarkConfig", "load_config", "run_monte_carlo", "run_single"]
