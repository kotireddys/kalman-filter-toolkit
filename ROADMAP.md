# Roadmap

## Core Modules

- [x] `filters/kf.py` linear Kalman filter with Joseph covariance update
- [x] `filters/ekf.py` extended Kalman filter with analytical and numerical Jacobians, plus First-Estimates Jacobians (`use_fej`) and dynamic state augmentation
- [x] `filters/ukf.py` unscented Kalman filter with Merwe scaled transform
- [x] `filters/eskf.py` error-state Kalman filter for quaternion IMU navigation
- [x] `filters/sq_root.py` square-root Kalman filter
- [x] `filters/adaptive.py` Sage-Husa and Mehra adaptive noise estimation
- [x] `filters/inekf.py` invariant EKF on SE₂(3) for IMU navigation, both error conventions
- [x] `filters/lie/` SO(3) and SE₂(3) exp/log/Adjoint utilities
- [x] `noise/allan_variance.py` Allan variance and coefficient fitting
- [x] `noise/process_noise.py` process-noise discretization utilities
- [x] `noise/measurement_noise.py` measurement-noise estimation utilities
- [x] `diagnostics/nis_nees.py` NIS and NEES consistency checks
- [x] `diagnostics/consistency.py` innovation mean and whiteness tests
- [x] `diagnostics/covariance_health.py` covariance health checks and repair

## Benchmark Harness

- [x] `benchmark/config.py` YAML config schema (trajectory, noise, sensors, init, divergence)
- [x] `benchmark/trajectory.py` benign/aggressive ground-truth trajectory generation
- [x] `benchmark/simulate.py` IMU + intermittent GNSS sensor simulation
- [x] `benchmark/estimators.py` uniform adapter over EKF/FEJ-EKF/ESKF/InEKF
- [x] `benchmark/metrics.py` ANEES/ANIS-vs-time, ATE/RTE, divergence, runtime
- [x] `benchmark/cache.py` Monte Carlo result caching keyed on config hash
- [x] `benchmark/report.py` CSV/Markdown summary tables (median + IQR)
- [x] `benchmark/figures.py` ANEES/ANIS bands and ablation-sweep figures
- [x] `benchmark/ablations.py` yaw / init-error / IMU-noise / GNSS-outage sweeps
- [x] `benchmark/generate_all.py` single entry point regenerating every figure

## Case Studies

- [x] `case_studies/01_tracking_intro`
- [x] `case_studies/02_imu_dead_reckoning`
- [x] `case_studies/03_adaptive_tuning`
- [x] `case_studies/04_multisensor_fusion`
- [x] `case_studies/05_filter_vs_smoother_vs_graph`

## Validation

- [x] Editable install via `pip install -e ".[dev]"`
- [x] Automated test suite in `tests/`

## Non-goals (this pass)

VIO, IMU preintegration, sliding-window MAP estimation, learned components, and
additional filter variants beyond EKF/FEJ-EKF/UKF/ESKF/InEKF/Square-Root/Adaptive
are planned for later phases.
