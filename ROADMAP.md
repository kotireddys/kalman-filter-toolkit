# Roadmap

## Core Modules

- [x] `filters/kf.py` linear Kalman filter with Joseph covariance update
- [x] `filters/ekf.py` extended Kalman filter with analytical and numerical Jacobians
- [x] `filters/ukf.py` unscented Kalman filter with Merwe scaled transform
- [x] `filters/eskf.py` error-state Kalman filter for quaternion IMU navigation
- [x] `filters/sq_root.py` square-root Kalman filter
- [x] `filters/adaptive.py` Sage-Husa and Mehra adaptive noise estimation
- [x] `noise/allan_variance.py` Allan variance and coefficient fitting
- [x] `noise/process_noise.py` process-noise discretization utilities
- [x] `noise/measurement_noise.py` measurement-noise estimation utilities
- [x] `diagnostics/nis_nees.py` NIS and NEES consistency checks
- [x] `diagnostics/consistency.py` innovation mean and whiteness tests
- [x] `diagnostics/covariance_health.py` covariance health checks and repair

## Case Studies

- [x] `case_studies/01_tracking_intro`
- [x] `case_studies/02_imu_dead_reckoning`
- [x] `case_studies/03_adaptive_tuning`
- [x] `case_studies/04_multisensor_fusion`
- [x] `case_studies/05_filter_vs_smoother_vs_graph`

## Validation

- [x] Editable install via `pip install -e ".[dev]"`
- [x] Automated test suite in `tests/`
