# Kalman Filter Toolkit

Kalman Filter Toolkit is a compact, test-driven repository for state estimation, noise modeling, and filter health checks. It is designed around a simple rule: the filter is not done until the noise model and diagnostics exist alongside it.

## What makes it different

- Noise characterization is first-class, including Allan variance, process-noise discretization, and measurement-noise estimation.
- Diagnostics are built in, with NIS/NEES, innovation consistency checks, and covariance health repair utilities.
- Case studies are scoped to real estimation problems so each filter earns its place.

## Repository Layout

```text
kalman-filter-toolkit/
├── filters/
├── noise/
├── diagnostics/
├── case_studies/
├── tests/
├── docs/
├── assets/
├── README.md
├── ROADMAP.md
├── pyproject.toml
└── LICENSE
```

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Minimal 2D Tracker Example

```python
import numpy as np
from filters.kf import KalmanFilter
from noise.process_noise import pcwn_1d

dt = 0.1
F_1d = np.array([[1.0, dt], [0.0, 1.0]])
Q_1d = pcwn_1d(dt, sigma_a=0.2)
F = np.block([[F_1d, np.zeros((2, 2))], [np.zeros((2, 2)), F_1d]])
Q = np.block([[Q_1d, np.zeros((2, 2))], [np.zeros((2, 2)), Q_1d]])
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
R = np.diag([2.0, 2.0])

kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=np.zeros(4), P0=np.eye(4) * 10.0)

measurement = np.array([10.0, 5.0])
kf.predict()
kf.update(measurement)
print(kf.x)
```

## Case Studies

| # | Problem | Filters Used | Key Concepts |
|---|---|---|---|
| 01 | 2D tracking with position measurements | KF, EKF | Covariance ellipses, NIS/NEES, Monte Carlo consistency |
| 02 | GNSS-denied IMU dead reckoning | ESKF | Allan variance, quaternion attitude, bias drift |
| 03 | Adaptive tuning under mismatch | Adaptive KF | Sage-Husa, Mehra R estimation, failure analysis |
| 04 | Tightly coupled multisensor fusion | ESKF, KF | Async updates, chi-squared gating, integrity checks |
| 05 | Filter vs smoother vs graph | EKF, RTS smoother, factor graph | ATE/RPE, batch vs recursive estimation |

## Design Principles

1. The filter does not exist until the case study needs it.
2. Noise modeling is not an afterthought.
3. Diagnostics are mandatory.
4. Real data should be used wherever possible.

## References

Foundational references include Bar-Shalom, Solà, Barfoot, Särkkä, Mehra, Sage-Husa, Wan-Merwe, Van Loan, Trawny-Roumeliotis, and IEEE 647-2006.
