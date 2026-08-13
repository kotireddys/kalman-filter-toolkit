# 04 Multisensor Fusion

## The Problem

Fuse IMU, GNSS, and barometer measurements asynchronously in a tightly coupled navigation stack.

## What You'll Learn

- How to handle sensors arriving at different rates.
- How chi-squared gating protects against outliers.
- How to reason about integrity in a multi-sensor estimator.

## Filters, Noise, and Diagnostics Used

- Filters: ESKF, linear KF for subproblems
- Noise: sensor-specific measurement models, HDOP scaling, heteroscedastic noise
- Diagnostics: chi-squared gating, covariance health, innovation consistency

## Data Source

Use synchronized synthetic sensor streams first, then a recorded navigation dataset with timestamped sensors.

## Deliverables

- [ ] Implement asynchronous measurement updates
- [ ] Add gating for GNSS and barometer outliers
- [ ] Plot fused trajectory against each sensor stream
- [ ] Track integrity metrics over time
- [ ] Compare tightly coupled and loosely coupled fusion
