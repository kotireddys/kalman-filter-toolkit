# 02 IMU Dead Reckoning

## The Problem

Estimate inertial navigation states without GNSS and understand how bias and noise accumulation drive drift.

## What You'll Learn

- Why error-state filtering is the standard approach for INS mechanization.
- How quaternion attitude propagation works in practice.
- How Allan variance supports IMU noise identification.

## Filters, Noise, and Diagnostics Used

- Filters: error-state Kalman filter
- Noise: Allan variance, IMU bias models, process noise discretization
- Diagnostics: covariance health, innovation consistency, drift analysis

## Data Source

Use real IMU logs when available, or replay a synthetic motion profile with injected sensor noise.

## Deliverables

- [ ] Compute Allan deviation on IMU data
- [ ] Estimate accelerometer and gyro noise terms
- [ ] Implement ESKF propagation and position update
- [ ] Plot position, velocity, and attitude drift
- [ ] Compare drift with and without bias estimation
