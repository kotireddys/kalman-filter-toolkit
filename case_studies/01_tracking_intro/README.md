# 01 Tracking Intro

## The Problem

Build a 2D constant-velocity tracker for position-only measurements and quantify whether the filter is statistically consistent.

## What You'll Learn

- How a linear Kalman filter behaves on a simple motion model.
- How covariance ellipses relate to uncertainty.
- How to evaluate NIS and NEES on Monte Carlo runs.

## Filters, Noise, and Diagnostics Used

- Filters: linear Kalman filter
- Noise: constant-velocity process noise, stationary measurement noise
- Diagnostics: NIS, NEES, chi-squared consistency tests

## Data Source

Use synthetic 2D motion data first, then replace it with logged tracking measurements if available.

## Deliverables

- [ ] Simulate a 2D constant-velocity trajectory
- [ ] Implement a position-only measurement model
- [ ] Plot covariance ellipses over time
- [ ] Run 50 Monte Carlo trials
- [ ] Report NIS and NEES consistency results
