# 05 Filter vs Smoother vs Graph

## The Problem

Compare recursive filtering, fixed-lag smoothing, and batch factor graphs on the same motion-estimation task.

## What You'll Learn

- How a forward filter differs from a backward smoother.
- Why batch optimization can improve trajectory quality.
- How to compare ATE and RPE across estimation methods.

## Filters, Noise, and Diagnostics Used

- Filters: EKF, RTS smoother, factor graph baseline
- Noise: process and measurement models shared across methods
- Diagnostics: ATE, RPE, residual analysis

## Data Source

Use a synthetic trajectory or a recorded odometry dataset with pose ground truth.

## Deliverables

- [ ] Implement an EKF baseline
- [ ] Add an RTS smoother pass
- [ ] Build a factor-graph comparison workflow
- [ ] Compare ATE and RPE across methods
- [ ] Document when smoothing or batching is worth the cost
