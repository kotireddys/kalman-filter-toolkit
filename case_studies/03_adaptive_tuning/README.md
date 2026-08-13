# 03 Adaptive Tuning

## The Problem

Compare fixed tuning with adaptive covariance estimation when the nominal noise model is wrong.

## What You'll Learn

- How Sage-Husa adaptation behaves under mismatch.
- How Mehra-style innovation-based R estimation works.
- Which failure modes appear when adaptation is too aggressive.

## Filters, Noise, and Diagnostics Used

- Filters: fixed KF, Sage-Husa adaptive KF
- Noise: innovation-based R estimation, covariance repair
- Diagnostics: innovation statistics, NIS, NEES, convergence plots

## Data Source

Use simulated measurements with deliberately mismatched Q and R, then test on logged data if available.

## Deliverables

- [ ] Implement fixed and adaptive filters on the same dataset
- [ ] Plot Q and R convergence over time
- [ ] Compare NIS and NEES under mismatch
- [ ] Identify at least one failure case for each adaptive method
- [ ] Summarize which adaptation is most stable
