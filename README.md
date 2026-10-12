# Compact gradient boosting vs adaptive model selection for drone trajectory prediction

## Motivation
The paper "D2dtracker: A framework for enabling real-time trajectory prediction for agile drone-to-drone tracking via adaptive model selection" (2024) predicts a target drone's future motion by adaptively choosing among several predictors. This experiment checks whether a single compact gradient-boosting model can do better than such a scheme, particularly in sharp maneuvers.

## Question
Can one compact LightGBM trajectory predictor beat an adaptive select-the-best-model-by-recent-error scheme at predicting a maneuvering drone's future position (0.5 s and 1 s ahead), especially during sharp maneuvers?

## Dataset
Simulated 3D drone trajectories at 10 Hz (smooth flight with random sharp turns, hard accelerations and stops, plus sensor noise), generated with numpy in `analysis.py`. It is **not** the paper's dataset (the paper's data is not stated). Train and test sets are split by whole trajectory.

## Baseline vs improved
- **Baseline:** a standard adaptive model-selection scheme (not the paper's exact code). The pool holds constant-velocity, constant-acceleration, damped-velocity and ridge autoregressive predictors. At each step the model with the lowest error over the last few steps is used.
- **Improved:** a compact LightGBM model on the same observed window. It uses velocity, acceleration and turn-rate features in a heading-aligned frame and predicts the future displacement. Hyper-parameters are tuned by grouped cross-validation on the training set only.

Metrics: mean position error (m) over both horizons on the test trajectories, the same error on sharp-maneuver windows only, and per-prediction latency (ms).

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On simulated 3D drone trajectories, a standard adaptive model-selection predictor reached 2.84 m mean position error versus 1.96 m for my compact LightGBM model on the same held-out test set (averaged over 0.5 s and 1 s horizons).

| Metric | Value |
|---|---|
| baseline_score | 2.842 |
| improved_score | 1.962 |
| baseline_score_std | 0 |
| improved_score_std | 0.0001804 |
| improvement | 0.8798 |
| baseline_sharp | 4.076 |
| improved_sharp | 3.076 |
| baseline_sharp_std | 0 |
| improved_sharp_std | 0.0004306 |
| baseline_h05 | 1.61 |
| improved_h05 | 1.166 |
| baseline_h05_std | 2.22e-16 |
| improved_h05_std | 0.0001476 |
| baseline_h10 | 4.074 |
| improved_h10 | 2.758 |
| baseline_h10_std | 0 |
| improved_h10_std | 0.0004984 |
| baseline_latency_ms | 0.001164 |
| improved_latency_ms | 0.04179 |
| pool_cv_score | 2.623 |
| pool_cv_sharp | 4.275 |
| pool_ca_score | 3.7 |
| pool_ca_sharp | 4.481 |
| pool_damped_score | 3.52 |
| pool_damped_sharp | 4.499 |
| pool_ridge_ar_score | 2.536 |
| pool_ridge_ar_sharp | 3.65 |
| n_sharp_test_windows | 17106 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic data: conclusions depend on the simulator's maneuver statistics and noise (0.3 m). Targets are true (noise-free) future positions; inputs are noisy. Sharp windows = mean true acceleration over the next 1 s above 3 m/s^2 (used only for evaluation). 'One' LightGBM model is a set of 6 output-wise regressors with shared hyper-parameters tuned on one target. Latency is batch-amortised per prediction on the same test windows (features included), 2 CPU threads for LightGBM. Baseline is deterministic (std 0); LightGBM seeds 42-44 differ only in subsampling. Windows overlap within a trajectory, so errors are correlated.

## Reference

This project is motivated by the following paper:

> M Abdelkader, AS Al-Batati, I Jarraya (2024). *D2dtracker: A framework for enabling real-time trajectory prediction for agile drone-to-drone tracking via adaptive model selection*. 2024 2nd International. https://ieeexplore.ieee.org/abstract/document/10467173/

Paper author: Prof. Anis Koubaa, Prince Sultan University (akoubaa@psu.edu.sa)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
