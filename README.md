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

## Reference

This project is motivated by the following paper:

> M Abdelkader, AS Al-Batati, I Jarraya (2024). *D2dtracker: A framework for enabling real-time trajectory prediction for agile drone-to-drone tracking via adaptive model selection*. 2024 2nd International. https://ieeexplore.ieee.org/abstract/document/10467173/

Paper author: Prof. Anis Koubaa, Prince Sultan University (akoubaa@psu.edu.sa)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
