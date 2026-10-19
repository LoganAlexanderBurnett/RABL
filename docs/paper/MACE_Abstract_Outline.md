# MACE Manuscript — Abstract Outline

**Purpose:** Guide a single-paragraph, self-contained abstract following the Introduction's progression from control-trajectory sampling, to recursive prediction, to adaptive conformal intervals, to operation-informed recalibration.

**Suggested length:** Approximately 190–230 words, subject to the journal's final abstract requirements. This document is an outline, not a full prose draft.

## Recommended sequence

### 1. Motivation and research gap (approximately 30–40 words)
- Establish the general problem of predicting coupled multivariate transients under **prescribed control trajectories** and differing physical time scales.
- Distinguish **complete recursive forecast accuracy** from **statistically reliable prediction intervals**; ensemble disagreement is not by itself calibrated uncertainty.
- Keep this opening general. The heat-pipe microreactor should enter as the evaluation setting rather than dominate the motivation.

### 2. Method and main innovations (approximately 55–70 words)
- Introduce the **Multivariate Adaptive Conformal Ensemble (MACE)** framework.
- Condense the three connected elements:
  - Temporally correlated control velocities are integrated into continuous stochastic **control trajectories** for simulator-generated training data.
  - A **ten-member autoregressive LSTM ensemble** advances each member through its *own predicted history*, with ensemble mean and spread computed from the resulting forecasts.
  - Conformal scores normalize prediction error by ensemble spread with a positive floor; distinct reductions target **Horizon**, **per-output Trajectory**, and **joint complete-multivariate trajectory** coverage.
- Identify **absolute-error conformal prediction** as the baseline under matched coverage objectives and the same forecast centers.
- Do not include equations, kernel hyperparameters, network layer sizes, or the conformal order-statistic formula.

### 3. Physical application and testing (approximately 25–35 words)
- Identify the reduced **heat-pipe microreactor and balance-of-plant simulator** as the reference model.
- Specify **15 output variables**, **200-s forecast trajectories**, and **501 time points**.
- Distinguish the original stochastic test distribution from an independently generated population of four kinds of **operation-informed control maneuvers**. The *same operational ensemble forecasts* are evaluated using two calibration distributions.
- Include sample counts only if space permits: **1,000 calibration** and **2,000 evaluation trajectories** for each applicable population.

### 4. Most informative quantitative results (approximately 65–75 words)
- Forecasting: pooled training-range-normalized MAE **0.00154** on the stochastic set and **0.00141** on the operation-informed set. The respective normalized RMSE values (**0.00359** and **0.00256**) are optional if space is tight.
- Matched stochastic calibration, at roughly the intended **95% coverage**:
  - **MACE-Horizon** reduced mean full interval width by **8.9%** and mean interval score by **34.6%** relative to Absolute CP-Horizon.
  - **MACE-Trajectory** reduced them by **43.9%** and **50.0%**, respectively, versus Absolute CP-Trajectory.
- Under transferred stochastic calibration, operational **MACE-Trajectory pooled per-output complete-trajectory coverage** was **0.6800**; independent operation-informed recalibration raised it to **0.9549**. This finding should usually take priority over listing more individual-output statistics.
- Recalibration does not uniformly narrow intervals. For MACE-Trajectory, normalized mean width rose **0.0121 → 0.0136** while coverage was recovered. Include the tradeoff only if the word budget allows.
- Joint calibration results (MACE joint coverage **0.9525** versus absolute **0.9445**) are supported, but usually do not need a separate numeric claim in the abstract.

### 5. Final implication and its boundary (approximately 20–30 words)
- Conclude that **ensemble disagreement is a useful forecast-dependent interval scale**, but the coverage reliability of the resulting intervals still depends on a calibration distribution appropriate for evaluation.
- Emphasize a general insight for **control-conditioned multivariate surrogate forecasts**, not a claim of universal distribution-shift robustness or deployment readiness.

## Style and accuracy checklist
- Use the manuscript's measured scientific tone. Arrange ideas as **control sampling → recursive forecasting → uncertainty calibration → distribution change**.
- Define MACE at first use. Use *output variable*, *control trajectory*, *forecast trajectory*, *calibration distribution*, and *evaluation distribution* consistently.
- Distinguish coverage events: **Horizon = a forecast horizon/output coordinate; Trajectory = all forecast times for one output; joint = all outputs and times**. The operational **0.6800** and **0.9549** values are *pooled averages of per-output trajectory coverage*, not joint coverage.
- Errors, widths, and interval scores are **normalized using the original training-set output ranges**, not stated in physical units.
- Do not imply transferred split-conformal guarantees under distribution change, calibrated conditional coverage, or physically validated operation of a real microreactor.
- Review against the current **MACE_Introduction.tex**, **MACE_Methodology.tex**, and **MACE_Results.tex** before final drafting.
