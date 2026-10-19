# MACE Manuscript — Conclusion Outline

**Purpose:** Close the argument developed in the Introduction and evaluated in Results. Synthesize the contributions, accurately qualify the statistical findings, and identify defensible next steps without repeating the Methodology.

**Suggested length:** Approximately 350–500 words in four or five connected scientific prose paragraphs. This document is a paragraph-level outline, not a full drafted Conclusion.

## Paragraph 1 — Research question and integrated framework
**Function:** Return to the Introduction's central question: how structured control-driven simulation data, accurate recursive forecasts, and calibrated adaptive intervals work together for multivariate transient prediction.

- Identify MACE as the combination of **temporally correlated control-trajectory sampling**, a **ten-member independently recurrent LSTM ensemble**, and **ensemble-spread-normalized split conformal prediction**.
- Recall the distinction between **Horizon**, **per-output complete-trajectory**, and **joint complete-multivariate trajectory** coverage as different statistical events.
- Place the evidence in the reduced **heat-pipe microreactor and balance-of-plant** simulation application: 15 output variables and 200-s transient forecasts.
- Avoid restating Gaussian-process covariance details, architecture configuration, or formulas for nonconformity scores.

**Transition:** Move from what was constructed to what independent test trajectories demonstrated.

## Paragraph 2 — Recursive predictive accuracy and matched stochastic calibration
**Function:** Summarize the strongest positive findings in terms of their actual evaluation objectives.

- On the stochastic test set, pooled normalized MAE/RMSE were **0.00154 / 0.00359**; on the operation-informed test set, **0.00141 / 0.00256**. These describe **complete autoregressive forecasts**, not teacher-forced one-step fits.
- At roughly nominal **95% empirical coverage** on the matched stochastic calibration/evaluation distribution, adaptive MACE intervals were narrower and had better interval scores than like-for-like Absolute CP intervals:
  - **MACE-Horizon:** mean width **8.9%** lower, interval score **34.6%** lower.
  - **MACE-Trajectory:** mean width **43.9%** lower, interval score **50.0%** lower.
- The reduced widths/scores held across all 15 output variables for the matched stochastic paired-method comparisons, although magnitudes varied by output.
- If retaining a joint-calibration result, use at most one sentence: MACE joint coverage **0.9525**, width **0.0169**, versus absolute joint coverage **0.9445**, width **0.0398**. Do **not** compare those as though joint and per-output Trajectory calibration addressed the same event.
- Infer **empirical interval efficiency for the evaluated matched populations**, not conditional validity for each individual trajectory.

**Transition:** Accurate central forecasts and matched calibration do not establish transferability of conformal thresholds.

## Paragraph 3 — Operational calibration transfer and recalibration
**Function:** Give appropriate emphasis to the most revealing result: reliable uncertainty depends on the calibration distribution even when forecast centers are unchanged.

- Describe the controlled comparison: thresholds from the original stochastic calibration set were transferred to the **same 2,000 operation-informed test forecasts** subsequently evaluated with independent operation-informed recalibration. Model weights, predicted means, and ensemble spreads remained fixed.
- With transferred thresholds, pooled **MACE-Trajectory per-output complete-trajectory coverage** was **0.6800**. Operation-informed recalibration raised it to **0.9549**; the paired change was **+0.2749**, with a descriptive 95% whole-trajectory bootstrap interval **[+0.2666, +0.2831]**.
- Acknowledge that recalibration did not uniformly improve every interval metric:
  - **MACE-Horizon:** coverage **0.9479 → 0.9559**, mean normalized width **0.0073 → 0.0062**, interval score **0.0092 → 0.0074**.
  - **MACE-Trajectory:** recovery of its more stringent time-simultaneous coverage objective required width **0.0121 → 0.0136**, and average interval score **0.0123 → 0.0137**.
- After matched operation-informed recalibration, MACE retained lower pooled mean width than its corresponding Absolute CP baseline: **0.0062 versus 0.0076** (Horizon) and **0.0136 versus 0.0246** (per-output Trajectory). Avoid claiming uniform per-variable dominance.
- The paired bootstrap resamples **whole evaluation trajectories** and is conditional on fixed models and calibration sets. It does not quantify retraining or recalibration-sample variability.

**Transition:** Relate the observed efficiency and distribution dependence to the distinct roles of spread and conformal thresholds.

## Paragraph 4 — Mechanistic interpretation and limitations
**Function:** State what the results support, and just as importantly, what they do not.

- The observed positive association between **ensemble spread and absolute forecast error** supports spread as a **forecast-dependent normalization scale**. Conformal thresholds, rather than spread alone, determine calibrated interval bounds.
- The standard split-conformal coverage argument requires **exchangeable calibration and evaluation trajectory samples** for the relevant coverage event. Dependence *within* each trajectory can be accommodated using whole-trajectory scores, but neither pooled coverage nor a spread–error association proves **conditional coverage** for individual maneuvers, outputs, or shifted populations.
- Scope the physical evidence: **one reduced simulator**, selected prescribed (not measured-in-service) control trajectories, **15 variables**, a fixed **200-s horizon**, and fixed ensemble/calibration archives. These results are not operating-reactor validation.
- Recalibration changes interval multipliers and bounds, **not the accuracy of the frozen ensemble forecast**. Do not blur those two sources of performance.

## Paragraph 5 — Future investigation and closing claim (optional)
**Function:** Provide a restrained outlook, then finish with a single generalizable finding.

- Frame extensions as prospective investigations: richer or higher-fidelity physical dynamics; broader, longer or unexpected control sequences; validation on further systems; variability across retrained ensembles and fresh calibration datasets; and shift-aware or sequential recalibration strategies.
- Acknowledge that operational use would require new evidence of distribution-appropriate interval coverage under realistic conditions and a justified calibration-update protocol.
- Final takeaway direction: **forecast-dependent ensemble scaling can sharpen conformal uncertainty bands, but dependable coverage requires calibration representative of the trajectories being evaluated**.
- If the eventual Discussion already treats limitations and future work extensively, condense Paragraphs 4 and 5 to prevent repetition.

## Final editing checklist
- Match the Introduction's four-part narrative: structured sampling, recursive forecasting, MACE/UQ comparisons, and operation-informed recalibration. The Conclusion should answer those contributions rather than introduce another agenda.
- Use the agreed terminology: *control trajectory*, *output variables*, *multivariate response trajectory*, *forecast trajectory*, *calibration distribution*, and *evaluation distribution*.
- Keep **prediction accuracy**, **empirical coverage**, **normalized mean interval width**, **interval score**, and **calibration transfer** distinct.
- When quoting 0.6800 and 0.9549, explicitly identify **mean per-output complete-trajectory coverage**. This is **not** 15-output simultaneous joint coverage.
- Avoid claims of uniform improvement from recalibration, per-maneuver conditional guarantees, distribution-free validity under arbitrary shifts, new physical validation, or improved point-forecast accuracy caused by recalibration.
- Do not propose maneuver-phase results: the final manuscript intentionally does not report that breakdown.
- Cross-check the current **MACE_Introduction.tex**, **MACE_Methodology.tex**, and **MACE_Results.tex** before converting the outline into prose.
