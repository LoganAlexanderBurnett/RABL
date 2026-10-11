# B/C/D/F operational-profile accuracy and MACE evaluation

This workflow evaluates the ten existing frozen members on **new physics simulations** over **0–200 seconds**. It does not train models, fit a scaler, select profiles using their outcomes, or tune the shared **0.001** floor. Original training, validation, calibration and test files, checkpoints, historical forecasts and previous studies remain inputs. Outputs go to a new directory.

## The two calibration arms

1. **Original calibration:** the same original 1,000 stochastic calibration profile identities, with fresh forecasts using corrected recursive feedback. Recalculate all quantiles using those forecasts. Application to B/C/D/F is a **nonexchangeable transfer diagnostic**, with no nominal coverage guarantee.
2. **Operational calibration:** 1,000 new B/C/D/F profiles sampled independently of the 2,000 new operational test profiles from the same prespecified distribution. Recalculate all quantiles using this calibration set. The evaluation uses exactly the same operational test forecasts as arm 1.

Each profile's family is an independent uniform draw from B, C, D, F. Parameter draws are independent within and between splits. Counts are exactly 1,000 calibration and 2,000 test profiles, but family counts fluctuate; they are not fixed quotas. This maintains the same iid operational mixture in both splits. Family-specific and phase-specific coverage are diagnostics; pooled calibration does not promise conditional coverage for each family or phase. The original 2,000 stochastic test profiles are retained and not used to choose the operational inputs.

The scaler-use audit found different X-state and Y-target coordinate frames. The new workflow maps each member's previous **Y-scaled** output back to **X-scaled** state input:

\[
x_{\mathrm{feedback}}=y_{\mathrm{prediction}}\frac{y_{\mathrm{span}}}{x_{\mathrm{span}}}+\frac{y_{\min}-x_{\min}}{x_{\mathrm{span}}}.
\]

Use native stored scaler arithmetic, then cast to float32 at the input boundary. Each member retains its own recursive history; subsequent measured states are never injected. Predictions remain in Y coordinates. Original prepared steady-state history is taken directly from the raw original calibration data and verified to be constant at the 45-degree starting condition. Shifted control windows match the existing dataset builder.

This correction is explicit in the new workflow helper. Historical production/audit helpers retain their previous behavior so they can still reproduce historical saved forecasts. **Do not reuse historical quantiles with corrected predictions**, or interpret this run as an isolated profile-shift comparison against the historical rollout. Both arms here share the corrected rollout, so their within-run calibration comparison isolates calibration distribution.

## Prespecified maneuver variations

All positions below are applied to the common drum-angle input, starting from **45 degrees**. Angles and schedules are proposed study choices, not recorded plant histories. Increasing RABL angle removes drum reactivity. Actual moves are piecewise linear in position and have constant signed speed; they are not instantaneous angle steps.

| Family | Variation distribution |
|---|---|
| B: single move/hold | Start uniformly 24–56 s; upward move 1–3 degrees; requested speed 0.05–0.15 degrees/s; hold through 200 s. |
| C: staircase | Three upward increments, each uniformly 0.5–1.25 degrees; starts 24–36, 74–86, 130–142 s; independent requested speeds 0.08–0.16 degrees/s. |
| D: slow move/recovery | Upward move 2–3 degrees starting 20–32 s, duration 40–60 s; return to 45 degrees starting 116–124 s, duration 40–60 s. |
| F: regulation trims | First target 45.25–45.75 degrees at 32–48 s; second target 44.25–44.75 degrees at 84–100 s; return to 45 degrees at 140–152 s; independent requested speeds 0.04–0.12 degrees/s. |

Draws are uniform within these ranges. Start times and movement durations are rounded to the 0.4 s evaluation grid, positions to six decimal places; applied speed is recalculated from the resulting endpoints. All applied speeds are checked to lie within 0.02–0.3 degrees/s, moves may not overlap, and profiles must finish in 200 s. Duplicate command sequences across either split cause a failure rather than a performance-based redraw. The recipe is fixed in `_mace_operational_profiles.py`, recorded in the manifest, and source-hashed.

Each Dymola input is a **MATLAB v4** table named `profile`, with columns time, angle, velocity, acceleration. Repeated event-time knots encode jumps in velocity while position stays continuous. Acceleration is zero on open segments; impulses at idealized corners are not represented. The reactor consumes the position signal; this is not a finite-acceleration actuator model. Simulation extraction is canonicalized to the **501-point, 0.4 s** grid and checked against the planned position signal. Input-table knots represent continuous position between samples exactly.

The earlier literature-informed profiles motivate these finite-speed move/hold families; the sampled amplitudes/times are prespecified experimental choices. These are drum-input challenges with the existing secondary-system boundary conditions. They do not represent independent load-following disturbances or validated closed-loop plant operation.

## Phase reporting

The disjoint phases are:

- **motion:** `[move start, move end)`;
- **early_settling:** `[move end, move end + 20 s)`, excluding any newer motion;
- **late_hold:** remaining post-maneuver points;
- **initial_hold:** points before the first move, reported separately so they cannot pad late holds.

Also report **all** points together. The 20 s settling window is a prespecified analysis window, not a claim that every state settles in 20 s. Change it only before generation if a different window is scientifically justified. Do not choose it from test performance.

Point coverage, mean full width, interval score and accuracy pool **sums and supported point counts**, so variable phase durations are handled explicitly. Aggregate width/score/MAE use equal-target averages in the original scaled coordinate system; aggregate RMSE is the square root of the equal-target mean normalized MSE. Target-specific width/score/MAE/RMSE use physical units (or the simulator's native units where units are not defined in metadata). MAE/RMSE compare the descaled ensemble mean with native physical simulator truth.

All coverage uses one **inclusive float32 scaled endpoint/truth convention**. Interval scores use float64 arithmetic on those endpoints/truth, then multiply by native Y span for physical units. The score at each interval is full width plus `2/alpha` times either miss distance. Reported mean interval score is a point-level score for each band's intervals, including trajectory/joint bands; it is not a separate multivariate trajectory scoring rule.

## Three separate coverage objectives

| Mode | Calibration scores and quantiles | Full-time primary coverage |
|---|---|---|
| horizon | Absolute/normalized residuals at each horizon–target pair; finite-sample quantile across calibration profiles separately for each pair. | Marginal coverage at each horizon/target; phase/all point averages are summaries, not a simultaneous objective. |
| target_trajectory | Each profile's maximum absolute/normalized residual over time, then a separate quantile for each target. | Every time point covered for the target; aggregate is equal-target mean of these target coverages. |
| joint_trajectory | Each profile's maximum absolute/normalized residual over **time and targets**, then one quantile. | Every time point and every target covered in the profile. |

Absolute CP is matched by calibration arm and objective. For MACE, normalize by that profile/time/target's ensemble spread plus the common floor, using `ddof=0`. Use rank `ceil((n_cal+1)*0.95)` with no interpolated quantiles. Quantiles are recalculated separately for both arms and all objectives. Never take the maximum of target-specific quantiles as a substitute for joint calibration.

For trajectory modes, phase coverage means all supported points of that phase in a profile; it is a diagnostic alongside the complete-profile result. A profile with no points in the phase is excluded from that phase's trajectory denominator. Joint-mode per-target coverage is a target trajectory diagnostic, not the joint objective. Full-system coverage is provided only by the joint-mode aggregate joint-coverage metric. Pooled calibration offers marginal objectives over the specified operational mixture, not guaranteed coverage conditional on a maneuver family.

## Paired whole-profile bootstrap

Use one common set of ordinary profile-with-replacement draws (10,000 replicates, PCG64, fixed seed) for **all** methods, arms, phases, targets and families. Entire trajectories travel together; never resample individual time points or targets. For family reports, filter each common global draw to that family, preserving pairing while allowing sampled family counts to fluctuate. Pool sums/counts again in each replicate. Undefined replicates, such as a family with no supported profiles, are counted and excluded from percentile calculations rather than replaced with artificial coverage.

Report descriptive 95% percentile intervals for accuracy and method metrics, paired MACE-minus-absolute differences, MACE/absolute width ratios and width reductions, and operational-minus-original-calibration differences/width ratios. For the latter comparison, ratio means operational width / original width. Calibration, checkpoints and forecasts remain fixed during bootstrap; this quantifies test-profile sampling uncertainty, not calibration-estimation uncertainty. Intervals are not simultaneous bounds across the many comparisons and do not establish exchangeability.

## Run from repository root in your Dymola/PyTorch environment

First fast-forward the branch:

```powershell
git pull --ff-only origin mace-floor-sensitivity
```

Review `scripts/mace_operational_validation_config.json`. Default paths match the previous workflows. `floor_study_dir` must refer to the completed original floor study with passed scaler verification. If the original files have been moved, use `input_overrides` for `external_scaled_h5_path`, `training_scaled_h5_path`, `raw_h5_path`, `reference_scaler_json_path`, and/or `ensemble_experiment_dir`; relocation must preserve file bytes. Paths are resolved against the checkout root. Use a **new** output directory.

Run these stages in order, each command on one continuous line:

```powershell
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_validation_config.json --stage generate
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_validation_config.json --stage simulate
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_validation_config.json --stage forecast
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_validation_config.json --stage evaluate
```

Or run the complete sequence in one command:

```powershell
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_validation_config.json --stage all
```

Generation saves the immutable manifest, phase point counts, v4 MAT tables and a plot of the first twelve test variations per family. Simulation uses the existing `DymolaBatchRunner` and `MicroreactorPK.Experiments.RunOneProfile`; Dymola must be installed/licensed, with the same Modelica dependencies as the original batch runs. The new physics workload is **3,000** independent runs. The forecast stage performs **4,000 profile rollouts × ten members**: original calibration plus operational calibration/test. Do not run ensemble training or dataset scaler-fitting scripts for this study.

An interrupted simulation can resume using the same `simulate` command only when manifest, configuration, runner and Modelica source still match. Every expected result is checked; failed profiles cannot be silently omitted. Forecast/evaluation stages refuse to overwrite or reuse partial outputs; if either is interrupted, preserve that attempt and use a new output directory for a clean run. The `all` stage is for a fresh run, not resumption. Completed simulation/forecast stages are not rerun when progressing to later stages. Source/config changes after generation are rejected rather than mixing versions within one result set.

### Recover the be406101 applied-angle validation failure

The original applied-angle check required agreement within 1e-8 degrees. For the first default profile, rounding trajectory time and angle to float32 and then applying the runner's linear interpolation reproduces a **2.107193843414734e-6 degree** discrepancy. Parsing that extracted CSV as float64 does not restore the lost precision.

The corrected check inspects the **native `data_2` dtype in `simulation/{cal,test}/run_*.mat`**. With verified float32 storage, it derives a pointwise bound from half an angle ULP plus the maximum prescribed angular speed times half a timestamp ULP, with a small float64 arithmetic allowance. It records the native dtype, raw MAT hash, maximum observed difference, derived bound, and agreement with a separately computed quantized regular-grid reference. Without verified float32 storage, the original stricter angle check remains. Timestamp checks remain at 1e-8 seconds. These are analytic-input versus extracted-signal diagnostics; saved forecast/scaler array checks still use exact equality and explicit precision conversions. Existing CSV control and truth values are not rewritten.

**Keep the current configuration, output directory, and raw `run_*.mat` files alongside their CSVs.** After pulling the fix, recover with these commands, each on one continuous line:

```powershell
git pull --ff-only origin mace-floor-sensitivity
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_validation_config.json --stage simulate --resume-after-validation-fix
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_validation_config.json --stage forecast
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_validation_config.json --stage evaluate
```

Do not run `generate` or `all` for this recovery. The explicit recovery flag allows only the two reviewed validation/orchestration files to replace their known be406101 versions, accepting both LF and Windows CRLF file hashes. All other recorded sources, configuration, profile MAT bytes, Modelica source, and Dymola runner must remain unchanged. Forecast/evaluation must not have begun. It archives the original run configuration and simulation metadata, records the migration, and validates cached CSVs before starting more simulations. A fully cached valid split is reused without opening Dymola. For the reported failure, this retains all 1,000 calibration simulations and proceeds to the 2,000 test simulations. Subsequent interrupted simulation attempts can use the ordinary `--stage simulate` command once migration is recorded.

This is a specific recovery path, not a general source-hash override. A changed checkpoint/scaler is still rejected by the frozen-input checks before forecasting; a changed previously validated CSV or native MAT is also rejected. If an angle fails the derived precision bound, inspect that profile's raw output rather than expanding the bound.

Saved member predictions, endpoints and profile statistics require several GB of disk for the default run. The paired bootstrap count matrix requires approximately 160 MB of RAM at 10,000 × 2,000; full test arrays and interval calculations add memory. Forecasts stream profile by profile; evaluation retains arrays and works one mode at a time. Small-count generation/evaluation fixtures are for software checks; production original-calibration forecasting deliberately requires the same original 1,000 profiles and 501 steps.

## Outputs to inspect/share

Within the configured output directory:

- `operational_manifest.json`, `run_configuration.json`, `profile_phase_counts.csv`, `operational_profile_variations.png`;
- `simulation_metadata.json` and `simulation/{cal,test}/results_drum_profile_*.csv`;
- `forecasts/forecast_metadata.json`, `precision_and_input_checks.json`, and corrected `original_cal.h5`, `operational_cal.h5`, `operational_test.h5`;
- `evaluation/accuracy_metrics.csv`: MAE/RMSE with profile-bootstrap intervals by phase/family/target;
- `evaluation/uq_method_metrics.csv`: coverage, full widths and interval scores with intervals, kept separate by objective and calibration arm;
- `evaluation/paired_comparisons.csv`: paired coverage/score/width differences, width ratios and reductions;
- `evaluation/horizon_target_coverage.csv`: actual per-horizon/per-target coverage (diagnostic, without bootstrap intervals);
- `evaluation/phase_support.csv`: profiles/points contributing to each phase/family;
- `evaluation/calibration_quantiles.h5`, `intervals_and_statistics.h5`, `paired_bootstrap_draws.h5`, `stored_array_checks.json`, `evaluation_metadata.json`.

Each saved-array comparison uses exact equality at matched precision. When precision differs, it records the explicit lower-precision conversion and native/post-conversion maximum differences. Solver grid/angle checks are separately labeled with their numerical bounds and native trajectory precision; they are not stored-array tolerance checks. Per-profile diagnostics are recorded in `simulation_metadata.json` under `result_validation`. Coverage is regenerated exactly from the saved endpoints and checked against the saved masks. Input datasets/checkpoints and forecast files are SHA-256 guarded before and after processing; git/runtime/source identities are recorded. Inspect `completed` status before interpreting results.

Run focused software checks with:

```powershell
python -m unittest discover -s tests -p "test_mace_*.py" -q
```

Software fixtures verify the pipeline's numerical behavior. They do not substitute for the actual reactor simulations or frozen production-model evaluation. Share the evaluation CSVs, metadata/check reports and manifest first; the larger HDF5 files preserve independently checkable endpoints, members, calibration scores and bootstrap draws.

## Plot three test forecasts per operational family

After `evaluate` completes, pull the plotting update and run this standalone command:

```powershell
python scripts/plot_mace_operational_forecasts.py
```

The default selects the first three sorted **test** profile IDs from each of B, C, D and F, without inspecting prediction errors or coverage. It produces 48 grids (12 profiles × 2 calibration arms × 2 trajectory views), each as PNG and PDF, in `outputs/ml_results/mace_operational_validation_v1/forecast_plots_4x4`. Selection and input fingerprints are recorded in `plot_metadata.json`; each rendered grid prints progress.

Each grid follows the existing 4×4 layout: drum angle, then Tf, Tm, Thp, TN2, Tsg, steam-outlet temperature, steam quality, six precursors, neutron level and reactivity. Truth is black and ensemble mean blue, with raw ensemble ±2σ in purple, saved MACE-Horizon bounds in green, and saved trajectory bounds in red. All targets use physical units. Target-trajectory and joint-trajectory bands have separate labeled views; original and operational calibration have separate folders. Bands are not clipped to physical limits, and raw ±2σ is explicitly uncalibrated.

For only the operational-calibration horizon/target-trajectory grids (12 grids):

```powershell
python scripts/plot_mace_operational_forecasts.py --calibration operational --trajectory-mode target_trajectory
```

Other options are `--results-dir`, `--output-dir`, `--profiles-per-family`, `--dpi`, `--formats png pdf`, and `--scaler-file` for a relocated byte-identical external scaler HDF5. Relative paths are resolved from the repository root. Existing plot directories are never overwritten; for another view or an interrupted plotting attempt, choose a fresh `--output-dir`.

The script reads `forecasts/operational_test.h5`, `evaluation/intervals_and_statistics.h5`, `evaluation/calibration_quantiles.h5`, completion metadata, the manifest and the original external scaler HDF5. Hashing the large forecast/scaler files can take a minute before rendering begins. It checks the recorded forecast/scaler hashes, profile and target order, native mean/spread, selected endpoints against saved quantiles, and coverage masks at exact stored precision. Precision conversions are explicitly reported. These files are checked for changes after plotting. The same selected profiles and mean forecasts appear in both calibration arms. No inference, simulation, fitting or recalibration is performed.

This standalone plotting addition does not change any source file hashed into the completed scientific run. Do not rerun `generate`, `simulate`, `forecast`, `evaluate`, or `all` merely to obtain these figures. The compact upload used for the initial results review omitted the large forecast/interval HDF5s, so plotting must run where those files are retained. The example figures illustrate individual forecasts; they do not replace aggregate coverage and bootstrap results.
