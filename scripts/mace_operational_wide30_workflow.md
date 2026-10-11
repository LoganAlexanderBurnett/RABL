# Final B/C/D/F study: +/-30 degrees around 45 degrees

Preserve `outputs/ml_results/mace_operational_validation_v1` and the existing local-study configuration. This separate study uses `scripts/mace_operational_wide30_config.json` and writes to **`outputs/ml_results/mace_operational_wide30_v1`**. Nothing in the retained study is overwritten. The original local recipe remains available and unchanged.

## Prespecified broad input distribution

Every profile starts from the same prepared **45-degree steady state**, spans **0–200 s**, and is sampled on the same **501-point / 0.4 s** grid. The allowed angle envelope is **15–75 degrees** (45 +/-30). A profile's initial direction is independently positive or negative with equal probability. Excursion amplitudes are independent uniform draws from **3–30 degrees**; not every profile reaches both endpoints. Family is an independent uniform draw from B/C/D/F in each split. Calibration/test are separate independent streams, with 1,000/2,000 profiles and no outcome selection or fixed family quotas. Profile seed is 20261012; IDs begin at `results_drum_profile_30001` to distinguish this study from the local study.

| Family | Targets relative to 45 degrees | Start windows [s] | Move durations [s] |
|---|---|---|---|
| B: move and hold | One signed excursion of 3–30 degrees, then hold. | 24–40 | 30–60 |
| C: staircase | Three increments in the same sampled direction, totaling 3–30 degrees. Three independent weights drawn from 0.6–1.4 are normalized to partition the total amplitude. | 20–28; 76–84; 132–140 | Each independently 24–32 |
| D: slow move/recovery | One signed excursion of 3–30 degrees, then return to 45 degrees. | 20–28; 120–128 | Each independently 45–60 |
| F: alternating regulation | A signed excursion of 3–30 degrees; an independently drawn excursion of 3–30 degrees on the opposite side of 45; return to 45. | 20–28; 76–84; 140–148 | 28–40; 40–48; 28–40 |

Start times and durations are rounded to 0.4 s; targets to six decimals. Velocity is constant within each move and computed from rounded endpoints/durations. Moves cannot overlap or finish after 200 s, and every target is checked against 15–75 degrees. Applied absolute speeds are checked against **0.01–1.5 degrees/s**. The previous local-study 0.3-degree/s cap would prevent large cross-envelope moves/returns within the chosen timeline, so this recipe samples duration rather than retaining that cap. This changes both amplitude and the possible move speeds; conclusions describe this broader input distribution, not an isolated amplitude-only experiment. These are prespecified stress inputs motivated by move/hold families, not validated plant control commands. Velocity changes are ideal instantaneous corners; finite acceleration limits are not modeled.

No simulations that fail or exceed numerical input-validation bounds may be silently removed or replaced with easier profiles. Inspect failures before completing the study.

## Run only the new study

Run from the repository root in the same Dymola/PyTorch environment used for the completed local study. Each command is one continuous line:

```powershell
git pull --ff-only origin mace-floor-sensitivity
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_wide30_config.json --stage new-only
```

`new-only` requires a fresh output directory and a completed original-calibration reuse source. Before generating anything or opening Dymola, it validates retained calibration provenance, original profile/target identities, dataset/scaler/checkpoint hashes, reviewed prediction/scaling source identities and runtime/precision settings. The provided path points to the completed local study. Keep its large `forecasts/original_cal.h5`, `forecast_metadata.json`, `evaluation/evaluation_metadata.json` and `operational_manifest.json`; the compact uploaded results ZIP is not a sufficient cache. Keep the original datasets and ten checkpoints at their existing paths, or use byte-preserving `input_overrides` as documented in the local workflow.

The command then generates/simulates **3,000 new profiles**, copies the original 1,000 corrected calibration forecasts **byte-for-byte**, forecasts only the new 1,000 calibration + 2,000 test profiles using the same ten trained members, and evaluates the two calibration arms. Original-calibration inference runs are **zero**. No old operational simulation or forecast is rerun. The cache's native HDF5 manifest identity remains intact; the new metadata records its origin, SHA-256 fingerprints and byte-identical copy. Saved means/spreads/truth and precision conversions are independently checked before reuse. Recalibration still happens for every objective and arm; copied forecasts do not mean copied legacy quantiles.

Runtime checks include CPU/GPU, library versions and numerical settings. If the preflight rejects a changed environment, restore the original environment rather than weakening the check. If original-calibration inference really must be recomputed in a different environment, use a separate fresh output directory, set `reuse_original_calibration_dir` to `null`, and use `--stage all`. That alternative recomputes original calibration; it is not `new-only`.

You can instead run each new-study stage separately, reviewing input profiles before simulation:

```powershell
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_wide30_config.json --stage generate
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_wide30_config.json --stage simulate
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_wide30_config.json --stage forecast
python scripts/run_mace_operational_validation.py --config scripts/mace_operational_wide30_config.json --stage evaluate
```

The original-calibration cache is checked at `forecast` when running individual stages. `new-only` adds the up-front cache check so a bad cache is caught before expensive simulations. After an interrupted simulation, rerun only `--stage simulate` with unchanged configuration/source/Modelica/runner; validated results are reused. Then advance to `forecast` and `evaluate`. These later stages refuse to overwrite partial outputs. `new-only` and `all` are fresh-run entry points, not resume commands. Pull once before generation and keep the source/configuration fixed for the complete run.

## Accuracy, calibrated intervals and uncertainty

The trained models, train-only external scaler, corrected own-member recursive Y-to-X feedback, alpha=0.05, common floor=0.001, and population ensemble spread (`ddof=0`) are fixed. Both arms use exactly the same new test forecasts:

1. **Original calibration:** use the retained corrected forecasts of the original 1,000 stochastic profiles. Recompute horizon–target, target-trajectory and joint-trajectory quantiles. Application to broad operational profiles is a nonexchangeable transfer diagnostic, without a nominal coverage guarantee.
2. **Broad operational calibration:** use the 1,000 new broad calibration profiles, independent of the 2,000 new tests from the same prespecified mixture. Recompute all quantiles from this set. Objectives are marginal over the pooled mixture, not guaranteed conditional coverage for every family or phase.

All accuracy, coverage, full-width ratios against matching absolute CP, interval scores and paired whole-profile bootstrap comparisons are the same as the [local workflow](mace_operational_validation_workflow.md). Report **all**, **motion**, **early settling (20 s)**, **late hold**, and **initial hold**, by pooled/family/target scopes. Keep horizon, target trajectory and joint trajectory separate. Target widths/scores/accuracy use physical units; aggregates use equally weighted scaled targets. Coverage uses the single inclusive float32 scaled endpoint/truth convention. Ten thousand paired whole-test-profile bootstrap draws give descriptive 95% intervals conditional on the fixed calibration sets. The seed is 20261013. Do not resample individual time points or use these intervals as simultaneous bounds over all comparisons.

Inspect `completed` and precision checks, `phase_support.csv`, `accuracy_metrics.csv`, `uq_method_metrics.csv`, `paired_comparisons.csv` and `horizon_target_coverage.csv`. Preserve the HDF5 members, endpoints, quantiles and bootstrap draws for reproducibility. The broad results do not replace the local results; compare the two studies as different prespecified operating distributions.

## Separate consolidated forecast reports

To regenerate the retained local plots as PNGs plus one main PDF, without running the scientific pipeline:

```powershell
python scripts/plot_mace_operational_forecasts.py
```

That writes `forecast_report_4x4/operational_forecasts.pdf` inside the local study. Existing scientific results and the earlier `forecast_plots_4x4` directory are retained.

After the broad evaluation completes, run:

```powershell
python scripts/plot_mace_operational_forecasts.py --results-dir outputs/ml_results/mace_operational_wide30_v1
```

This writes **`forecast_report_4x4/operational_wide30_forecasts.pdf`** inside the broad study and individual PNGs in calibration/objective/family subfolders. No individual PDFs are created. The default has 48 grids: the first three sorted test profiles per B/C/D/F family, shown under both calibration arms and both separate target/joint trajectory views; horizon bounds appear in each view. A cover and 16 provenance section pages organize the 65-page report and identify each profile's seed, simulator/forecast origin, angle range, prescribed moves and speeds. `plot_metadata.json` records exact saved-array checks and the page index. Selection is independent of accuracy/coverage. The plotting script is read-only with respect to scientific results; it never simulates, infers or recalibrates. Existing plot report folders are protected; use a fresh `--output-dir` for an additional attempt.

To review results here, upload the broad study's evaluation CSVs, metadata/check reports, manifest, profile-variation PNG and consolidated PDF. Retain the large forecast/interval HDF5s locally; they are needed to regenerate the report.
