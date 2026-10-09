# Fixed-run profile bootstrap, horizon diagnostics, and joint trajectory bands

Run from the repository root, using the environment containing NumPy and h5py
(and matplotlib when plots are enabled):

```bash
python scripts/run_mace_profile_validation.py --config scripts/mace_profile_validation_config.json
```

The default input is the completed `outputs/ml_results/mace_floor_sensitivity_v1`
study. The default new output is `outputs/ml_results/mace_profile_validation_v1`.
No existing output directory is overwritten. No model is loaded, trained, or
used for new inference. The fixed main floor remains **1e-3**, alpha=.05, with
10,000 bootstrap replicates and deterministic seed 20261009.

## Required local inputs and provenance

Keep the completed floor study and its original audit calibration/test forecast
HDF5 files and external scaled dataset available. The runner reads the original
paths from the prior study's configuration; paths are relative to the current
working directory. If files were moved, set an `input_overrides` object with any
of `external_scaled_h5_path`, `calibration_forecasts_h5_path`, or
`test_forecasts_h5_path`. Override files must have the same SHA-256 as the prior
study's recorded inputs. It is unnecessary to repeat raw-training scaler fitting.

The prior study must be complete with a passing scaler verification and intact
inputs. The four profile lists must remain unique and disjoint. Input fingerprints,
profile names, configuration, source hashes, seed, code revision and completion
status are saved. Newly read forecasts are verified against the dataset and
saved ensemble members. Current horizon and targetwise trajectory quantiles and
all calibration profile maxima are independently regenerated once and must
match the completed study **exactly**. They remain frozen during resampling.

## Paired whole-profile bootstrap

For each replicate, draw N test-profile indices uniformly **with replacement**,
where N is 2,000 in the production run. Every method, output and metric uses
those same indices. A profile retains all 501 horizons and 15 targets together.
The script averages widths, scores and pointwise coverage within each profile,
and computes targetwise/all-target trajectory events before resampling.
These per-profile sufficient statistics produce exactly the same estimands as
resampling complete equal-length trajectories, without repeatedly allocating
all forecast arrays. Replicate draws and arithmetic are invariant to batch size;
`bootstrap_batch_size` controls temporary memory, not the statistical procedure.
The SHA-256 of the common integer index stream is recorded for reproducibility.

Bootstrap confidence intervals are the 2.5th and 97.5th percentiles of each
replicate statistic. They quantify test-profile sampling uncertainty conditional
on the fitted ensemble and fixed calibration set. They do not include variability
from training or recalibration. Test profiles must be independent sampling units;
if profiles share a simulation family that induces dependence, the sampling unit
must instead be that family. Cross-horizon and cross-target dependence within
profiles is preserved. Intervals are **marginal/descriptive**, not simultaneous
across all targets, metrics, modes, and horizons. Percentile intervals can be
unstable or degenerate for very rare/no observed misses; report the actual event
counts and interval limitations rather than interpreting a degenerate interval
as proof of certainty.

Report each method's coverage, mean width and mean pointwise interval score, plus:

- Width and interval-score differences: **MACE minus absolute CP**; negative
  differences favor MACE for these metrics.
- Relative reductions: `1 - mean(MACE)/mean(absolute)`; positive reductions favor
  MACE. Compute ratios of resampled means, never averages of profile ratios.
- Coverage differences: MACE minus absolute CP, in fractions; multiply by 100
  for percentage points. Positive differences do not alone prove better overall
  interval performance, nor does an interval containing zero prove equivalence.

Aggregate widths/scores remain in scaled coordinates. Target-specific widths
and scores are converted using the native target span and labeled with physical
units where known. Coverage and relative reductions are fractions. Undefined
ratios from zero baseline means are stored as NaN in HDF5 and counted explicitly
in CSV via `valid_replicates`/`undefined_replicates`; their CI is conditional on
finite replicates and must not be treated as an unconditional interval. Empty
finite distributions have blank bounds. Ordinary pointwise interval scores are
averaged over time even for trajectory bands; this is not a proper score for a
joint trajectory set.

## Low-coverage horizon diagnostics

Both MACE-Horizon and absolute horizon CP are evaluated using the same inclusive
**float32 scaled endpoints** as the completed floor study. Per-target/per-horizon
rows include test coverage, descriptive 95% Wilson bounds over profiles,
calibration endpoint coverage, paired method disagreements, physical widths and
errors, spread/floor diagnostics, the calibrated quantile, and control changes.
Contiguous low-coverage regions are recorded for two separate descriptive
criteria: coverage below nominal and an **unadjusted** Wilson upper endpoint
below nominal. These cells are correlated and the intervals are not corrected
for multiplicity. They are diagnostic flags, not independent hypothesis tests.

The default selects the three lowest-coverage cells per target for each method
and records up to ten largest miss distances per selected cell, with profile
identity, scaled truth/endpoints, physical error and control context. Selection
uses the observed test data; these examples are exploratory, not fresh validation.

Control regimes are defined independently of prediction errors:

- `initial`: horizon index zero.
- `control_changing`: successive physical control change greater than 1e-4 degrees.
- `control_holding`: successive change at most that threshold, after index zero.

The threshold is explicit in config/metadata. Holding the control does not imply
that the physical system has reached steady state; changing it does not exhaust
all types of reactor transients. Regime coverage pools time points descriptively;
it has no point-level binomial CI or conditional coverage guarantee. Inspect
these summaries and miss examples to assess whether low coverage aligns with
control transitions, late rollout, or other behavior; do not infer causation from
these tables alone.

## Optional whole-system joint trajectory comparison

`include_joint: true` adds **separate** normalized and absolute joint bands.
For each calibration profile, take the maximum residual score across **all
horizons and all targets**, then the corrected conformal order statistic across
profiles. The production one-based rank is 951:

- MACE: `max_{t,j} abs(y_scaled - mean_scaled)/(std_scaled + 1e-3)`.
- Absolute baseline: `max_{t,j} abs(y_scaled - mean_scaled)`.

The absolute joint baseline uses unit scale in each target's train-minmax-scaled
coordinates, making its cross-target maximum dimensionless. Apply one scalar
quantile per method to all endpoints using the same respective scales. Do **not**
take the maximum of separately calibrated target quantiles. Calibrate both joint
methods once, then hold their quantiles fixed for the bootstrap.

The primary joint objective is the fraction of profiles covered at **every time
and every output**. Per-target joint-band rows report trajectory coverage as a
secondary diagnostic. Horizon and existing targetwise trajectory constructions
remain unchanged and have their own matching absolute baselines. The joint
comparison assesses whether efficiency persists at an explicitly matched
whole-system objective; it introduces no new-method novelty claim. Setting
`include_joint: false` omits the joint calibration and comparison outputs.

## Outputs

| File | Contents |
|---|---|
| `validation_metadata.json` | Fixed settings, input/source hashes, RNG/index-stream hash, profile lists, completion flag and inferential scope |
| `cal_forecast_verification.json`, `test_forecast_verification.json` | Repeated fixed-input forecast consistency checks |
| `profile_statistics.h5` | Aligned per-profile widths, scores, pointwise fractions, targetwise and joint trajectory events |
| `bootstrap_method_metrics.csv` | Point estimates and percentile intervals for each method, aggregate and target scopes |
| `bootstrap_comparisons.csv` | Paired differences and ratio-of-mean reductions, estimates, bounds and valid-replicate counts |
| `bootstrap_distributions.h5` | All method/comparison replicate distributions, using common profile draws |
| `joint_calibration.h5` | Optional scalar joint quantiles, per-profile joint scores and calibration identities |
| `horizon_diagnostics.csv` | Every current horizon/target diagnostic for MACE and absolute horizon CP |
| `horizon_target_summary.csv` | Worst cells, flagged-cell counts and longest low-coverage regions |
| `low_coverage_regions.csv` | Contiguous regions under each descriptive criterion |
| `worst_horizon_miss_examples.csv` | Miss examples from selected lowest-coverage cells |
| `control_regime_metrics.csv` | Descriptive coverage/error/width/score by control regime |
| `bootstrap_width_reductions.pdf/.png`, `horizon_coverage.pdf/.png` | Optional bootstrap interval plot and paired coverage heatmaps |

Zip the new output directory for interpretation after the run. Actual production
results require the original local forecasts; synthetic tests are implementation
checks, not scientific results. A folder with `completed: false` is incomplete.

Focused tests:

```bash
python -m unittest discover -s tests -p 'test_mace*.py' -v
```
