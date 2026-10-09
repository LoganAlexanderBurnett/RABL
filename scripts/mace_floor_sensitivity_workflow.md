# Fixed-forecast MACE floor sensitivity study

Run from the repository root on the computer containing the original ensemble,
forecast HDF5 files, and datasets:

```bash
python scripts/run_lstm_floor_sensitivity.py --config scripts/lstm_floor_sensitivity_config.json
```

Only NumPy and h5py are needed for the calculations; matplotlib is needed when
`plots` is true. There is no torch import, checkpoint deserialization, training,
new forecast generation, or new test-set selection. The ten checkpoint files
are read only to calculate their SHA-256 hashes. Every input HDF5 is opened
read-only. Existing outputs are never overwritten.
Checkpoint paths are `ensemble_experiment_dir/ensemble/model_0/model.pt`
through `ensemble_experiment_dir/ensemble/model_9/model.pt`, matching training.

## Inputs and fixed design

The example JSON uses the paths from the current saved run and original ensemble
training script. Adjust paths if these files were moved. Paths are interpreted
relative to the current working directory, not relative to the JSON file.
The raw dataset is the original **unscaled prepared X/Y dataset**, which contains
the steady-state history used in training, not an inverse transform of scaled
data. Keep this input independent of the scaler being checked.

The original ensemble's `datasets/scaled_dataset.h5` is compared directly with
the external scaled HDF5 used by the UQ run. The JSON reference must be the
ensemble's `scaling_statistics.json`, whose values were exported after the
original loader converted scaler arrays to float32. Its source precision is
recorded explicitly. This is different from a generic frozen scaler JSON.

By default the script requires 5,000 training, 1,000 validation, 1,000 calibration,
and 2,000 test profiles, 501 indexed time steps, ten members, population ensemble
standard deviation (`ddof=0`), and alpha=0.05. Counts/steps in the JSON allow small
integration fixtures, not a different scientific study. The saved split manifests
from the ensemble experiment are used instead of regenerating repository manifests.
All four splits must be disjoint and match both scaled HDF5 datasets. Calibration
identity, alpha, and the common current floor are checked against saved UQ metadata.

The seven fixed common **additive** floors are:

| Floor in scaled space |
|---:|
| 1e-6 |
| 1e-5 |
| 1e-4 |
| 3e-4 |
| 1e-3 |
| 3e-3 |
| 1e-2 |

There is no per-target tuning or test-based floor selection. Physical floor
contributions differ by target span and are recorded in `target_metrics.csv`.

At every floor, compute `abs(truth - saved_mean) / (saved_std + floor)` anew.
MACE-Horizon selects the conformal order statistic separately at every
horizon–target pair. MACE-Trajectory first takes the maximum over time for each
profile and target, then selects each target's order statistic across profiles.
For 1,000 calibration profiles and alpha=.05 the one-based rank is 951.
The two matching absolute-residual CP calibrations are independently recomputed
once from the same forecasts. The ten saved member predictions are checked for
finite values and compared with the saved aggregates; reported differences do
not replace the original saved means or spreads.

## Numerical convention and independent checks

Calibration, floors, quantiles, and interval endpoints use float32 to match the
existing implementation. **All new coverage results use inclusive comparisons
of scaled float32 truth and endpoints**, including horizon, target trajectory,
and complete multivariate trajectory coverage. Widths and interval scores use
float64 endpoint arithmetic. Per-target physical widths/scores are scaled
widths/scores times the native HDF5 target span; offsets are not rounded into
widths. Cross-target aggregates remain scaled. No mixed-unit physical average
is reported. Interval score is full width plus `(2/alpha)` times the lower/upper
miss distances. It is the ordinary pointwise interval score averaged over the
trajectory, not a proper score for a joint trajectory prediction set.

Scaler verification runs before the sweep. It checks:

1. Feature order and native min/max/span arrays between the external and training
   HDF5 files, plus min/span against the float32 saved run scaler.
2. Independent min/max/span recomputation over **only raw training profiles**,
   flattening all X windows and Y rows, matching the original fitting scope.
   Constant-feature spans use the documented replacement of zero by one.
3. Actual raw-to-stored-scaled values in **both scaled HDF5 datasets** via `(physical-min)/span`, for steam
   quality, TN2 (the first target), and the control input. First/middle/last rows
   of first/middle/last profiles from every split are checked. Forecast truth is
   also compared with the full stored scaled Y arrays for every calibration/test
   profile, and the control with the dataset's last-window control channel.

Same-precision arrays require exact equality. Different precisions are compared
after explicitly rounding both to the lower precision; native and converted
maximum differences and mismatch counts are reported. No `allclose` or broad
tolerance is used by the study. A scaler/sample mismatch stops the run after
writing the diagnostic report. Matching extrema establishes consistency with
train-only fitting; it cannot prove historical provenance if including held-out
profiles happened to produce identical extrema.

At 1e-3 and for both absolute baselines, the script compares regenerated quantiles
with saved calibration metadata. If current method forecast HDF5 files are
available, lower/upper arrays are compared exactly at matching precision.
These large method files are optional; their absence is recorded explicitly.
Regenerated scaled coverage is compared with legacy physical float32 coverage,
including disagreement counts, affected targets, and example profile/time
indices. Saved metrics are compared against a reconstruction of the previous
mixed-space convention. This distinguishes endpoint-rounding changes from a
different calibration or dataset. Every maximum numeric difference is reported;
small differences from floating reduction order are not hidden behind a tolerance.
Coverage fractions are additionally converted to integer covered-event counts
using their known denominators; these counts must match exactly. Original float
differences remain in the report. Quantile, covered-event count, or saved
interval-array mismatches stop successful completion.

## Outputs and interpretation

| File | Contents |
|---|---|
| `study_metadata.json` | Fixed settings, profile lists, input/checkpoint/source hashes, code revision, completion flag |
| `scaler_verification.json` | Native precision comparisons, raw extrema and actual transformation samples |
| `forecast_verification.json` | Saved aggregate, truth and control comparisons for every profile |
| `recalibration.h5` | Quantiles for every floor and absolute CP, all targetwise calibration trajectory maxima, rank and effective float32 floors |
| `aggregate_metrics.csv` | Separate horizon/trajectory coverage, scaled full widths, scores and matching baseline ratios |
| `target_metrics.csv` | Target coverage, physical widths/scores and ratios, width percentiles, quantile summaries and target trajectory confidence intervals |
| `horizon_target_metrics.csv` | Every target/horizon coverage, physical mean width, width ratio and pointwise confidence interval |
| `floor_fraction_distributions.json` | Per-target calibration/test distributions and histograms of tau/(sigma+tau) |
| `saved_run_reconciliation.json` | Current-floor and absolute quantile/interval comparisons and legacy coverage reconciliation |
| `floor_sensitivity.pdf`, `.png` | Aggregate width ratios and steam-quality coverage/width diagnostics |

Coverage intervals are descriptive 95% Wilson intervals over independent test
profiles at a **fixed horizon–target pair**, or over whole-profile target
trajectory events. Times are not counted as independent observations. These
intervals are not simultaneous across all targets, horizons or floors. Horizon
coverage averaged over time is a summary; inspect every horizon–target cell for
the actual calibration objective. Targetwise trajectory coverage does not imply
95% coverage of every output over an entire multivariate trajectory.

Assess 3e-4, 1e-3 and 3e-3 as a neighborhood: do width and interval-score gains
persist, do target coverages remain compatible with the intended objective,
and do steam-quality quantiles and width distributions vary predictably?
The script reports all floors and does not declare success or choose a winner.
When tau dominates spread, recalibrated MACE tends toward matching absolute CP.
Quantiles change with tau, so increasing tau need not increase every width.

Focused validation (no model dependencies):

```bash
python -m unittest discover -s tests -p test_mace_floor_sensitivity.py -v
```

Tests cover independent profile-max order statistics, exact parity with the
existing numerical calibrators at all seven floors, the large-floor limit,
precision-sensitive comparisons, scaled coverage events, corruption despite
matching scaler metadata, and the complete HDF5 workflow with immutable inputs.
