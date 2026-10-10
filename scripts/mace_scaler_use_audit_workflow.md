# Frozen-model external scaler and scaler-use audit

Run from the repository root in the existing PyTorch environment:

```bash
python scripts/run_mace_scaler_use_audit.py --config scripts/mace_scaler_use_audit_config.json
```

The default new output is `outputs/ml_results/mace_scaler_use_audit_v1`.
Zip that complete directory for review, including when the audit flags a
coordinate mismatch or numerical replay difference. Existing output directories
are never overwritten; choose a new `output_dir` for a subsequent execution.

## What is fixed

The completed `outputs/ml_results/mace_floor_sensitivity_v1` study provides the
original dataset, forecast, scaler, split and checkpoint identities. SHA-256
checks require the same scientific input bytes and all ten original checkpoints.
Models are loaded for inference only. The audit does not train, refit a scaler,
change profile splits, alter the common 1e-3 floor, replace saved forecasts,
or modify production rollout/conformal code.

Three calibration and three test profiles are selected by evenly spaced
positions in the fixed manifest order, independently of observed performance.
All time indices, target channels and ensemble members are replayed. The actual
stored input-window length is used: a configured lookback need not equal the
number of rows in a prepared input window.

The original raw prepared `X` windows and `Y` rows, training scaled HDF5,
external scaled HDF5, experiment scaler JSON, member metadata, audit forecasts,
and saved calibration/test intervals must be available locally. The paths are
read from the preceding study's config. The additional checkpoint metadata paths
are `ensemble/model_0/member_metadata.json` through member 9 within that
experiment. Architecture parameters come from those metadata files, and strict
checkpoint loading verifies the weights against that architecture.

## External scaler verification

By default `recompute_train_extrema: true` independently recomputes minima,
maxima and spans from raw **training** profiles only, checks the external scaler
against training metadata and the float32 JSON export, and repeats actual
raw-to-scaled sample checks for another target, steam quality and the control.
This reuses the independent arithmetic in `_mace_scaler_audit.py`; it does not
call the production fitting helper. Prepared steady-state history is included
because it was included in the original fitting domain.

For every selected calibration/test profile the audit also checks complete
`X` and `Y` arrays, including all history positions and control windows, against
the independent native-precision formula:

```text
scaled = (raw_physical - saved_training_min) / saved_training_span
```

The production scaling function is evaluated against this independent formula.
The actual production `ProfileDataset` outputs from both scaled datasets are
then compared against the reconstructed arrays after an explicit float32 cast,
matching the model input boundary. The production scaler loader's float32
conversion is recorded and compared with the native HDF5 values.

Setting `recompute_train_extrema: false` can skip the repeated full training
scan. That mode explicitly reuses the preceding successful independent scaler
audit and still requires matching SHA-256 values for raw/scaled datasets,
scaler JSON and checkpoints. It still checks all selected profile arrays and
execution inputs. It is not a substitute for an initially completed scaler audit.

## Recursive feedback must use the input coordinate frame

The current production `rolling_forecast` directly appends each member's
output prediction to its state-input history. That is valid only when the
input-state and output-target scaling frames agree. Independently fitting
`X` and `Y` extrema can produce different frames even when target names agree.

The audit checks minima and spans exactly for corresponding channels. It
records the required coordinate conversion:

```text
X_scaled = Y_scaled * (Y_span / X_span) + (Y_min - X_min) / X_span
```

The mapping uses native saved scaler arithmetic, followed by explicit float32
conversion at the model input boundary. Its identity form preserves exact
values when the frames agree. `feedback_coordinates.json` contains both frames,
the per-target multipliers and offsets, and their relationship to the current
direct-feedback implementation.

Forward hooks capture the tensors actually passed to each production model
at every horizon, and its output at each step. The audit independently checks
the initial raw-reconstructed state history, the complete future control
window, and each member's own previous outputs. A separate comparison maps
those previous outputs into the required X coordinates. This separates two
questions:

- Does the current code reproduce the recorded historical recurrence?
- Does that recurrence preserve the intended physical coordinate transformation?

Exact historical replay can coexist with a failed coordinate check. Do not
treat reproducing old forecasts as proof that the physical feedback is correct.
No corrected-rollout forecasts or corrected-method performance claims are
generated by this audit. Fixing feedback and evaluating its consequences would
be a separate, explicit step, with new forecasts and recalibration of all
affected conformal methods.

## Forecasts, descaling and intervals

The audit compares production member trajectories against an independently
implemented recurrence using the same frozen weights and raw-reconstructed
float32 inputs. It checks population standard deviations with ddof=0 and member
ordering, then compares member predictions, ensemble means and spreads against
the original stored audit arrays.

The production saved-forecast loader and UQ application functions are exercised
directly. The current checkout's source is used rather than a stale installed
RABL package; actual function locations, line ranges and SHA-256 values are
reported in `source_trace.json`. Production files and audit inputs are checked
for changes during execution.

For all four absolute/MACE horizon/targetwise trajectory methods, the audit
checks the full-calibration quantiles against the preceding study and saved UQ
metadata. Quantiles are **not** recalibrated on the few selected profiles. The
saved common floor is checked at 1e-3 across every target. Independent endpoint
arithmetic is compared with both the production application and saved physical
and scaled interval arrays for the selected calibration and test profiles.
The MACE effective scale and normalized errors are also compared directly.

Means and endpoints are descaled with the target offset and span; controls use
their input offset and span. Standard deviations transform with the span only.
When stored physical two-sigma endpoints are present, their original
float32 member-descaling/std calculation is independently reproduced. A
diagnostic comparison reports the rounding difference between taking the
standard deviation after float32 member descaling and scaling the original
standard deviation. It never treats the offset as an additive standard deviation.

Another diagnostic reports physical-endpoint width versus scaled full width
times the native target span. These can differ because physical float32
endpoints were rounded separately. Canonical widths and scores in the preceding
studies use float64 scaled endpoint subtraction and the native physical span.
Control raw-versus-descaled roundtrip differences are also reported rather
than incorrectly requiring every lossy float32 roundtrip to be exact.

## Numerical convention and audit statuses

No `allclose`, relative tolerance, absolute tolerance, or automatic numerical
drift acceptance is used. Equal stored precision requires exact array equality.
Different stored precision is compared after explicitly documented conversion
to the lower floating precision, with the native and converted maximum
differences and differing-entry counts both retained. Per-feature maximum
differences are recorded where the last array dimension identifies a target
or input channel.

`device: auto` uses CUDA if available, otherwise CPU. `device` can explicitly
be `cpu`, `cuda` or a CUDA device such as `cuda:0`. CPU replay of a historical
CUDA run may differ even with identical inputs and weights. Python, NumPy,
h5py, PyTorch, CUDA/cuDNN versions, device, thread count and backend precision
flags are recorded. Existing backend precision choices are preserved; the
audit does not silently force different numerical kernels or precision.
Replay differences require review, not a silently broadened threshold.

| Status | Meaning |
|---|---|
| `passed` | Required input, coordinate, production/reference and stored replay comparisons all matched. |
| `input_output_coordinate_mismatch` | Required deterministic comparisons matched, but X/Y frames or actual recursive feedback differ from the required input coordinates. |
| `replay_review_required` | Structural and coordinate checks matched, but historical stored forecasts did not replay exactly under the recorded environment. |
| `audit_failed` | A required invariant comparison failed; inspect the individual records. |
| `execution_failed` | The audit could not finish, for example because an original input hash changed, a file is missing or an array is invalid. |

Completed audit statuses other than `passed` return shell exit code **2** while
preserving the full output for review. `completed: true` means the audit ran
to completion, not that every check passed. Execution failures leave
`completed: false` and include the error in metadata. Diagnostic-only rounding
differences do not cause a failure. If multiple categories fail, metadata lists
all counts even though the status prioritizes invariant, coordinate, then
historical replay mismatches.

## Outputs

- `audit_metadata.json`: completion/pass status, scientific identities/hashes,
  selected profiles, checkpoint architectures, runtime and category counts.
- `scaler_verification.json`: repeated independent train-only fitting and
  sample checks, present when the default full scan is enabled.
- `source_trace.json`: the source functions actually exercised and their hashes.
- `feedback_coordinates.json`: native input-state/output-target frames and the
  required affine feedback mapping.
- `comparisons.json`: exact comparisons, explicit precision conversions,
  maximum differences and per-feature diagnostics.
- `replays.h5`: audit-only replay member trajectories, means/spreads, and actual
  versus required model inputs at first/middle/last horizon indices per member.
- `replay_coverage_effects.csv`: changed coverage events and width/score effects
  of numerical replay differences at the original full-calibration quantiles.
- `audit_summary.md`: readable completion status and interpretation.

Replay coverage effects use the same inclusive float32 **scaled** endpoints as
the floor and bootstrap studies. Physical widths are per-target; scaled widths
and scores use float64 arithmetic. These effects concern only the selected
profiles and frozen quantiles. They are not a new calibrated method evaluation,
a test-set aggregate performance estimate, or an assessment of a corrected
feedback method. Horizon and trajectory rows remain separate. This audit does
not modify or re-evaluate the optional joint method.

For relocated inputs, `input_overrides` can contain the original six dataset,
scaler and forecast config keys, plus `ensemble_experiment_dir` and
`saved_uq_comparison_dir`. The original scientific-input SHA-256 values must
still match. No checkpoint substitution is permitted.

## Tests

```bash
python -m unittest discover -s tests -p 'test_mace*.py' -v
```

New tests use actual tiny PyTorch LSTMs and strict checkpoint reloads. They
verify exact end-to-end reproduction, immutable inputs, one-ULP mismatch
detection, explicit precision conversion, actual control corruption despite
matching metadata, changed-checkpoint rejection, source/input provenance, and
the case where historical replay is exact but feedback coordinates are wrong.
