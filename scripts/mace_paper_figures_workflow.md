# MACE paper figure generation

Run from the repository root in the existing `pymola` environment. This is a
read-only plotting workflow: no Dymola startup, training, model inference,
scaler refitting, floor selection, or conformal recalibration. The state notation
in the companion introduction/methodology outline is now **x(t)**.

## Commands

Each command below is one continuous line, suitable for PowerShell.

```powershell
git fetch origin
git switch mace-floor-sensitivity
git pull --ff-only origin mace-floor-sensitivity
python scripts/generate_mace_paper_figures.py --list
python scripts/generate_mace_paper_figures.py --config scripts/mace_paper_figures_config.json --figures all
```

If plotting dependencies are missing:

```powershell
python -m pip install numpy scipy matplotlib pillow h5py
```

Only selected figures (IDs can be space-separated or comma-separated):

```powershell
python scripts/generate_mace_paper_figures.py --figures sampling dataset operational_controls
python scripts/generate_mace_paper_figures.py --figures sampling --sampling-mode stored
python scripts/generate_mace_paper_figures.py --figures dataset --dataset-mode stored_image
python scripts/generate_mace_paper_figures.py --figures operational_forecasts
python scripts/generate_mace_paper_figures.py --figures representative_forecast --calibration-source operational --trajectory-mode target_trajectory
```

After editing figure settings or code, replace only selected figure folders:

```powershell
python scripts/generate_mace_paper_figures.py --figures sampling operational_controls --overwrite
```

Use a separate destination instead of overwriting existing figure versions:

```powershell
python scripts/generate_mace_paper_figures.py --figures all --output-dir outputs/paper/mace_figures_v2
```

## Available figures

| Stable ID | Output | Source behavior |
|---|---|---|
| `sampling` | Three-panel Figure 1, PNG/PDF; sampling data NPZ | First eight sorted original batch-3 control CSVs. If absent in `auto` mode, explicitly labeled illustrative GP draws; `stored` requires actual tables. |
| `dataset` | Required 4×4 Figure 5, PNG/PDF | All batch-3 simulator CSVs if present; defaults require exactly 5,000. Raw regeneration shows the variable name as each subplot title, with no repeated variable label on the y-axis. `stored_image` copies the pre-existing image unchanged. |
| `operational_controls` | Optional operation-informed controls PNG/PDF; data NPZ | Lowest three evaluation IDs in each retained maneuver type, reconstructed from saved moves. Plot legends now show angular excursions and not the numeric suffixes of profile IDs. Figure 6 is omitted from the manuscript. |
| `existing_assets` | Original plant, LSTM architecture, autoregression, and Rolling Forecast graphics | Exact copied source PNGs plus raster PDF wrappers. These are explicitly marked as needing manual revisions. |
| `operational_forecasts` | Existing 4×4 forecast PNGs and one sectioned PDF | Calls the existing saved-result renderer; original/operational calibration and per-output/joint trajectory views stay separate. |
| `representative_forecast` | `mace_representative_test_forecast.png/.pdf` | One saved operational test trajectory with simulator truth, ensemble mean, uncalibrated ±2σ, MACE-Horizon, and MACE-Trajectory (or joint-trajectory) bands. Selection by fixed test ID, independent of prediction error. |

`--figures all` builds the standard registered figures except the optional
`operational_controls` renderer for omitted Figure 6. Request that stable ID
explicitly when revisiting the control-trajectory illustrations. Pure flowcharts, the MACE event diagram, and manual artwork corrections
are excluded from automatic scientific rendering. Figure 6 is no longer included
in the Methodology. Its archived optional control plot remains accessible via
`operational_controls`. The runner does not present uncorrected source graphics
as final manuscript figures.

The operational forecast report retains the earlier output rule: individual PNGs
and **one consolidated PDF**, not a separate PDF per 4×4 grid. Main manuscript
scientific figures are individually exportable PDF/PNG masters. Optional SVGs
can be requested with `--formats png pdf svg`; dense dataset paths are rasterized
inside vector files to avoid enormous exports.

## Configuration and source paths

Edit `scripts/mace_paper_figures_config.json` if the saved results live elsewhere.
Relative paths resolve against the repository root. Paths beginning with
`outputs/` honor `RABL_OUTPUT_ROOT` if set; supplied asset paths remain relative
to the repository. An explicit output directory follows the same rule.

Default input locations:

- `outputs/variography_profiles/batch_0003/drum_profile_*.csv`.
- `outputs/sim_profiles/batch_0003/results_drum_profile_*.csv`.
- `outputs/ml_results/mace_operational_validation_v1/operational_manifest.json`.
- For forecasts, also `forecasts/operational_test.h5`, forecast/evaluation metadata,
  `evaluation/intervals_and_statistics.h5`, `evaluation/calibration_quantiles.h5`,
  and the exact frozen scaler file. The default scaler path matches the existing
  floor-study configuration; change `operational.scaler_file` if the identical
  file has been relocated. Saved hash checks still apply.
- `docs/paper/assets/` contains the supplied images, so those exports and the
  dataset-image fallback work after pulling without manually copying attachments.

A results ZIP containing summaries/images without the large forecast/interval
HDF5 files can generate control panels and asset exports, but cannot regenerate
the forecast report. Point the script at the full saved run on your computer.
Missing inputs produce a named failure and exit status 1. Other requested figure
generators continue, and successfully generated figures remain available. Thus
`all` never quietly claims success for an incomplete set.

The sampling figure uses the repository's covariance and variance-design helper.
Its conditioned dispersion uses the actual regularized velocity transformation
and exact zero first velocity. The 2.5° design target is for the unconditioned
covariance. In illustrative mode, a Cholesky sampler produces reproducible
illustrations; it does not claim to reproduce the original generator's SVD-based
random draws or the training dataset. Settings, seed, selected IDs, grid/integration
differences, and curve data are recorded. Confirm original generation records
before treating configured sampling parameters as all-batch dataset provenance.

The broader-angle recipe is rejected explicitly for both operational generators.
Control panels use identical 44–49° axes and 0–200 s axes. Dataset panels use the
supplied display order and physical power P, while the forecast report continues
to display the model's normalized power output n. No bounds are clipped.
All regenerated 4×4 plots use each variable name only as its subplot title.
Variable names are not repeated on the y-axis or x-axis. A shared figure-level
label identifies the time axis. The six precursor variables appear as $c_1$ through
$c_6$ without a misleading "(normalized)" suffix.

### Regenerate and select a representative forecast

The separate Results-ready representative figure uses only archived data from
`operational_test.h5` and its saved conformal quantiles and intervals. It does
not recalculate scores or select a favorable example. By default, it selects
the lexically first saved operational test profile ID. To select a different
predefined test case, specify the exact ID from `operational_manifest.json`.

```powershell
python scripts/generate_mace_paper_figures.py --figures representative_forecast --calibration-source operational --trajectory-mode target_trajectory --overwrite
$manifest = Get-Content outputs/ml_results/mace_operational_validation_v1/operational_manifest.json -Raw | ConvertFrom-Json
$testId = ($manifest.profiles | Where-Object { $_.split -eq "test" } | Select-Object -First 1).id
python scripts/generate_mace_paper_figures.py --figures representative_forecast --profile-id $testId --calibration-source original --trajectory-mode joint_trajectory --overwrite
```

The second run demonstrates the explicit profile-ID override using a test ID
read from the actual manifest. Replace `$testId` with any other test-profile ID
to select a particular recorded maneuver. Original calibration represents
the nonexchangeable transfer diagnostic. Operational calibration uses matching
independent samples from the prespecified maneuver mixture. The plot metadata
records selected test IDs, control moves, hashes, and exact array-consistency
checks. Each 4×4 plot includes prescribed drum angle and all 15 model outputs.

### Rebuild the manuscript dataset image

```powershell
python scripts/generate_mace_paper_figures.py --figures dataset --dataset-mode raw --overwrite
Copy-Item outputs\paper\mace_figures\dataset\fig05_generated_transient_dataset.png docs\paper\assets\timeseries_batch_0003_ALL_PROFILES.png -Force
```

The copy operation is optional until you have visually inspected the regenerated
image. It updates the raster asset already referenced by `MACE_Methodology.tex`.
It should only be used after the batch-3 simulation CSVs have passed the
complete-profile count check. `--dataset-mode stored_image` does **not** update
labels because it preserves the original image exactly.

## Outputs, reruns, and provenance

Default destination: `outputs/paper/mace_figures/<figure_id>/`. Each successful
folder contains the figure(s), `caption.md`, and `figure_metadata.json` with source
hashes, renderer/configuration identities, effective settings, selected sample
IDs where relevant, and output hashes. `figure_index.json` records the last run
and retains other successful figure entries across selected reruns.

Existing selected folders require `--overwrite` or a new `--output-dir`. Rendering
and source validation happen in a temporary folder before replacing a selected
result. An input/validation failure leaves its previous result untouched. Other
figure folders and the simulation/forecast sources are never overwritten.

## Adding figures for later manuscript sections

1. Add a renderer in `_mace_paper_figures.py` with signature
   `(config, out, path, options)`. Read frozen source files; write only under `out`.
2. Return `files` (paths relative to `out`), `inputs` (records from `fingerprint`),
   `caption`, and any sample-selection/unit/method metadata. Use `save_figure`
   for common formats and styling. An imported asset may stay raster; label that.
3. Add its stable descriptive ID, renderer name, and description to `REGISTRY`
   in `generate_mace_paper_figures.py`. Add its configuration block.
4. Test data selection, units, source integrity, and the scientifically important
   arithmetic. Check a rendered output at publication size.

This keeps results figures independent of manuscript numbering. Future accuracy,
coverage, width, interval-score, and floor figures should consume archived metrics
or endpoints using the paper's single recurrence/numerical convention. The current
runner does not invent missing corrected stochastic-study results or recalculate
calibration to obtain a more favorable figure.


## Results figures and copy-paste LaTeX tables

**Run only after the corrected stochastic forecasts, floor study, and profile validation finish.**
The Results renderers only read the saved studies. They do not simulate, train, forecast, recalibrate, or resample.

```powershell
python scripts/generate_mace_paper_figures.py --figures stochastic_representative stochastic_uq_performance operational_representatives spread_error_relationship
python scripts/generate_mace_results_tables.py --config scripts/mace_paper_figures_config.json
```

Figure PDF/PNG files are written to separate descriptive-ID directories under outputs/paper/mace_figures.
Tables are written as complete copy-paste-ready LaTeX environments in outputs/paper/mace_results_tables_v1:
- table05_forecast_accuracy.tex: pooled stochastic and operational normalized MAE/RMSE.
- table06_operational_uq.tex: both operational-test MACE-Horizon calibration arms, paired bootstrap intervals, and a paired difference.

All manuscript figures use a fixed source-inventory check and save input hashes, selected profile IDs and captions.
The source reference x_steam_out (not predictions) must be <1 for >=10% of 501 forecast samples.
One middle-sorted eligible stochastic test ID and one from each operational family is used; no ranking by forecast error.
If an operational family lacks a qualifying test profile, rendering fails rather than selecting a different kind of profile.
Figure 6 follows Figure 5's 4x4 variable order and titles and includes two MACE and two matched absolute CP methods, with no uncalibrated ensemble band.
Figure 7 makes Horizon-to-Horizon and Trajectory-to-Trajectory comparisons.
Figure 8 uses trajectory-group colors, solid simulator curves, dashed forecasts, and ONLY operationally calibrated MACE-Horizon intervals.
Figure 9 defines bins from saved ensemble spread, then aggregates errors and widths from all stochastic test predictions.

The figure generator does not overwrite canonical docs/paper/figures assets; visually inspect the rendered outputs before including them.
For new table versions change results.results_tables_output_dir in the JSON config (existing table directories are protected).
The tabular LaTeX files use no nonstandard package; adjust table* float width in the Overleaf project if needed.

For all four new Results figures alone, use `python scripts/generate_mace_paper_figures.py --figures results`. The historical `--figures all` selection continues to mean the legacy standard figure set, without the new Results renderers.
