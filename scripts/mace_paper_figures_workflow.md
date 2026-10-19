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
| `operational_controls` | Figure 6: operation-informed control maneuvers PNG/PDF; data NPZ | Three evaluation trajectories per retained maneuver type, reconstructed from saved moves. Panel titles omit family letters; legend entries are Trajectory 1–3, not source IDs. |
| `existing_assets` | Original plant, LSTM architecture, autoregression, and Rolling Forecast graphics | Exact copied source PNGs plus raster PDF wrappers. These are explicitly marked as needing manual revisions. |
| `operational_forecasts` | Existing 4×4 forecast PNGs and one sectioned PDF | Calls the existing saved-result renderer; original/operational calibration and per-output/joint trajectory views stay separate. |
| `representative_forecast` | `mace_representative_test_forecast.png/.pdf` | One saved operational test trajectory with simulator truth, ensemble mean, uncalibrated ±2σ, MACE-Horizon, and MACE-Trajectory (or joint-trajectory) bands. Selection by fixed test ID, independent of prediction error. |

The `--figures all` command preserves the historical Methodology/report defaults.
Use `--figures results` to generate the manuscript Results Figures 6–10,
**including** the reinstated Figure 6 operational control maneuvers.
Pure flowcharts, the MACE event diagram, and manual artwork corrections
are excluded from automatic scientific rendering.

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

Wait for the completed corrected stochastic forecasts, floor calibration,
and profile validation. The figure renderers read archived results, and do
not train, run new simulations or reestimate calibration quantiles.

Generate every Results figure and the Overleaf-ready numerical tables:

```powershell
python scripts/generate_mace_paper_figures.py --figures results --representative-count 10 --overwrite
python scripts/generate_mace_results_tables.py --config scripts/mace_paper_figures_config.json
```

The Results figure ordering is:

| Figure | Stable renderer ID | Saved output |
|---|---|---|
| 6: Operation-informed control trajectories | `operational_controls` | `fig06_operational_controls.pdf` |
| 7: Stochastic representative forecasts | `stochastic_representative` | `fig07_stochastic_representative_01.pdf` … `_10.pdf` |
| 8: Stochastic UQ performance | `stochastic_uq_performance` | `fig08_stochastic_uq_performance.pdf` |
| 9: Operational representatives | `operational_representatives` | `fig09_operational_representatives_01.pdf` … `_10.pdf` |
| 10: Spread vs. error and width | `spread_error_relationship` | `fig10_spread_error_relationship.pdf` |

Individual CLI figure IDs remain stable; only manuscript figure numbers and
filenames change. Figure 6 titles are (a)–(d) and describe each maneuver
without letters B/C/D/F. Legends use 'Trajectory 1', 'Trajectory 2', and
'Trajectory 3'; original profile IDs and angle extrema stay in metadata.
The Figure 10 spread binning and line plots are unchanged from the
formerly numbered Figure 9.

Figure 7 remains a 4×4 simulator/ensemble-mean/four calibrated UQ-method
comparison, with the previous shading and slimmer envelope edges and
gray ensemble-mean line. Crucially, **every** chosen profile is audited at
every forecast time and for every output and conformal method. Plotted
endpoints are reconstructed in float64 from the exact archived fitted
quantiles and spreads to avoid float32 endpoint rounding; independent
analytic physical-unit widths are compared against those actually plotted.
The older float32 pathway's zero-width rate and >5% relative loss rate
are also reported. Inspect the saved
`fig07_representative_interval_width_audit.csv`
in the `stochastic_representative` figure directory. It contains
per-profile, per-output, per-method width summaries in physical units,
alongside saved original quantile ranges. Recalibration, prediction,
ensemble mean/spread, or interval scoring are NOT redone.

Figure 8 coverage panels (a) and (b) use y limits 0.9–1.0.
Panel (b) uses the actual 15 symbolic variable names on the x-axis with
tilted labels. Panels (c)–(d) place each bar's exact value rounded to three
decimal places above the bar.

Figure 9 still overlays the four operational test groups with group colors,
using only operational-calibrated MACE-Horizon uncertainty. For the selected
test profiles in families B, C and D, reference x_steam_out must be below
unity at least once. Group F is sampled from ALL its saved evaluation profiles
without this requirement (because the retained family-F profiles may not
reach x_steam_out<1). The distinct rule is recorded in both metadata and
caption, alongside profile IDs and actual unsaturated fractions.

Set the number of selected stochastic and operational plots via
`--representative-count N` (default N=10) and the
seed via `--representative-seed 20261016`. Selection is
uniform without replacement among qualifying trajectories. If any group
has fewer than N eligible test profiles, the corresponding renderer fails
with an eligibility count, without changing its selection criteria.

The Results tables are now transposed and divided by output group, with
**variable symbols across columns** and each UQ method spanning three
separate rows: empirical coverage, normalized full width, and normalized
interval score. All 15 output variables are retained in exactly one of:

- five reactor and thermal-node temperatures,
- six delayed-neutron precursor concentrations,
- reactor power, reactivity, steam outlet temperature, steam outlet quality.

The pooled mean across all 15 outputs is in the last group, clearly
distinguished from the four variable-specific columns. The 95% nominal
coverage event differs for Horizon, per-output Trajectory and joint
multivariate Trajectory calibration, as defined in Methodology.

In `docs/paper/MACE_Results.tex`, three separate accuracy tables appear
in Section 3.1; three stochastic UQ tables plus one small joint UQ table
appear in Section 3.2; and three transferred-calibration tables, three
independently recalibrated tables and one paired-bootstrap comparison
appear near their respective discussions in Section 3.3. Each is a
standalone LaTeX float and has its own citation label. The manuscript
tables are embedded, so **Overleaf requires no extra table inputs**.

`scripts/generate_mace_results_tables.py` generates the exact same
**layout** from frozen results, writing two files containing grouped
float environments, plus the detailed data CSV. The new output version
is `outputs/paper/mace_results_tables_v3` to preserve v2:

- `table05_forecast_accuracy.tex`: three accuracy floats, three stochastic
  UQ floats, and one joint multivariate coverage float
- `table06_operational_uq.tex`: three transferred-calibration UQ floats,
  three operation-informed calibration UQ floats, and the pooled paired
  bootstrap differences
- `table_metrics_detail.csv`: exact point estimates and available 95%
  whole-trajectory bootstrap bounds, including paired comparisons by output

**Units:** coverage is a fraction; the MAE, RMSE, mean full interval width,
and mean interval score displayed are dimensionless after normalization
by each output's original training-set range. Display of power in MW in
plots does not change the normalization used for tables. For stochastic
and operational UQ, the archived per-output physical widths/scores are
divided by that original training span.

To regenerate:

```powershell
python -m unittest discover -s tests -p "test_mace_*.py" -q
python scripts/generate_mace_results_tables.py --config scripts/mace_paper_figures_config.json
```

The generator still rejects existing output directories and never
recomputes model forecasts, conformal quantiles, or bootstrap samples.

## Publication figure refinements

- Figure 6: All four operation-informed maneuver titles are descriptive and
  omit internal letter codes. Three example commands are shown in a
  colorblind-conscious navy/teal/coral palette (line styles also differ).
  The legend appears ONLY in the top-left corner of panel (b), stacked in
  three rows, and the archive still records the original profile IDs.
- Figure 7: The same four calibrated conformal bands and their audit are
  preserved; the two-line figure legend is closer to the first row without
  overlapping. The attached audit from 10 representative profiles showed
  no >5% float32 width loss and no zero-width loss. Plotting is unchanged
  scientifically.
- Figure 8: Output-variable symbols in the trajectory coverage subplot
  are displayed at a larger font size.
- Figure 9: Six operational panels in this order: row 1 = drum angle,
  P (MW), reactivity in dollars; row 2 = fuel temperature, outlet steam
  temperature, outlet steam quality. Colors identify distinct maneuver
  trajectories, not output variables. A thin black stroke outlines each
  dash of the corresponding colored ensemble mean. The legend uses
  descriptive maneuver types, omitting internal family letter codes.
- Figure 10: The empirical spread-versus-error curve is dark gray to
  distinguish it from the colored MACE and absolute-CP intervals; the
  plotted bin values, definitions, and comparative panel are unchanged.

### Power-display consistency

Native LSTM target `n` is the dimensionless neutron-power multiplier.
For EVERY physical-power figure, its reference, point forecast and interval
endpoints are rendered as `P [MW] = 6 n` using the common plotting-unit
constant `POWER_MW_PER_N = 6.0`. This includes the manuscript Results
Figures 7 and 9, the legacy 4x4 operational forecast report, and the
independent 4x4 operational representative plot. The Figure 5 simulator
dataset already contains `P_MW` and is **not** multiplied again.
Figure 8's coverage is dimensionless so its `n` column is only relabeled
as `P`, with no transformation to coverage or loss statistics.
Saved forecasts, CP quantiles, and normalized calibration/evaluation metrics
stay in their original coordinates.

### Plotting refinements and optional Figure 10 backup

The scientific archives and nominal calibration objectives are unchanged.

- **Figure 6 controls:** Every maneuver panel automatically scales its own
  y-axis to the actual saved control-angle traces plus the 45-degree
  reference line, using a small data-relative margin. No shared y-axis,
  universal 44–49 degree limit, or fixed y-tick array is imposed.
- **Figure 7 representatives:** All four CP envelopes have equal thin
  boundary strokes; the simulator reference, control-angle trace, and
  dashed gray ensemble-mean trace use thin lines as well. The existing
  interval-width audit remains active and unchanged.
- **Figure 8 coverage:** Panel (b) retains the symbolic, rotated output
  names as tick labels but no longer adds an 'Output variable' axis label.
- **Figure 9 operational forecasts:** The four maneuver-color MACE-Horizon
  fills are more opaque (alpha 0.22, previously 0.10); the top row moves
  closer to the figure legend with space retained above the plots.
- **Figure 10 spread/error:** The original two-panel publication figure
  remains **unchanged** as `fig10_spread_error_relationship.pdf` (and PNG).
  Every invocation now also produces
  `fig10_spread_error_relationship_three_panel.pdf` (and PNG). Its
  **left-hand histogram** counts whole stochastic test trajectories
  according to their per-trajectory mean absolute error in the frozen
  output-scaled coordinates. For each of the 2,000 saved evaluation
  trajectories, MAE averages absolute errors equally over all 501 forecast
  times and all 15 targets (one observation per trajectory). The other
  panels exactly reproduce the original spread/error and interval-width
  comparisons, renamed (b) and (c) in the optional version. A matching
  `fig10_trajectory_scaled_mae.csv` saves the trajectory IDs and 2,000
  per-trajectory MAE values for review. Histogram bin count is 30; this
  does not alter the spread-quantile binning or any CP result.

Use the normal Results command to regenerate both variants:

    python scripts/generate_mace_paper_figures.py --figures results --representative-count 10 --overwrite

Or regenerate Figure 10 alone:

    python scripts/generate_mace_paper_figures.py --figures spread_error_relationship --overwrite

### Figure 8 legibility update

The Horizon coverage subplot (a) now uses the vertical range 0.93–1.00,
while the per-output trajectory coverage subplot (b) uses 0.92–0.98.
Both internal legends use 8.5-point text, and the numerical labels above
panels (c) and (d) use 8-point text with three decimal places. These
are plotting-only settings: the nominal 95% reference, saved coverage,
widths, interval scores, and method comparisons are unchanged.
