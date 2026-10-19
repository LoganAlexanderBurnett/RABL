# MACE Methodology manuscript and figures

The section is in [MACE_Methodology.tex](MACE_Methodology.tex). It describes the Multivariate Adaptive Conformal Ensemble (MACE) approach. This is a
LaTeX section for insertion after the existing [Introduction](MACE_Introduction.tex),
not a replacement for the full article. The methods add one citation in
[MACE_Methodology_References.bib](MACE_Methodology_References.bib). Compile
with both this file and `MACE_Introduction_References.bib`.

## Figure status

The manuscript uses descriptive maneuver names in prose and figures. Internal
B/C/D/F identifiers are retained only in the archived simulation manifests and
filenames so the original datasets remain reproducible.

| Figure | Manuscript asset | Source / required action |
|---|---|---|
| 1. Correlated velocity sampling | `outputs/paper/mace_figures/sampling/fig01_control_trajectory_sampling.pdf` | Existing validated scientific renderer; **run using stored original batch-3 controls** before publication |
| 2. LSTM / autoregressive feedback | `docs/paper/figures/fig02_lstm_autoregression.tikz.tex` | New editable vector diagram, included with `\input` |
| 3. MACE normalized scores / event sets | `docs/paper/figures/fig03_mace_calibration_events.tikz.tex` | New editable vector diagram; schematic event cells, not results |
| 4. Updated reduced microreactor plant | `docs/paper/figures/fig04_microreactor_plant.tikz.tex` | New editable vector diagram based on current Modelica equations and wrapper; earlier raster retained as source at `docs/paper/assets/hpmr_bop_viz.png` |
| 5. Original 4×4 simulator dataset | `docs/paper/assets/timeseries_batch_0003_ALL_PROFILES.png` | Supplied image preserved, not misrepresented as forecasting results. Re-render from raw batch data to verify trace count |
| A1. Recursive forecast algorithm | `docs/paper/figures/figA1_recursive_forecast_algorithm.tikz.tex` | Source ready for a later appendix; not inserted into Methodology |

**Figure 6 is excluded from the Methodology and the standard manuscript figure build.**
The previously developed study schematic and operational-control plots remain
available as optional repository resources but are not required or cited.
The control recipe and calibration comparison are retained in manuscript tables.

**No archived production control CSVs are committed to this branch.** Figure 1
must therefore be built on a workstation with the archived data, and its
sidecar metadata reviewed. The existing plot runner can generate an
*illustrative* Figure 1 when the CSVs are unavailable; it labels this
explicitly and it must not be presented as sampled training data.

The manuscript embeds the data-dependent Figure 1 by its production path,
rather than committing substitute traces or invented sample identifiers.
Generate this asset before full manuscript compilation. Figure 5 uses the
existing source image.

## Commands

From repository root (one continuous command per line):

```bash
python -m pip install numpy scipy matplotlib pillow h5py pymupdf
python scripts/build_mace_methodology_figures.py --diagrams-only
python scripts/build_mace_methodology_figures.py
```

Use `pdflatex` (and TikZ) to typeset the editable conceptual figures, or run the
build script for standalone vector PDFs and 300-dpi PNG previews. The build script
does not launch Dymola, retrain models, change scaler files, generate physics
samples, or recalibrate intervals. The second command uses the repository's
existing data-figure renderer, which is documented in
[scripts/mace_paper_figures_workflow.md](../../scripts/mace_paper_figures_workflow.md).
Stored-data mode is the default. An optional `--illustrative-sampling` flag
must be passed explicitly to generate labeled, non-empirical sampling previews.

## LaTeX preamble

```tex
\usepackage{amsmath,amssymb,graphicx,booktabs,array,tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,matrix}
```

Figure ` \input ` paths and scientific-plot paths are relative to **repository
root**. Run LaTeX from that directory or adjust `\graphicspath` and `\input`
paths for the final paper's build directory. The manuscript carries no hardcoded
performance results.

## Evidence remaining

- Verify source manifests and original sampling/velocity-table hashes for every
  stochastic batch; the generated sampler rejects *angle* violations, but does
  not implement a hard speed-rejection check. The sub-1-degree/s criterion is
  currently the retained-data constraint, pending original archive verification.
- Check full partition genealogy and duplicate profiles, including possible
  shared branches, before asserting independence of the original draws.
- Regenerate/reconcile original stochastic evaluation and calibration forecasts
  using the corrected Y-to-X feedback recurrence. Historical stochastic metrics
  produced by direct feedback cannot be combined with corrected operational
  results as though they share one forecast definition.
- Check executed run-time simulator parameter overrides, solver identity/version,
  and the normalized precursor labeling used in the figure before submission.
- Regenerate Figure 5 from original batch arrays if precise population counts
  or readable labels are required; do not infer a count from the existing raster.

## Operation-informed control figure

The Methodology now includes Figure 6 immediately after introducing the
operation-informed maneuver types, with the exact Overleaf-relative asset
path `figures/fig06_operational_controls.pdf`. Copy the existing generated
figure to that Overleaf location; the manuscript intentionally does not
link to a local `outputs/` figure path. Results Figure 8 remains unchanged;
joint complete-multivariate trajectory coverage is instead summarized in
Results Table 5. The manuscript does not promise a maneuver-phase UQ
breakdown or claim per-maneuver conformal guarantees.
