# MACE Methodology manuscript and figures

The section is in [MACE_Methodology.tex](MACE_Methodology.tex). This is a
LaTeX section for insertion after the existing [Introduction](MACE_Introduction.tex),
not a replacement for the full article. The methods add one citation in
[MACE_Methodology_References.bib](MACE_Methodology_References.bib). Compile
with both this file and `MACE_Introduction_References.bib`.

## Figure status

| Figure | Manuscript asset | Source / required action |
|---|---|---|
| 1. Correlated velocity sampling | `outputs/paper/mace_figures/sampling/fig01_control_trajectory_sampling.pdf` | Existing validated scientific renderer; **run using stored original batch-3 controls** before publication |
| 2. LSTM / autoregressive feedback | `docs/paper/figures/fig02_lstm_autoregression.tikz.tex` | New editable vector diagram, included with `\input` |
| 3. MACE normalized scores / event sets | `docs/paper/figures/fig03_mace_calibration_events.tikz.tex` | New editable vector diagram; schematic event cells, not results |
| 4. Updated reduced microreactor plant | `docs/paper/figures/fig04_microreactor_plant.tikz.tex` | New editable vector diagram based on current Modelica equations and wrapper; earlier raster retained as source at `docs/paper/assets/hpmr_bop_viz.png` |
| 5. Original 4×4 simulator dataset | `docs/paper/assets/timeseries_batch_0003_ALL_PROFILES.png` | Supplied image preserved, not misrepresented as forecasting results. Re-render from raw batch data to verify trace count |
| 6a. Frozen calibration comparison | `docs/paper/figures/fig06_calibration_design.tikz.tex` | New editable vector study schematic |
| 6b–e. Operational control families | `outputs/paper/mace_figures/operational_controls/fig06_bcde_operational_controls.pdf` | Existing renderer reads retained `operational_manifest.json`; selects lowest three evaluation IDs per family |
| A1. Recursive forecast algorithm | `docs/paper/figures/figA1_recursive_forecast_algorithm.tikz.tex` | Source ready for a later appendix; not inserted into Methodology |

**No archived production control CSVs or operational manifest are committed to
this branch.** Figure 1 and Figure 6(b–e) must therefore be built on a workstation
with the archived data, and their sidecar metadata reviewed. The existing plot
runner can generate an *illustrative* Figure 1 when the CSVs are unavailable; it
labels this explicitly and it must not be presented as sampled training data.

The manuscript embeds data-dependent figures by their production paths, rather
than committing substitute traces or invented sample identifiers. To compile
the full manuscript, generate these assets first.

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
