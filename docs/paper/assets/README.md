# Supplied manuscript figure sources

These five PNGs preserve the user's supplied images byte-for-byte. Their filenames
are normalized only to make paths portable. They are original source assets,
not automatically corrected publication figures.

| File | Supplied source | Intended use |
|---|---|---|
| `hpmr_bop_viz.png` | `hpmr-bop-viz(1).png` | Figure 4; review labels against current Modelica plant |
| `lstm_architecture.png` | `LSTM Architecture.png` | Figure 2a; correct actual dimensions/architecture |
| `neural_autoregression.png` | `NeuralAutoregression.png` | Figure 2b; correct timing and Y-to-X feedback mapping |
| `rolling_forecast.png` | `Rolling Forecast.png` | Appendix A1 after pseudocode correction |
| `timeseries_batch_0003_ALL_PROFILES.png` | Same name | Figure 5; fallback when original simulation CSVs are unavailable |

The quantitative runner copies source graphics and can wrap raster images in
PDF/SVG containers. That does not improve resolution or make raster labels editable.
The manual corrections are specified in `../MACE_Introduction_Methodology_Outline.md`.
No broader-angle study figure is included.
