![RABL_LOGO](misc/rabl_logo.png)

# RABL

RABL is a Python + Modelica workflow for microreactor transient studies. It combines three pieces in one repository:

1. **Variography-based profile generation** for stochastic control-drum trajectories.
2. **Batch Dymola execution** for running those profiles through Modelica experiments.
3. **Machine-learning dataset tooling** for scaling, splitting, and preparing LSTM-ready datasets.

---

## What is in this repo

```text
.
├── src/
│   ├── rabl/
│   │   ├── variography/       # Drum profile generation + branching
│   │   ├── interface/         # Dymola batch orchestration
│   │   └── machine_learning/  # Dataset prep + LSTM pipeline utilities
│   └── modelica/              # Modelica package(s)
├── scripts/                   # End-to-end and plotting CLIs
├── tests/                     # Test fixtures and generated reference artifacts
├── misc/                      # Images/figures/logo
└── README.md
```

> `outputs/` is created at runtime by scripts (for generated profiles, simulation outputs, and dataset artifacts).

---

## Installation

RABL is packaged via `pyproject.toml` and currently requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Dependencies and environment notes

- The Python package currently declares `torch` as a dependency.
- Variography and dataset tooling rely on common scientific packages (NumPy/SciPy/Matplotlib/H5 tools), which should be installed in your environment.
- Dymola workflows require a licensed Dymola install plus the `dymola` Python interface available to the active interpreter.

---

## Core workflows

## 1) Generate stochastic drum profiles (Python API)

```python
import numpy as np
from rabl.variography import DrumProfileGenerator

t_grid = np.linspace(0.0, 200.0, 2001)

gen = DrumProfileGenerator(kernel="matern52", ell=10.0)
profiles = gen.generate(
    t_grid=t_grid,
    n_realizations=3,
    baseline_angle_deg=45.0,
    seed=123,
)

profiles[0].save_csv("outputs/variography_profiles/example/drum_profile_00001.csv")
profiles[0].save_mat("outputs/variography_profiles/example/drum_profile_00001.mat")
```

## 2) Batch Dymola from generated profiles (Python API)

```python
from rabl.interface import BatchConfig, DymolaBatchRunner

cfg = BatchConfig(
    profiles_dir="../../../outputs/variography_profiles/batch_0003",
    out_dir="../../../outputs/sim_profiles/batch_0003",
    output_interval=0.4,
)

runner = DymolaBatchRunner(cfg)
runner.start()
try:
    runner.run_all()
finally:
    runner.close()
```

This writes profile-level result files and a batch summary CSV under the configured output folder.

## 3) End-to-end batch script (profiles + Dymola)

The main automation script is:

```bash
python scripts/run_dymola_batch.py
```

It reads `scripts/config.py`, generates a new `outputs/variography_profiles/batch_XXXX` folder, then runs Dymola and writes to `outputs/sim_profiles/batch_XXXX`.

---

## Machine-learning dataset utilities

Common entry points:

```bash
python scripts/build_lstm_dataset.py --help
python scripts/scale_lstm_dataset.py --help
python scripts/run_lstm_pipeline_forecast.py --help
```

### Split mode support in scaling

`scripts/scale_lstm_dataset.py` supports:

- `--split-mode sample` (default): train/val split at sample level while holding out unseen profiles for test.
- `--split-mode profile`: full-profile-disjoint train/val/test split.

Example:

```bash
python scripts/scale_lstm_dataset.py data/my_dataset.h5 --scaling-type standard --split-mode sample
```

Scaled outputs include split metadata so downstream tooling can detect the split strategy.

---

## Additional scripts

Other frequently used scripts in `scripts/` include:

- `generate_branched_control_profiles.py` (profile branching + visualization)
- `run_recursive_branching.py` (recursive branching experiments)
- `plot_sim_batches.py` / `plot_variograms.py` / `plot_scaled_splits.py` (analysis plotting)
- `print_best_hyperparameters.py` (hyperparameter summary helper)

Use `--help` where available for exact CLI options.

---

## Development notes

- Source code lives under `src/` with package imports rooted at `rabl.*`.
- The interface layer (`rabl.interface`) intentionally treats Dymola as optional at import time; non-Dymola tooling can still be used without it.
- Tests folder currently contains both fixtures and generated/reference artifacts used for workflow validation.

---

## License

MIT. See [LICENSE](LICENSE).
