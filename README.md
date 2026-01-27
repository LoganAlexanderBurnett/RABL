![RABL_LOGO](misc/rabl_logo.png)

RABL is a Python and Modelica toolbox for microreactor dynamics studies. It brings together:

- **Variography tooling** to generate stochastic drum motion profiles.
- **Batch Dymola automation** to run large sweeps of Modelica experiments.
- **Machine-learning utilities** to scale, inspect, and prepare simulation data for LSTM workflows.

The repository keeps the Modelica models, Python package, and scripts in one place so you can
move from profile generation → simulation → dataset preparation without jumping between tools.

## Capabilities

### Variography-driven drum profiles
The `rabl.variography` package provides a Gaussian-process based generator for
control drum angle trajectories. It builds velocity covariances with Matérn kernels, integrates
them into angle profiles, and derives velocities/accelerations for simulation-ready inputs.
Generated profiles can be saved as CSV or MAT files for downstream workflows.

### Dymola batch simulations
The `rabl.interface` package wraps the Dymola API to run a batch of Modelica
simulations against the generated drum profiles. It handles per-profile stop times, result
extraction, and summary logging.

### Machine-learning dataset utilities
Modules under `src/rabl/machine_learning` and scripts under `scripts/` provide helpers to:

- scale raw simulation outputs,
- visualize feature correlations,
- plot variograms and batch results,
- build scaled LSTM datasets.

## Repository layout

```
.
├── misc/                     # Branding assets
├── outputs/                  # Generated profiles and simulation outputs
├── scripts/                  # CLI entry points and plotting scripts
├── src/
│   ├── rabl/                   # Core Python package
│   │   ├── interface/          # Dymola batch runner
│   │   ├── variography/         # Drum profile generation
│   │   └── machine_learning/    # Dataset prep/analysis utilities
│   └── modelica/              # Modelica package(s) for Dymola
└── tests/                     # Automated tests (if added)
```

## Installation

Create a Python 3.10+ environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The Dymola batch runner depends on a licensed Dymola installation with the
`dymola` Python package available in your environment.

## Quick start

### Generate drum profiles

```python
from rabl.variography import DrumProfileGenerator

# Define a time grid (seconds)
import numpy as np

profile_gen = DrumProfileGenerator(kernel="matern52", ell=5.0)
time_s = np.linspace(0.0, 100.0, 501)

# Draw a profile
profile = profile_gen.generate(time_s, n_realizations=1, baseline_angle_deg=0.0)[0]

# Save for downstream tools
profile.save_csv("outputs/variography/drum_profile_00001.csv")
profile.save_mat("outputs/variography/drum_profile_00001.mat")
```

### Run a Dymola batch

```python
from rabl.interface import BatchConfig, DymolaBatchRunner

config = BatchConfig(
    profiles_dir="../../../outputs/variography/test_batch",
    out_dir="../../../outputs/sim/test_batch",
)

runner = DymolaBatchRunner(config)
runner.start()
runner.run_batch()
runner.close()
```

The batch runner writes per-profile CSV results and a `batch_summary.csv` file in the
configured output directory.

### Build scaled datasets

Example command-line workflows live in `scripts/`. A common flow is:

```bash
python scripts/scale_lstm_dataset.py --help
python -m rabl.machine_learning.build_lstm_dataset --help
```

Use the `--help` flag on each script to see required inputs/outputs and configuration options.

## Configuration notes

- **Profile naming**: The batch runner expects profile filenames like
  `drum_profile_00001.mat` so it can extract the profile index and generate deterministic
  result names.
- **Modelica package path**: Update `BatchConfig.package_mo` if the Modelica package is
  moved or renamed.
- **Outputs**: Simulation results are written as CSV files alongside a summary file in the
  `out_dir` specified by the batch configuration.

## Development tips

- The Python package lives under `src/` and can be imported after installing in editable mode.
- Plotting utilities and dataset builders are in `scripts/` and can be executed directly.
- Add new Modelica experiments under `src/modelica/` and update the batch config to point to
  them.

## License

MIT. See [LICENSE](LICENSE).
