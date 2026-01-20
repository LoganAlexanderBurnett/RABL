import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_VARS = [
    "drumAngleDeg",
    "TN2",
    "dTN2",
    "Tm",
    "dTm",
    "Thp",
    "dThp",
    "Tf",
    "dTf",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "dc[1]",
    "dc[2]",
    "dc[3]",
    "dc[4]",
    "dc[5]",
    "dc[6]",
    "P_MW",
    "n",
    "dn",
    "rho_dollars",
    "m_dot_steam",
    "Q_to_steam",
]


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"Missing config file: {config_path}")

    spec = importlib.util.spec_from_file_location("sim_plot_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load config file: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return {
        "batch_numbers": getattr(config_module, "PLOT_BATCH_NUMBERS", None),
    }


def _validate_config(config: dict) -> dict:
    batch_numbers = config["batch_numbers"]
    if not isinstance(batch_numbers, (list, tuple)) or not batch_numbers:
        raise SystemExit("PLOT_BATCH_NUMBERS must be a non-empty list in config.py.")
    if not all(isinstance(n, int) and n >= 0 for n in batch_numbers):
        raise SystemExit("PLOT_BATCH_NUMBERS must contain non-negative integers in config.py.")
    return config


def _plot_results(results_csv: Path, output_path: Path) -> None:
    data = np.genfromtxt(results_csv, delimiter=",", names=True, dtype=float)
    if "t" not in data.dtype.names:
        raise SystemExit(f"Missing 't' column in {results_csv}")

    missing = [var for var in PLOT_VARS if var not in data.dtype.names]
    if missing:
        raise SystemExit(f"Missing variables in {results_csv.name}: {missing}")

    t = data["t"]

    fig, axes = plt.subplots(9, 3, figsize=(18, 24), sharex=True)
    axes = axes.flatten()

    for ax, var in zip(axes, PLOT_VARS, strict=True):
        ax.plot(t, data[var])
        ax.set_title(var)
        ax.set_ylabel(var)

    for ax in axes[-3:]:
        ax.set_xlabel("t (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.py"
    config = _validate_config(_load_config(config_path))

    repo_root = script_dir.parent
    sim_root = repo_root / "outputs" / "sim"

    for batch_number in config["batch_numbers"]:
        batch_name = f"batch_{batch_number:04d}"
        sim_dir = sim_root / batch_name
        if not sim_dir.exists():
            raise SystemExit(f"Simulation batch folder not found: {sim_dir}")

        results_csvs = sorted(sim_dir.glob("results_drum_profile_*.csv"))
        if not results_csvs:
            raise SystemExit(f"No results_drum_profile_*.csv found in {sim_dir}")

        for results_csv in results_csvs:
            output_path = sim_dir / f"timeseries_{results_csv.stem}.png"
            _plot_results(results_csv, output_path)
            print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
