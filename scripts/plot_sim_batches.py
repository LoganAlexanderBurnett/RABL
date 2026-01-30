import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_VARS = [
    "drumAngleDeg",
    "drumVelDeg_s",
    "TN2",
    "Tm",
    "Thp",
    "Tf",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "P_MW",
    "rho_dollars",
    "rho_drums_dollars",
    "rho_fuel_dollars",
    "rho_moderator_dollars",
    "Q_to_steam",
]


# -------------------------------------------------------------------------
# Color rules (requested)
# -------------------------------------------------------------------------
COLOR_MAP = {
    "drumAngleDeg": "black",
    "drumVelDeg_s": "black",

    # Temperatures (red)
    "TN2": "#d62728",
    "Tm": "#d62728",
    "Thp": "#d62728",
    "Tf": "#d62728",

    # c[i] dark green
    "c[1]": "#006400",
    "c[2]": "#006400",
    "c[3]": "#006400",
    "c[4]": "#006400",
    "c[5]": "#006400",
    "c[6]": "#006400",

    # P_MW in pink
    "P_MW": "#e377c2",

    # rho dollars in dark blue
    "rho_dollars": "#1f3a93",
    "rho_drums_dollars": "#1f3a93",
    "rho_fuel_dollars": "#1f3a93",
    "rho_moderator_dollars": "#1f3a93",

    # Q_to_steam in gray
    "Q_to_steam": "#7f7f7f",
}


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


def _read_results_csv(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    if "t" not in df.columns:
        raise SystemExit(f"Missing 't' column in {results_csv}")

    missing = [v for v in PLOT_VARS if v not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing variables in {results_csv.name}: {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows where time is missing
    df = df.dropna(subset=["t"])

    return df


def _plot_all_profiles(results_csvs: list[Path], output_path: Path) -> None:
    # Read all data first
    dfs: list[tuple[str, pd.DataFrame]] = []
    for p in results_csvs:
        dfs.append((p.stem, _read_results_csv(p)))

    rows = 3
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12), sharex=True)
    axes = axes.flatten()

    for ax, var in zip(axes, PLOT_VARS, strict=False):
        color = COLOR_MAP.get(var, "black")

        # Overlay every profile on this subplot
        for _, df in dfs:
            ax.plot(
                df["t"].to_numpy(),
                df[var].to_numpy(),
                color=color,
                linewidth=1.0,
                alpha=0.10,
            )

        ax.set_title(var)
        ax.set_ylabel(var)
        ax.grid(True, which="both", alpha=0.35)

    for ax in axes[len(PLOT_VARS):]:
        ax.set_axis_off()

    for ax in axes[-cols:]:
        ax.set_xlabel("t (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.py"
    config = _validate_config(_load_config(config_path))

    repo_root = script_dir.parent
    sim_root = repo_root / "outputs" / "sim_profiles"

    for batch_number in config["batch_numbers"]:
        batch_name = f"batch_{batch_number:04d}"
        sim_dir = sim_root / batch_name
        if not sim_dir.exists():
            raise SystemExit(f"Simulation batch folder not found: {sim_dir}")

        results_csvs = sorted(sim_dir.glob("results_drum_profile_*.csv"))
        if not results_csvs:
            raise SystemExit(f"No results_drum_profile_*.csv found in {sim_dir}")

        # ONE output PNG for the entire batch
        output_path = sim_dir / f"timeseries_{batch_name}_ALL_PROFILES.png"
        _plot_all_profiles(results_csvs, output_path)
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
