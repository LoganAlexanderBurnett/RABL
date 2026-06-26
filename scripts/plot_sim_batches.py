import importlib.util
from pathlib import Path

from rabl.paths import resolve_output_root

import matplotlib.pyplot as plt
import pandas as pd


PLOT_VARS = [
    "drumAngleDeg",
    "TN2",
    "Tm",
    "Thp",
    "Tf",
    "Tsg",
    "T_steam_out",
    "x_steam_out",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "P_MW",
    "rho_dollars",
]


# -------------------------------------------------------------------------
# Color rules (requested)
# -------------------------------------------------------------------------
COLOR_MAP = {
    "drumAngleDeg": "black",

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

    # rho dollars total in black, components in distinct colors
    "rho_dollars": "black",
    "rho_drums_dollars": "#1f77b4",
    "rho_fuel_dollars": "#2ca02c",
    "rho_moderator_dollars": "#ff7f0e",

    # SG outputs
    "Tsg": "#d62728",
    "T_steam_out": "#8c564b",
    "x_steam_out": "#7f7f7f",
}


def _pretty_var_label(var_name: str) -> str:
    mapping = {
        "drumAngleDeg": r"$\theta_{\mathrm{drum}}$ (deg)",
        "TN2": r"$T_{N2}$",
        "Tm": r"$T_m$",
        "Thp": r"$T_{hp}$",
        "Tf": r"$T_f$",
        "Tsg": r"$T_{sg}$",
        "T_steam_out": r"$T_{\mathrm{steam,out}}$",
        "x_steam_out": r"$x_{\mathrm{steam,out}}$",
        "P_MW": r"$P$ (MW)",
        "rho_dollars": r"$\rho_{\$}$",
        "rho_drums_dollars": r"$\rho_{\mathrm{drums},\$}$",
        "rho_fuel_dollars": r"$\rho_{\mathrm{fuel},\$}$",
        "rho_moderator_dollars": r"$\rho_{\mathrm{moderator},\$}$",
    }
    if var_name in mapping:
        return mapping[var_name]
    if var_name.startswith("c[") and var_name.endswith("]"):
        return rf"$c_{{{var_name[2:-1]}}}$"
    return var_name.replace("_", r"\_")


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
    plt.rcParams.update({"font.size": 18})
    # Read all data first
    dfs: list[tuple[str, pd.DataFrame]] = []
    for p in results_csvs:
        dfs.append((p.stem, _read_results_csv(p)))

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16), sharex=True)
    axes = axes.flatten()

    for ax, var in zip(axes, PLOT_VARS, strict=False):
        color = COLOR_MAP.get(var, "black")
        pretty_label = _pretty_var_label(var)

        # Overlay every profile on this subplot
        for _, df in dfs:
            ax.plot(
                df["t"].to_numpy(),
                df[var].to_numpy(),
                color=color,
                linewidth=1.0,
                alpha=0.10,
            )

        ax.set_title(pretty_label)
        ax.set_ylabel(pretty_label)
        ax.grid(True, which="both", alpha=0.2)

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

    sim_root = resolve_output_root() / "sim_profiles"

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
