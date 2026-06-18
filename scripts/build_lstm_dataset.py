import argparse
from pathlib import Path

from rabl.paths import resolve_output_root
from rabl.machine_learning import build_lstm_dataset


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "scripts" / "config.py"
    output_root = resolve_output_root()
    default_sim_root = output_root / "sim_profiles"
    default_output_dir = output_root / "datasets" / "unscaled_unsplit"

    parser = argparse.ArgumentParser(
        description="Build an LSTM-ready dataset from simulation outputs."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        required=True,
        help="Number of past timesteps to include."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to scripts/config.py (default: {default_config}).",
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=default_sim_root,
        help=f"Directory containing batch simulation outputs (default: {default_sim_root}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory to write the merged dataset (default: {default_output_dir}).",
    )
    parser.add_argument(
        "--batches",
        type=str,
        nargs="+",
        required=True,
        help="Batch IDs to include (e.g., --batches 0001 0002 0004).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-profile sample-count messages while building the dataset.",
    )

    args = parser.parse_args()
    config_path: Path = args.config
    sim_root: Path = args.sim_root
    output_dir: Path = args.output_dir



    config = build_lstm_dataset._validate_config(build_lstm_dataset._load_config(config_path))
    build_lstm_dataset.build_dataset(
        sim_root,
        output_dir,
        config["steady_state"],
        args.lookback,
        args.batches,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
