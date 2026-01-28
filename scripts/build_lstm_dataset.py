import argparse
from pathlib import Path

from rabl.machine_learning import build_lstm_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an LSTM-ready dataset from simulation outputs."
    )
    parser.add_argument("--lookback", type=int, required=True, help="Number of past timesteps to include.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to scripts/config.py (defaults to repo scripts/config.py).",
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=None,
        help="Directory containing batch simulation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the merged dataset.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (repo_root / "scripts" / "config.py")
    sim_root = args.sim_root or (repo_root / "outputs" / "sim_profiles")
    output_dir = args.output_dir or (repo_root / "outputs" / "datasets")

    config = build_lstm_dataset._validate_config(build_lstm_dataset._load_config(config_path))
    build_lstm_dataset.build_dataset(sim_root, output_dir, config["steady_state"], args.lookback)


if __name__ == "__main__":
    main()
