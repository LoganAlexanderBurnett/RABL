import argparse
from pathlib import Path

from rabl.machine_learning import LSTMDatasetScalerSplitter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scale an LSTM dataset H5 file using LSTMDatasetScalerSplitter."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input .h5 dataset.",
    )
    parser.add_argument(
        "--scaling-type",
        dest="scaling_type",
        required=True,
        choices=("standard", "minmax", "none"),
        help="Scaling type to apply.",
    )
    parser.add_argument(
        "--split-mode",
        dest="split_mode",
        default="sample",
        choices=("sample", "profile"),
        help="Split strategy used for train/val/test partitioning.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path: Path = args.input_path
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".h5":
        raise SystemExit(f"Expected an .h5 file, got: {input_path}")

    splitter = LSTMDatasetScalerSplitter(
        input_path=input_path,
        scaling_type=args.scaling_type,
        split_mode=args.split_mode,
    )
    output_path = splitter.run()
    print(f"Saved scaled dataset to {output_path}")


if __name__ == "__main__":
    main()
