import argparse
from pathlib import Path

from rabl.machine_learning import plot_scaled_features

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input .h5 dataset.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output_path = plot_scaled_features(args.input_path)
    print(f"Saved combined split plot to {output_path}")


if __name__ == "__main__":
    main()
