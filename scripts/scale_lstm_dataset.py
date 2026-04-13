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
    parser.add_argument(
        "--holdout-manifest",
        dest="holdout_manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest with fixed val/test profile keys.",
    )
    parser.add_argument(
        "--val-count",
        dest="val_count",
        type=int,
        default=None,
        help="Optional exact number of profiles to place in val split.",
    )
    parser.add_argument(
        "--test-count",
        dest="test_count",
        type=int,
        default=None,
        help="Optional exact number of profiles to place in test split.",
    )
    parser.add_argument(
        "--save-holdout-manifest",
        dest="save_holdout_manifest",
        type=Path,
        default=None,
        help="Optional path to save sampled val/test profile IDs as a reusable manifest.",
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
    if args.holdout_manifest is not None and (args.val_count is not None or args.test_count is not None):
        raise SystemExit("--holdout-manifest is mutually exclusive with --val-count/--test-count.")
    if (args.val_count is None) ^ (args.test_count is None):
        raise SystemExit("Both --val-count and --test-count must be provided together.")
    if args.save_holdout_manifest is not None and args.holdout_manifest is not None:
        raise SystemExit("--save-holdout-manifest cannot be used with --holdout-manifest.")

    splitter = LSTMDatasetScalerSplitter(
        input_path=input_path,
        scaling_type=args.scaling_type,
        split_mode=args.split_mode,
        holdout_manifest_path=args.holdout_manifest,
        val_count=args.val_count,
        test_count=args.test_count,
        save_holdout_manifest_path=args.save_holdout_manifest,
    )
    output_path = splitter.run()
    print(f"Saved scaled dataset to {output_path}")

    import h5py

    with h5py.File(output_path, "r") as h5f:
        split_strategy = str(h5f.attrs.get("split_strategy", "fractional"))
        train_profiles = list(h5f["train"]["files"].keys())
        val_profiles = list(h5f["val"]["files"].keys())
        test_profiles = list(h5f["test"]["files"].keys())
        print(
            "Split summary: "
            f"strategy={split_strategy}, "
            f"train_profiles={len(train_profiles)}, "
            f"val_profiles={len(val_profiles)}, "
            f"test_profiles={len(test_profiles)}"
        )


if __name__ == "__main__":
    main()
