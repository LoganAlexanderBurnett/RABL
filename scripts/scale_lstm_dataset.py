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
        "--test-manifest",
        dest="test_manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest with fixed test profile keys.",
    )
    parser.add_argument(
        "--val-manifest",
        dest="val_manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest with fixed validation profile keys.",
    )
    parser.add_argument(
        "--cal-manifest",
        dest="cal_manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest with fixed calibration profile keys.",
    )
    parser.add_argument(
        "--cal-frac",
        dest="cal_frac",
        type=float,
        default=0.0,
        help=(
            "Calibration fraction. With fixed test/val manifests and no --cal-manifest, "
            "this fraction carves calibration profiles from the remaining non-test/non-val pool."
        ),
    )
    parser.add_argument(
        "--test-count",
        dest="test_count",
        type=int,
        default=None,
        help="Optional exact number of profiles to place in test split.",
    )
    parser.add_argument(
        "--save-test-manifest",
        dest="save_test_manifest",
        type=Path,
        default=None,
        help="Optional path to save sampled test profile IDs as a reusable manifest.",
    )
    parser.add_argument(
        "--train-profile-limit-with-manifests",
        dest="train_profile_limit_with_manifests",
        type=int,
        default=None,
        help=(
            "Only when both --test-manifest and --val-manifest are provided: "
            "use only the first N remaining profiles for the train split."
        ),
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
    if args.test_manifest is not None and args.test_count is not None:
        raise SystemExit("--test-manifest is mutually exclusive with --test-count.")
    if args.val_manifest is not None and args.test_count is not None:
        raise SystemExit("--val-manifest is mutually exclusive with --test-count.")
    if args.cal_manifest is not None and args.test_count is not None:
        raise SystemExit("--cal-manifest is mutually exclusive with --test-count.")
    if args.save_test_manifest is not None and args.test_manifest is not None:
        raise SystemExit("--save-test-manifest cannot be used with --test-manifest.")
    if args.save_test_manifest is not None and args.val_manifest is not None:
        raise SystemExit("--save-test-manifest cannot be used with --val-manifest.")
    if args.save_test_manifest is not None and args.test_count is None:
        raise SystemExit("--save-test-manifest requires --test-count.")
    if args.train_profile_limit_with_manifests is not None:
        if args.test_manifest is None or args.val_manifest is None:
            raise SystemExit(
                "--train-profile-limit-with-manifests requires both --test-manifest and --val-manifest."
            )

    splitter = LSTMDatasetScalerSplitter(
        input_path=input_path,
        scaling_type=args.scaling_type,
        split_mode=args.split_mode,
        test_manifest_path=args.test_manifest,
        val_manifest_path=args.val_manifest,
        cal_manifest_path=args.cal_manifest,
        cal_frac=args.cal_frac,
        train_profile_limit_with_manifests=args.train_profile_limit_with_manifests,
        test_count=args.test_count,
        save_test_manifest_path=args.save_test_manifest,
    )
    output_path = splitter.run()
    print(f"Saved scaled dataset to {output_path}")

    import h5py

    with h5py.File(output_path, "r") as h5f:
        split_strategy = str(h5f.attrs.get("split_strategy", "fractional"))
        train_profiles = list(h5f["train"]["files"].keys())
        val_profiles = list(h5f["val"]["files"].keys())
        cal_profiles = list(h5f["cal"]["files"].keys()) if "cal" in h5f else []
        test_profiles = list(h5f["test"]["files"].keys())
        print(
            "Split summary: "
            f"strategy={split_strategy}, "
            f"train_profiles={len(train_profiles)}, "
            f"val_profiles={len(val_profiles)}, "
            f"cal_profiles={len(cal_profiles)}, "
            f"test_profiles={len(test_profiles)}"
        )


if __name__ == "__main__":
    main()
