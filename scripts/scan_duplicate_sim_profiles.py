"""Scan simulation batch folders for duplicate control profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

CSV_GLOB = "results_drum_profile_*.csv"
BATCH_GLOB = "batch_????"
TIME_COLUMN = "t"
CONTROL_COLUMN = "drumAngleDeg"


def _read_time_control(csv_path: Path) -> tuple[list[float], list[float]]:
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        missing = [col for col in (TIME_COLUMN, CONTROL_COLUMN) if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV file {csv_path} is missing required column(s): {missing}")
        times: list[float] = []
        controls: list[float] = []
        for row in reader:
            times.append(float(row[TIME_COLUMN]))
            controls.append(float(row[CONTROL_COLUMN]))
    if not times:
        raise ValueError(f"CSV file has no data rows: {csv_path}")
    return times, controls


def _control_profile_signature(csv_path: Path, *, decimals: int, include_time: bool) -> str:
    times, controls = _read_time_control(csv_path)
    digest = hashlib.sha256()
    digest.update(f"decimals={decimals};include_time={include_time};n={len(controls)}\n".encode("utf-8"))
    values = zip(times, controls, strict=True) if include_time else ((0.0, value) for value in controls)
    for time_value, control_value in values:
        if include_time:
            digest.update(f"{round(time_value, decimals):.{decimals}f},".encode("utf-8"))
        digest.update(f"{round(control_value, decimals):.{decimals}f}\n".encode("utf-8"))
    return digest.hexdigest()


def scan_duplicate_control_profiles(
    sim_root: Path,
    *,
    decimals: int = 12,
    include_time: bool = True,
    also_check_filenames: bool = False,
) -> dict[str, object]:
    """Return duplicate control trajectories across batch_XXXX result CSVs."""
    sim_root = Path(sim_root)
    if not sim_root.is_dir():
        raise FileNotFoundError(f"Simulation profile root does not exist or is not a directory: {sim_root}")
    if decimals < 0:
        raise ValueError("decimals must be nonnegative.")

    control_paths_by_signature: dict[str, list[str]] = defaultdict(list)
    profile_paths_by_stem: dict[str, list[str]] = defaultdict(list)
    batch_dirs = sorted(path for path in sim_root.glob(BATCH_GLOB) if path.is_dir())
    for batch_dir in batch_dirs:
        for csv_path in sorted(batch_dir.glob(CSV_GLOB)):
            signature = _control_profile_signature(csv_path, decimals=decimals, include_time=include_time)
            control_paths_by_signature[signature].append(str(csv_path))
            if also_check_filenames:
                profile_paths_by_stem[csv_path.stem].append(str(csv_path))

    duplicate_controls = {
        signature: paths for signature, paths in sorted(control_paths_by_signature.items()) if len(paths) > 1
    }
    duplicate_stems = {
        stem: paths for stem, paths in sorted(profile_paths_by_stem.items()) if len(paths) > 1
    }
    return {
        "sim_root": str(sim_root),
        "batch_count": len(batch_dirs),
        "profile_csv_count": sum(len(paths) for paths in control_paths_by_signature.values()),
        "unique_control_profile_count": len(control_paths_by_signature),
        "duplicate_control_profiles": duplicate_controls,
        "duplicate_profile_stems": duplicate_stems,
        "filename_check_enabled": bool(also_check_filenames),
        "decimals": int(decimals),
        "include_time": bool(include_time),
    }


def _print_duplicate_group(header: str, groups: dict[str, list[str]]) -> None:
    if not groups:
        print(f"{header}: none")
        return
    print(f"{header}: {len(groups)} group(s)")
    for signature, paths in groups.items():
        print(f"  {signature}")
        for path in paths:
            print(f"    - {path}")


def _print_summary(result: dict[str, object]) -> None:
    print(f"Simulation root: {result['sim_root']}")
    print(f"Batch directories scanned: {result['batch_count']}")
    print(f"Profile CSV files scanned: {result['profile_csv_count']}")
    print(f"Unique control profiles: {result['unique_control_profile_count']}")
    print(f"Rounded decimal places: {result['decimals']}")
    print(f"Time included in duplicate signature: {result['include_time']}")
    print()
    _print_duplicate_group("Duplicate control profiles", result["duplicate_control_profiles"])  # type: ignore[arg-type]
    if result["filename_check_enabled"]:
        _print_duplicate_group("Duplicate profile filenames/stems", result["duplicate_profile_stems"])  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan outputs/sim_profiles/batch_XXXX folders for duplicated control "
            "profiles in results_drum_profile_XXXXX.csv files. A duplicate means "
            "two CSVs have the same drumAngleDeg trajectory, with time included "
            "in the signature by default."
        )
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("outputs/sim_profiles"),
        help="Root containing batch_XXXX folders (default: outputs/sim_profiles).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=12,
        help="Round time/control values to this many decimals before hashing (default: 12).",
    )
    parser.add_argument(
        "--ignore-time",
        action="store_true",
        help="Detect duplicates using only the control values, ignoring the time grid.",
    )
    parser.add_argument(
        "--also-check-filenames",
        action="store_true",
        help="Also report duplicate results_drum_profile filename stems.",
    )
    args = parser.parse_args()

    result = scan_duplicate_control_profiles(
        args.sim_root,
        decimals=args.decimals,
        include_time=not args.ignore_time,
        also_check_filenames=args.also_check_filenames,
    )
    _print_summary(result)
    if result["duplicate_control_profiles"] or result["duplicate_profile_stems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
