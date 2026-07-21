"""Scan simulation batch folders for duplicate results_drum_profile CSV files."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

CSV_GLOB = "results_drum_profile_*.csv"
BATCH_GLOB = "batch_????"


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_duplicate_profiles(sim_root: Path, *, check_content: bool = False) -> dict[str, object]:
    """Return duplicate profile stems, and optionally duplicate file contents."""
    sim_root = Path(sim_root)
    if not sim_root.is_dir():
        raise FileNotFoundError(f"Simulation profile root does not exist or is not a directory: {sim_root}")

    profile_paths_by_stem: dict[str, list[str]] = defaultdict(list)
    content_paths_by_hash: dict[str, list[str]] = defaultdict(list)
    batch_dirs = sorted(path for path in sim_root.glob(BATCH_GLOB) if path.is_dir())
    for batch_dir in batch_dirs:
        for csv_path in sorted(batch_dir.glob(CSV_GLOB)):
            profile_paths_by_stem[csv_path.stem].append(str(csv_path))
            if check_content:
                content_paths_by_hash[_file_sha256(csv_path)].append(str(csv_path))

    duplicate_stems = {
        stem: paths for stem, paths in sorted(profile_paths_by_stem.items()) if len(paths) > 1
    }
    duplicate_content = {
        digest: paths for digest, paths in sorted(content_paths_by_hash.items()) if len(paths) > 1
    }
    return {
        "sim_root": str(sim_root),
        "batch_count": len(batch_dirs),
        "profile_csv_count": sum(len(paths) for paths in profile_paths_by_stem.values()),
        "unique_profile_stem_count": len(profile_paths_by_stem),
        "duplicate_profile_stems": duplicate_stems,
        "duplicate_content_hashes": duplicate_content,
        "content_hash_checked": bool(check_content),
    }


def _print_summary(result: dict[str, object]) -> None:
    print(f"Simulation root: {result['sim_root']}")
    print(f"Batch directories scanned: {result['batch_count']}")
    print(f"Profile CSV files scanned: {result['profile_csv_count']}")
    print(f"Unique profile stems: {result['unique_profile_stem_count']}")

    duplicate_stems = result["duplicate_profile_stems"]
    if duplicate_stems:
        assert isinstance(duplicate_stems, dict)
        print(f"\nDuplicate profile filenames/stems found: {len(duplicate_stems)}")
        for stem, paths in duplicate_stems.items():
            print(f"  {stem}")
            for path in paths:
                print(f"    - {path}")
    else:
        print("\nNo duplicate profile filenames/stems found.")

    duplicate_content = result["duplicate_content_hashes"]
    if result["content_hash_checked"]:
        if duplicate_content:
            assert isinstance(duplicate_content, dict)
            print(f"\nDuplicate file contents found: {len(duplicate_content)} hash group(s)")
            for digest, paths in duplicate_content.items():
                print(f"  sha256={digest}")
                for path in paths:
                    print(f"    - {path}")
        else:
            print("No duplicate file contents found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan outputs/sim_profiles/batch_XXXX folders for duplicate "
            "results_drum_profile_XXXXX.csv files. By default duplicates are "
            "detected by profile filename/stem; --check-content also flags files "
            "with identical SHA-256 contents."
        )
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("outputs/sim_profiles"),
        help="Root containing batch_XXXX folders (default: outputs/sim_profiles).",
    )
    parser.add_argument(
        "--check-content",
        action="store_true",
        help="Also compute SHA-256 hashes and report files with identical content.",
    )
    args = parser.parse_args()

    result = scan_duplicate_profiles(args.sim_root, check_content=args.check_content)
    _print_summary(result)
    if result["duplicate_profile_stems"] or result["duplicate_content_hashes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
