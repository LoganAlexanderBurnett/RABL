"""Small CLI driver for Dymola batch runner modes.

Examples
--------
# Independent flat MAT workflow (existing behavior)
python scripts/run_pymola_mode.py --mode flat_mat --out outputs/sim_profiles/manual --profiles outputs/variography_profiles/test_batch

# Branched HDF5 workflow
python scripts/run_pymola_mode.py --mode branched_hdf5 --h5 tests/recursive_branching/profiles.h5 --out outputs/sim_profiles/branched_manual
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    # Import directly from pymola so dependency/import errors surface clearly.
    from rabl.interface.pymola import BatchConfig, DymolaBatchRunner
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Unable to import Dymola interface dependencies. "
        "Make sure your Python environment can import both "
        "'dymola.dymola_interface' and required scientific packages (numpy/scipy/h5py)."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_rel(path: str | None, fallback: str) -> str:
    if not path:
        return fallback
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return str(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dymola batch workflow in flat or branched mode.")
    parser.add_argument("--mode", choices=("flat_mat", "branched_hdf5"), required=True, help="Execution workflow mode.")
    parser.add_argument("--h5", default=None, help="Path to profiles.h5 (required for --mode branched_hdf5).")
    parser.add_argument("--out", required=True, help="Output directory for run artifacts/results.")
    parser.add_argument(
        "--profiles",
        default=None,
        help="Directory of drum_profile_*.mat files (used only for --mode flat_mat).",
    )
    parser.add_argument("--output-interval", type=float, default=0.1, help="Dymola output interval in seconds.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Disable skip-existing behavior.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = _repo_rel(args.out, "../../../outputs/sim_profiles/test_batch")
    profiles_dir = _repo_rel(args.profiles, "../../../outputs/variography_profiles/test_batch")
    h5_path = _repo_rel(args.h5, "../../../tests/recursive_branching/profiles.h5")

    if args.mode == "branched_hdf5" and args.h5 is None:
        raise SystemExit("--h5 is required when --mode=branched_hdf5")

    cfg = BatchConfig(
        profile_mode=args.mode,
        out_dir=out_dir,
        profiles_dir=profiles_dir,
        branched_hdf5_path=h5_path,
        output_interval=args.output_interval,
        canonical_output_interval=args.output_interval,
        skip_existing=not args.no_skip_existing,
    )

    runner = DymolaBatchRunner(cfg)
    runner.start()
    try:
        if args.mode == "branched_hdf5":
            runner.run_branched_hdf5()
        else:
            runner.run_all()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
