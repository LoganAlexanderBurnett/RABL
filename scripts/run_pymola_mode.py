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
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
PLOT_VARS = [
    "drumAngleDeg",
    "drumVelDeg_s",
    "TN2",
    "Tm",
    "Thp",
    "Tf",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "P_MW",
    "rho_dollars",
    "rho_drums_dollars",
    "rho_fuel_dollars",
    "rho_moderator_dollars",
    "Q_to_steam",
]

def _repo_rel(path: str | None, fallback: str) -> str:
    if not path:
        return fallback
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return str(p)




def _next_batch_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^batch_(\d{4})$")

    max_idx = 0
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1
    candidate = base_dir / f"batch_{next_idx:04d}"
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def _resolve_output_dir(args: argparse.Namespace) -> str:
    if args.run_mode == "testing":
        if args.out is None:
            raise SystemExit("--out is required when --run-mode=testing")
        return _repo_rel(args.out, "../../../outputs/sim_profiles/test_batch")

    production_base = (REPO_ROOT / "outputs" / "sim_profiles").resolve()
    out_dir = _next_batch_dir(production_base)
    print(f"[run-mode=production] Created output directory: {out_dir}")
    return str(out_dir)


def _cleanup_production_artifacts(out_dir: Path) -> None:
    removed = 0
    for ext in (".txt", ".c", ".e"):
        for file_path in out_dir.rglob(f"*{ext}"):
            if file_path.is_file():
                file_path.unlink(missing_ok=True)
                removed += 1
    print(f"[run-mode=production] Cleanup complete. Removed {removed} files (.txt/.c/.e).")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dymola batch workflow in flat or branched mode.")
    parser.add_argument("--mode", choices=("flat_mat", "branched_hdf5"), required=True, help="Execution workflow mode.")
    parser.add_argument("--h5", default=None, help="Path to profiles.h5 (required for --mode branched_hdf5).")
    parser.add_argument("--run-mode", choices=("testing", "production"), default="testing", help="Run mode. testing requires --out; production auto-creates next batch dir.")
    parser.add_argument("--out", default=None, help="Output directory for run artifacts/results (required in testing mode).")
    parser.add_argument(
        "--profiles",
        default=None,
        help="Directory of drum_profile_*.mat files (used only for --mode flat_mat).",
    )
    parser.add_argument("--output-interval", type=float, default=0.1, help="Dymola output interval in seconds.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Disable skip-existing behavior.")
    parser.add_argument("--branch-check-atol", type=float, default=1e-6, help="Absolute tolerance for parent/child shared-prefix checks.")
    parser.add_argument("--branch-check-rtol", type=float, default=1e-5, help="Relative tolerance for parent/child shared-prefix checks.")
    parser.add_argument("--branch-check-strict", action="store_true", help="Fail run if any shared-prefix check fails.")
    return parser.parse_args()


def _read_results_csv(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    if "t" not in df.columns:
        raise RuntimeError(f"Missing 't' column in {results_csv}")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["t"])
    return df


def _extract_root_id_from_results_stem(stem: str) -> str:
    # e.g. "results_root_001__profile_000012" -> "root_001"
    parts = stem.split("__")
    if len(parts) >= 2 and parts[0].startswith("results_"):
        return parts[0].replace("results_", "", 1)
    return "unknown_root"


def _plot_all_profiles(results_csvs: list[Path], output_path: Path) -> None:
    dfs: list[tuple[str, str, pd.DataFrame]] = []
    for p in results_csvs:
        root_id = _extract_root_id_from_results_stem(p.stem)
        dfs.append((p.stem, root_id, _read_results_csv(p)))

    root_ids = sorted({root_id for _, root_id, _ in dfs})
    base_root_colors = [
        "crimson",
        "gold",
        "black",
    ]
    rng = np.random.default_rng()
    root_colors: dict[str, tuple[float, float, float, float] | str] = {}
    for i, root_id in enumerate(root_ids):
        if i < len(base_root_colors):
            root_colors[root_id] = base_root_colors[i]
        else:
            root_colors[root_id] = (float(rng.random()), float(rng.random()), float(rng.random()), 1.0)

    rows = 3
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(30, 12), sharex=True)
    axes = axes.flatten()

    for ax, var in zip(axes, PLOT_VARS, strict=False):
        for _, root_id, df in dfs:
            if var not in df.columns:
                continue
            ax.plot(
                df["t"].to_numpy(),
                df[var].to_numpy(),
                color=root_colors[root_id],
                linewidth=1.0,
                alpha=0.15,
            )
        ax.set_title(var)
        ax.set_ylabel(var)
        ax.grid(True, which="both", alpha=0.35)

    for ax in axes[len(PLOT_VARS):]:
        ax.set_axis_off()
    for ax in axes[-cols:]:
        ax.set_xlabel("t (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _run_shared_prefix_checks(stitched_dir: Path, h5_path: Path, *, atol: float, rtol: float) -> list[str]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("Shared-prefix checks require h5py.") from exc

    failures: list[str] = []
    with h5py.File(h5_path, "r") as h5f:
        for root_id in sorted(h5f.keys()):
            root_grp = h5f[root_id]
            for profile_id in sorted(root_grp.keys()):
                grp = root_grp[profile_id]
                parent_id = str(grp.attrs.get("parent_profile_id", "")).strip()
                if not parent_id:
                    continue
                branch_time = float(grp.attrs.get("branch_time", np.nan))
                if np.isnan(branch_time):
                    failures.append(f"{root_id}/{profile_id}: missing branch_time")
                    continue

                parent_csv = stitched_dir / f"results_{root_id}__{parent_id}.csv"
                child_csv = stitched_dir / f"results_{root_id}__{profile_id}.csv"
                if not parent_csv.exists() or not child_csv.exists():
                    failures.append(f"{root_id}/{profile_id}: missing stitched csv parent={parent_csv.exists()} child={child_csv.exists()}")
                    continue

                p_df = _read_results_csv(parent_csv)
                c_df = _read_results_csv(child_csv)
                p_pref = p_df[p_df["t"] <= branch_time].copy()
                c_pref = c_df[c_df["t"] <= branch_time].copy()

                if p_pref.empty or c_pref.empty:
                    failures.append(f"{root_id}/{profile_id}: empty shared prefix")
                    continue

                t_common = np.intersect1d(p_pref["t"].to_numpy(), c_pref["t"].to_numpy())
                if t_common.size == 0:
                    failures.append(f"{root_id}/{profile_id}: no common time grid in shared prefix")
                    continue

                p_idx = p_pref.set_index("t")
                c_idx = c_pref.set_index("t")
                vars_to_check = [v for v in PLOT_VARS if v in p_idx.columns and v in c_idx.columns]
                for v in vars_to_check:
                    p_vals = p_idx.loc[t_common, v].to_numpy(dtype=float)
                    c_vals = c_idx.loc[t_common, v].to_numpy(dtype=float)
                    if not np.allclose(p_vals, c_vals, atol=atol, rtol=rtol, equal_nan=True):
                        max_abs = float(np.nanmax(np.abs(p_vals - c_vals)))
                        failures.append(
                            f"{root_id}/{profile_id}: shared-prefix mismatch in '{v}' at t<={branch_time:.6f}, max_abs={max_abs:.3e}"
                        )
                        break
    return failures


def main() -> None:
    args = parse_args()

    out_dir = _resolve_output_dir(args)
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
            stitched_dir = Path(out_dir) / cfg.stitched_results_dir
            failures = _run_shared_prefix_checks(
                stitched_dir=stitched_dir,
                h5_path=Path(h5_path),
                atol=args.branch_check_atol,
                rtol=args.branch_check_rtol,
            )
            if failures:
                print("\nShared-prefix check failures:")
                for msg in failures:
                    print(f"  - {msg}")
                if args.branch_check_strict:
                    raise SystemExit("Shared-prefix checks failed (strict mode).")
            else:
                print("Shared-prefix checks passed for all parent/child pairs.")

            results_csvs = sorted(stitched_dir.glob("results_*.csv"))
            if not results_csvs:
                raise SystemExit(f"No stitched CSVs found in: {stitched_dir}")
            plot_path = stitched_dir / "timeseries_stitched_ALL_PROFILES.png"
            _plot_all_profiles(results_csvs, plot_path)
            print(f"Saved stitched plot to {plot_path}")
        else:
            runner.run_all()
    finally:
        runner.close()

    if args.run_mode == "production":
        _cleanup_production_artifacts(Path(out_dir))


if __name__ == "__main__":
    main()
