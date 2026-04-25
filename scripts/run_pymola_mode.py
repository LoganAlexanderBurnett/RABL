"""Small CLI driver for Dymola batch runner modes.

Examples
--------
# Independent flat MAT workflow (existing behavior)
python scripts/run_pymola_mode.py --mode flat_mat --out outputs/sim_profiles/manual --profiles outputs/variography_profiles/test_batch

# Branched MAT+manifest workflow
python scripts/run_pymola_mode.py --mode branched_mat --profiles outputs/variography_profiles/batch_0001 --out outputs/sim_profiles/branched_manual
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    # Import directly from pymola so dependency/import errors surface clearly.
    from rabl.interface.pymola import BatchConfig, DymolaBatchRunner
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Unable to import Dymola interface dependencies. "
        "Make sure your Python environment can import both "
        "'dymola.dymola_interface' and required scientific packages (numpy/scipy/h5py)."
    ) from exc

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


def _find_latest_global_result_index(sim_root: Path) -> int:
    if not sim_root.exists():
        return 0
    pattern = re.compile(r"^results_drum_profile_(\d{5})$")
    max_idx = 0
    for result_path in sim_root.glob("batch_*/results_drum_profile_*.*"):
        if result_path.suffix.lower() not in {".csv", ".mat"}:
            continue
        match = pattern.match(result_path.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx


def _copy_branched_results_to_batch_root_with_global_numbering(out_dir: Path, sim_root: Path) -> None:
    branched_results_dir = out_dir / "branched_results"
    if not branched_results_dir.exists():
        print("[run-mode=production] No branched_results directory found; skipping copy.")
        return

    sources = sorted(branched_results_dir.glob("results_*.csv"))
    if not sources:
        print("[run-mode=production] No branched CSV results found; skipping copy.")
        return

    next_idx = _find_latest_global_result_index(sim_root) + 1
    copied = 0
    for csv_src in sources:
        mat_src = csv_src.with_suffix(".mat")
        dst_stem = f"results_drum_profile_{next_idx:05d}"
        csv_dst = out_dir / f"{dst_stem}.csv"
        mat_dst = out_dir / f"{dst_stem}.mat"
        shutil.copy2(csv_src, csv_dst)
        if mat_src.exists():
            shutil.copy2(mat_src, mat_dst)
        copied += 1
        next_idx += 1
    print(f"[run-mode=production] Copied {copied} branched result pairs to batch root with global profile numbering.")


def _load_branch_manifest_index(batch_dir: Path) -> dict[tuple[str, str], tuple[str | None, float | None]]:
    manifest_path = batch_dir / "branched_profiles_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest for branched_mat mode: {manifest_path}")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], tuple[str | None, float | None]] = {}
    for entry in entries:
        root_id = str(entry["root_group_name"])
        profile_id = str(entry["profile_id"])
        parent = str(entry.get("parent_profile_id", "")).strip() or None
        branch_time_val = entry.get("branch_time", None)
        branch_time = None if branch_time_val is None else float(branch_time_val)
        out[(root_id, profile_id)] = (parent, branch_time)
    return out


def _cleanup_production_artifacts(out_dir: Path) -> None:
    generated_profiles_dir = out_dir / "generated_profiles"
    if generated_profiles_dir.exists():
        shutil.rmtree(generated_profiles_dir, ignore_errors=True)

    branched_results_dir = out_dir / "branched_results"
    removed = 0
    for file_path in out_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if branched_results_dir in file_path.parents:
            continue
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".c", ".exe", ".mat"}:
            file_path.unlink(missing_ok=True)
            removed += 1

    summary_path = out_dir / "batch_summary.csv"
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            statuses = summary_df.get("status")
            if statuses is not None and not statuses.empty and statuses.astype(str).eq("OK").all():
                shutil.rmtree(out_dir / "logs", ignore_errors=True)
                shutil.rmtree(out_dir / "restart_results", ignore_errors=True)
                print("[run-mode=production] Removed logs/ and restart_results/ because all batch_summary statuses are OK.")
        except Exception as exc:
            print(f"[run-mode=production] Warning: unable to evaluate batch_summary.csv for conditional cleanup: {exc}")

    print("[run-mode=production] Cleanup complete. Removed generated_profiles/, *.txt, *.c, *.exe, and non-branched-results *.mat files.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dymola batch workflow in flat or branched mode.")
    parser.add_argument("--mode", choices=("flat_mat", "branched_mat"), required=True, help="Execution workflow mode.")
    parser.add_argument("--run-mode", choices=("testing", "production"), default="testing", help="Run mode. testing requires --out; production auto-creates next batch dir.")
    parser.add_argument("--out", default=None, help="Output directory for run artifacts/results (required in testing mode).")
    parser.add_argument(
        "--profiles",
        default=None,
        help=(
            "For --mode flat_mat: directory of drum_profile_*.mat files. "
            "For --mode branched_mat: directory containing branched "
            "drum_profile_*.mat files and branched_profiles_manifest.json."
        ),
    )
    parser.add_argument("--output-interval", type=float, default=0.1, help="Dymola output interval in seconds.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Disable skip-existing behavior.")
    parser.add_argument("--branch-check-atol", type=float, default=1e-6, help="Absolute tolerance for branch-start-time checks.")
    parser.add_argument("--branch-check-rtol", type=float, default=1e-5, help="Relative tolerance for branch-start-time checks.")
    parser.add_argument("--branch-check-strict", action="store_true", help="Fail run if any branch-start-time check fails.")
    return parser.parse_args()


def _read_results_csv(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    if "t" not in df.columns:
        raise RuntimeError(f"Missing 't' column in {results_csv}")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["t"])
    return df


def _extract_root_id_from_results_stem(stem: str) -> str:
    # e.g. "results_root_001__profile_00012" -> "root_001"
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


def _run_branch_start_time_checks(branched_results_dir: Path, branch_index: dict[tuple[str, str], tuple[str | None, float | None]], *, atol: float, rtol: float) -> list[str]:
    failures: list[str] = []
    for (root_id, profile_id), (parent_id, branch_time) in sorted(branch_index.items()):
        result_csv = branched_results_dir / f"results_{root_id}__{profile_id}.csv"
        if not result_csv.exists():
            failures.append(f"{root_id}/{profile_id}: missing result csv")
            continue
        df = _read_results_csv(result_csv)
        if df.empty:
            failures.append(f"{root_id}/{profile_id}: empty result csv")
            continue
        first_t = float(df["t"].iloc[0])

        if not parent_id:
            if not np.isclose(first_t, 0.0, atol=atol, rtol=rtol):
                failures.append(f"{root_id}/{profile_id}: root should start at t=0, got t={first_t:.6f}")
            continue
        if branch_time is None or np.isnan(branch_time):
            failures.append(f"{root_id}/{profile_id}: missing branch_time")
            continue
        if first_t + max(atol, abs(branch_time) * rtol) < branch_time:
            failures.append(
                f"{root_id}/{profile_id}: branch result starts before branch_time "
                f"(first_t={first_t:.6f}, branch_time={branch_time:.6f})"
            )
    return failures


def main() -> None:
    args = parse_args()

    out_dir = _resolve_output_dir(args)
    profiles_dir = _repo_rel(args.profiles, "../../../outputs/variography_profiles/test_batch")
    branch_index: dict[tuple[str, str], tuple[str | None, float | None]] = {}

    if args.mode == "branched_mat":
        if args.profiles is None:
            raise SystemExit("--profiles is required when --mode=branched_mat")
        mat_batch_dir = Path(_repo_rel(args.profiles, "../../../outputs/variography_profiles/test_batch"))
        branch_index = _load_branch_manifest_index(mat_batch_dir)

    cfg = BatchConfig(
        profile_mode=args.mode,
        out_dir=out_dir,
        profiles_dir=profiles_dir,
        output_interval=args.output_interval,
        canonical_output_interval=args.output_interval,
        skip_existing=not args.no_skip_existing,
    )

    runner = DymolaBatchRunner(cfg)
    runner.start()
    try:
        if args.mode == "branched_mat":
            runner.run_branched_mat()
            branched_results_dir = Path(out_dir) / cfg.branched_results_dir
            failures = _run_branch_start_time_checks(
                branched_results_dir=branched_results_dir,
                branch_index=branch_index,
                atol=args.branch_check_atol,
                rtol=args.branch_check_rtol,
            )
            if failures:
                print("\nBranch-start-time check failures:")
                for msg in failures:
                    print(f"  - {msg}")
                if args.branch_check_strict:
                    raise SystemExit("Branch-start-time checks failed (strict mode).")
            else:
                print("Branch-start-time checks passed for all profiles.")

            results_csvs = sorted(branched_results_dir.glob("results_*.csv"))
            if not results_csvs:
                raise SystemExit(f"No branched CSVs found in: {branched_results_dir}")
            plot_path = branched_results_dir / "timeseries_branched_ALL_PROFILES.png"
            _plot_all_profiles(results_csvs, plot_path)
            print(f"Saved branched-results plot to {plot_path}")
        else:
            runner.run_all()
    finally:
        runner.close()

    if args.run_mode == "production":
        out_dir_path = Path(out_dir)
        _cleanup_production_artifacts(out_dir_path)
        if args.mode == "branched_mat":
            sim_root = (REPO_ROOT / "outputs" / "sim_profiles").resolve()
            _copy_branched_results_to_batch_root_with_global_numbering(out_dir_path, sim_root)


if __name__ == "__main__":
    main()
