"""Single-experiment driver starting from raw simulation CSV batch folders.

Workflow:
1) Build unscaled LSTM dataset from selected /outputs/sim_profiles/batch_xxxx folders.
2) Split+scale dataset with fixed test set (saved manifest on first pass).
3) Tune hyperparameters on full train split.
4) Train bagged ensemble (M models) using tuned hyperparameters.
5) Sample new controls (branching or random), simulate them, and add new sim batch.
6) Repeat from step 1 for retraining cycles, keeping test fixed and re-sampling train/val.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rabl.machine_learning import build_lstm_dataset
from rabl.machine_learning.dataset_scaling import LSTMDatasetScalerSplitter
from rabl.machine_learning.tuner import GridSearchConfig, run_grid_search
from rabl.machine_learning.bagging_ensemble import run_bagging_ensemble
from rabl.machine_learning.lstm_pipeline import save_forecast_profiles_pdf
from rabl.machine_learning.recursive_branching import (
    RecursiveBranchingBatchConfig,
    run_recursive_branching_batch,
)
from rabl.interface.pymola import BatchConfig, DymolaBatchRunner
from rabl.variography.DrumVariography import DrumProfileGenerator


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    strategy: str  # branching | random
    seed: int
    retrain_cycles: int
    initial_sim_batches: list[str]
    lookback: int
    scaling_type: str
    split_mode: str
    test_count: int
    n_models: int
    bag_fraction: float
    prefer_gpu: bool
    hp_grid: dict[str, Any]
    branching: dict[str, Any]
    dymola: dict[str, Any]
    config_py_path: str = str(REPO_ROOT / "scripts" / "config.py")
    sim_root: str = str(REPO_ROOT / "outputs" / "sim_profiles")
    variography_root: str = str(REPO_ROOT / "outputs" / "variography_profiles")


def _load_cfg(path: Path) -> ExperimentConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig(**data)


def _extract_created_batch_dir(output_text: str) -> Path:
    match = re.search(r"Created output directory:\s*(.+)", output_text)
    if not match:
        raise RuntimeError("Could not parse created batch directory from subprocess output.")
    return Path(match.group(1).strip())


def _next_batch_dir(base_dir: Path) -> Path:
    base_dir = base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^batch_(\d{4})$")
    max_idx = 0
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    out = base_dir / f"batch_{max_idx + 1:04d}"
    out.mkdir(parents=False, exist_ok=False)
    return out


def _build_from_batches(
    *,
    sim_root: Path,
    batch_ids: list[str],
    lookback: int,
    cfg_py: Path,
    out_dir: Path,
) -> Path:
    cfg = build_lstm_dataset._validate_config(build_lstm_dataset._load_config(cfg_py))
    out_dir.mkdir(parents=True, exist_ok=True)
    return build_lstm_dataset.build_dataset(
        sim_root=sim_root,
        output_dir=out_dir,
        steady_state=cfg["steady_state"],
        k=lookback,
        batch_ids=batch_ids,
    )


def _scale_with_fixed_test(
    *,
    unscaled_h5: Path,
    out_dir: Path,
    scaling_type: str,
    split_mode: str,
    test_manifest: Path,
    test_count: int,
    seed: int,
) -> Path:
    splitter = LSTMDatasetScalerSplitter(
        input_path=unscaled_h5,
        scaling_type=scaling_type,
        split_mode=split_mode,
        train_frac=0.5,
        val_frac=0.5,
        test_frac=0.0,
        seed=seed,
        test_manifest_path=test_manifest if test_manifest.exists() else None,
        test_count=None if test_manifest.exists() else test_count,
        save_test_manifest_path=test_manifest if not test_manifest.exists() else None,
        output_dir=out_dir,
    )
    return splitter.run()


def _tune(scaled_h5: Path, out_dir: Path, seed: int, grid: dict[str, Any]) -> Any:
    tune_cfg = GridSearchConfig(
        lookback_datasets={int(grid["lookback"]): scaled_h5},
        learning_rates=list(grid["learning_rates"]),
        batch_sizes=list(grid["batch_sizes"]),
        n_lstm_values=list(grid["n_lstm_values"]),
        hidden_lstm_values=list(grid["hidden_lstm_values"]),
        hidden_fc_values=list(grid["hidden_fc_values"]),
        n_fc=int(grid.get("n_fc", 1)),
        epochs=int(grid.get("epochs", 20)),
        seed=seed,
        out_dir=out_dir,
        prefer_gpu=bool(grid.get("prefer_gpu", True)),
        preload_train_to_device=True,
    )
    _results, best = run_grid_search(tune_cfg)
    return best


def _sample_random_profiles(batch_dir: Path, *, n_profiles: int, T: float, dt: float, seed: int, baseline: float, kernel: str) -> None:
    t_grid = np.arange(0.0, T + dt, dt, dtype=float)
    if not np.isclose(t_grid[-1], T):
        t_grid = np.append(t_grid, T)
    gen = DrumProfileGenerator(kernel=kernel)
    profiles = gen.generate(
        t_grid=t_grid,
        n_realizations=n_profiles,
        baseline_angle_deg=baseline,
        seed=seed,
    )
    for idx, p in enumerate(profiles):
        p.save_mat(str(batch_dir / f"drum_profile_{idx:05d}.mat"))


def _expected_new_profiles(nr: int, nk: int, nb: int) -> int:
    return int(nr * ((nb + 1) ** nk))


def _find_latest_global_result_index(sim_root: Path) -> int:
    pattern = re.compile(r"^results_drum_profile_(\d{5})$")
    max_idx = 0
    for result_path in sim_root.glob("batch_*/results_drum_profile_*.*"):
        if result_path.suffix.lower() not in {".csv", ".mat"}:
            continue
        m = pattern.match(result_path.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def _copy_stitched_results_to_batch_root_with_global_numbering(out_dir: Path, sim_root: Path) -> None:
    stitched = out_dir / "stitched_results"
    if not stitched.exists():
        return
    sources = sorted(stitched.glob("results_*.csv"))
    if not sources:
        return
    next_idx = _find_latest_global_result_index(sim_root) + 1
    for csv_src in sources:
        mat_src = csv_src.with_suffix(".mat")
        stem = f"results_drum_profile_{next_idx:05d}"
        csv_dst = out_dir / f"{stem}.csv"
        mat_dst = out_dir / f"{stem}.mat"
        csv_dst.write_bytes(csv_src.read_bytes())
        if mat_src.exists():
            mat_dst.write_bytes(mat_src.read_bytes())
        next_idx += 1


def _run_recursive_branching_internal(
    *,
    cfg: ExperimentConfig,
    model_paths: list[str],
    bagged_h5_path: Path,
    lstm_hidden: int,
    n_lstm: int,
    fc_hidden: int,
    seed: int,
    variography_root: Path,
) -> Path:
    out_dir = _next_batch_dir(variography_root)
    print(f"[step] Running recursive branching into {out_dir}")
    run_cfg = RecursiveBranchingBatchConfig(
        model_paths=tuple(Path(p) for p in model_paths),
        bagged_h5_path=Path(bagged_h5_path),
        output_dir=out_dir,
        T=float(cfg.branching["T"]),
        dt=float(cfg.branching["dt"]),
        Nk=int(cfg.branching["N_k"]),
        Nb=int(cfg.branching["N_b"]),
        Nr=int(cfg.branching["N_r"]),
        baseline_angle_deg=float(cfg.branching["baseline_angle_deg"]),
        seed=int(seed),
        lookback=int(cfg.lookback),
        n_lstm=int(n_lstm),
        lstm_hidden=int(lstm_hidden),
        n_fc=1,
        fc_hidden=(int(fc_hidden),),
        kernel=str(cfg.branching["kernel"]),
        device=str(cfg.branching.get("device", "cpu")),
        config_path=Path(cfg.config_py_path),
    )
    run_recursive_branching_batch(run_cfg)
    return out_dir


def _run_dymola_internal(
    *,
    mode: str,
    profiles_dir: Path,
    output_interval: float,
    sim_root: Path,
) -> Path:
    out_dir = _next_batch_dir(sim_root).resolve()
    profiles_dir = profiles_dir.resolve()
    print(f"[step] Running Dymola simulation mode={mode} into {out_dir}")
    batch_cfg = BatchConfig(
        profile_mode=mode,
        out_dir=str(out_dir),
        profiles_dir=str(profiles_dir),
        output_interval=float(output_interval),
        canonical_output_interval=float(output_interval),
        skip_existing=True,
    )
    runner = DymolaBatchRunner(batch_cfg)
    runner.start()
    try:
        if mode == "branched_mat":
            runner.run_branched_mat()
        else:
            runner.run_all()
    finally:
        runner.close()
    if mode == "branched_mat":
        _plot_stitched_results(out_dir / "stitched_results")
        _copy_stitched_results_to_batch_root_with_global_numbering(out_dir, sim_root)
        _cleanup_production_artifacts(out_dir)
    return out_dir


def _cleanup_production_artifacts(out_dir: Path) -> None:
    generated_profiles = out_dir / "generated_profiles"
    if generated_profiles.exists():
        shutil.rmtree(generated_profiles, ignore_errors=True)

    stitched_dir = out_dir / "stitched_results"
    for file_path in out_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if stitched_dir in file_path.parents:
            continue
        if file_path.suffix.lower() in {".txt", ".c", ".exe", ".mat"}:
            file_path.unlink(missing_ok=True)


def _plot_stitched_results(stitched_dir: Path) -> None:
    if not stitched_dir.exists():
        return
    csv_paths = sorted(stitched_dir.glob("results_*.csv"))
    if not csv_paths:
        return

    series: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    vars_to_plot: list[str] = []
    for idx, csv_path in enumerate(csv_paths):
        with csv_path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
        if not rows:
            continue
        keys = list(rows[0].keys())
        if "t" not in keys:
            continue
        t = np.asarray([float(r["t"]) for r in rows], dtype=float)
        payload: dict[str, np.ndarray] = {}
        if idx == 0:
            vars_to_plot = [k for k in keys if k in {"drumAngleDeg", "TN2", "Tm", "Thp", "Tf", "Q_to_steam"}]
        for k in vars_to_plot:
            if k in keys:
                payload[k] = np.asarray([float(r[k]) for r in rows], dtype=float)
        series.append((t, payload))
    if not series or not vars_to_plot:
        return

    n = len(vars_to_plot)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 3.5 * rows_n), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, var in zip(axes_flat, vars_to_plot):
        for t, payload in series:
            if var in payload:
                ax.plot(t, payload[var], alpha=0.25, linewidth=1.0)
        ax.set_title(var)
        ax.grid(alpha=0.3)
    for ax in axes_flat[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    out_path = stitched_dir / "timeseries_stitched_ALL_PROFILES.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[step] Saved stitched plot: {out_path}")


def _summarize_forecasts(forecast_h5: Path) -> dict[str, float]:
    all_err: list[np.ndarray] = []
    with h5py.File(forecast_h5, "r") as h5f:
        for profile_name in h5f.keys():
            grp = h5f[profile_name]
            table = np.asarray(grp["data"][()], dtype=float)
            cols_raw = grp.attrs.get("columns", [])
            cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in cols_raw]
            true_cols = [idx for idx, c in enumerate(cols) if c.startswith("x_true(t)_")]
            pred_cols = [idx for idx, c in enumerate(cols) if c.startswith("x_mean(t)_")]
            if not true_cols or len(true_cols) != len(pred_cols):
                continue
            err = table[:, pred_cols] - table[:, true_cols]
            all_err.append(err)
    if not all_err:
        return {"rmse": float("nan"), "mae": float("nan"), "max_abs": float("nan")}
    cat = np.concatenate(all_err, axis=0)
    return {
        "rmse": float(np.sqrt(np.mean(cat**2))),
        "mae": float(np.mean(np.abs(cat))),
        "max_abs": float(np.max(np.abs(cat))),
    }


def _compute_uncertainty_metrics(forecast_h5: Path) -> dict[str, Any]:
    z_by_level = {
        "50": 0.67448975,
        "80": 1.28155157,
        "95": 1.95996398,
    }
    all_err: list[np.ndarray] = []
    all_sigma95: list[np.ndarray] = []
    with h5py.File(forecast_h5, "r") as h5f:
        for profile_name in h5f.keys():
            grp = h5f[profile_name]
            table = np.asarray(grp["data"][()], dtype=float)
            cols_raw = grp.attrs.get("columns", [])
            cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in cols_raw]
            true_cols = [idx for idx, c in enumerate(cols) if c.startswith("x_true(t)_")]
            mean_cols = [idx for idx, c in enumerate(cols) if c.startswith("x_mean(t)_")]
            sigma_cols = [idx for idx, c in enumerate(cols) if c.startswith("x_2sigma(t)_")]
            if not true_cols or len(true_cols) != len(mean_cols) or len(sigma_cols) != len(true_cols):
                continue
            y_true = table[:, true_cols]
            y_mean = table[:, mean_cols]
            sigma95 = table[:, sigma_cols]
            all_err.append(np.abs(y_mean - y_true))
            all_sigma95.append(sigma95)

    if not all_err:
        return {
            "coverage": {"50": float("nan"), "80": float("nan"), "95": float("nan")},
            "interval_width": {"50": float("nan"), "80": float("nan"), "95": float("nan")},
            "calibration_error": float("nan"),
        }

    err = np.concatenate(all_err, axis=0)
    sigma95 = np.concatenate(all_sigma95, axis=0)
    std = sigma95 / 2.0

    coverage: dict[str, float] = {}
    width: dict[str, float] = {}
    cal_terms: list[float] = []
    for level, z in z_by_level.items():
        half_width = z * std
        cov = float(np.mean(err <= half_width))
        coverage[level] = cov
        width[level] = float(np.mean(2.0 * half_width))
        cal_terms.append(abs(cov - (float(level) / 100.0)))

    return {
        "coverage": coverage,
        "interval_width": width,
        "calibration_error": float(np.mean(cal_terms)),
    }


def _plot_metrics_over_cycles(cycle_metrics: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cycles = [int(m["cycle"]) for m in cycle_metrics]
    rmse = [float(m["forecast_quality"]["rmse"]) for m in cycle_metrics]
    mae = [float(m["forecast_quality"]["mae"]) for m in cycle_metrics]
    cal = [float(m["uncertainty_quality"]["calibration_error"]) for m in cycle_metrics]
    cov50 = [float(m["uncertainty_quality"]["coverage"]["50"]) for m in cycle_metrics]
    cov80 = [float(m["uncertainty_quality"]["coverage"]["80"]) for m in cycle_metrics]
    cov95 = [float(m["uncertainty_quality"]["coverage"]["95"]) for m in cycle_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(cycles, rmse, marker="o", label="RMSE")
    ax.plot(cycles, mae, marker="o", label="MAE")
    ax.set_title("Forecast error vs cycle")
    ax.set_xlabel("Cycle")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(cycles, cov50, marker="o", label="50%")
    ax.plot(cycles, cov80, marker="o", label="80%")
    ax.plot(cycles, cov95, marker="o", label="95%")
    ax.set_title("Empirical coverage vs cycle")
    ax.set_xlabel("Cycle")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(cycles, cal, marker="o", color="C3")
    ax.set_title("Calibration error vs cycle")
    ax.set_xlabel("Cycle")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    sharp95 = [float(m["uncertainty_quality"]["interval_width"]["95"]) for m in cycle_metrics]
    ax.plot(cycles, sharp95, marker="o", color="C2")
    ax.set_title("95% interval width (sharpness) vs cycle")
    ax.set_xlabel("Cycle")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "metrics_vs_cycle.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full experiment starting from sim batch folders.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-output-dir", type=Path, default=REPO_ROOT / "outputs" / "experiments")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    if cfg.strategy not in {"branching", "random"}:
        raise SystemExit("strategy must be branching or random.")

    run_dir = args.base_output_dir / cfg.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    test_manifest = run_dir / "test_manifest.json"

    sim_root = Path(cfg.sim_root).resolve()
    var_root = Path(cfg.variography_root).resolve()
    cfg_py = Path(cfg.config_py_path).resolve()
    known_batches = list(cfg.initial_sim_batches)
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.__dict__,
        "cycles": [],
    }

    for cycle in range(cfg.retrain_cycles):
        cycle_dir = run_dir / f"cycle_{cycle + 1:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n==== Cycle {cycle + 1}/{cfg.retrain_cycles}: build dataset ====")

        unscaled_h5 = _build_from_batches(
            sim_root=sim_root,
            batch_ids=known_batches,
            lookback=cfg.lookback,
            cfg_py=cfg_py,
            out_dir=cycle_dir / "unscaled",
        )

        scaled_h5 = _scale_with_fixed_test(
            unscaled_h5=unscaled_h5,
            out_dir=cycle_dir / "scaled",
            scaling_type=cfg.scaling_type,
            split_mode=cfg.split_mode,
            test_manifest=test_manifest,
            test_count=cfg.test_count,
            seed=cfg.seed + cycle,
        )
        print(f"[step] Built scaled dataset: {scaled_h5}")

        print(f"[step] Hyperparameter tuning...")
        best = _tune(
            scaled_h5=scaled_h5,
            out_dir=cycle_dir / "tuning",
            seed=cfg.seed + cycle,
            grid=cfg.hp_grid,
        )
        print(f"[step] Best trial: lr={best.learning_rate}, bs={best.batch_size}, n_lstm={best.n_lstm}, hl={best.hidden_lstm}, hf={best.hidden_fc}")

        print(f"[step] Training bagged ensemble...")
        ensemble = run_bagging_ensemble(
            scaled_h5,
            out_dir=cycle_dir / "ensemble",
            n_models=cfg.n_models,
            bag_fraction=cfg.bag_fraction,
            seed=cfg.seed + cycle,
            batch_size=best.batch_size,
            epochs=int(cfg.hp_grid.get("epochs", 20)),
            learning_rate=best.learning_rate,
            n_lstm=best.n_lstm,
            lstm_hidden=best.hidden_lstm,
            n_fc=1,
            fc_hidden=(best.hidden_fc,),
            prefer_gpu=cfg.prefer_gpu,
        )
        model_paths = [str(Path(d) / "model.pt") for d in ensemble["model_dirs"]]
        bagged_h5_path = Path(ensemble["bagged_h5_path"])
        forecast_h5 = Path(ensemble["forecast_output_path"])
        point_metrics = _summarize_forecasts(forecast_h5)
        unc_metrics = _compute_uncertainty_metrics(forecast_h5)
        metrics = {
            "forecast_quality": point_metrics,
            "uncertainty_quality": unc_metrics,
        }
        print(
            f"[cycle {cycle + 1}] Ensemble test summary: "
            f"RMSE={point_metrics['rmse']:.6f}, MAE={point_metrics['mae']:.6f}, MAX_ABS={point_metrics['max_abs']:.6f}, "
            f"COV50={unc_metrics['coverage']['50']:.4f}, COV80={unc_metrics['coverage']['80']:.4f}, "
            f"COV95={unc_metrics['coverage']['95']:.4f}, CAL_ERR={unc_metrics['calibration_error']:.4f}"
        )

        forecast_pdf = cycle_dir / "ensemble" / "ensemble_test_forecasts.pdf"
        save_forecast_profiles_pdf(
            forecast_h5_path=forecast_h5,
            output_pdf_path=forecast_pdf,
            mode="ensemble",
        )
        print(f"[cycle {cycle + 1}] Saved forecast visualization PDF: {forecast_pdf}")
        metrics_json = cycle_dir / "ensemble" / "ensemble_metrics.json"
        metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[cycle {cycle + 1}] Saved ensemble metrics JSON: {metrics_json}")

        # No need to generate/simulate new data after last training cycle.
        if cycle < cfg.retrain_cycles - 1:
            if cfg.strategy == "branching":
                var_batch = _run_recursive_branching_internal(
                    cfg=cfg,
                    model_paths=model_paths,
                    bagged_h5_path=bagged_h5_path,
                    lstm_hidden=int(best.hidden_lstm),
                    n_lstm=int(best.n_lstm),
                    fc_hidden=int(best.hidden_fc),
                    seed=cfg.seed + cycle,
                    variography_root=var_root,
                )
                pymola_mode = "branched_mat"
            else:
                n_new = _expected_new_profiles(
                    int(cfg.branching["N_r"]), int(cfg.branching["N_k"]), int(cfg.branching["N_b"])
                )
                var_batch = _next_batch_dir(var_root)
                _sample_random_profiles(
                    var_batch,
                    n_profiles=n_new,
                    T=float(cfg.branching["T"]),
                    dt=float(cfg.branching["dt"]),
                    seed=cfg.seed + cycle,
                    baseline=float(cfg.branching["baseline_angle_deg"]),
                    kernel=str(cfg.branching["kernel"]),
                )
                pymola_mode = "flat_mat"

            sim_out_dir = _run_dymola_internal(
                mode=pymola_mode,
                profiles_dir=Path(var_batch),
                output_interval=float(cfg.dymola["output_interval"]),
                sim_root=sim_root,
            )
            sim_batch = sim_out_dir.name
            known_batches.append(sim_batch.replace("batch_", ""))
        else:
            var_batch = None
            sim_batch = None

        metadata["cycles"].append(
            {
                "cycle": cycle + 1,
                "input_batches": list(known_batches),
                "unscaled_h5": str(unscaled_h5),
                "scaled_h5": str(scaled_h5),
                "best_trial": {
                    "learning_rate": best.learning_rate,
                    "batch_size": best.batch_size,
                    "n_lstm": best.n_lstm,
                    "hidden_lstm": best.hidden_lstm,
                    "hidden_fc": best.hidden_fc,
                },
                "bagged_h5_path": str(bagged_h5_path),
                "model_paths": model_paths,
                "forecast_h5": str(forecast_h5),
                "forecast_pdf": str(forecast_pdf),
                "ensemble_test_metrics": metrics,
                "ensemble_metrics_json": str(metrics_json),
                "new_variography_batch": None if var_batch is None else str(var_batch),
                "new_sim_batch": sim_batch,
            }
        )

    _plot_metrics_over_cycles(
        [
            {"cycle": row["cycle"], **row["ensemble_test_metrics"]}
            for row in metadata["cycles"]
        ],
        out_dir=run_dir / "metrics_plots",
    )

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
