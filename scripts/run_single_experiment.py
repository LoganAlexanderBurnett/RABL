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
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rabl.machine_learning import build_lstm_dataset
from rabl.machine_learning.dataset_scaling import LSTMDatasetScalerSplitter
from rabl.machine_learning.tuner import GridSearchConfig, run_grid_search, run_hyperband_search
from rabl.machine_learning.bagging_ensemble import run_bagging_ensemble
from rabl.machine_learning.lstm_pipeline import save_forecast_profiles_pdf
from rabl.machine_learning.recursive_branching import (
    RecursiveBranchingBatchConfig,
    run_recursive_branching_batch,
)
from rabl.interface.pymola import BatchConfig, DymolaBatchRunner
from rabl.variography.DrumVariography import DrumProfileGenerator

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


def _print_block_header(title: str, *, width: int = 96, fill: str = "=") -> None:
    bar = fill * width
    print(f"\n{bar}\n{title.center(width)}\n{bar}\n")


def _print_step_banner(cycle: int, label: str, *, width: int = 96) -> None:
    bar = "-" * width
    print(f"\n{bar}\n[cycle {cycle:02d}] {label}\n{bar}")


def _print_step_result(cycle: int, label: str, detail: str | None = None, *, width: int = 96) -> None:
    bar = "." * width
    print(f"{bar}\n[cycle {cycle:02d}] ✅ {label}")
    if detail:
        print(detail)
    print(f"{bar}\n")


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


def _load_variography_params(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"Missing config file: {config_path}")
    spec = importlib.util.spec_from_file_location("single_experiment_variography_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load config file: {config_path}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    required_keys = (
        "ELL",
        "SIGMA_THETA_TARGET",
        "NUGGET_V_DEG2_S2",
        "KERNEL",
        "T_GRID_DURATION",
        "T_GRID_INTERVALS",
        "BASELINE_ANGLE_DEG",
    )
    params = {key: getattr(config_module, key, None) for key in required_keys}
    missing = [key for key, value in params.items() if value is None]
    if missing:
        raise SystemExit(f"config.py is missing required variography keys: {missing}")
    return params


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
        train_frac=0.6,
        val_frac=0.4,
        test_frac=0.0,
        seed=seed,
        test_manifest_path=test_manifest if test_manifest.exists() else None,
        test_count=None if test_manifest.exists() else test_count,
        save_test_manifest_path=test_manifest if not test_manifest.exists() else None,
        output_dir=out_dir,
    )
    return splitter.run()


def _tune(scaled_h5: Path, out_dir: Path, seed: int, grid: dict[str, Any]) -> Any:
    method = str(grid.get("method", "grid")).strip().lower()
    if method not in {"grid", "hyperband"}:
        raise ValueError("hp_grid.method must be either 'grid' or 'hyperband'.")

    min_epochs_raw = grid.get("min_epochs")
    max_epochs_raw = grid.get("max_epochs")
    reduction_factor_raw = grid.get("reduction_factor", 3)
    if method == "hyperband":
        if min_epochs_raw is None or max_epochs_raw is None:
            raise ValueError("Hyperband tuning requires hp_grid.min_epochs and hp_grid.max_epochs.")
        min_epochs = int(min_epochs_raw)
        max_epochs = int(max_epochs_raw)
        reduction_factor = int(reduction_factor_raw)
    else:
        min_epochs = None
        max_epochs = None
        reduction_factor = int(reduction_factor_raw)

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
        preload_val_to_device=True,
        early_stopping_patience=(
            None
            if grid.get("early_stopping_patience") is None
            else int(grid["early_stopping_patience"])
        ),
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        reduction_factor=reduction_factor,
    )
    if method == "hyperband":
        _results, best = run_hyperband_search(tune_cfg)
    else:
        _results, best = run_grid_search(tune_cfg)
    return best


def _sample_random_profiles(batch_dir: Path, *, n_profiles: int, seed: int, variography_params: dict[str, Any]) -> None:
    duration = float(variography_params["T_GRID_DURATION"])
    intervals = int(variography_params["T_GRID_INTERVALS"])
    if intervals < 1:
        raise ValueError("T_GRID_INTERVALS must be >= 1.")
    dt = duration / intervals
    baseline = float(variography_params["BASELINE_ANGLE_DEG"])
    kernel = str(variography_params["KERNEL"])
    ell = float(variography_params["ELL"])
    sigma_target = float(variography_params["SIGMA_THETA_TARGET"])
    nugget = float(variography_params["NUGGET_V_DEG2_S2"])
    t_grid = np.arange(0.0, duration + dt, dt, dtype=float)
    if not np.isclose(t_grid[-1], duration):
        t_grid = np.append(t_grid, duration)
    gen = DrumProfileGenerator(kernel=kernel, ell=ell, nugget_v_deg2_s2=nugget)
    gen.solve_params_for_sigma_theta(
        t_grid=t_grid,
        sigma_theta_target=sigma_target,
        ell=ell,
        nugget=nugget,
        update_instance=True,
    )
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


def _copy_branched_results_to_batch_root_with_global_numbering(out_dir: Path, sim_root: Path) -> None:
    branched_results = out_dir / "branched_results"
    if not branched_results.exists():
        return
    sources = sorted(branched_results.glob("results_*.csv"))
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
    variography_params: dict[str, Any],
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
    duration = float(variography_params["T_GRID_DURATION"])
    intervals = int(variography_params["T_GRID_INTERVALS"])
    if intervals < 1:
        raise ValueError("T_GRID_INTERVALS must be >= 1.")
    dt = duration / intervals
    run_cfg = RecursiveBranchingBatchConfig(
        model_paths=tuple(Path(p) for p in model_paths),
        bagged_h5_path=Path(bagged_h5_path),
        output_dir=out_dir,
        T=duration,
        dt=dt,
        Nk=int(cfg.branching["N_k"]),
        Nb=int(cfg.branching["N_b"]),
        Nr=int(cfg.branching["N_r"]),
        baseline_angle_deg=float(variography_params["BASELINE_ANGLE_DEG"]),
        seed=int(seed),
        lookback=int(cfg.lookback),
        n_lstm=int(n_lstm),
        lstm_hidden=int(lstm_hidden),
        n_fc=1,
        fc_hidden=(int(fc_hidden),),
        kernel=str(variography_params["KERNEL"]),
        ell=float(variography_params["ELL"]),
        sigma_theta_target=float(variography_params["SIGMA_THETA_TARGET"]),
        nugget_v_deg2_s2=float(variography_params["NUGGET_V_DEG2_S2"]),
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
        _plot_branched_results(out_dir / "branched_results")
        _copy_branched_results_to_batch_root_with_global_numbering(out_dir, sim_root)
        _cleanup_production_artifacts(out_dir)
    return out_dir


def _cleanup_production_artifacts(out_dir: Path) -> None:
    generated_profiles = out_dir / "generated_profiles"
    if generated_profiles.exists():
        shutil.rmtree(generated_profiles, ignore_errors=True)

    branched_results_dir = out_dir / "branched_results"
    for file_path in out_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if branched_results_dir in file_path.parents:
            continue
        if file_path.suffix.lower() in {".txt", ".c", ".exe", ".mat"}:
            file_path.unlink(missing_ok=True)


def _plot_branched_results(branched_results_dir: Path) -> None:
    if not branched_results_dir.exists():
        return
    csv_paths = sorted(branched_results_dir.glob("results_*.csv"))
    if not csv_paths:
        return

    series: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = []
    vars_to_plot = list(PLOT_VARS)
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
        for k in vars_to_plot:
            if k in keys:
                payload[k] = np.asarray([float(r[k]) for r in rows], dtype=float)
        root_id = _extract_root_id_from_results_stem(csv_path.stem)
        series.append((root_id, t, payload))
    if not series or not vars_to_plot:
        return

    root_ids = sorted({root_id for root_id, _, _ in series})
    base_root_colors = ["crimson", "gold", "black"]
    rng = np.random.default_rng()
    root_colors: dict[str, tuple[float, float, float, float] | str] = {}
    for i, root_id in enumerate(root_ids):
        if i < len(base_root_colors):
            root_colors[root_id] = base_root_colors[i]
        else:
            root_colors[root_id] = (float(rng.random()), float(rng.random()), float(rng.random()), 1.0)

    n = len(vars_to_plot)
    cols = 6
    rows_n = 3
    fig, axes = plt.subplots(rows_n, cols, figsize=(30, 12), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, var in zip(axes_flat, vars_to_plot):
        for root_id, t, payload in series:
            if var in payload:
                ax.plot(t, payload[var], alpha=0.15, linewidth=1.0, color=root_colors[root_id])
        ax.set_title(var)
        ax.grid(alpha=0.3)
    for ax in axes_flat[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    out_path = branched_results_dir / "timeseries_branched_ALL_PROFILES.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[step] Saved branched-results plot: {out_path}")


def _normalize_batch_name(batch: str) -> str:
    batch = str(batch).strip()
    if batch.startswith("batch_"):
        return batch
    return f"batch_{int(batch):04d}"


def _plot_cycle_colored_batches(
    *,
    sim_root: Path,
    initial_sim_batches: list[str],
    cycle_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    batch_to_color: dict[str, tuple[float, float, float, float] | str] = {}
    batch_to_label: dict[str, str] = {}

    for raw_batch in initial_sim_batches:
        batch_name = _normalize_batch_name(raw_batch)
        batch_to_color[batch_name] = "black"
        batch_to_label[batch_name] = "initial_sim_batches"

    cmap = plt.get_cmap("tab10")
    for cycle_row in cycle_rows:
        new_batch = cycle_row.get("new_sim_batch")
        if not new_batch:
            continue
        cycle_idx = int(cycle_row.get("cycle", 0))
        batch_name = _normalize_batch_name(str(new_batch))
        batch_to_color[batch_name] = cmap((cycle_idx - 1) % 10)
        batch_to_label[batch_name] = f"cycle_{cycle_idx:02d}_added_profiles"

    series: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = []
    for batch_name, color in batch_to_color.items():
        batch_dir = sim_root / batch_name
        if not batch_dir.exists():
            print(f"[warn] Missing batch directory for cycle-colored plot: {batch_dir}")
            continue
        csv_paths = sorted(batch_dir.glob("results_drum_profile_*.csv"))
        for csv_path in csv_paths:
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
            for key in PLOT_VARS:
                if key in keys:
                    payload[key] = np.asarray([float(r[key]) for r in rows], dtype=float)
            series.append((batch_name, t, payload))

    if not series:
        print("[warn] No CSV series found for cycle-colored batch plot; skipping.")
        return

    fig, axes = plt.subplots(3, 6, figsize=(30, 12), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, var in zip(axes_flat, PLOT_VARS):
        for batch_name, t, payload in series:
            if var in payload:
                ax.plot(
                    t,
                    payload[var],
                    alpha=0.10,
                    linewidth=1.0,
                    color=batch_to_color[batch_name],
                )
        ax.set_title(var)
        ax.grid(alpha=0.3)
    for ax in axes_flat[len(PLOT_VARS):]:
        ax.set_axis_off()

    legend_handles = [
        Line2D([0], [0], color="black", lw=2, label="initial_sim_batches")
    ]
    seen_labels = {"initial_sim_batches"}
    for cycle_row in cycle_rows:
        new_batch = cycle_row.get("new_sim_batch")
        if not new_batch:
            continue
        cycle_idx = int(cycle_row.get("cycle", 0))
        label = f"cycle_{cycle_idx:02d}_added_profiles"
        if label in seen_labels:
            continue
        seen_labels.add(label)
        legend_handles.append(
            Line2D([0], [0], color=cmap((cycle_idx - 1) % 10), lw=2, label=label)
        )
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[step] Saved cycle-colored profiles plot: {output_path}")


def _extract_root_id_from_results_stem(stem: str) -> str:
    parts = stem.split("__")
    if len(parts) >= 2 and parts[0].startswith("results_"):
        return parts[0].replace("results_", "", 1)
    return "unknown_root"


def _compute_forecast_metrics(forecast_h5: Path) -> dict[str, Any]:
    y_true_profiles: list[np.ndarray] = []
    y_pred_profiles: list[np.ndarray] = []
    target_names: list[str] | None = None
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
            profile_target_names = [cols[idx].replace("x_true(t)_", "", 1) for idx in true_cols]
            if target_names is None:
                target_names = profile_target_names
            y_true = table[:, true_cols]
            y_pred = table[:, pred_cols]
            y_true_profiles.append(y_true)
            y_pred_profiles.append(y_pred)

    if not y_true_profiles or target_names is None:
        return {
            "per_target": {},
            "aggregated": {"smape": float("nan"), "nrmse": float("nan")},
            "horizon": {"rmse": {}, "mae": {}},
        }

    y_true_cat = np.concatenate(y_true_profiles, axis=0)
    y_pred_cat = np.concatenate(y_pred_profiles, axis=0)
    err_cat = y_pred_cat - y_true_cat
    abs_err_cat = np.abs(err_cat)

    rmse_by_target = np.sqrt(np.mean(err_cat**2, axis=0))
    mae_by_target = np.mean(abs_err_cat, axis=0)
    bias_by_target = np.mean(err_cat, axis=0)
    true_mean = np.mean(y_true_cat, axis=0)
    sst = np.sum((y_true_cat - true_mean) ** 2, axis=0)
    sse = np.sum(err_cat**2, axis=0)
    r2_by_target = np.where(sst > 0.0, 1.0 - (sse / sst), np.nan)

    denom = np.abs(y_true_cat) + np.abs(y_pred_cat)
    smape = float(np.mean(200.0 * abs_err_cat / np.maximum(denom, 1e-12)))
    target_range = np.max(y_true_cat, axis=0) - np.min(y_true_cat, axis=0)
    nrmse_by_target = np.where(target_range > 0.0, rmse_by_target / target_range, np.nan)
    nrmse = float(np.nanmean(nrmse_by_target))

    max_horizon = max(profile.shape[0] for profile in y_true_profiles)
    horizon_rmse_by_target = np.full((max_horizon, len(target_names)), np.nan, dtype=float)
    horizon_mae_by_target = np.full((max_horizon, len(target_names)), np.nan, dtype=float)
    for horizon_idx in range(max_horizon):
        horizon_true: list[np.ndarray] = []
        horizon_pred: list[np.ndarray] = []
        for true_profile, pred_profile in zip(y_true_profiles, y_pred_profiles):
            if horizon_idx < true_profile.shape[0]:
                horizon_true.append(true_profile[horizon_idx : horizon_idx + 1, :])
                horizon_pred.append(pred_profile[horizon_idx : horizon_idx + 1, :])
        if not horizon_true:
            continue
        h_true = np.concatenate(horizon_true, axis=0)
        h_pred = np.concatenate(horizon_pred, axis=0)
        h_err = h_pred - h_true
        horizon_rmse_by_target[horizon_idx, :] = np.sqrt(np.mean(h_err**2, axis=0))
        horizon_mae_by_target[horizon_idx, :] = np.mean(np.abs(h_err), axis=0)

    per_target: dict[str, dict[str, Any]] = {}
    for idx, target_name in enumerate(target_names):
        per_target[target_name] = {
            "rmse": float(rmse_by_target[idx]),
            "mae": float(mae_by_target[idx]),
            "bias": float(bias_by_target[idx]),
            "r2": float(r2_by_target[idx]),
            "nrmse": float(nrmse_by_target[idx]),
            "horizon_rmse": [float(v) for v in horizon_rmse_by_target[:, idx].tolist()],
            "horizon_mae": [float(v) for v in horizon_mae_by_target[:, idx].tolist()],
        }

    horizon_metrics = {
        "rmse": {target_name: per_target[target_name]["horizon_rmse"] for target_name in target_names},
        "mae": {target_name: per_target[target_name]["horizon_mae"] for target_name in target_names},
    }
    return {
        "per_target": per_target,
        "aggregated": {
            "smape": smape,
            "nrmse": nrmse,
        },
        "horizon": horizon_metrics,
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
    smape = [float(m["forecast_quality"]["aggregated"]["smape"]) for m in cycle_metrics]
    nrmse = [float(m["forecast_quality"]["aggregated"]["nrmse"]) for m in cycle_metrics]
    cal = [float(m["uncertainty_quality"]["calibration_error"]) for m in cycle_metrics]
    cov50 = [float(m["uncertainty_quality"]["coverage"]["50"]) for m in cycle_metrics]
    cov80 = [float(m["uncertainty_quality"]["coverage"]["80"]) for m in cycle_metrics]
    cov95 = [float(m["uncertainty_quality"]["coverage"]["95"]) for m in cycle_metrics]
    target_names = list(cycle_metrics[0]["forecast_quality"]["per_target"].keys())
    rmse_matrix = np.asarray(
        [
            [float(c["forecast_quality"]["per_target"][target]["rmse"]) for target in target_names]
            for c in cycle_metrics
        ],
        dtype=float,
    )
    mae_matrix = np.asarray(
        [
            [float(c["forecast_quality"]["per_target"][target]["mae"]) for target in target_names]
            for c in cycle_metrics
        ],
        dtype=float,
    )
    bias_matrix = np.asarray(
        [
            [float(c["forecast_quality"]["per_target"][target]["bias"]) for target in target_names]
            for c in cycle_metrics
        ],
        dtype=float,
    )
    r2_matrix = np.asarray(
        [
            [float(c["forecast_quality"]["per_target"][target]["r2"]) for target in target_names]
            for c in cycle_metrics
        ],
        dtype=float,
    )

    horizon_rmse_series: list[np.ndarray] = []
    horizon_mae_series: list[np.ndarray] = []
    for cycle_row in cycle_metrics:
        horizon_rmse_targets = cycle_row["forecast_quality"]["horizon"]["rmse"]
        horizon_mae_targets = cycle_row["forecast_quality"]["horizon"]["mae"]
        rmse_rows = np.asarray([horizon_rmse_targets[target] for target in target_names], dtype=float).T
        mae_rows = np.asarray([horizon_mae_targets[target] for target in target_names], dtype=float).T
        horizon_rmse_series.append(np.nanmean(rmse_rows, axis=1))
        horizon_mae_series.append(np.nanmean(mae_rows, axis=1))

    fig, axes = plt.subplots(3, 3, figsize=(22, 14))
    ax = axes[0, 0]
    ax.plot(cycles, smape, marker="o", label="sMAPE (%)")
    ax_twin = ax.twinx()
    ax_twin.plot(cycles, nrmse, marker="s", color="C2", label="NRMSE")
    ax.set_title("Aggregated performance vs cycle")
    ax.set_xlabel("Cycle")
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    ax = axes[0, 1]
    ax.plot(cycles, cov50, marker="o", label="50%")
    ax.plot(cycles, cov80, marker="o", label="80%")
    ax.plot(cycles, cov95, marker="o", label="95%")
    ax.set_title("Empirical coverage vs cycle")
    ax.set_xlabel("Cycle")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(cycles, cal, marker="o", color="C3")
    ax.set_title("Calibration error vs cycle")
    ax.set_xlabel("Cycle")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    sharp95 = [float(m["uncertainty_quality"]["interval_width"]["95"]) for m in cycle_metrics]
    ax.plot(cycles, sharp95, marker="o", color="C2")
    ax.set_title("95% interval width (sharpness) vs cycle")
    ax.set_xlabel("Cycle")
    ax.grid(alpha=0.3)

    eps = 1e-12

    def _log_norm(arr: np.ndarray) -> LogNorm:
        positive = arr[np.isfinite(arr) & (arr > 0)]
        if positive.size == 0:
            return LogNorm(vmin=eps, vmax=1.0)
        vmin = max(float(np.min(positive)), eps)
        vmax = max(float(np.max(positive)), vmin * 10.0)
        return LogNorm(vmin=vmin, vmax=vmax)

    ax = axes[1, 1]
    im = ax.imshow(
        np.maximum(rmse_matrix, eps),
        aspect="auto",
        interpolation="nearest",
        norm=_log_norm(rmse_matrix),
    )
    ax.set_title("Per-target RMSE (log scale)")
    ax.set_ylabel("Cycle")
    ax.set_yticks(range(len(cycles)))
    ax.set_yticklabels(cycles)
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    im = ax.imshow(
        np.maximum(mae_matrix, eps),
        aspect="auto",
        interpolation="nearest",
        norm=_log_norm(mae_matrix),
    )
    ax.set_title("Per-target MAE (log scale)")
    ax.set_ylabel("Cycle")
    ax.set_yticks(range(len(cycles)))
    ax.set_yticklabels(cycles)
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2, 0]
    im = ax.imshow(
        np.maximum(np.abs(bias_matrix), eps),
        aspect="auto",
        interpolation="nearest",
        norm=_log_norm(np.abs(bias_matrix)),
    )
    ax.set_title("Per-target |Bias| (log scale)")
    ax.set_ylabel("Cycle")
    ax.set_yticks(range(len(cycles)))
    ax.set_yticklabels(cycles)
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2, 1]
    im = ax.imshow(r2_matrix, aspect="auto", interpolation="nearest", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("Per-target R²")
    ax.set_ylabel("Cycle")
    ax.set_yticks(range(len(cycles)))
    ax.set_yticklabels(cycles)
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2, 2]
    for cycle_idx, cycle_num in enumerate(cycles):
        ax.plot(horizon_rmse_series[cycle_idx], alpha=0.8, label=f"C{cycle_num} RMSE")
        ax.plot(horizon_mae_series[cycle_idx], alpha=0.8, linestyle="--", label=f"C{cycle_num} MAE")
    ax.set_title("Horizon-wise mean RMSE/MAE (across targets)")
    ax.set_xlabel("Horizon step")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    if len(cycles) <= 5:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "metrics_vs_cycle.png", dpi=150)
    plt.close(fig)


def _plot_timing_metrics_over_cycles(cycle_rows: list[dict[str, Any]], output_path: Path) -> None:
    timing_keys = [
        "build_unscaled_dataset_sec",
        "scale_split_dataset_sec",
        "hyperparameter_tuning_sec",
        "ensemble_training_sec",
        "ensemble_metrics_compute_sec",
        "forecast_pdf_render_sec",
        "profile_generation_sec",
        "dymola_simulation_sec",
        "cycle_total_sec",
    ]
    cycle_indices = [int(cycle_row.get("cycle", i + 1)) for i, cycle_row in enumerate(cycle_rows)]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes_flat = axes.ravel()
    for ax, key in zip(axes_flat, timing_keys):
        values = [float(cycle_row["timing"].get(key, float("nan"))) for cycle_row in cycle_rows]
        ax.plot(cycle_indices, values, marker="o")
        ax.set_title(key)
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.3)
    for ax in axes_flat[-3:]:
        ax.set_xlabel("cycle")
    fig.suptitle("run_single_experiment timing metrics across cycles")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _print_timing_overview(cycle_rows: list[dict[str, Any]], total_runtime_sec: float) -> None:
    timing_keys = [
        "build_unscaled_dataset_sec",
        "scale_split_dataset_sec",
        "hyperparameter_tuning_sec",
        "ensemble_training_sec",
        "ensemble_metrics_compute_sec",
        "forecast_pdf_render_sec",
        "profile_generation_sec",
        "dymola_simulation_sec",
        "cycle_total_sec",
    ]
    if not cycle_rows:
        print("\n[timing-overview] No cycle timing data available.")
        return

    col_metric = 34
    col_val = 12
    header_bar = "=" * (col_metric + (col_val * 4) + 6)
    print(f"\n{header_bar}")
    print("TIMING OVERVIEW".center(len(header_bar)))
    print(header_bar)
    print(
        f"{'Metric':<{col_metric}} | {'Mean (s)':>{col_val}} | {'Min (s)':>{col_val}} | "
        f"{'Max (s)':>{col_val}} | {'Sum (s)':>{col_val}}"
    )
    print("-" * len(header_bar))

    key_to_values: dict[str, list[float]] = {}
    for key in timing_keys:
        values = [float(row.get('timing', {}).get(key, float('nan'))) for row in cycle_rows]
        values = [v for v in values if np.isfinite(v)]
        key_to_values[key] = values
        if not values:
            print(f"{key:<{col_metric}} | {'n/a':>{col_val}} | {'n/a':>{col_val}} | {'n/a':>{col_val}} | {'n/a':>{col_val}}")
            continue
        print(
            f"{key:<{col_metric}} | {np.mean(values):>{col_val}.2f} | {np.min(values):>{col_val}.2f} | "
            f"{np.max(values):>{col_val}.2f} | {np.sum(values):>{col_val}.2f}"
        )

    print("-" * len(header_bar))
    print(f"{'total_runtime_sec':<{col_metric}} | {'':>{col_val}} | {'':>{col_val}} | {'':>{col_val}} | {total_runtime_sec:>{col_val}.2f}")
    print(header_bar)

    stack_keys = [k for k in timing_keys if k != "cycle_total_sec"]
    total_stack = sum(np.sum(key_to_values[k]) for k in stack_keys if key_to_values[k])
    if total_stack <= 0:
        return
    print("Cycle-time composition (share of summed step times):")
    for key in stack_keys:
        values = key_to_values[key]
        if not values:
            continue
        portion = float(np.sum(values) / total_stack)
        bar = "█" * int(round(portion * 40))
        print(f"  - {key:<32} {portion:>6.1%}  {bar}")


def _save_forecast_pdf_subset(
    *,
    forecast_h5_path: Path,
    output_pdf_path: Path,
    max_profiles: int,
) -> None:
    with h5py.File(forecast_h5_path, "r") as src:
        names = sorted(src.keys())[:max(0, int(max_profiles))]
        if not names:
            print("[step] No forecast profiles selected for PDF; skipping PDF generation.")
            return
        subset_h5 = output_pdf_path.with_suffix(".subset_tmp.h5")
        with h5py.File(subset_h5, "w") as dst:
            for name in names:
                src.copy(name, dst)
        save_forecast_profiles_pdf(
            forecast_h5_path=subset_h5,
            output_pdf_path=output_pdf_path,
            mode="ensemble",
        )
    subset_h5.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full experiment starting from sim batch folders.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-output-dir", type=Path, default=REPO_ROOT / "outputs" / "experiments")
    parser.add_argument(
        "--plot-n-forecasts",
        type=int,
        default=10,
        help="Number of forecast profiles to include in the ensemble forecast PDF per cycle.",
    )
    args = parser.parse_args()
    if args.plot_n_forecasts < 0:
        raise SystemExit("--plot-n-forecasts must be >= 0.")

    cfg = _load_cfg(args.config)
    if cfg.strategy not in {"branching", "random"}:
        raise SystemExit("strategy must be branching or random.")

    run_dir = args.base_output_dir / cfg.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    test_manifest = run_dir / "test_manifest.json"

    sim_root = Path(cfg.sim_root).resolve()
    var_root = Path(cfg.variography_root).resolve()
    cfg_py = Path(cfg.config_py_path).resolve()
    variography_params = _load_variography_params(cfg_py)
    known_batches = list(cfg.initial_sim_batches)
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.__dict__,
        "cycles": [],
    }
    seed_manifest: dict[str, Any] = {
        "timestamp_utc": metadata["timestamp_utc"],
        "base_seed": int(cfg.seed),
        "cycles": [],
    }

    all_start = perf_counter()
    for cycle in range(cfg.retrain_cycles):
        cycle_seed = int(cfg.seed + cycle)
        cycle_start = perf_counter()
        step_times: dict[str, float] = {}
        cycle_input_batches = list(known_batches)
        cycle_dir = run_dir / f"cycle_{cycle + 1:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        _print_block_header(
            f"CYCLE {cycle + 1}/{cfg.retrain_cycles}  |  SEED={cycle_seed}  |  OUTPUT={cycle_dir}"
        )

        _print_step_banner(cycle + 1, "Build unscaled dataset")
        t0 = perf_counter()
        unscaled_h5 = _build_from_batches(
            sim_root=sim_root,
            batch_ids=known_batches,
            lookback=cfg.lookback,
            cfg_py=cfg_py,
            out_dir=cycle_dir / "unscaled",
        )
        step_times["build_unscaled_dataset_sec"] = perf_counter() - t0
        _print_step_result(cycle + 1, "Build unscaled dataset complete", f"Output: {unscaled_h5}")

        _print_step_banner(cycle + 1, "Scale + split dataset")
        t0 = perf_counter()
        scaled_h5 = _scale_with_fixed_test(
            unscaled_h5=unscaled_h5,
            out_dir=cycle_dir / "scaled",
            scaling_type=cfg.scaling_type,
            split_mode=cfg.split_mode,
            test_manifest=test_manifest,
            test_count=cfg.test_count,
            seed=cycle_seed,
        )
        step_times["scale_split_dataset_sec"] = perf_counter() - t0
        _print_step_result(cycle + 1, "Scale + split complete", f"Output: {scaled_h5}")

        _print_step_banner(cycle + 1, "Hyperparameter tuning")
        t0 = perf_counter()
        best = _tune(
            scaled_h5=scaled_h5,
            out_dir=cycle_dir / "tuning",
            seed=cycle_seed,
            grid=cfg.hp_grid,
        )
        step_times["hyperparameter_tuning_sec"] = perf_counter() - t0
        _print_step_result(
            cycle + 1,
            "Hyperparameter tuning complete",
            (
                f"Best trial: lr={best.learning_rate}, bs={best.batch_size}, "
                f"n_lstm={best.n_lstm}, hl={best.hidden_lstm}, hf={best.hidden_fc}"
            ),
        )

        _print_step_banner(cycle + 1, "Train bagged ensemble")
        t0 = perf_counter()
        ensemble = run_bagging_ensemble(
            scaled_h5,
            out_dir=cycle_dir / "ensemble",
            n_models=cfg.n_models,
            bag_fraction=cfg.bag_fraction,
            seed=cycle_seed,
            batch_size=best.batch_size,
            epochs=int(cfg.hp_grid.get("epochs", 20)),
            learning_rate=best.learning_rate,
            n_lstm=best.n_lstm,
            lstm_hidden=best.hidden_lstm,
            n_fc=1,
            fc_hidden=(best.hidden_fc,),
            prefer_gpu=cfg.prefer_gpu,
            preload_val_to_device=True,
        )
        step_times["ensemble_training_sec"] = perf_counter() - t0
        model_paths = [str(Path(d) / "model.pt") for d in ensemble["model_dirs"]]
        bagged_h5_path = Path(ensemble["bagged_h5_path"])
        forecast_h5 = Path(ensemble["forecast_output_path"])
        t0 = perf_counter()
        point_metrics = _compute_forecast_metrics(forecast_h5)
        unc_metrics = _compute_uncertainty_metrics(forecast_h5)
        step_times["ensemble_metrics_compute_sec"] = perf_counter() - t0
        metrics = {
            "forecast_quality": point_metrics,
            "uncertainty_quality": unc_metrics,
        }
        per_target = point_metrics.get("per_target", {})
        target_rmse_values = [float(per_target[name]["rmse"]) for name in per_target]
        target_mae_values = [float(per_target[name]["mae"]) for name in per_target]
        mean_target_rmse = float(np.nanmean(target_rmse_values)) if target_rmse_values else float("nan")
        mean_target_mae = float(np.nanmean(target_mae_values)) if target_mae_values else float("nan")
        print(
            f"[cycle {cycle + 1}] Ensemble test summary: "
            f"mean_target_RMSE={mean_target_rmse:.6f}, mean_target_MAE={mean_target_mae:.6f}, "
            f"sMAPE={point_metrics['aggregated']['smape']:.4f}%, NRMSE={point_metrics['aggregated']['nrmse']:.6f}, "
            f"COV50={unc_metrics['coverage']['50']:.4f}, COV80={unc_metrics['coverage']['80']:.4f}, "
            f"COV95={unc_metrics['coverage']['95']:.4f}, CAL_ERR={unc_metrics['calibration_error']:.4f}"
        )

        forecast_pdf = cycle_dir / "ensemble" / "ensemble_test_forecasts.pdf"
        t0 = perf_counter()
        _save_forecast_pdf_subset(
            forecast_h5_path=forecast_h5,
            output_pdf_path=forecast_pdf,
            max_profiles=args.plot_n_forecasts,
        )
        step_times["forecast_pdf_render_sec"] = perf_counter() - t0
        _print_step_result(cycle + 1, "Ensemble training + evaluation complete", f"Forecast PDF: {forecast_pdf}")
        metrics_json = cycle_dir / "ensemble" / "ensemble_metrics.json"
        metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[cycle {cycle + 1}] Saved ensemble metrics JSON: {metrics_json}")

        # No need to generate/simulate new data after last training cycle.
        if cycle < cfg.retrain_cycles - 1:
            if cfg.strategy == "branching":
                _print_step_banner(cycle + 1, "Generate new profiles via recursive branching")
                t0 = perf_counter()
                var_batch = _run_recursive_branching_internal(
                    cfg=cfg,
                    variography_params=variography_params,
                    model_paths=model_paths,
                    bagged_h5_path=bagged_h5_path,
                    lstm_hidden=int(best.hidden_lstm),
                    n_lstm=int(best.n_lstm),
                    fc_hidden=int(best.hidden_fc),
                    seed=cycle_seed,
                    variography_root=var_root,
                )
                step_times["profile_generation_sec"] = perf_counter() - t0
                pymola_mode = "branched_mat"
                _print_step_result(cycle + 1, "Recursive branching profile generation complete", f"Profiles dir: {var_batch}")
            else:
                _print_step_banner(cycle + 1, "Generate new random profiles")
                t0 = perf_counter()
                n_new = _expected_new_profiles(
                    int(cfg.branching["N_r"]), int(cfg.branching["N_k"]), int(cfg.branching["N_b"])
                )
                var_batch = _next_batch_dir(var_root)
                _sample_random_profiles(
                    var_batch,
                    n_profiles=n_new,
                    seed=cycle_seed,
                    variography_params=variography_params,
                )
                step_times["profile_generation_sec"] = perf_counter() - t0
                pymola_mode = "flat_mat"
                _print_step_result(cycle + 1, "Random profile generation complete", f"Profiles dir: {var_batch}")

            _print_step_banner(cycle + 1, f"Run Dymola simulation ({pymola_mode})")
            t0 = perf_counter()
            sim_out_dir = _run_dymola_internal(
                mode=pymola_mode,
                profiles_dir=Path(var_batch),
                output_interval=float(cfg.dymola["output_interval"]),
                sim_root=sim_root,
            )
            step_times["dymola_simulation_sec"] = perf_counter() - t0
            sim_batch = sim_out_dir.name
            known_batches.append(sim_batch.replace("batch_", ""))
            _print_step_result(cycle + 1, "Dymola simulation complete", f"New simulation batch: {sim_batch}")
        else:
            var_batch = None
            sim_batch = None
            step_times["profile_generation_sec"] = 0.0
            step_times["dymola_simulation_sec"] = 0.0

        cycle_total = perf_counter() - cycle_start
        step_times["cycle_total_sec"] = cycle_total
        _print_block_header(f"CYCLE {cycle + 1} COMPLETE  |  TOTAL={cycle_total:.2f}s", fill="#")
        for key in (
            "build_unscaled_dataset_sec",
            "scale_split_dataset_sec",
            "hyperparameter_tuning_sec",
            "ensemble_training_sec",
            "ensemble_metrics_compute_sec",
            "forecast_pdf_render_sec",
            "profile_generation_sec",
            "dymola_simulation_sec",
        ):
            print(f"[timing] cycle {cycle + 1} {key}: {step_times.get(key, 0.0):.2f} s")

        metadata["cycles"].append(
            {
                "cycle": cycle + 1,
                "input_batches": cycle_input_batches,
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
                "forecast_profiles_plotted": int(max(args.plot_n_forecasts, 0)),
                "ensemble_test_metrics": metrics,
                "ensemble_metrics_json": str(metrics_json),
                "new_variography_batch": None if var_batch is None else str(var_batch),
                "new_sim_batch": sim_batch,
                "timing": step_times,
            }
        )
        cycle_seed_info: dict[str, Any] = {
            "cycle": cycle + 1,
            "cycle_seed": cycle_seed,
            "scale_split_dataset_seed": cycle_seed,
            "hyperparameter_tuning_seed": cycle_seed,
            "ensemble_training_seed": cycle_seed,
        }
        if cycle < cfg.retrain_cycles - 1:
            if cfg.strategy == "branching":
                cycle_seed_info["profile_generation_seed"] = cycle_seed
                cycle_seed_info["recursive_branching_root_seeds"] = [
                    cycle_seed + root_idx for root_idx in range(int(cfg.branching["N_r"]))
                ]
            else:
                cycle_seed_info["profile_generation_seed"] = cycle_seed
                cycle_seed_info["random_profile_generation_seed"] = cycle_seed
        seed_manifest["cycles"].append(cycle_seed_info)

    _plot_metrics_over_cycles(
        [
            {"cycle": row["cycle"], **row["ensemble_test_metrics"]}
            for row in metadata["cycles"]
        ],
        out_dir=run_dir / "metrics_plots",
    )
    _plot_timing_metrics_over_cycles(
        cycle_rows=metadata["cycles"],
        output_path=run_dir / "metrics_plots" / "timing_metrics_vs_cycle.png",
    )
    _plot_cycle_colored_batches(
        sim_root=sim_root,
        initial_sim_batches=list(cfg.initial_sim_batches),
        cycle_rows=metadata["cycles"],
        output_path=run_dir / "metrics_plots" / "profiles_by_cycle_color.png",
    )

    metadata_path = run_dir / "run_metadata.json"
    seed_manifest_path = run_dir / "seed_manifest.json"
    metadata["total_runtime_sec"] = perf_counter() - all_start
    print(f"[timing] all cycles total: {metadata['total_runtime_sec']:.2f} s")
    _print_timing_overview(metadata["cycles"], total_runtime_sec=float(metadata["total_runtime_sec"]))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    seed_manifest_path.write_text(json.dumps(seed_manifest, indent=2), encoding="utf-8")
    print(f"Done. Metadata: {metadata_path}")
    print(f"Done. Seed manifest: {seed_manifest_path}")


if __name__ == "__main__":
    main()
