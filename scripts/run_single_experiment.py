"""Run a single branching-vs-random experiment harness execution.

This driver is config-driven and focuses on one experiment unit at a time:
    - fixed unseen target(s)
    - one (N_r, N_k, N_b) tuple
    - one strategy {branching, random}
    - one seed
    - one retraining-cycle count

It orchestrates:
    1) profile generation (branching or random),
    2) Dymola simulation,
    3) optional ingest/tune/retrain/evaluate command hooks,
    4) metrics/logging/artifact bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
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

from rabl.variography.DrumVariography import DrumProfileGenerator
from rabl.machine_learning.dataset_scaling import LSTMDatasetScalerSplitter
from rabl.machine_learning.tuner import GridSearchConfig, run_grid_search
from rabl.machine_learning.bagging_ensemble import run_bagging_ensemble


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    strategy: str  # branching | random
    seed: int
    retrain_cycles: int
    N_r: int
    N_k: int
    N_b: int
    model_paths: list[str]
    bagged_h5_path: str
    lstm_hidden: int
    fc_hidden: list[int]
    lookback: int = 12
    n_features: int = 13
    n_lstm: int = 1
    n_fc: int = 1
    baseline_angle_deg: float = 45.0
    T: float = 200.0
    dt: float = 0.4
    kernel: str = "matern52"
    device: str = "cpu"
    output_interval: float = 0.1
    run_pymola_mode: str = "testing"
    paired_branching_profiles_h5: str | None = None
    paired_branching_profile_dir: str | None = None
    metrics_input_json: str | None = None
    retrain_policy: str = "retune"  # retune | no-retune
    ingest_command: list[str] | None = None
    tune_command: list[str] | None = None
    retrain_command: list[str] | None = None
    evaluate_command: list[str] | None = None
    initial_unscaled_h5: str | None = None
    session_unscaled_h5: list[str] | None = None


def _load_config(config_path: Path) -> ExperimentConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "strategy",
        "seed",
        "retrain_cycles",
        "N_r",
        "N_k",
        "N_b",
        "model_paths",
        "bagged_h5_path",
        "lstm_hidden",
        "fc_hidden",
    }
    missing = sorted(required - set(data.keys()))
    if missing:
        raise SystemExit(f"Missing required config keys: {missing}")
    strategy = str(data["strategy"]).strip().lower()
    if strategy not in {"branching", "random"}:
        raise SystemExit("config.strategy must be 'branching' or 'random'.")
    data["strategy"] = strategy
    retrain_policy = str(data.get("retrain_policy", "retune")).strip().lower()
    if retrain_policy not in {"retune", "no-retune"}:
        raise SystemExit("config.retrain_policy must be one of {'retune', 'no-retune'}.")
    data["retrain_policy"] = retrain_policy
    return ExperimentConfig(**data)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _count_profiles_in_profiles_h5(profiles_h5_path: Path) -> int:
    if not profiles_h5_path.exists():
        raise FileNotFoundError(f"profiles.h5 not found: {profiles_h5_path}")
    total = 0
    with h5py.File(profiles_h5_path, "r") as h5f:
        for root_name in h5f.keys():
            grp = h5f[root_name]
            total += len(grp.keys())
    if total < 1:
        raise RuntimeError(f"No profiles found in {profiles_h5_path}")
    return total


def _extract_reference_time_grid_and_baseline(profiles_h5_path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(profiles_h5_path, "r") as h5f:
        root_names = sorted(h5f.keys())
        if not root_names:
            raise RuntimeError(f"No root groups found in {profiles_h5_path}")
        first_root = h5f[root_names[0]]
        profile_names = sorted(first_root.keys())
        if not profile_names:
            raise RuntimeError(f"No profile groups found in root '{root_names[0]}'")
        p0 = first_root[profile_names[0]]
        t_grid = np.asarray(p0["t"][()], dtype=float)
        theta = np.asarray(p0["theta_deg"][()], dtype=float)
        if t_grid.ndim != 1 or t_grid.size < 2:
            raise RuntimeError("Invalid t-grid found in profiles.h5")
        baseline = float(theta[0])
        return t_grid, baseline


def _write_random_mat_profiles(
    out_dir: Path,
    *,
    n_profiles: int,
    t_grid: np.ndarray,
    baseline_angle_deg: float,
    kernel: str,
    seed: int,
) -> list[Path]:
    generator = DrumProfileGenerator(kernel=kernel)
    profiles = generator.generate(
        t_grid=t_grid,
        n_realizations=n_profiles,
        baseline_angle_deg=baseline_angle_deg,
        seed=seed,
    )
    written: list[Path] = []
    for idx, profile in enumerate(profiles):
        out_path = out_dir / f"drum_profile_{idx:05d}.mat"
        profile.save_mat(str(out_path))
        written.append(out_path)
    return written


def _prepare_branching_profiles(cfg: ExperimentConfig, strategy_dir: Path) -> tuple[Path, int]:
    branching_out = _ensure_dir(strategy_dir / "branching_profiles")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_recursive_branching.py"),
        "--run-mode",
        "testing",
        "--output-dir",
        str(branching_out),
        "--model-path",
        *[str(Path(p)) for p in cfg.model_paths],
        "--bagged-h5-path",
        str(Path(cfg.bagged_h5_path)),
        "--lstm-hidden",
        str(cfg.lstm_hidden),
        "--fc-hidden",
        *[str(v) for v in cfg.fc_hidden],
        "--lookback",
        str(cfg.lookback),
        "--n-features",
        str(cfg.n_features),
        "--n-lstm",
        str(cfg.n_lstm),
        "--n-fc",
        str(cfg.n_fc),
        "--T",
        str(cfg.T),
        "--dt",
        str(cfg.dt),
        "--Nk",
        str(cfg.N_k),
        "--Nb",
        str(cfg.N_b),
        "--Nr",
        str(cfg.N_r),
        "--baseline-angle-deg",
        str(cfg.baseline_angle_deg),
        "--kernel",
        cfg.kernel,
        "--device",
        cfg.device,
        "--visualize",
        "1",
        "--seed",
        str(cfg.seed),
    ]
    _run_cmd(cmd, cwd=REPO_ROOT)
    profiles_h5 = branching_out / "profiles.h5"
    n_profiles = _count_profiles_in_profiles_h5(profiles_h5)
    return branching_out, n_profiles


def _prepare_random_profiles(cfg: ExperimentConfig, strategy_dir: Path) -> tuple[Path, int]:
    if not cfg.paired_branching_profiles_h5:
        raise SystemExit(
            "Random strategy requires config.paired_branching_profiles_h5 to enforce budget matching."
        )
    ref_profiles_h5 = Path(cfg.paired_branching_profiles_h5)
    n_profiles = _count_profiles_in_profiles_h5(ref_profiles_h5)
    t_grid, baseline = _extract_reference_time_grid_and_baseline(ref_profiles_h5)
    baseline_angle_deg = cfg.baseline_angle_deg if cfg.baseline_angle_deg is not None else baseline

    random_profile_dir = _ensure_dir(strategy_dir / "random_profiles")
    written = _write_random_mat_profiles(
        random_profile_dir,
        n_profiles=n_profiles,
        t_grid=t_grid,
        baseline_angle_deg=baseline_angle_deg,
        kernel=cfg.kernel,
        seed=cfg.seed,
    )
    if len(written) != n_profiles:
        raise RuntimeError("Random profile generation failed budget matching.")
    return random_profile_dir, n_profiles


def _simulate(cfg: ExperimentConfig, strategy: str, profiles_dir: Path, strategy_dir: Path) -> Path:
    out_dir = _ensure_dir(strategy_dir / "sim_profiles")
    mode = "branched_mat" if strategy == "branching" else "flat_mat"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_pymola_mode.py"),
        "--mode",
        mode,
        "--run-mode",
        cfg.run_pymola_mode,
        "--out",
        str(out_dir),
        "--profiles",
        str(profiles_dir),
        "--output-interval",
        str(cfg.output_interval),
    ]
    _run_cmd(cmd, cwd=REPO_ROOT)
    return out_dir


def _render_command_template(template: list[str], values: dict[str, str]) -> list[str]:
    return [part.format(**values) for part in template]


def _run_optional_hook(name: str, template: list[str] | None, values: dict[str, str]) -> None:
    if not template:
        print(f"[hook:{name}] skipped (not configured).")
        return
    cmd = _render_command_template(template, values)
    _run_cmd(cmd, cwd=REPO_ROOT)


def _load_metrics_input(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metrics_input_json must be a JSON object: {path}")
    return payload


def _plot_if_available(metrics: dict[str, Any], out_dir: Path) -> dict[str, str]:
    plots: dict[str, str] = {}

    perf = metrics.get("performance_vs_budget")
    if isinstance(perf, list) and perf:
        budgets = [float(row["budget"]) for row in perf]
        values = [float(row["rmse"]) for row in perf]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(budgets, values, marker="o")
        ax.set_xlabel("Simulation budget (profiles)")
        ax.set_ylabel("RMSE")
        ax.set_title("Performance vs simulation budget")
        ax.grid(alpha=0.3)
        out = out_dir / "performance_vs_budget.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        plots["performance_vs_budget"] = str(out)

    cal = metrics.get("calibration_curve")
    if isinstance(cal, list) and cal:
        nominal = [float(row["nominal"]) for row in cal]
        empirical = [float(row["empirical"]) for row in cal]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.plot(nominal, empirical, marker="o")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title("Calibration curve")
        ax.grid(alpha=0.3)
        out = out_dir / "calibration_curve.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        plots["calibration_curve"] = str(out)

    delta = metrics.get("branching_minus_random_delta")
    if isinstance(delta, list) and delta:
        x = np.arange(len(delta))
        y = np.asarray([float(v) for v in delta], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0.0, color="gray", linestyle="--")
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Metric index")
        ax.set_ylabel("Delta (branching - random)")
        ax.set_title("Branching-minus-random deltas")
        ax.grid(alpha=0.3)
        out = out_dir / "branching_minus_random_delta.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        plots["branching_minus_random_delta"] = str(out)

    return plots


def _append_summary_row(summary_csv: Path, row: dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = [
        "timestamp_utc",
        "experiment_id",
        "strategy",
        "retrain_policy",
        "seed",
        "N_r",
        "N_k",
        "N_b",
        "retrain_cycles",
        "target_spec",
        "budget_profiles",
        "run_dir",
    ]
    for key in ordered_keys:
        row.setdefault(key, "")

    line = ",".join(str(row[key]).replace(",", ";") for key in ordered_keys)
    if not summary_csv.exists():
        header = ",".join(ordered_keys)
        summary_csv.write_text(f"{header}\n{line}\n", encoding="utf-8")
    else:
        with summary_csv.open("a", encoding="utf-8") as fp:
            fp.write(f"{line}\n")


def _decode_colnames(raw: np.ndarray) -> list[str]:
    out: list[str] = []
    for item in raw:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def ingest_command(
    *,
    input_unscaled_h5: Path,
    output_dir: Path,
    holdout_manifest: Path,
    test_count: int = 2000,
    save_manifest_if_missing: bool = False,
) -> Path:
    """Scale + split dataset with fixed test and resampled train/val for retraining cycles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest_path = holdout_manifest if save_manifest_if_missing else None
    splitter = LSTMDatasetScalerSplitter(
        input_path=input_unscaled_h5,
        scaling_type="standard",
        split_mode="profile",
        train_frac=0.5,
        val_frac=0.5,
        test_frac=0.0,
        test_manifest_path=holdout_manifest if holdout_manifest.exists() else None,
        test_count=None if holdout_manifest.exists() else test_count,
        save_test_manifest_path=save_manifest_path,
    )
    scaled_h5 = splitter.run()
    return scaled_h5


def evaluate_command(
    *,
    forecast_h5_path: Path,
    output_json_path: Path,
) -> dict[str, Any]:
    """Compute simple forecast + uncertainty metrics from ensemble rolling-forecast HDF5."""
    if not forecast_h5_path.exists():
        raise FileNotFoundError(f"Forecast HDF5 not found: {forecast_h5_path}")

    all_err: list[np.ndarray] = []
    all_abs: list[np.ndarray] = []
    all_sigma2: list[np.ndarray] = []
    coverage_95_count = 0
    coverage_95_total = 0

    with h5py.File(forecast_h5_path, "r") as h5f:
        for profile_name in h5f.keys():
            grp = h5f[profile_name]
            data = np.asarray(grp["data"][()], dtype=float)
            columns = _decode_colnames(grp.attrs["column_names"])
            true_cols = [i for i, c in enumerate(columns) if c.startswith("x_true(t)_")]
            mean_cols = [i for i, c in enumerate(columns) if c.startswith("x_mean(t)_")]
            sigma_cols = [i for i, c in enumerate(columns) if c.startswith("x_2sigma(t)_")]
            if not true_cols or not mean_cols:
                continue
            y_true = data[:, true_cols]
            y_mean = data[:, mean_cols]
            if y_true.shape != y_mean.shape:
                continue
            err = y_mean - y_true
            all_err.append(err)
            all_abs.append(np.abs(err))

            if sigma_cols and len(sigma_cols) == y_true.shape[1]:
                sigma2 = data[:, sigma_cols]
                all_sigma2.append(sigma2)
                coverage_95_count += int(np.sum(np.abs(err) <= sigma2))
                coverage_95_total += int(err.size)

    if not all_err:
        metrics = {
            "forecast_quality": {},
            "uncertainty_quality": {},
            "robustness": {},
            "warning": "No usable forecast groups found.",
        }
    else:
        err_cat = np.concatenate(all_err, axis=0)
        abs_cat = np.concatenate(all_abs, axis=0)
        rmse = float(np.sqrt(np.mean(err_cat**2)))
        mae = float(np.mean(abs_cat))
        max_abs = float(np.max(abs_cat))
        iae = float(np.sum(abs_cat))
        metrics = {
            "forecast_quality": {
                "rmse": rmse,
                "mae": mae,
                "max_abs_error": max_abs,
                "integrated_abs_error": iae,
            },
            "uncertainty_quality": {
                "coverage_95_empirical": (
                    float(coverage_95_count / coverage_95_total) if coverage_95_total > 0 else None
                ),
                "mean_interval_width_95": (
                    float(np.mean(np.concatenate(all_sigma2, axis=0))) if all_sigma2 else None
                ),
            },
            "robustness": {
                "error_variance": float(np.var(err_cat)),
            },
        }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one single experiment harness execution.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON experiment config.")
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiments",
        help="Base directory for experiment artifacts.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if cfg.initial_unscaled_h5 is None:
        raise SystemExit("config.initial_unscaled_h5 is required.")
    if cfg.session_unscaled_h5 is None or len(cfg.session_unscaled_h5) < 2:
        raise SystemExit("config.session_unscaled_h5 must contain exactly 2 session dataset paths.")
    if len(cfg.session_unscaled_h5) != 2:
        raise SystemExit("This scripted scenario expects exactly 2 retraining sessions.")
    if cfg.N_r != 8 or cfg.N_k != 3 or cfg.N_b != 2:
        raise SystemExit("This scripted scenario is fixed to Nr=8, Nk=3, Nb=2 (216 profiles/session).")
    if cfg.retrain_cycles != 2:
        raise SystemExit("This scripted scenario requires retrain_cycles=2.")

    cfg_dir = _ensure_dir(args.base_output_dir / f"cfg_{cfg.N_r}_{cfg.N_k}_{cfg.N_b}")
    seed_dir = _ensure_dir(cfg_dir / f"seed_{cfg.seed}")
    strategy_dir = _ensure_dir(seed_dir / cfg.strategy)
    run_dir = _ensure_dir(strategy_dir / cfg.experiment_id)
    plots_dir = _ensure_dir(run_dir / "plots")

    run_meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "paths": {
            "cfg_dir": str(cfg_dir),
            "seed_dir": str(seed_dir),
            "strategy_dir": str(strategy_dir),
            "run_dir": str(run_dir),
        },
        "stages": {},
    }

    holdout_manifest = run_dir / "holdout_manifest.json"

    initial_scaled = ingest_command(
        input_unscaled_h5=Path(cfg.initial_unscaled_h5),
        output_dir=run_dir / "cycle_00" / "dataset",
        holdout_manifest=holdout_manifest,
        test_count=2000,
        save_manifest_if_missing=True,
    )
    with h5py.File(initial_scaled, "r") as h5f:
        train_profiles = len(h5f["train"]["files"].keys())
        val_profiles = len(h5f["val"]["files"].keys())
        test_profiles = len(h5f["test"]["files"].keys())
    if train_profiles != 500 or val_profiles != 500 or test_profiles != 2000:
        raise SystemExit(
            "Initial split counts do not match requested train/val/test = 500/500/2000. "
            f"Got train={train_profiles}, val={val_profiles}, test={test_profiles}."
        )

    all_cycle_metrics: list[dict[str, Any]] = []
    budget_profiles = 216
    for cycle in range(cfg.retrain_cycles):
        cycle_dir = _ensure_dir(run_dir / f"cycle_{cycle + 1:02d}")
        if cfg.strategy == "branching":
            profiles_dir, observed_budget = _prepare_branching_profiles(cfg, cycle_dir)
        else:
            profiles_dir, observed_budget = _prepare_random_profiles(cfg, cycle_dir)
        if observed_budget != budget_profiles:
            raise SystemExit(f"Expected {budget_profiles} profiles/session, observed {observed_budget}.")
        sim_out = _simulate(cfg, cfg.strategy, profiles_dir, cycle_dir)

        scaled_h5 = ingest_command(
            input_unscaled_h5=Path(cfg.session_unscaled_h5[cycle]),
            output_dir=cycle_dir / "dataset",
            holdout_manifest=holdout_manifest,
            test_count=2000,
            save_manifest_if_missing=False,
        )

        tuning_dir = _ensure_dir(cycle_dir / "tuning")
        tune_cfg = GridSearchConfig(
            lookback_datasets={cfg.lookback: scaled_h5},
            learning_rates=[1e-3, 5e-4, 1e-4],  # coarse 3-combination grid
            batch_sizes=[64],
            n_lstm_values=[1],
            hidden_lstm_values=[cfg.lstm_hidden],
            hidden_fc_values=[cfg.fc_hidden[0]],
            n_fc=cfg.n_fc,
            epochs=20,
            seed=cfg.seed,
            out_dir=tuning_dir,
            prefer_gpu=True,
        )
        _trial_results, best_trial = run_grid_search(tune_cfg)

        ensemble_dir = _ensure_dir(cycle_dir / "ensemble")
        ensemble = run_bagging_ensemble(
            scaled_h5,
            out_dir=ensemble_dir,
            n_models=3,
            seed=cfg.seed,
            n_lstm=best_trial.n_lstm,
            lstm_hidden=best_trial.hidden_lstm,
            n_fc=1,
            fc_hidden=(best_trial.hidden_fc,),
            learning_rate=best_trial.learning_rate,
            batch_size=best_trial.batch_size,
            epochs=20,
        )

        cycle_metrics = evaluate_command(
            forecast_h5_path=Path(ensemble["forecast_output_path"]),
            output_json_path=cycle_dir / "metrics.json",
        )
        cycle_metrics["cycle"] = cycle + 1
        cycle_metrics["budget_profiles"] = budget_profiles
        cycle_metrics["sim_profiles_dir"] = str(sim_out)
        all_cycle_metrics.append(cycle_metrics)

    metrics = {"cycles": all_cycle_metrics}
    metrics.setdefault("forecast_quality", {})
    metrics.setdefault("uncertainty_quality", {})
    metrics.setdefault("robustness", {})
    metrics["budget_profiles"] = int(budget_profiles)
    metrics["strategy"] = cfg.strategy
    metrics["seed"] = int(cfg.seed)
    metrics["params"] = {"N_r": cfg.N_r, "N_k": cfg.N_k, "N_b": cfg.N_b}
    metrics["artifacts"] = _plot_if_available(metrics, plots_dir)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    run_meta["paths"]["metrics_json"] = str(metrics_path)

    run_meta_path = run_dir / "run_metadata.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    summary_row = {
        "timestamp_utc": run_meta["timestamp_utc"],
        "experiment_id": cfg.experiment_id,
        "strategy": cfg.strategy,
        "retrain_policy": cfg.retrain_policy,
        "seed": cfg.seed,
        "N_r": cfg.N_r,
        "N_k": cfg.N_k,
        "N_b": cfg.N_b,
        "retrain_cycles": cfg.retrain_cycles,
        "target_spec": "configured_external",
        "budget_profiles": budget_profiles,
        "run_dir": str(run_dir),
    }
    _append_summary_row(args.base_output_dir / "summary.csv", summary_row)

    print("\nSingle experiment run complete.")
    print(f"Run dir: {run_dir}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
