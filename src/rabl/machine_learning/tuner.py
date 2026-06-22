"""Grid-search tuner for the LSTM training pipeline.

This module performs a Cartesian-product hyperparameter search over pre-built
LSTM datasets keyed by lookback window size.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from rabl.paths import resolve_output_root
from rabl.machine_learning.lstm_pipeline import (
    build_datasets,
    build_model,
    cleanup_cuda,
    TARGET_NAMES,
    test_and_save_forecasts,
    train_with_fallback,
    _load_scaling_stats,
)


@dataclass(slots=True)
class GridSearchConfig:
    lookback_datasets: dict[int, Path]
    learning_rates: list[float]
    batch_sizes: list[int]
    n_lstm_values: list[int]
    hidden_lstm_values: list[int]
    hidden_fc_values: list[int]
    n_fc_values: list[int]
    epochs: int = 20
    seed: int = 123
    out_dir: Path = resolve_output_root() / "ml_tuning"
    prefer_gpu: bool = True
    lstm_dropout: float = 0.0
    preload_train_to_device: bool = True
    preload_val_to_device: bool = True
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = True
    test_output_dirname: str = "best_model_test"
    step_lr_step_size: int = 30
    step_lr_gamma: float = 0.5
    use_tqdm: bool = True
    verbose: int = 1
    min_epochs: int | None = None
    max_epochs: int | None = None
    reduction_factor: int = 3
    prune_strategy: str = "successive_halving"

    def __post_init__(self) -> None:
        if not self.n_fc_values:
            raise ValueError("n_fc_values must contain at least one value.")
        if any(v < 1 for v in self.n_fc_values):
            raise ValueError("All n_fc values must be >= 1.")
        if self.min_epochs is None and self.max_epochs is None:
            return
        if self.min_epochs is None or self.max_epochs is None:
            raise ValueError("Both min_epochs and max_epochs must be provided for Hyperband tuning.")
        if self.min_epochs < 1:
            raise ValueError("min_epochs must be >= 1.")
        if self.max_epochs < self.min_epochs:
            raise ValueError("max_epochs must be >= min_epochs.")
        if self.reduction_factor <= 1:
            raise ValueError("reduction_factor must be > 1.")
        if self.prune_strategy != "successive_halving":
            raise ValueError("prune_strategy must be 'successive_halving'.")

    @property
    def use_hyperband(self) -> bool:
        return self.min_epochs is not None and self.max_epochs is not None


@dataclass(slots=True)
class TrialResult:
    lookback: int
    dataset_path: str
    learning_rate: float
    batch_size: int
    n_lstm: int
    n_fc: int
    hidden_lstm: int
    hidden_fc: int
    best_val_loss: float
    final_val_loss: float
    used_device: str
    trial_dir: str
    weights_path: str
    hyperband: dict[str, Any] | None = None


@dataclass(slots=True)
class HyperbandBracket:
    bracket_index: int
    initial_trial_count: int
    rung_count: int
    rung_budgets: list[int]


def _parse_lookback_entry(raw: str) -> tuple[int, Path]:
    """Parse one lookback mapping entry in the form ``LOOKBACK=DATASET_PATH``."""
    if "=" not in raw:
        raise ValueError(
            f"Invalid lookback mapping '{raw}'. Use LOOKBACK=DATASET_PATH (e.g. 8=data/lookback8.h5)."
        )
    lookback_str, path_str = raw.split("=", 1)
    lookback = int(lookback_str)
    dataset_path = Path(path_str).expanduser().resolve()
    return lookback, dataset_path


def _parse_lookback_mapping(values: list[str]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for raw in values:
        lookback, dataset_path = _parse_lookback_entry(raw)
        if lookback in mapping:
            raise ValueError(f"Duplicate lookback entry for {lookback}.")
        mapping[lookback] = dataset_path
    return mapping


def count_grid_combinations(config: GridSearchConfig) -> int:
    """Return the total number of grid-search trials."""
    return (
        len(config.lookback_datasets)
        * len(config.learning_rates)
        * len(config.batch_sizes)
        * len(config.n_lstm_values)
        * len(config.hidden_lstm_values)
        * len(config.n_fc_values)
        * len(config.hidden_fc_values)
    )


def construct_hyperband_brackets(config: GridSearchConfig) -> list[HyperbandBracket]:
    """Build full Hyperband bracket definitions for configured budgets."""
    if not config.use_hyperband:
        return []

    assert config.min_epochs is not None
    assert config.max_epochs is not None
    eta = config.reduction_factor
    s_max = int(math.floor(math.log(config.max_epochs / config.min_epochs, eta)))
    if s_max < 0:
        raise ValueError("Hyperband requires max_epochs >= min_epochs.")
    total_budget = (s_max + 1) * config.max_epochs
    brackets: list[HyperbandBracket] = []

    for s in range(s_max, -1, -1):
        n = int(math.ceil((total_budget / config.max_epochs) * (eta**s) / (s + 1)))
        r = config.max_epochs * (eta ** (-s))
        rung_budgets = [min(config.max_epochs, int(math.ceil(r * (eta**i)))) for i in range(s + 1)]
        rung_budgets = sorted(set(rung_budgets))
        if not rung_budgets:
            continue
        brackets.append(
            HyperbandBracket(
                bracket_index=s,
                initial_trial_count=max(1, n),
                rung_count=len(rung_budgets),
                rung_budgets=rung_budgets,
            )
        )
    return brackets


def hyperband_initial_trial_count(min_epochs: int, max_epochs: int, reduction_factor: int) -> int:
    """Return total Hyperband initial trials (N_init) implied by (min,max,eta)."""
    if min_epochs < 1:
        raise ValueError("min_epochs must be >= 1.")
    if max_epochs < min_epochs:
        raise ValueError("max_epochs must be >= min_epochs.")
    if reduction_factor <= 1:
        raise ValueError("reduction_factor must be > 1.")
    s_max = int(math.floor(math.log(max_epochs / min_epochs, reduction_factor)))
    if s_max < 0:
        return 0
    total_budget = (s_max + 1) * max_epochs
    total_initial = 0
    for s in range(s_max, -1, -1):
        n = int(math.ceil((total_budget / max_epochs) * (reduction_factor**s) / (s + 1)))
        total_initial += max(1, n)
    return total_initial


def _all_trial_params(config: GridSearchConfig) -> list[dict[str, Any]]:
    trial_grid = product(
        sorted(config.lookback_datasets.items(), key=lambda item: item[0]),
        config.learning_rates,
        config.batch_sizes,
        config.n_lstm_values,
        config.hidden_lstm_values,
        config.n_fc_values,
        config.hidden_fc_values,
    )
    params: list[dict[str, Any]] = []
    for (lookback, dataset_path), learning_rate, batch_size, n_lstm, hidden_lstm, n_fc, hidden_fc in trial_grid:
        params.append(
            {
                "lookback": lookback,
                "dataset_path": dataset_path,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_lstm": n_lstm,
                "n_fc": n_fc,
                "hidden_lstm": hidden_lstm,
                "hidden_fc": hidden_fc,
            }
        )
    return params


def _stage_train_trial(
    config: GridSearchConfig,
    params: dict[str, Any],
    trial_dir: Path,
    epochs_to_run: int,
    resume_state_path: Path | None,
) -> tuple[dict[str, Any], Path, Path]:
    stage_t0 = perf_counter()
    datasets = build_datasets(
        h5_path=params["dataset_path"],
        batch_size=params["batch_size"],
        seed=config.seed,
    )
    model, history, used_device = train_with_fallback(
        datasets,
        epochs=epochs_to_run,
        out_dir=trial_dir,
        n_lstm=params["n_lstm"],
        lstm_hidden=params["hidden_lstm"],
        lstm_dropout=config.lstm_dropout,
        n_fc=params["n_fc"],
        fc_hidden=tuple([params["hidden_fc"]] * int(params["n_fc"])),
        learning_rate=params["learning_rate"],
        step_lr_step_size=config.step_lr_step_size,
        step_lr_gamma=config.step_lr_gamma,
        verbose=config.verbose,
        prefer_gpu=config.prefer_gpu,
        preload_train_to_device=config.preload_train_to_device,
        preload_val_to_device=config.preload_val_to_device,
        deterministic_seed=config.seed,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        restore_best_weights=config.restore_best_weights,
        use_tqdm=config.use_tqdm,
        resume_from_weights=resume_state_path,
    )
    checkpoint_path = trial_dir / f"checkpoint_stage_{epochs_to_run}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    weights_path = trial_dir / "best_model_weights.pt"
    torch.save(model.state_dict(), weights_path)
    best_val_loss = float(min(history["val_loss"]))
    actual_epochs = len(history["val_loss"])
    cleanup_cuda(model, used_device)
    del model
    stage_duration_s = perf_counter() - stage_t0
    return (
        {
            "best_val_loss": best_val_loss,
            "final_val_loss": float(history["val_loss"][-1]),
            "actual_epochs_trained": int(actual_epochs),
            "used_device": str(used_device),
            "early_stopped": bool(actual_epochs < epochs_to_run),
            "duration_s": float(stage_duration_s),
        },
        checkpoint_path,
        weights_path,
    )


def _save_timing_summary(
    out_dir: Path,
    *,
    run_type: str,
    total_duration_s: float,
    trial_timings: list[dict[str, Any]],
) -> Path:
    payload = {
        "run_type": run_type,
        "total_duration_s": float(total_duration_s),
        "num_timed_trials": len(trial_timings),
        "trial_timings": trial_timings,
    }
    out_path = out_dir / "tuning_timing.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _compute_forecast_quality_metrics(forecast_h5: Path, *, scaled_h5: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    scaling_stats = _load_scaling_stats(scaled_h5)

    y_true_profiles: list[np.ndarray] = []
    y_pred_profiles: list[np.ndarray] = []
    target_names: list[str] | None = None
    with h5py.File(forecast_h5, "r") as h5f:
        for profile_name in h5f.keys():
            grp = h5f[profile_name]
            table = np.asarray(grp["data"][()], dtype=float)
            cols_raw = grp.attrs.get("columns", [])
            cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in cols_raw]
            ensemble_true = [idx for idx, c in enumerate(cols) if c.startswith("x_true(t)_")]
            ensemble_pred = [idx for idx, c in enumerate(cols) if c.startswith("x_mean(t)_")]
            single_true = [idx for idx, c in enumerate(cols) if c.startswith("x(t)_")]
            single_pred = [idx for idx, c in enumerate(cols) if c.startswith("x^~(t)_")]

            true_cols: list[int]
            pred_cols: list[int]
            target_prefix: str
            if ensemble_true and len(ensemble_true) == len(ensemble_pred):
                true_cols = ensemble_true
                pred_cols = ensemble_pred
                target_prefix = "x_true(t)_"
            elif single_true and len(single_true) == len(single_pred):
                true_cols = single_true
                pred_cols = single_pred
                target_prefix = "x(t)_"
            else:
                continue
            profile_target_names = [cols[idx].replace(target_prefix, "", 1) for idx in true_cols]
            if target_names is None:
                target_names = profile_target_names
            y_true_profiles.append(table[:, true_cols])
            y_pred_profiles.append(table[:, pred_cols])

    if not y_true_profiles or target_names is None:
        return {"per_target": {}, "aggregated": {"scaled_mae": float("nan"), "scaled_rmse": float("nan")}}

    y_true_cat = np.concatenate(y_true_profiles, axis=0)
    y_pred_cat = np.concatenate(y_pred_profiles, axis=0)
    err_cat = y_pred_cat - y_true_cat
    abs_err_cat = np.abs(err_cat)

    rmse_by_target = np.sqrt(np.mean(err_cat**2, axis=0))
    mae_by_target = np.mean(abs_err_cat, axis=0)
    target_indices = [TARGET_NAMES.index(name) for name in target_names]
    y_stats = scaling_stats["y"]
    if scaling_stats["type"] == "standard":
        target_scale = np.asarray(y_stats["std"], dtype=float)[target_indices]
    elif scaling_stats["type"] == "minmax":
        target_scale = np.asarray(y_stats["span"], dtype=float)[target_indices]
    else:
        raise ValueError(f"Unsupported scaling type: {scaling_stats['type']}")
    if target_scale.shape[-1] != len(target_names):
        raise ValueError(f"Target scale length mismatch: expected {len(target_names)}, got shape {target_scale.shape}.")
    if not np.all(np.isfinite(target_scale)) or not np.all(target_scale > 0.0):
        raise ValueError(f"Target scales must all be finite and positive. Got: {target_scale!r}")
    err_scaled = err_cat / target_scale
    scaled_mae_by_target = np.nanmean(np.abs(err_scaled), axis=0)
    scaled_rmse_by_target = np.sqrt(np.nanmean(err_scaled**2, axis=0))
    scaled_mae = float(np.nanmean(scaled_mae_by_target))
    scaled_rmse = float(np.nanmean(scaled_rmse_by_target))
    if not np.isclose(scaled_mae, float(np.nanmean(scaled_mae_by_target)), rtol=1e-12, atol=1e-12):
        raise AssertionError("Aggregate scaled_mae does not equal the mean of per-target scaled MAE values.")
    if not np.isclose(scaled_rmse, float(np.nanmean(scaled_rmse_by_target)), rtol=1e-12, atol=1e-12):
        raise AssertionError("Aggregate scaled_rmse does not equal the mean of per-target scaled RMSE values.")

    per_target: dict[str, dict[str, float]] = {}
    for idx, target_name in enumerate(target_names):
        per_target[target_name] = {
            "rmse": float(rmse_by_target[idx]),
            "mae": float(mae_by_target[idx]),
            "scaled_mae": float(scaled_mae_by_target[idx]),
            "scaled_rmse": float(scaled_rmse_by_target[idx]),
        }

    return {
        "per_target": per_target,
        "aggregated": {"scaled_mae": scaled_mae, "scaled_rmse": scaled_rmse},
    }


def _print_tuning_overview(config: GridSearchConfig, best_result: TrialResult, test_metrics: dict[str, float]) -> None:
    summary_path = config.out_dir / "grid_search_results.json"
    timing_path = config.out_dir / "tuning_timing.json"
    if not summary_path.exists():
        print(f"WARNING: could not find tuning summary at {summary_path}; skipping overview.")
        return
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary_payload.get("results", [])
    run_type = "hyperband" if "hyperband" in summary_payload else "grid"
    timing_payload: dict[str, Any] = {}
    if timing_path.exists():
        timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print("TUNING OVERVIEW (STORY MODE)")
    print("=" * 100)
    print(f"1) We started a {run_type} search over {count_grid_combinations(config)} candidate combinations.")
    print(f"2) The run completed with {len(results)} successful trained trials recorded in the summary.")
    if timing_payload:
        print(
            "3) Timing snapshot: "
            f"total_duration_s={float(timing_payload.get('total_duration_s', float('nan'))):.2f}, "
            f"timed_entries={int(timing_payload.get('num_timed_trials', 0))}."
        )
    else:
        print("3) Timing snapshot: no timing summary file found.")

    if run_type == "hyperband":
        print("4) Hyperband progression details:")
        for bracket in summary_payload.get("hyperband", {}).get("brackets", []):
            bracket_index = int(bracket.get("bracket_index", -1))
            initial_trials = int(bracket.get("initial_trial_count", 0))
            budgets = bracket.get("rung_budgets", [])
            print(
                f"   - Bracket s={bracket_index} began with {initial_trials} sampled trials "
                f"and budgets {budgets}."
            )
            trials = bracket.get("trials", [])
            rung_indices = sorted(
                {
                    int(metric.get("rung_index", -1))
                    for trial in trials
                    for metric in trial.get("rung_metrics", [])
                    if metric.get("rung_index") is not None
                }
            )
            for rung_idx in rung_indices:
                advanced: list[str] = []
                removed: list[str] = []
                failed: list[str] = []
                for trial in trials:
                    trial_idx = int(trial.get("trial_index", -1))
                    hp = trial.get("sampled_hyperparameters", {})
                    trial_name = (
                        f"trial_{trial_idx:04d}(lr={hp.get('learning_rate')},bs={hp.get('batch_size')},"
                        f"nl={hp.get('n_lstm')},nfc={hp.get('n_fc')},hl={hp.get('hidden_lstm')},hf={hp.get('hidden_fc')})"
                    )
                    rung_metrics = trial.get("rung_metrics", [])
                    match = next((m for m in rung_metrics if int(m.get("rung_index", -1)) == rung_idx), None)
                    if match is None:
                        continue
                    decision = str(match.get("decision", "unknown"))
                    if decision == "promoted":
                        advanced.append(trial_name)
                    elif decision == "failed":
                        failed.append(trial_name)
                    else:
                        removed.append(f"{trial_name}:{decision}")
                print(
                    f"     • Rung {rung_idx}: advanced={len(advanced)}, removed={len(removed)}, failed={len(failed)}"
                )
                if advanced:
                    print("       advanced -> " + ", ".join(advanced))
                if removed:
                    print("       removed  -> " + ", ".join(removed))
                if failed:
                    print("       failed   -> " + ", ".join(failed))
    else:
        print("4) Grid progression details:")
        for idx, result in enumerate(results, start=1):
            print(
                f"   - Trial {idx:04d}: "
                f"lb={result.get('lookback')}, lr={result.get('learning_rate')}, bs={result.get('batch_size')}, "
                f"nl={result.get('n_lstm')}, nfc={result.get('n_fc')}, hl={result.get('hidden_lstm')}, hf={result.get('hidden_fc')} "
                f"-> best_val_loss={result.get('best_val_loss')}"
            )

    print(
        "5) The winning configuration was: "
        f"lookback={best_result.lookback}, lr={best_result.learning_rate:g}, "
        f"batch={best_result.batch_size}, n_lstm={best_result.n_lstm}, n_fc={best_result.n_fc}, "
        f"hidden_lstm={best_result.hidden_lstm}, hidden_fc={best_result.hidden_fc}."
    )
    print(f"6) Its best validation loss was {best_result.best_val_loss:.6e}.")
    print(f"7) The selected model artifacts live in: {best_result.trial_dir}")
    print(
        "8) Final best-model test timing metrics: "
        + ", ".join(f"{k}={float(v):.6f}" for k, v in sorted(test_metrics.items()))
    )
    print("=" * 100)


def run_grid_search(config: GridSearchConfig) -> tuple[list[TrialResult], TrialResult]:
    """Run all tuning trials and return all trial metrics plus the best trial."""
    total_trials = count_grid_combinations(config)
    if total_trials <= 0:
        raise ValueError("Grid search has zero combinations. Please provide non-empty parameter lists.")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Total combinations: {total_trials}")
    print("\n" + "=" * 100)
    print("STARTING GRID SEARCH TUNING RUN")
    print("=" * 100)

    trial_grid = product(
        sorted(config.lookback_datasets.items(), key=lambda item: item[0]),
        config.learning_rates,
        config.batch_sizes,
        config.n_lstm_values,
        config.hidden_lstm_values,
        config.n_fc_values,
        config.hidden_fc_values,
    )

    results: list[TrialResult] = []
    trial_timings: list[dict[str, Any]] = []
    run_t0 = perf_counter()

    for trial_index, (
        (lookback, dataset_path),
        learning_rate,
        batch_size,
        n_lstm,
        hidden_lstm,
        n_fc,
        hidden_fc,
    ) in enumerate(trial_grid, start=1):
        print("\n" + "#" * 100)
        print(f"GRID TRIAL {trial_index}/{total_trials}")
        print("#" * 100)
        trial_t0 = perf_counter()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found for lookback={lookback}: {dataset_path}")

        trial_name = (
            f"trial_{trial_index:04d}"
            f"_lb{lookback}"
            f"_lr{learning_rate:g}"
            f"_bs{batch_size}"
            f"_nl{n_lstm}"
            f"_nfc{n_fc}"
            f"_hl{hidden_lstm}"
            f"_hf{hidden_fc}"
        )
        trial_dir = config.out_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{trial_index}/{total_trials}] lookback={lookback}, lr={learning_rate:g}, "
            f"batch_size={batch_size}, n_lstm={n_lstm}, n_fc={n_fc}, hidden_lstm={hidden_lstm}, hidden_fc={hidden_fc}"
        )

        datasets = build_datasets(h5_path=dataset_path, batch_size=batch_size, seed=config.seed)
        model, history, used_device = train_with_fallback(
            datasets,
            epochs=config.epochs,
            out_dir=trial_dir,
            n_lstm=n_lstm,
            lstm_hidden=hidden_lstm,
            lstm_dropout=config.lstm_dropout,
            n_fc=n_fc,
            fc_hidden=tuple([hidden_fc] * int(n_fc)),
            learning_rate=learning_rate,
            step_lr_step_size=config.step_lr_step_size,
            step_lr_gamma=config.step_lr_gamma,
            verbose=config.verbose,
            prefer_gpu=config.prefer_gpu,
            preload_train_to_device=config.preload_train_to_device,
            preload_val_to_device=config.preload_val_to_device,
            deterministic_seed=config.seed,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            restore_best_weights=config.restore_best_weights,
            use_tqdm=config.use_tqdm,
        )

        weights_path = trial_dir / "best_model_weights.pt"
        torch.save(model.state_dict(), weights_path)

        best_val_loss = min(history["val_loss"])
        final_val_loss = history["val_loss"][-1]
        result = TrialResult(
            lookback=lookback,
            dataset_path=str(dataset_path),
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_lstm=n_lstm,
            n_fc=n_fc,
            hidden_lstm=hidden_lstm,
            hidden_fc=hidden_fc,
            best_val_loss=float(best_val_loss),
            final_val_loss=float(final_val_loss),
            used_device=str(used_device),
            trial_dir=str(trial_dir),
            weights_path=str(weights_path),
        )
        results.append(result)

        cleanup_cuda(model, used_device)
        del model
        trial_duration_s = perf_counter() - trial_t0
        trial_timings.append(
            {
                "trial_index": trial_index,
                "trial_dir": str(trial_dir),
                "duration_s": float(trial_duration_s),
                "lookback": lookback,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_lstm": n_lstm,
                "hidden_lstm": hidden_lstm,
                "hidden_fc": hidden_fc,
            }
        )
        print(f"GRID TRIAL DURATION: {trial_duration_s:.2f} sec")

    best_result = min(results, key=lambda item: item.best_val_loss)

    summary_payload = {
        "config": {
            **asdict(config),
            "lookback_datasets": {k: str(v) for k, v in config.lookback_datasets.items()},
            "out_dir": str(config.out_dir),
        },
        "num_trials": len(results),
        "best_trial": asdict(best_result),
        "results": [asdict(item) for item in results],
    }
    with (config.out_dir / "grid_search_results.json").open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    total_duration_s = perf_counter() - run_t0
    timing_path = _save_timing_summary(
        config.out_dir,
        run_type="grid",
        total_duration_s=total_duration_s,
        trial_timings=trial_timings,
    )

    print("\nBest trial:")
    print(json.dumps(asdict(best_result), indent=2))
    print(f"Saved tuning summary to: {config.out_dir / 'grid_search_results.json'}")
    print(f"Saved timing summary to: {timing_path}")
    print(f"TOTAL GRID SEARCH DURATION: {total_duration_s:.2f} sec")

    return results, best_result


def run_hyperband_search(config: GridSearchConfig) -> tuple[list[TrialResult], TrialResult]:
    """Run full Hyperband schedule and return trial metrics plus best trial."""
    if not config.use_hyperband:
        raise ValueError("Hyperband search requires min_epochs/max_epochs configuration.")
    if config.prune_strategy != "successive_halving":
        raise ValueError("Only successive_halving prune_strategy is supported.")

    brackets = construct_hyperband_brackets(config)
    if not brackets:
        raise ValueError("No Hyperband brackets were constructed from the provided config.")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    all_params = _all_trial_params(config)
    if not all_params:
        raise ValueError("Hyperparameter space is empty.")
    rng = random.Random(config.seed)
    shuffled_params = list(all_params)
    rng.shuffle(shuffled_params)
    param_cursor = 0

    required_initial_trials = sum(bracket.initial_trial_count for bracket in brackets)
    if required_initial_trials > len(shuffled_params):
        raise ValueError(
            "Hyperband without-replacement sampling requires at least as many unique combinations as "
            f"initial bracket samples. required={required_initial_trials}, available={len(shuffled_params)}."
        )

    def _next_param_sample() -> dict[str, Any]:
        nonlocal param_cursor
        if param_cursor >= len(shuffled_params):
            raise RuntimeError("No remaining hyperparameter combinations for without-replacement sampling.")
        sampled = dict(shuffled_params[param_cursor])
        param_cursor += 1
        return sampled

    run_t0 = perf_counter()
    trial_timings: list[dict[str, Any]] = []
    print("\n" + "=" * 100)
    print("STARTING HYPERBAND TUNING RUN")
    print("=" * 100)
    print(f"Hyperband brackets: {len(brackets)}")

    results: list[TrialResult] = []
    hyperband_brackets_meta: list[dict[str, Any]] = []
    trial_counter = 0

    for bracket in brackets:
        bracket_t0 = perf_counter()
        print("\n" + "*" * 100)
        print(
            f"HYPERBAND BRACKET s={bracket.bracket_index} | "
            f"initial_trials={bracket.initial_trial_count} | "
            f"rungs={bracket.rung_count} | budgets={bracket.rung_budgets}"
        )
        print("*" * 100)
        bracket_trials: list[dict[str, Any]] = []
        for _ in range(bracket.initial_trial_count):
            sampled = _next_param_sample()
            trial_counter += 1
            trial_dir = config.out_dir / f"hb_b{bracket.bracket_index}_trial_{trial_counter:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            bracket_trials.append(
                {
                    "trial_index": trial_counter,
                    "params": sampled,
                    "trial_dir": trial_dir,
                    "checkpoint_path": None,
                    "weights_path": None,
                    "rungs": [],
                    "status": "pending",
                    "used_device": "",
                    "best_val_loss": float("inf"),
                    "final_val_loss": float("inf"),
                    "cumulative_epochs": 0,
                }
            )

        active_trials = bracket_trials
        rung_meta: list[dict[str, Any]] = []
        for rung_idx, allocated_epochs in enumerate(bracket.rung_budgets):
            print("\n" + "-" * 100)
            print(
                f"BRACKET s={bracket.bracket_index} RUNG {rung_idx + 1}/{bracket.rung_count} "
                f"target_cumulative_epochs={allocated_epochs}"
            )
            print("-" * 100)
            rung_records: list[dict[str, Any]] = []
            completed_trials: list[dict[str, Any]] = []
            for trial in active_trials:
                params = trial["params"]
                resume_path = trial["checkpoint_path"]
                try:
                    epochs_to_run = max(0, allocated_epochs - int(trial["cumulative_epochs"]))
                    if epochs_to_run <= 0:
                        raise ValueError(
                            f"Invalid cumulative rung budget ordering for trial {trial['trial_index']}."
                        )
                    metrics, checkpoint_path, weights_path = _stage_train_trial(
                        config=config,
                        params=params,
                        trial_dir=trial["trial_dir"],
                        epochs_to_run=epochs_to_run,
                        resume_state_path=resume_path,
                    )
                    trial["checkpoint_path"] = checkpoint_path
                    trial["weights_path"] = weights_path
                    trial["status"] = "completed"
                    trial["best_val_loss"] = min(trial["best_val_loss"], metrics["best_val_loss"])
                    trial["final_val_loss"] = metrics["final_val_loss"]
                    trial["used_device"] = metrics["used_device"]
                    trial["cumulative_epochs"] = int(trial["cumulative_epochs"]) + int(metrics["actual_epochs_trained"])
                    rung_entry = {
                        "rung_index": rung_idx,
                        "allocated_epochs": allocated_epochs,
                        "actual_epochs_trained": trial["cumulative_epochs"],
                        "stage_epochs_trained": metrics["actual_epochs_trained"],
                        "best_val_loss": metrics["best_val_loss"],
                        "final_val_loss": metrics["final_val_loss"],
                        "checkpoint_path": str(checkpoint_path),
                        "resumed_from_checkpoint": bool(resume_path is not None),
                        "resume_checkpoint_path": str(resume_path) if resume_path is not None else None,
                        "early_stopped": metrics["early_stopped"],
                        "decision": "completed",
                        "duration_s": metrics["duration_s"],
                    }
                    trial["rungs"].append(rung_entry)
                    completed_trials.append(trial)
                    rung_records.append({"trial_index": trial["trial_index"], **rung_entry})
                    trial_timings.append(
                        {
                            "run_type": "hyperband",
                            "bracket_index": bracket.bracket_index,
                            "rung_index": rung_idx,
                            "trial_index": int(trial["trial_index"]),
                            "trial_dir": str(trial["trial_dir"]),
                            "duration_s": float(metrics["duration_s"]),
                            "allocated_epochs": allocated_epochs,
                            "actual_epochs_trained": int(trial["cumulative_epochs"]),
                        }
                    )
                except Exception as exc:
                    rung_entry = {
                        "rung_index": rung_idx,
                        "allocated_epochs": allocated_epochs,
                        "actual_epochs_trained": 0,
                        "stage_epochs_trained": 0,
                        "best_val_loss": None,
                        "final_val_loss": None,
                        "checkpoint_path": None,
                        "resumed_from_checkpoint": bool(resume_path is not None),
                        "resume_checkpoint_path": str(resume_path) if resume_path is not None else None,
                        "early_stopped": False,
                        "decision": "failed",
                        "failure_reason": repr(exc),
                    }
                    trial["rungs"].append(rung_entry)
                    trial["status"] = "failed"
                    rung_records.append({"trial_index": trial["trial_index"], **rung_entry})

            completed_trials.sort(key=lambda item: item["best_val_loss"])
            promotion_count = 0
            promoted_trial_ids: set[int] = set()
            if rung_idx < bracket.rung_count - 1 and completed_trials:
                promotion_count = max(1, len(completed_trials) // config.reduction_factor)
                promoted = completed_trials[:promotion_count]
                promoted_trial_ids = {int(item["trial_index"]) for item in promoted}
            for trial in completed_trials:
                decision = "completed" if rung_idx == bracket.rung_count - 1 else "pruned"
                if int(trial["trial_index"]) in promoted_trial_ids:
                    decision = "promoted"
                elif bool(trial["rungs"][-1].get("early_stopped")):
                    decision = "early_stopped"
                trial["rungs"][-1]["decision"] = decision

            rung_meta.append(
                {
                    "rung_index": rung_idx,
                    "allocated_epochs": allocated_epochs,
                    "promotion_count": promotion_count,
                    "trial_records": rung_records,
                }
            )
            active_trials = [trial for trial in completed_trials if int(trial["trial_index"]) in promoted_trial_ids]
            print(
                f"Completed rung with {len(completed_trials)} successful trials; "
                f"promoted={len(promoted_trial_ids)}"
            )

        hyperband_brackets_meta.append(
            {
                "bracket_index": bracket.bracket_index,
                "initial_trial_count": bracket.initial_trial_count,
                "rung_count": bracket.rung_count,
                "rung_budgets": bracket.rung_budgets,
                "rungs": rung_meta,
                "trials": [
                    {
                        "trial_index": trial["trial_index"],
                        "sampled_hyperparameters": {
                            **trial["params"],
                            "dataset_path": str(trial["params"]["dataset_path"]),
                        },
                        "trial_dir": str(trial["trial_dir"]),
                        "status": trial["status"],
                        "checkpoint_resume_status": "resumed" if any(
                            rung.get("resumed_from_checkpoint") for rung in trial["rungs"]
                        ) else "fresh",
                        "rung_metrics": trial["rungs"],
                    }
                    for trial in bracket_trials
                ],
            }
        )
        bracket_duration_s = perf_counter() - bracket_t0
        print(f"BRACKET s={bracket.bracket_index} DURATION: {bracket_duration_s:.2f} sec")

        for trial in bracket_trials:
            if trial["status"] == "failed":
                continue
            params = trial["params"]
            weights_path = Path(trial["weights_path"]) if trial["weights_path"] is not None else trial["trial_dir"] / "best_model_weights.pt"
            result = TrialResult(
                lookback=params["lookback"],
                dataset_path=str(params["dataset_path"]),
                learning_rate=float(params["learning_rate"]),
                batch_size=int(params["batch_size"]),
                n_lstm=int(params["n_lstm"]),
                n_fc=int(params["n_fc"]),
                hidden_lstm=int(params["hidden_lstm"]),
                hidden_fc=int(params["hidden_fc"]),
                best_val_loss=float(trial["best_val_loss"]),
                final_val_loss=float(trial["final_val_loss"]),
                used_device=str(trial["used_device"]),
                trial_dir=str(trial["trial_dir"]),
                weights_path=str(weights_path),
                hyperband={
                    "bracket_index": bracket.bracket_index,
                    "rung_metrics": trial["rungs"],
                },
            )
            results.append(result)

    if not results:
        raise RuntimeError("Hyperband search finished without successful trials.")
    best_result = min(results, key=lambda item: item.best_val_loss)

    summary_payload = {
        "config": {
            **asdict(config),
            "lookback_datasets": {k: str(v) for k, v in config.lookback_datasets.items()},
            "out_dir": str(config.out_dir),
        },
        "num_trials": len(results),
        "best_trial": asdict(best_result),
        "results": [asdict(item) for item in results],
        "hyperband": {
            "config": {
                "min_epochs": config.min_epochs,
                "max_epochs": config.max_epochs,
                "reduction_factor": config.reduction_factor,
                "prune_strategy": config.prune_strategy,
            },
            "brackets": hyperband_brackets_meta,
            "best_trial": asdict(best_result),
        },
    }
    with (config.out_dir / "grid_search_results.json").open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)
    total_duration_s = perf_counter() - run_t0
    timing_path = _save_timing_summary(
        config.out_dir,
        run_type="hyperband",
        total_duration_s=total_duration_s,
        trial_timings=trial_timings,
    )

    print("\nBest trial:")
    print(json.dumps(asdict(best_result), indent=2))
    print(f"Saved tuning summary to: {config.out_dir / 'grid_search_results.json'}")
    print(f"Saved timing summary to: {timing_path}")
    print(f"TOTAL HYPERBAND DURATION: {total_duration_s:.2f} sec")
    return results, best_result


def test_best_model(config: GridSearchConfig, best_result: TrialResult) -> dict[str, float]:
    """Load the best model weights and run test-time rolling forecasts."""
    dataset_path = Path(best_result.dataset_path)
    weights_path = Path(best_result.weights_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Best-trial dataset path not found: {dataset_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Best-model weights path not found: {weights_path}")

    datasets = build_datasets(
        h5_path=dataset_path,
        batch_size=best_result.batch_size,
        seed=config.seed,
    )
    timesteps = int(datasets["sample_shape"][1])
    num_features = int(datasets["sample_shape"][2])
    num_targets = int(datasets["target_shape"][1])


    model = build_model(
        timesteps=timesteps,
        num_features=num_features,
        num_targets=num_targets,
        n_lstm=best_result.n_lstm,
        lstm_hidden=best_result.hidden_lstm,
        lstm_dropout=config.lstm_dropout,
        n_fc=best_result.n_fc,
        fc_hidden=tuple([best_result.hidden_fc] * int(best_result.n_fc)),
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    test_out_dir = Path(best_result.trial_dir) / config.test_output_dirname
    test_metrics = test_and_save_forecasts(
        model,
        datasets["test_profile_ds"],
        out_dir=test_out_dir,
        h5_path=datasets["h5_path"],
        use_tqdm=config.use_tqdm,
    )
    forecast_h5 = test_out_dir / "rolling_forecasts.h5"
    if forecast_h5.exists():
        forecast_quality = _compute_forecast_quality_metrics(forecast_h5, scaled_h5=datasets["h5_path"])
        per_target = forecast_quality.get("per_target", {})
        target_rmse_values = [float(per_target[name]["rmse"]) for name in per_target]
        target_mae_values = [float(per_target[name]["mae"]) for name in per_target]
        if target_rmse_values:
            mean_target_rmse = sum(target_rmse_values) / len(target_rmse_values)
            mean_target_mae = sum(target_mae_values) / len(target_mae_values)
        else:
            mean_target_rmse = float("nan")
            mean_target_mae = float("nan")
        print(
            "BEST MODEL TEST SUMMARY: "
            f"mean_target_RMSE={mean_target_rmse:.6f}, mean_target_MAE={mean_target_mae:.6f}, "
            f"scaled_MAE={forecast_quality['aggregated']['scaled_mae']:.6f}, "
            f"scaled_RMSE={forecast_quality['aggregated']['scaled_rmse']:.6f}"
        )
        metrics_payload = {
            "forecast_quality": forecast_quality,
            "timing": test_metrics,
        }
        metrics_json = test_out_dir / "best_model_metrics_summary.json"
        metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        print(f"Saved best-model metrics summary to: {metrics_json}")
    else:
        print(f"WARNING: rolling_forecasts.h5 not found at {forecast_h5}; skipping metrics summary.")
    print(f"Saved best-model test outputs to: {test_out_dir}")
    return test_metrics


def _load_trial_result_from_summary(config: GridSearchConfig, trial_dir: Path) -> tuple[int, TrialResult]:
    summary_path = config.out_dir / "grid_search_results.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Grid-search summary not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    trial_dir_resolved = str(trial_dir.resolve())
    for trial_idx, raw_result in enumerate(payload.get("results", []), start=1):
        raw_trial_dir = str(raw_result.get("trial_dir", ""))
        if str(Path(raw_trial_dir).resolve()) == trial_dir_resolved:
            return trial_idx, TrialResult(**raw_result)

    raise ValueError(
        f"Trial folder '{trial_dir}' was not found in summary results at {summary_path}."
    )


def test_model_from_trial_dir(config: GridSearchConfig, trial_dir: Path) -> dict[str, float]:
    """Load a specific trial folder model and run test-time rolling forecasts."""
    resolved_trial_dir = Path(trial_dir).expanduser().resolve()
    trial_number, trial_result = _load_trial_result_from_summary(config, resolved_trial_dir)
    print(
        f"Testing trial #{trial_number}: {resolved_trial_dir} "
        f"(lookback={trial_result.lookback})"
    )
    return test_best_model(config, trial_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search tuner for rabl.machine_learning.lstm_pipeline")
    parser.add_argument(
        "--lookback-dataset",
        nargs="+",
        required=True,
        metavar="LOOKBACK=H5_PATH",
        help=(
            "Lookback-to-dataset mapping entries. "
            "Example: --lookback-dataset 4=data/lb4.h5 8=data/lb8.h5 12=data/lb12.h5"
        ),
    )
    parser.add_argument("--learning-rates", type=float, nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--n-lstm", type=int, nargs="+", required=True, dest="n_lstm_values")
    parser.add_argument("--hidden-lstm", type=int, nargs="+", required=True, dest="hidden_lstm_values")
    parser.add_argument("--hidden-fc", type=int, nargs="+", required=True, dest="hidden_fc_values")
    parser.add_argument(
        "--n-fc",
        type=int,
        nargs="+",
        default=[1],
        dest="n_fc_values",
        help="One or more FC-layer counts for the model head.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help="Enable Hyperband with this minimum rung budget (cumulative epochs).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Enable Hyperband with this maximum rung budget (cumulative epochs).",
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=3,
        help="Hyperband downsampling factor eta (>1).",
    )
    parser.add_argument(
        "--prune-strategy",
        type=str,
        default="successive_halving",
        help="Rung pruning strategy (currently only successive_halving).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=resolve_output_root() / "ml_tuning")
    parser.add_argument(
        "--lstm-dropout",
        type=float,
        default=0.0,
        help="Dropout between stacked LSTM layers (ignored when n_lstm=1).",
    )
    parser.add_argument(
        "--step-lr-step-size",
        type=int,
        default=30,
        help="StepLR step size passed to train_with_fallback/train_model.",
    )
    parser.add_argument(
        "--step-lr-gamma",
        type=float,
        default=0.5,
        help="StepLR gamma passed to train_with_fallback/train_model.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars for training and evaluation.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity passed through to training and evaluation routines.",
    )
    parser.add_argument(
        "--preload-train-to-device",
        dest="preload_train_to_device",
        action="store_true",
        default=True,
        help="Preload training batches to the selected training device (default: enabled).",
    )
    parser.add_argument(
        "--no-preload-train-to-device",
        dest="preload_train_to_device",
        action="store_false",
        help="Disable preloading training batches to device.",
    )
    parser.add_argument(
        "--preload-val-to-device",
        dest="preload_val_to_device",
        action="store_true",
        default=True,
        help="Preload validation batches to the selected training device (default: enabled).",
    )
    parser.add_argument(
        "--no-preload-val-to-device",
        dest="preload_val_to_device",
        action="store_false",
        help="Disable preloading validation batches to device.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Enable early stopping with this validation-patience value.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--no-restore-best-weights",
        action="store_true",
        help="Do not restore best validation-loss weights at the end of training.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU training for all trials.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print the number of combinations and exit.",
    )
    parser.add_argument(
        "--test-output-dirname",
        type=str,
        default="best_model_test",
        help="Subdirectory under the best trial folder where test forecasts are saved.",
    )
    parser.add_argument(
        "--test-trial-dir",
        type=Path,
        default=None,
        help=(
            "Test a specific trial directory from an existing tuning run. "
            "When provided, grid search is skipped and weights are loaded from this trial folder."
        ),
    )
    return parser.parse_args()


def _helper_args_from_argv(argv: list[str]) -> argparse.Namespace:
    helper_parser = argparse.ArgumentParser(add_help=False)
    helper_parser.add_argument("--helper", action="store_true", help="Print Hyperband N_init and exit.")
    helper_parser.add_argument("--min", dest="helper_min", type=int, default=None)
    helper_parser.add_argument("--max", dest="helper_max", type=int, default=None)
    helper_parser.add_argument("--eta", dest="helper_eta", type=int, default=None)
    helper_args, _ = helper_parser.parse_known_args(argv)
    return helper_args


def main() -> None:
    helper_args = _helper_args_from_argv(sys.argv[1:])
    if helper_args.helper:
        if helper_args.helper_min is None or helper_args.helper_max is None or helper_args.helper_eta is None:
            raise ValueError("--helper requires --min, --max, and --eta.")
        n_init = hyperband_initial_trial_count(
            min_epochs=helper_args.helper_min,
            max_epochs=helper_args.helper_max,
            reduction_factor=helper_args.helper_eta,
        )
        print(
            json.dumps(
                {
                    "min_epochs": helper_args.helper_min,
                    "max_epochs": helper_args.helper_max,
                    "reduction_factor": helper_args.helper_eta,
                    "n_init": n_init,
                },
                indent=2,
            )
        )
        return

    args = parse_args()
    lookback_datasets = _parse_lookback_mapping(args.lookback_dataset)

    config = GridSearchConfig(
        lookback_datasets=lookback_datasets,
        learning_rates=list(args.learning_rates),
        batch_sizes=list(args.batch_sizes),
        n_lstm_values=list(args.n_lstm_values),
        hidden_lstm_values=list(args.hidden_lstm_values),
        hidden_fc_values=list(args.hidden_fc_values),
        n_fc_values=list(args.n_fc_values),
        epochs=args.epochs,
        seed=args.seed,
        out_dir=args.out_dir,
        prefer_gpu=not args.cpu_only,
        lstm_dropout=args.lstm_dropout,
        step_lr_step_size=args.step_lr_step_size,
        step_lr_gamma=args.step_lr_gamma,
        use_tqdm=not args.no_tqdm,
        verbose=args.verbose,
        preload_train_to_device=args.preload_train_to_device,
        preload_val_to_device=args.preload_val_to_device,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_weights=not args.no_restore_best_weights,
        test_output_dirname=args.test_output_dirname,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        reduction_factor=args.reduction_factor,
        prune_strategy=args.prune_strategy,
    )

    num_combinations = count_grid_combinations(config)
    print(f"Total combinations: {num_combinations}")
    if args.count_only:
        return

    if args.test_trial_dir is not None:
        test_model_from_trial_dir(config, args.test_trial_dir)
        return

    if config.use_hyperband:
        _results, best_result = run_hyperband_search(config)
    else:
        _results, best_result = run_grid_search(config)
    print("\n" + "=" * 100)
    print("STARTING FINAL BEST-MODEL TEST EVALUATION")
    print("=" * 100)
    test_metrics = test_best_model(config, best_result)
    _print_tuning_overview(config, best_result, test_metrics)


if __name__ == "__main__":
    main()
