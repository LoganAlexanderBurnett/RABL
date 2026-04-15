"""Grid-search tuner for the LSTM training pipeline.

This module performs a Cartesian-product hyperparameter search over pre-built
LSTM datasets keyed by lookback window size.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch

from rabl.machine_learning.lstm_pipeline import (
    build_datasets,
    build_model,
    cleanup_cuda,
    test_and_save_forecasts,
    train_with_fallback,
)


@dataclass(slots=True)
class GridSearchConfig:
    lookback_datasets: dict[int, Path]
    learning_rates: list[float]
    batch_sizes: list[int]
    n_lstm_values: list[int]
    hidden_lstm_values: list[int]
    hidden_fc_values: list[int]
    n_fc: int = 1
    epochs: int = 20
    seed: int = 123
    out_dir: Path = Path("outputs") / "ml_tuning"
    prefer_gpu: bool = True
    lstm_dropout: float = 0.0
    preload_train_to_device: bool = False
    preload_val_to_device: bool = False
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


def _all_trial_params(config: GridSearchConfig) -> list[dict[str, Any]]:
    trial_grid = product(
        sorted(config.lookback_datasets.items(), key=lambda item: item[0]),
        config.learning_rates,
        config.batch_sizes,
        config.n_lstm_values,
        config.hidden_lstm_values,
        config.hidden_fc_values,
    )
    params: list[dict[str, Any]] = []
    for (lookback, dataset_path), learning_rate, batch_size, n_lstm, hidden_lstm, hidden_fc in trial_grid:
        params.append(
            {
                "lookback": lookback,
                "dataset_path": dataset_path,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_lstm": n_lstm,
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
        n_fc=config.n_fc,
        fc_hidden=(params["hidden_fc"],),
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
    return (
        {
            "best_val_loss": best_val_loss,
            "final_val_loss": float(history["val_loss"][-1]),
            "actual_epochs_trained": int(actual_epochs),
            "used_device": str(used_device),
            "early_stopped": bool(actual_epochs < epochs_to_run),
        },
        checkpoint_path,
        weights_path,
    )


def run_grid_search(config: GridSearchConfig) -> tuple[list[TrialResult], TrialResult]:
    """Run all tuning trials and return all trial metrics plus the best trial."""
    total_trials = count_grid_combinations(config)
    if total_trials <= 0:
        raise ValueError("Grid search has zero combinations. Please provide non-empty parameter lists.")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Total combinations: {total_trials}")

    trial_grid = product(
        sorted(config.lookback_datasets.items(), key=lambda item: item[0]),
        config.learning_rates,
        config.batch_sizes,
        config.n_lstm_values,
        config.hidden_lstm_values,
        config.hidden_fc_values,
    )

    results: list[TrialResult] = []

    for trial_index, (
        (lookback, dataset_path),
        learning_rate,
        batch_size,
        n_lstm,
        hidden_lstm,
        hidden_fc,
    ) in enumerate(trial_grid, start=1):
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found for lookback={lookback}: {dataset_path}")

        trial_name = (
            f"trial_{trial_index:04d}"
            f"_lb{lookback}"
            f"_lr{learning_rate:g}"
            f"_bs{batch_size}"
            f"_nl{n_lstm}"
            f"_hl{hidden_lstm}"
            f"_hf{hidden_fc}"
        )
        trial_dir = config.out_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{trial_index}/{total_trials}] lookback={lookback}, lr={learning_rate:g}, "
            f"batch_size={batch_size}, n_lstm={n_lstm}, hidden_lstm={hidden_lstm}, hidden_fc={hidden_fc}"
        )

        datasets = build_datasets(h5_path=dataset_path, batch_size=batch_size, seed=config.seed)
        model, history, used_device = train_with_fallback(
            datasets,
            epochs=config.epochs,
            out_dir=trial_dir,
            n_lstm=n_lstm,
            lstm_hidden=hidden_lstm,
            lstm_dropout=config.lstm_dropout,
            n_fc=config.n_fc,
            fc_hidden=(hidden_fc,),
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

    print("\nBest trial:")
    print(json.dumps(asdict(best_result), indent=2))
    print(f"Saved tuning summary to: {config.out_dir / 'grid_search_results.json'}")

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

    results: list[TrialResult] = []
    hyperband_brackets_meta: list[dict[str, Any]] = []
    trial_counter = 0

    for bracket in brackets:
        bracket_trials: list[dict[str, Any]] = []
        for _ in range(bracket.initial_trial_count):
            sampled = dict(rng.choice(all_params))
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
                    }
                    trial["rungs"].append(rung_entry)
                    completed_trials.append(trial)
                    rung_records.append({"trial_index": trial["trial_index"], **rung_entry})
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

    print("\nBest trial:")
    print(json.dumps(asdict(best_result), indent=2))
    print(f"Saved tuning summary to: {config.out_dir / 'grid_search_results.json'}")
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
        n_fc=config.n_fc,
        fc_hidden=(best_result.hidden_fc,),
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
    parser.add_argument("--n-fc", type=int, default=1, help="Number of FC layers in the model head.")
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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "ml_tuning")
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
        action="store_true",
        help="Preload training batches to the selected training device.",
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


def main() -> None:
    args = parse_args()
    lookback_datasets = _parse_lookback_mapping(args.lookback_dataset)

    config = GridSearchConfig(
        lookback_datasets=lookback_datasets,
        learning_rates=list(args.learning_rates),
        batch_sizes=list(args.batch_sizes),
        n_lstm_values=list(args.n_lstm_values),
        hidden_lstm_values=list(args.hidden_lstm_values),
        hidden_fc_values=list(args.hidden_fc_values),
        n_fc=args.n_fc,
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
    test_best_model(config, best_result)


if __name__ == "__main__":
    main()
