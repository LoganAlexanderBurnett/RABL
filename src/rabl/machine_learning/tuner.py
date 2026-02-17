"""Grid-search tuner for the LSTM training pipeline.

This module performs a Cartesian-product hyperparameter search over pre-built
LSTM datasets keyed by lookback window size.
"""

from __future__ import annotations

import argparse
import json
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
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = True
    test_output_dirname: str = "best_model_test"
    step_lr_step_size: int = 30
    step_lr_gamma: float = 0.5
    verbose: int = 1


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
            deterministic_seed=config.seed,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            restore_best_weights=config.restore_best_weights,
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
    )
    print(f"Saved best-model test outputs to: {test_out_dir}")
    return test_metrics


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
        verbose=args.verbose,
        preload_train_to_device=args.preload_train_to_device,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_weights=not args.no_restore_best_weights,
        test_output_dirname=args.test_output_dirname,
    )

    num_combinations = count_grid_combinations(config)
    print(f"Total combinations: {num_combinations}")
    if args.count_only:
        return

    _results, best_result = run_grid_search(config)
    test_best_model(config, best_result)


if __name__ == "__main__":
    main()
