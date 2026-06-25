"""End-to-end LSTM training plus conformal prediction from batch directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rabl.machine_learning import build_lstm_dataset
from rabl.machine_learning.conformal_prediction import (
    calibrate_autoregressive_conformal,
    conformal_rolling_forecast_profile,
    plot_conformal_forecast_profile_grid,
    save_conformal_forecasts_hdf5,
)
from rabl.machine_learning.dataset_scaling import LSTMDatasetScalerSplitter
from rabl.machine_learning.lstm_pipeline import (
    LSTMPipeline,
    LSTMPipelineConfig,
    STATE_DIM,
    TARGET_NAMES,
    _descale_feature_from_stats,
    _load_scaling_stats,
    cleanup_cuda,
    test_and_save_forecasts,
)


@dataclass(frozen=True)
class LSTMConformalRunConfig:
    sim_root: str
    batches: list[str]
    lookback: int
    config_py_path: str
    unscaled_out_dir: str
    scaled_out_dir: str
    out_dir: str
    unscaled_output_name: str | None = None
    scaled_output_name: str | None = None
    quiet_dataset_build: bool = False
    scaling_type: str = "standard"
    split_mode: str = "profile"
    train_frac: float = 0.65
    val_frac: float = 0.15
    cal_frac: float = 0.05
    test_frac: float = 0.15
    test_manifest_path: str | None = None
    val_manifest_path: str | None = None
    cal_manifest_path: str | None = None
    train_profile_limit_with_manifests: int | None = None
    batch_size: int = 256
    epochs: int = 100
    seed: int = 123
    learning_rate: float = 1e-3
    n_lstm: int = 1
    lstm_hidden: int = 64
    lstm_dropout: float = 0.0
    n_fc: int = 1
    fc_hidden: list[int] | tuple[int, ...] = (64,)
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0
    prefer_gpu: bool = True
    alpha: float = 0.05
    horizon_mode: str = "per_horizon"
    max_plots: int = 5

    def __post_init__(self) -> None:
        if not self.batches:
            raise ValueError("batches must contain at least one batch id.")
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1.")
        if self.scaling_type not in {"standard", "minmax", "none"}:
            raise ValueError("scaling_type must be 'standard', 'minmax', or 'none'.")
        if self.split_mode not in {"profile", "sample"}:
            raise ValueError("split_mode must be 'profile' or 'sample'.")
        if self.horizon_mode not in {"per_horizon", "global"}:
            raise ValueError("horizon_mode must be 'per_horizon' or 'global'.")
        if len(tuple(self.fc_hidden)) != self.n_fc:
            raise ValueError("fc_hidden must provide exactly n_fc values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end LSTM conformal prediction from a JSON config: build datasets, "
            "train one LSTM, evaluate test forecasts, and produce conformal UQ."
        ),
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to LSTM conformal JSON config.")
    return parser.parse_args()


def _load_cfg(path: Path) -> LSTMConformalRunConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return LSTMConformalRunConfig(**data)


def _optional_path(path: str | None) -> Path | None:
    return None if path in (None, "") else Path(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _compute_coverage_and_width(forecasts: list[dict[str, Any]], target_names: list[str]) -> dict[str, Any]:
    y_true = np.vstack([entry["y_true"] for entry in forecasts])
    lower = np.vstack([entry["lower"] for entry in forecasts])
    upper = np.vstack([entry["upper"] for entry in forecasts])
    coverage = np.mean((lower <= y_true) & (y_true <= upper), axis=0)
    width = np.mean(upper - lower, axis=0)
    if not np.all(np.isfinite(coverage)):
        raise ValueError("Empirical coverage contains non-finite values.")
    if not np.all(np.isfinite(width)):
        raise ValueError("Average conformal interval width contains non-finite values.")
    return {
        "coverage_by_target": {name: float(val) for name, val in zip(target_names, coverage)},
        "average_width_by_target": {name: float(val) for name, val in zip(target_names, width)},
        "mean_coverage": float(np.mean(coverage)),
        "mean_average_width": float(np.mean(width)),
    }


def _build_unscaled_dataset(args: LSTMConformalRunConfig) -> Path:
    config = build_lstm_dataset._validate_config(build_lstm_dataset._load_config(Path(args.config_py_path)))
    return build_lstm_dataset.build_dataset(
        Path(args.sim_root),
        Path(args.unscaled_out_dir),
        config["steady_state"],
        args.lookback,
        args.batches,
        output_name=args.unscaled_output_name,
        verbose=not args.quiet_dataset_build,
    )


def _scale_and_split_dataset(args: LSTMConformalRunConfig, unscaled_path: Path) -> Path:
    splitter = LSTMDatasetScalerSplitter(
        input_path=unscaled_path,
        scaling_type=args.scaling_type,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        cal_frac=args.cal_frac,
        test_frac=args.test_frac,
        output_dir=Path(args.scaled_out_dir),
        output_name=args.scaled_output_name,
        seed=args.seed,
        split_mode=args.split_mode,
        test_manifest_path=_optional_path(args.test_manifest_path),
        val_manifest_path=_optional_path(args.val_manifest_path),
        cal_manifest_path=_optional_path(args.cal_manifest_path),
        train_profile_limit_with_manifests=args.train_profile_limit_with_manifests,
    )
    return splitter.run()


def _train_model(args: LSTMConformalRunConfig, scaled_h5_path: Path) -> tuple[LSTMPipeline, torch.nn.Module, torch.device, Path]:
    if len(args.fc_hidden) != args.n_fc:
        raise ValueError("--fc-hidden must provide exactly --n-fc values.")
    config = LSTMPipelineConfig(
        h5_path=scaled_h5_path,
        batch_size=args.batch_size,
        seed=args.seed,
        n_lstm=args.n_lstm,
        lstm_hidden=args.lstm_hidden,
        lstm_dropout=args.lstm_dropout,
        n_fc=args.n_fc,
        fc_hidden=tuple(args.fc_hidden),
        learning_rate=args.learning_rate,
    )
    pipeline = LSTMPipeline(config)
    pipeline.build()
    pipeline.inspect()
    model, _history, used_device = pipeline.train(
        epochs=args.epochs,
        out_dir=Path(args.out_dir),
        prefer_gpu=args.prefer_gpu,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_weights=True,
    )
    weights_path = pipeline.save_model_pt(model, Path(args.out_dir) / "model.pt")
    print(f"Saved trained model weights to: {weights_path}")
    return pipeline, model, used_device, weights_path


def _run_conformal_uq(
    args: LSTMConformalRunConfig,
    *,
    pipeline: LSTMPipeline,
    model: torch.nn.Module,
    scaled_h5_path: Path,
    weights_path: Path,
) -> None:
    if pipeline.datasets is None:
        raise ValueError("Pipeline datasets were not built before conformal prediction.")
    datasets = pipeline.datasets
    cal_profile_ds = datasets.get("cal_profile_ds")
    if cal_profile_ds is None or not datasets.get("cal_profile_names"):
        raise ValueError(
            f"Scaled HDF5 dataset {scaled_h5_path} has no cal split; set cal_manifest_path or cal_frac > 0."
        )

    target_shape = datasets["target_shape"]
    num_targets = int(target_shape[1]) if len(target_shape) > 1 else len(TARGET_NAMES)
    target_names = TARGET_NAMES[:num_targets]
    scaling_stats = _load_scaling_stats(scaled_h5_path)

    conformal_result = calibrate_autoregressive_conformal(
        model,
        cal_profile_ds,
        alpha=args.alpha,
        horizon_mode=args.horizon_mode,
        state_dim=STATE_DIM,
    )
    q_hat = np.asarray(conformal_result["q_hat"])
    expected_shape = (
        (int(conformal_result["n_horizons"]), num_targets)
        if args.horizon_mode == "per_horizon"
        else (num_targets,)
    )
    if q_hat.shape != expected_shape:
        raise ValueError(f"q_hat shape check failed: expected {expected_shape}, got {q_hat.shape}.")

    cal_forecasts = [
        conformal_rolling_forecast_profile(
            model,
            str(profile_name),
            x_profile.numpy(),
            y_profile.numpy(),
            conformal_result=conformal_result,
            scaling_stats=scaling_stats,
            state_dim=STATE_DIM,
            control_channel=0,
        )
        for profile_name, x_profile, y_profile in datasets["cal_profile_ds"]
    ]
    cal_metrics = _compute_coverage_and_width(cal_forecasts, target_names)

    conformal_out_dir = Path(args.out_dir) / "conformal"
    conformal_out_dir.mkdir(parents=True, exist_ok=True)
    forecasts: list[dict[str, Any]] = []
    for idx, (profile_name, x_profile, y_profile) in enumerate(datasets["test_profile_ds"]):
        x_np = x_profile.numpy()
        entry = conformal_rolling_forecast_profile(
            model,
            str(profile_name),
            x_np,
            y_profile.numpy(),
            conformal_result=conformal_result,
            scaling_stats=scaling_stats,
            state_dim=STATE_DIM,
            control_channel=0,
        )
        if not np.all(entry["scaled"]["lower"] <= entry["scaled"]["upper"]):
            raise ValueError(f"Scaled lower <= upper check failed for profile {profile_name}.")
        forecasts.append(entry)
        if idx < args.max_plots:
            x_plot = x_np.copy()
            control_idx = STATE_DIM
            x_plot[:, :, control_idx] = _descale_feature_from_stats(
                scaling_stats,
                x_plot[:, :, control_idx],
                control_idx,
            )
            plot_conformal_forecast_profile_grid(
                x_profile=x_plot,
                y_true=entry["y_true"],
                y_pred=entry["y_pred"],
                lower=entry["lower"],
                upper=entry["upper"],
                target_names=target_names,
                title=f"Conformal Rolling Forecast - {profile_name}",
                save_path=conformal_out_dir / f"conformal_forecast_{profile_name}.png",
            )

    if not forecasts:
        raise ValueError("No test profiles were available for conformal forecast evaluation.")

    output_h5 = conformal_out_dir / "conformal_forecasts.h5"
    save_conformal_forecasts_hdf5(forecasts, output_path=output_h5, target_names=target_names)
    metrics = _compute_coverage_and_width(forecasts, target_names)

    metadata = {
        "alpha": float(args.alpha),
        "horizon_mode": args.horizon_mode,
        "q_hat_shape": list(q_hat.shape),
        "n_cal_profiles": int(conformal_result["n_cal_profiles"]),
        "n_horizons": int(conformal_result["n_horizons"]),
        "target_names": target_names,
        "empirical_calibration_coverage": cal_metrics["coverage_by_target"],
        "empirical_calibration_mean_coverage": cal_metrics["mean_coverage"],
        "weights_path": str(weights_path),
        "scaled_h5_path": str(scaled_h5_path),
    }
    (conformal_out_dir / "conformal_calibration_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2))
    (conformal_out_dir / "conformal_coverage_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2))

    print("\nConformal coverage summary (test set):")
    print(f"{'target':<18} {'coverage':>10} {'avg_width':>14}")
    for name in target_names:
        print(f"{name:<18} {metrics['coverage_by_target'][name]:10.4f} {metrics['average_width_by_target'][name]:14.6g}")
    print(f"Saved conformal forecasts to: {output_h5}")


def main() -> None:
    cli_args = parse_args()
    args = _load_cfg(cli_args.config)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    unscaled_h5_path = _build_unscaled_dataset(args)
    scaled_h5_path = _scale_and_split_dataset(args, unscaled_h5_path)
    print(f"Using scaled train/val/cal/test dataset: {scaled_h5_path}")

    pipeline, model, used_device, weights_path = _train_model(args, scaled_h5_path)

    print("Evaluating deterministic rolling forecasts on the test split...")
    test_and_save_forecasts(
        model,
        pipeline.datasets["test_profile_ds"],
        out_dir=Path(args.out_dir),
        h5_path=scaled_h5_path,
        state_dim=pipeline.config.state_dim,
        control_channel=pipeline.config.control_channel,
        target_names=pipeline.config.target_names,
        max_plots=0,
        plot_callback=None,
        use_tqdm=pipeline.config.use_tqdm,
    )

    _run_conformal_uq(
        args,
        pipeline=pipeline,
        model=model,
        scaled_h5_path=scaled_h5_path,
        weights_path=weights_path,
    )

    run_metadata = {
        "unscaled_h5_path": str(unscaled_h5_path),
        "scaled_h5_path": str(scaled_h5_path),
        "weights_path": str(weights_path),
        "config_path": str(cli_args.config),
        "config": args.__dict__,
    }
    (Path(args.out_dir) / "end_to_end_conformal_run_metadata.json").write_text(json.dumps(_json_safe(run_metadata), indent=2))

    if "cuda" in str(used_device).lower():
        cleanup_cuda(model, used_device)


if __name__ == "__main__":
    main()
