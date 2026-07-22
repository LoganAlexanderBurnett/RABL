"""End-to-end LSTM training plus conformal prediction from batch directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from rabl.machine_learning import build_lstm_dataset
from rabl.machine_learning.conformal_prediction import (
    calibrate_autoregressive_conformal,
    conformal_rolling_forecast_profile,
    joint_ensemble_conformal_forecast_profile,
    joint_ensemble_coverage_metrics,
    calibrate_joint_ensemble_normalized_conformal,
    plot_conformal_forecast_profile_grid,
    save_conformal_forecasts_hdf5,
    save_joint_ensemble_conformal_forecasts_hdf5,
    DEFAULT_UQ_METHODS,
    UQ_METHODS,
    apply_uq_method,
    calibrate_absolute_conformal,
    calibrate_ensemble_normalized_conformal,
    compute_ensemble_profile_forecast,
    compute_uq_coverage_metrics,
    save_shared_ensemble_predictions_hdf5,
    save_uq_forecasts_hdf5,
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
    ProfileDataset,
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
    conformal_method: str = "per_horizon_absolute"
    n_models: int = 5
    bag_fraction: float = 0.70
    bag_split_mode: str = "profile"
    sigma_floor: float | list[float] = 1e-6
    ensemble_ddof: int = 0
    save_member_forecasts: bool = True
    ensemble_source: str = "train"
    ensemble_checkpoint_paths: list[str] | tuple[str, ...] = ()
    ensemble_bagged_h5_path: str | None = None
    uq_methods: list[str] | tuple[str, ...] | None = None
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
        if self.conformal_method not in {"per_horizon_absolute", "global_absolute", "joint_ensemble_normalized"}:
            raise ValueError("Unsupported conformal_method.")
        if self.ensemble_source not in {"train", "checkpoints"}:
            raise ValueError("ensemble_source must be 'train' or 'checkpoints'.")
        if self.conformal_method == "joint_ensemble_normalized" and self.ensemble_source == "train" and self.n_models < 2:
            raise ValueError("joint_ensemble_normalized training mode requires n_models >= 2.")
        if self.conformal_method == "joint_ensemble_normalized" and self.ensemble_source == "checkpoints" and len(self.ensemble_checkpoint_paths) < 2:
            raise ValueError("joint_ensemble_normalized checkpoint mode requires at least two ensemble_checkpoint_paths.")
        if not (0.0 < self.bag_fraction <= 1.0):
            raise ValueError("bag_fraction must be in (0, 1].")
        if self.bag_split_mode not in {"profile", "sample"}:
            raise ValueError("bag_split_mode must be 'profile' or 'sample'.")
        methods = tuple(DEFAULT_UQ_METHODS if self.uq_methods is None else self.uq_methods)
        unknown_methods = sorted(set(methods) - set(DEFAULT_UQ_METHODS))
        if unknown_methods:
            raise ValueError(f"Unsupported uq_methods: {unknown_methods}")
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

    absolute_horizon_mode = (
        "global" if args.conformal_method == "global_absolute" else
        "per_horizon" if args.conformal_method == "per_horizon_absolute" else args.horizon_mode
    )
    conformal_result = calibrate_autoregressive_conformal(
        model, cal_profile_ds, alpha=args.alpha, horizon_mode=absolute_horizon_mode, state_dim=STATE_DIM,
    )
    q_hat = np.asarray(conformal_result["q_hat"])
    expected_shape = (
        (int(conformal_result["n_horizons"]), num_targets)
        if absolute_horizon_mode == "per_horizon"
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
        "conformal_method": args.conformal_method,
        "horizon_mode": absolute_horizon_mode,
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



def _choose_device(prefer_gpu: bool) -> torch.device:
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def _scaling_stats_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("type") != right.get("type"):
        return False
    for group in ("x", "y"):
        if set(left[group]) != set(right[group]):
            return False
        for key in left[group]:
            if not np.array_equal(np.asarray(left[group][key]), np.asarray(right[group][key])):
                return False
    return True


def _bag_training_profile_names(bagged_h5_path: Path, n_members: int) -> list[list[str]]:
    if not bagged_h5_path.exists():
        raise FileNotFoundError(f"ensemble_bagged_h5_path does not exist: {bagged_h5_path}")
    bag_profile_names: list[list[str]] = []
    with h5py.File(bagged_h5_path, "r") as h5f:
        if "scaling" not in h5f:
            raise ValueError(f"Bagged HDF5 is missing required scaling metadata: {bagged_h5_path}")
        for idx in range(n_members):
            split_path = f"train/bag_{idx}/files"
            if split_path not in h5f:
                raise ValueError(f"Bagged HDF5 is missing required bag membership group: {split_path}")
            names = sorted(str(name) for name in h5f[split_path].keys())
            if not names:
                raise ValueError(f"Bagged HDF5 bag_{idx} contains no training profiles.")
            bag_profile_names.append(names)
    return bag_profile_names


def _assert_no_bag_leakage(
    bag_profile_names: list[list[str]],
    *,
    cal_profile_names: list[str],
    test_profile_names: list[str],
) -> dict[str, Any]:
    cal_names = set(cal_profile_names)
    test_names = set(test_profile_names)
    cal_leaks = sorted({name for bag in bag_profile_names for name in set(bag).intersection(cal_names)})
    test_leaks = sorted({name for bag in bag_profile_names for name in set(bag).intersection(test_names)})
    if cal_leaks or test_leaks:
        raise ValueError(f"Training bags leak held-out profiles: cal={cal_leaks[:5]}, test={test_leaks[:5]}")
    return {
        "train_calibration_overlap_detected": False,
        "train_test_overlap_detected": False,
        "checked_training_bag_count": len(bag_profile_names),
    }


def _load_joint_ensemble_from_checkpoints(
    args: LSTMConformalRunConfig,
    *,
    scaled_h5_path: Path,
    datasets: dict[str, Any],
    scaling_stats: dict[str, Any],
) -> dict[str, Any]:
    from rabl.machine_learning.bagging_ensemble import load_bagged_lstm_ensemble_checkpoints

    checkpoint_paths = [Path(path) for path in args.ensemble_checkpoint_paths]
    if len(checkpoint_paths) < 2:
        raise ValueError("ensemble_checkpoint_paths must contain at least two checkpoints.")
    missing = [path for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ensemble checkpoint path(s): {missing}")
    if args.ensemble_bagged_h5_path in (None, ""):
        raise ValueError("ensemble_bagged_h5_path is required when ensemble_source='checkpoints'.")
    bagged_h5_path = Path(args.ensemble_bagged_h5_path)
    bag_scaling_stats = _load_scaling_stats(bagged_h5_path)
    scaling_match = _scaling_stats_equal(scaling_stats, bag_scaling_stats)
    if not scaling_match:
        raise ValueError("Current scaled HDF5 scaling statistics do not exactly match ensemble_bagged_h5_path.")

    bag_profile_names = _bag_training_profile_names(bagged_h5_path, len(checkpoint_paths))
    overlap = _assert_no_bag_leakage(
        bag_profile_names,
        cal_profile_names=datasets["cal_profile_names"],
        test_profile_names=datasets["test_profile_names"],
    )
    x_shape = datasets["sample_shape"]
    y_shape = datasets["target_shape"]
    device = _choose_device(args.prefer_gpu)
    models = load_bagged_lstm_ensemble_checkpoints(
        checkpoint_paths,
        timesteps=int(x_shape[1]),
        num_features=int(x_shape[2]),
        num_targets=int(y_shape[1]),
        n_lstm=args.n_lstm,
        lstm_hidden=args.lstm_hidden,
        lstm_dropout=args.lstm_dropout,
        n_fc=args.n_fc,
        fc_hidden=tuple(args.fc_hidden),
        device=device,
    )
    return {
        "models": models,
        "model_paths": checkpoint_paths,
        "bagged_h5_path": bagged_h5_path,
        "bag_profile_names": bag_profile_names,
        "used_devices": [str(device)] * len(models),
        "ensemble_source": "checkpoints",
        "architecture_validation": {
            "strict_state_dict_load": True,
            "timesteps": int(x_shape[1]),
            "num_features": int(x_shape[2]),
            "num_targets": int(y_shape[1]),
            "n_lstm": args.n_lstm,
            "lstm_hidden": args.lstm_hidden,
            "lstm_dropout": args.lstm_dropout,
            "n_fc": args.n_fc,
            "fc_hidden": list(args.fc_hidden),
        },
        "scaling_match": scaling_match,
        "overlap_checks": overlap,
    }

def _assert_profile_disjoint(datasets: dict[str, Any]) -> None:
    sets = {name: set(datasets.get(f"{name}_profile_names", [])) for name in ("train", "val", "cal", "test")}
    for left in sets:
        for right in sets:
            if left < right and sets[left].intersection(sets[right]):
                raise ValueError(f"Profile splits overlap: {left} and {right}.")


def _method_calibration_result(method_id: str, cal_forecasts: list[dict[str, Any]], args: LSTMConformalRunConfig) -> dict[str, Any] | None:
    if method_id == "ensemble_conformal_target_trajectory":
        return calibrate_ensemble_normalized_conformal(
            cal_forecasts, alpha=args.alpha, sigma_floor=np.asarray(args.sigma_floor), temporal_mode="trajectory",
        )
    if method_id == "ensemble_conformal_target_horizon":
        return calibrate_ensemble_normalized_conformal(
            cal_forecasts, alpha=args.alpha, sigma_floor=np.asarray(args.sigma_floor), temporal_mode="per_horizon",
        )
    if method_id == "absolute_conformal_target_horizon":
        return calibrate_absolute_conformal(cal_forecasts, alpha=args.alpha, temporal_mode="per_horizon")
    if method_id == "absolute_conformal_target_trajectory":
        return calibrate_absolute_conformal(cal_forecasts, alpha=args.alpha, temporal_mode="trajectory")
    if method_id == "raw_ensemble_2sigma":
        return None
    raise ValueError(f"Unsupported UQ method: {method_id}")


def _calibration_metadata_json(calibration: dict[str, Any] | None, *, method_id: str, method_dir: Path) -> dict[str, Any]:
    info = UQ_METHODS[method_id]
    if calibration is None:
        return {
            "method_id": method_id,
            "method_label": info["label"],
            "temporal_mode": info["temporal_mode"],
            "residual_type": info["residual_type"],
            "uses_conformal_quantile": False,
            "no_conformal_guarantee": True,
        }
    payload = {
        "method_id": method_id,
        "method_label": info["label"],
        "alpha": calibration["alpha"],
        "temporal_mode": calibration["temporal_mode"],
        "residual_type": calibration["residual_type"],
        "calibration_profile_names": calibration["calibration_profile_names"],
    }
    for key in ("q_by_target", "q_by_horizon_target", "quantile_index_by_target", "score_count_by_target", "calibration_count_by_horizon", "quantile_index_by_horizon_target", "sigma_floor"):
        if key in calibration:
            payload[key] = _json_safe(calibration[key])
    scores_path = method_dir / "calibration_scores.h5"
    with h5py.File(scores_path, "w") as h5f:
        for key in ("calibration_scores_by_profile_target",):
            if key in calibration:
                h5f.create_dataset(key, data=calibration[key], compression="gzip", chunks=True)
    payload["calibration_scores_h5"] = str(scores_path)
    return payload


def _forecast_shared_profiles(models, profile_ds, *, scaling_stats, ddof: int) -> list[dict[str, Any]]:
    return [
        compute_ensemble_profile_forecast(
            models, str(name), x.numpy(), y.numpy(), scaling_stats=scaling_stats,
            state_dim=STATE_DIM, ddof=ddof,
        )
        for name, x, y in profile_ds
    ]


def _run_joint_ensemble_uq(args: LSTMConformalRunConfig, scaled_h5_path: Path, *, config_path: Path | None = None) -> None:
    from rabl.machine_learning.bagging_ensemble import run_bagging_ensemble
    from rabl.machine_learning.lstm_pipeline import build_datasets

    if args.bag_split_mode != "profile":
        raise ValueError("joint_ensemble_normalized requires profile-level bagging.")
    datasets = build_datasets(scaled_h5_path, args.batch_size, args.seed)
    if not datasets.get("cal_profile_names"):
        raise ValueError("joint_ensemble_normalized requires a non-empty profile-disjoint cal split.")
    _assert_profile_disjoint(datasets)
    methods = list(DEFAULT_UQ_METHODS if args.uq_methods is None else args.uq_methods)
    comparison_dir = Path(args.out_dir) / "uq_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    scaling_stats = _load_scaling_stats(scaled_h5_path)
    if args.ensemble_source == "train":
        result = run_bagging_ensemble(
            scaled_h5_path, out_dir=comparison_dir / "ensemble_training", n_models=args.n_models,
            bag_fraction=args.bag_fraction, bag_split_mode="profile", seed=args.seed,
            batch_size=args.batch_size, epochs=args.epochs, early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta, learning_rate=args.learning_rate,
            n_lstm=args.n_lstm, lstm_hidden=args.lstm_hidden, lstm_dropout=args.lstm_dropout,
            n_fc=args.n_fc, fc_hidden=tuple(args.fc_hidden), prefer_gpu=args.prefer_gpu,
            save_member_forecasts=args.save_member_forecasts,
        )
        result["ensemble_source"] = "train"
        result["architecture_validation"] = {"strict_state_dict_load": False, "trained_in_current_run": True}
        result["scaling_match"] = True
        result["overlap_checks"] = _assert_no_bag_leakage(
            result["bag_profile_names"],
            cal_profile_names=datasets["cal_profile_names"],
            test_profile_names=datasets["test_profile_names"],
        )
    else:
        result = _load_joint_ensemble_from_checkpoints(
            args, scaled_h5_path=scaled_h5_path, datasets=datasets, scaling_stats=scaling_stats,
        )
    target_names = list(TARGET_NAMES[:int(datasets["target_shape"][1])])
    models = result["models"]
    cal_ds = ProfileDataset(scaled_h5_path, datasets["cal_profile_names"], "cal")
    test_ds = ProfileDataset(scaled_h5_path, datasets["test_profile_names"], "test")

    cal_shared = _forecast_shared_profiles(models, cal_ds, scaling_stats=scaling_stats, ddof=args.ensemble_ddof)
    test_shared = _forecast_shared_profiles(models, test_ds, scaling_stats=scaling_stats, ddof=args.ensemble_ddof)
    save_shared_ensemble_predictions_hdf5(
        cal_shared, output_path=comparison_dir / "calibration_ensemble_predictions.h5",
        target_names=target_names, ensemble_ddof=args.ensemble_ddof,
    )
    save_shared_ensemble_predictions_hdf5(
        test_shared, output_path=comparison_dir / "test_ensemble_predictions.h5",
        target_names=target_names, ensemble_ddof=args.ensemble_ddof,
    )
    shared_metadata = {
        "ensemble_source": result["ensemble_source"],
        "model_checkpoint_paths": [str(path) for path in result["model_paths"]],
        "ensemble_bagged_h5_path": str(result["bagged_h5_path"]),
        "architecture_validation": result["architecture_validation"],
        "scaling_match": bool(result["scaling_match"]),
        "overlap_checks": result["overlap_checks"],
        "scaled_h5_path": str(scaled_h5_path),
        "target_names": target_names,
        "calibration_profile_names": datasets["cal_profile_names"],
        "test_profile_names": datasets["test_profile_names"],
        "alpha": float(args.alpha),
        "nominal_coverage": 1.0 - float(args.alpha),
        "sigma_floor": _json_safe(np.asarray(args.sigma_floor)),
        "ensemble_ddof": int(args.ensemble_ddof),
        "ensemble_member_count": len(models),
        "config_path": None if config_path is None else str(config_path),
    }
    (comparison_dir / "shared_ensemble_metadata.json").write_text(json.dumps(_json_safe(shared_metadata), indent=2))

    manifest_methods: list[dict[str, Any]] = []
    for method_id in methods:
        info = UQ_METHODS[method_id]
        method_dir = comparison_dir / method_id
        method_dir.mkdir(parents=True, exist_ok=True)
        calibration = _method_calibration_result(method_id, cal_shared, args)
        cal_entries = [
            apply_uq_method(forecast, method_id=method_id, calibration_result=calibration, scaling_stats=scaling_stats,
                            alpha=args.alpha, sigma_floor=np.asarray(args.sigma_floor), include_member_predictions=args.save_member_forecasts)
            for forecast in cal_shared
        ]
        test_entries = [
            apply_uq_method(forecast, method_id=method_id, calibration_result=calibration, scaling_stats=scaling_stats,
                            alpha=args.alpha, sigma_floor=np.asarray(args.sigma_floor), include_member_predictions=args.save_member_forecasts)
            for forecast in test_shared
        ]
        h5_metadata = {
            "method_id": method_id,
            "method_label": info["label"],
            "alpha": float(args.alpha),
            "nominal_coverage": 1.0 - float(args.alpha),
            "ensemble_member_count": len(models),
            "ensemble_ddof": int(args.ensemble_ddof),
            "residual_space": "scaled",
            "temporal_calibration_mode": info["temporal_mode"],
            "uses_ensemble_normalization": bool(info["uses_ensemble_normalization"]),
        }
        calibration_forecasts_path = method_dir / "calibration_forecasts.h5"
        test_forecasts_path = method_dir / "test_forecasts.h5"
        save_uq_forecasts_hdf5(cal_entries, output_path=calibration_forecasts_path, metadata=h5_metadata, target_names=target_names)
        save_uq_forecasts_hdf5(test_entries, output_path=test_forecasts_path, metadata=h5_metadata, target_names=target_names)
        metrics = compute_uq_coverage_metrics(
            test_entries, target_names, alpha=args.alpha, primary_coverage_type=info["primary_coverage_type"],
            no_conformal_guarantee=(method_id == "raw_ensemble_2sigma"),
        )
        metrics_path = method_dir / "coverage_metrics.json"
        metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2))
        metadata_name = "method_metadata.json" if method_id == "raw_ensemble_2sigma" else "calibration_metadata.json"
        metadata_path = method_dir / metadata_name
        metadata = _calibration_metadata_json(calibration, method_id=method_id, method_dir=method_dir)
        metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2))
        manifest_methods.append({
            "method_id": method_id,
            "method_label": info["label"],
            "temporal_mode": info["temporal_mode"],
            "residual_type": info["residual_type"],
            "normalized": bool(info["uses_ensemble_normalization"]),
            "calibration_metadata_path": str(metadata_path),
            "calibration_forecasts_path": str(calibration_forecasts_path),
            "test_forecasts_path": str(test_forecasts_path),
            "metrics_path": str(metrics_path),
        })
        print(f"{method_id}: primary coverage={metrics['primary_empirical_coverage']:.4f}, mean width={metrics['mean_interval_width_overall']:.6g}")

    manifest = {**shared_metadata, "experiment": Path(args.out_dir).name, "uq_methods": manifest_methods}
    (comparison_dir / "uq_methods_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2))
    print(f"Saved UQ comparison manifest to: {comparison_dir / 'uq_methods_manifest.json'}")

def main() -> None:
    cli_args = parse_args()
    args = _load_cfg(cli_args.config)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    unscaled_h5_path = _build_unscaled_dataset(args)
    scaled_h5_path = _scale_and_split_dataset(args, unscaled_h5_path)
    print(f"Using scaled train/val/cal/test dataset: {scaled_h5_path}")

    weights_path: Path | None = None
    used_device: torch.device | None = None
    if args.conformal_method == "joint_ensemble_normalized":
        _run_joint_ensemble_uq(args, scaled_h5_path, config_path=cli_args.config)
    else:
        pipeline, model, used_device, weights_path = _train_model(args, scaled_h5_path)
        print("Evaluating deterministic rolling forecasts on the test split...")
        test_and_save_forecasts(model, pipeline.datasets["test_profile_ds"], out_dir=Path(args.out_dir),
            h5_path=scaled_h5_path, state_dim=pipeline.config.state_dim,
            control_channel=pipeline.config.control_channel, target_names=pipeline.config.target_names,
            max_plots=0, plot_callback=None, use_tqdm=pipeline.config.use_tqdm)
        _run_conformal_uq(args, pipeline=pipeline, model=model, scaled_h5_path=scaled_h5_path, weights_path=weights_path)

    run_metadata = {
        "unscaled_h5_path": str(unscaled_h5_path), "scaled_h5_path": str(scaled_h5_path),
        "weights_path": None if weights_path is None else str(weights_path),
        "config_path": str(cli_args.config), "config": args.__dict__,
    }
    (Path(args.out_dir) / "end_to_end_conformal_run_metadata.json").write_text(json.dumps(_json_safe(run_metadata), indent=2))

    if used_device is not None and "cuda" in str(used_device).lower():
        cleanup_cuda(model, used_device)


if __name__ == "__main__":
    main()
