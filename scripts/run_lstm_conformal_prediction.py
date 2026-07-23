"""Apply LSTM ensemble conformal UQ from saved audit ensemble forecasts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from rabl.machine_learning.conformal_prediction import (
    DEFAULT_UQ_METHODS,
    UQ_METHODS,
    apply_uq_method,
    calibrate_absolute_conformal,
    calibrate_ensemble_normalized_conformal,
    compute_uq_coverage_metrics,
    save_shared_ensemble_predictions_hdf5,
    save_uq_forecasts_hdf5,
)
from rabl.machine_learning.lstm_pipeline import _descale_targets_from_stats, _load_scaling_stats


@dataclass(frozen=True)
class LSTMConformalRunConfig:
    """Configuration for applying UQ to saved ensemble audit forecasts.

    This script intentionally does not build datasets, train models, load model
    checkpoints, or run ensemble inference.  It consumes saved calibration and
    test ensemble forecast HDF5 files, typically the ``*_audit.h5`` files written
    by ``scripts/train_lstm_ensemble.py``.
    """

    scaled_h5_path: str
    calibration_ensemble_forecasts_audit_h5_path: str
    test_ensemble_forecasts_audit_h5_path: str
    uq_output_dir: str
    train_manifest_path: str
    val_manifest_path: str
    cal_manifest_path: str
    test_manifest_path: str
    alpha: float = 0.05
    sigma_floor: float | list[float] = 1e-6
    ensemble_ddof: int = 0
    save_member_forecasts: bool = True
    uq_methods: list[str] | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if int(self.ensemble_ddof) < 0:
            raise ValueError("ensemble_ddof must be non-negative.")
        methods = tuple(DEFAULT_UQ_METHODS if self.uq_methods is None else self.uq_methods)
        unknown_methods = sorted(set(methods) - set(DEFAULT_UQ_METHODS))
        if unknown_methods:
            raise ValueError(f"Unsupported uq_methods: {unknown_methods}")
        for label, value in (
            ("scaled_h5_path", self.scaled_h5_path),
            ("calibration_ensemble_forecasts_audit_h5_path", self.calibration_ensemble_forecasts_audit_h5_path),
            ("test_ensemble_forecasts_audit_h5_path", self.test_ensemble_forecasts_audit_h5_path),
            ("uq_output_dir", self.uq_output_dir),
            ("train_manifest_path", self.train_manifest_path),
            ("val_manifest_path", self.val_manifest_path),
            ("cal_manifest_path", self.cal_manifest_path),
            ("test_manifest_path", self.test_manifest_path),
        ):
            if value in (None, ""):
                raise ValueError(f"{label} is required.")
        sigma_floor = np.asarray(self.sigma_floor, dtype=float)
        if not np.all(np.isfinite(sigma_floor)) or np.any(sigma_floor < 0.0):
            raise ValueError("sigma_floor must contain finite nonnegative values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LSTM ensemble conformal UQ from saved calibration/test ensemble "
            "audit forecast HDF5 files. This script does not train models, load "
            "checkpoints, build datasets, or rerun ensemble inference."
        ),
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to LSTM conformal JSON config.")
    return parser.parse_args()


_DEPRECATED_CONFIG_KEYS = {
    "sim_root",
    "batches",
    "lookback",
    "config_py_path",
    "unscaled_out_dir",
    "scaled_out_dir",
    "out_dir",
    "unscaled_output_name",
    "scaled_output_name",
    "quiet_dataset_build",
    "scaling_type",
    "split_mode",
    "train_frac",
    "val_frac",
    "cal_frac",
    "test_frac",
    "train_profile_limit_with_manifests",
    "batch_size",
    "epochs",
    "seed",
    "learning_rate",
    "n_lstm",
    "lstm_hidden",
    "lstm_dropout",
    "n_fc",
    "fc_hidden",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "prefer_gpu",
    "horizon_mode",
    "conformal_method",
    "n_models",
    "bag_fraction",
    "bag_split_mode",
    "ensemble_source",
    "ensemble_checkpoint_paths",
    "ensemble_bagged_h5_path",
}
_ALLOWED_CONFIG_KEYS = set(LSTMConformalRunConfig.__dataclass_fields__)  # type: ignore[attr-defined]


def _load_cfg(path: Path) -> LSTMConformalRunConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    deprecated = sorted(set(data).intersection(_DEPRECATED_CONFIG_KEYS))
    if deprecated:
        raise ValueError(
            "run_lstm_conformal_prediction.py now consumes saved ensemble audit forecasts only. "
            f"Remove unsupported dataset/training/checkpoint keys from the config: {deprecated}"
        )
    unknown = sorted(set(data) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown config fields: {unknown}")
    return LSTMConformalRunConfig(**data)


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


def _decode_attr_strings(value: Any) -> list[str]:
    arr = np.asarray(value)
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in arr.tolist()]


def _load_manifest_profiles(path: Path, field: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    profiles = data.get(field)
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"Manifest {path} must contain a non-empty list field '{field}'.")
    names = [str(name) for name in profiles]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Manifest {path} contains duplicate profile names: {duplicates[:10]}")
    return names


def _load_manifests(args: LSTMConformalRunConfig) -> dict[str, list[str]]:
    manifests = {
        "train": _load_manifest_profiles(Path(args.train_manifest_path), "train_profiles"),
        "val": _load_manifest_profiles(Path(args.val_manifest_path), "val_profiles"),
        "cal": _load_manifest_profiles(Path(args.cal_manifest_path), "cal_profiles"),
        "test": _load_manifest_profiles(Path(args.test_manifest_path), "test_profiles"),
    }
    for left in manifests:
        for right in manifests:
            if left < right:
                overlap = sorted(set(manifests[left]).intersection(manifests[right]))
                if overlap:
                    raise ValueError(f"Profile manifests overlap between {left} and {right}: {overlap[:10]}")
    return manifests


def _ensure_file(path: str, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _profile_names_from_h5(path: Path) -> list[str]:
    with h5py.File(path, "r") as h5f:
        return sorted(str(name) for name in h5f.keys())


def _validate_profile_set(path: Path, actual: list[str], expected: list[str], label: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"{label} forecast profiles in {path} do not match the {label} manifest. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )


def _validate_forecast_pair(cal_path: Path, test_path: Path) -> tuple[list[str], int, int]:
    with h5py.File(cal_path, "r") as cal_h5, h5py.File(test_path, "r") as test_h5:
        cal_targets = _decode_attr_strings(cal_h5.attrs.get("target_names", []))
        test_targets = _decode_attr_strings(test_h5.attrs.get("target_names", []))
        if not cal_targets or cal_targets != test_targets:
            raise ValueError("Calibration and test forecast HDF5 files must contain identical target_names attributes.")
        cal_members = int(cal_h5.attrs.get("member_count", cal_h5.attrs.get("ensemble_member_count", -1)))
        test_members = int(test_h5.attrs.get("member_count", test_h5.attrs.get("ensemble_member_count", -1)))
        if cal_members < 2 or cal_members != test_members:
            raise ValueError("Calibration and test forecast HDF5 files must contain the same member count >= 2.")
        cal_ddof = int(cal_h5.attrs.get("ensemble_std_ddof", cal_h5.attrs.get("ensemble_ddof", 0)))
        test_ddof = int(test_h5.attrs.get("ensemble_std_ddof", test_h5.attrs.get("ensemble_ddof", 0)))
        if cal_ddof != test_ddof:
            raise ValueError("Calibration and test forecast HDF5 files must use the same ensemble ddof.")
        return cal_targets, cal_members, cal_ddof


def _dataset_alias(group: h5py.Group, *names: str) -> np.ndarray:
    for name in names:
        if name in group:
            return group[name][()]
    raise ValueError(f"Group {group.name} is missing required dataset; tried aliases {names}.")


def _load_audit_ensemble_forecasts(
    path: Path,
    *,
    scaling_stats: dict[str, Any],
    expected_profiles: list[str],
    target_names: list[str],
    expected_ddof: int,
) -> list[dict[str, Any]]:
    forecasts: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5f:
        h5_targets = _decode_attr_strings(h5f.attrs.get("target_names", []))
        if h5_targets != target_names:
            raise ValueError(f"target_names mismatch in {path}: expected {target_names}, got {h5_targets}")
        h5_ddof = int(h5f.attrs.get("ensemble_std_ddof", h5f.attrs.get("ensemble_ddof", 0)))
        if h5_ddof != int(expected_ddof):
            raise ValueError(f"ensemble ddof mismatch in {path}: expected {expected_ddof}, got {h5_ddof}")
        for profile_name in expected_profiles:
            if profile_name not in h5f:
                raise ValueError(f"Forecast file {path} is missing profile group {profile_name}.")
            group = h5f[profile_name]
            y_scaled = np.asarray(_dataset_alias(group, "y_true_scaled"), dtype=np.float32)
            members = np.asarray(_dataset_alias(group, "member_predictions_scaled"), dtype=np.float32)
            mean_scaled = np.asarray(_dataset_alias(group, "ensemble_mean_scaled", "mean_scaled"), dtype=np.float32)
            spread_scaled = np.asarray(_dataset_alias(group, "ensemble_std_scaled", "spread_scaled"), dtype=np.float32)
            if members.ndim != 3:
                raise ValueError(f"member_predictions_scaled for {profile_name} must have shape (members, steps, targets).")
            if mean_scaled.shape != y_scaled.shape or spread_scaled.shape != y_scaled.shape:
                raise ValueError(f"Forecast shape mismatch for {profile_name} in {path}.")
            if members.shape[1:] != y_scaled.shape:
                raise ValueError(f"Member forecast shape mismatch for {profile_name} in {path}.")
            if y_scaled.shape[1] != len(target_names):
                raise ValueError(f"Target count mismatch for {profile_name} in {path}.")
            for name, array in {
                "y_true_scaled": y_scaled,
                "member_predictions_scaled": members,
                "mean_scaled": mean_scaled,
                "spread_scaled": spread_scaled,
            }.items():
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} for {profile_name} in {path} contains non-finite values.")
            if np.any(spread_scaled < 0.0):
                raise ValueError(f"spread_scaled for {profile_name} in {path} contains negative values.")
            t = np.asarray(_dataset_alias(group, "t"), dtype=np.float32)
            u = np.asarray(_dataset_alias(group, "control", "u"), dtype=np.float32)
            y_true = (
                np.asarray(group["y_true_physical"][()], dtype=np.float32)
                if "y_true_physical" in group
                else _descale_targets_from_stats(scaling_stats, y_scaled).astype(np.float32)
            )
            y_pred = (
                np.asarray(group["ensemble_mean_physical"][()], dtype=np.float32)
                if "ensemble_mean_physical" in group
                else _descale_targets_from_stats(scaling_stats, mean_scaled).astype(np.float32)
            )
            forecasts.append(
                {
                    "profile": str(profile_name),
                    "t": t,
                    "u": u,
                    "y_true_scaled": y_scaled,
                    "member_predictions_scaled": members,
                    "mean_scaled": mean_scaled,
                    "spread_scaled": spread_scaled,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
    return forecasts


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
    for key in (
        "q_by_target",
        "q_by_horizon_target",
        "quantile_index_by_target",
        "score_count_by_target",
        "calibration_count_by_horizon",
        "quantile_index_by_horizon_target",
        "sigma_floor",
    ):
        if key in calibration:
            payload[key] = _json_safe(calibration[key])
    scores_path = method_dir / "calibration_scores.h5"
    with h5py.File(scores_path, "w") as h5f:
        for key in ("calibration_scores_by_profile_target",):
            if key in calibration:
                h5f.create_dataset(key, data=calibration[key], compression="gzip", chunks=True)
    payload["calibration_scores_h5"] = str(scores_path)
    return payload


def _run_uq_from_saved_forecasts(args: LSTMConformalRunConfig, *, config_path: Path | None = None) -> None:
    scaled_h5_path = _ensure_file(args.scaled_h5_path, "scaled_h5_path")
    cal_forecasts_path = _ensure_file(
        args.calibration_ensemble_forecasts_audit_h5_path,
        "calibration_ensemble_forecasts_audit_h5_path",
    )
    test_forecasts_path = _ensure_file(
        args.test_ensemble_forecasts_audit_h5_path,
        "test_ensemble_forecasts_audit_h5_path",
    )
    manifests = _load_manifests(args)
    cal_profiles = manifests["cal"]
    test_profiles = manifests["test"]
    _validate_profile_set(cal_forecasts_path, _profile_names_from_h5(cal_forecasts_path), cal_profiles, "calibration")
    _validate_profile_set(test_forecasts_path, _profile_names_from_h5(test_forecasts_path), test_profiles, "test")

    target_names, member_count, h5_ddof = _validate_forecast_pair(cal_forecasts_path, test_forecasts_path)
    if h5_ddof != int(args.ensemble_ddof):
        raise ValueError(
            f"Config ensemble_ddof={args.ensemble_ddof} does not match saved forecast ddof={h5_ddof}."
        )
    scaling_stats = _load_scaling_stats(scaled_h5_path)
    cal_shared = _load_audit_ensemble_forecasts(
        cal_forecasts_path,
        scaling_stats=scaling_stats,
        expected_profiles=cal_profiles,
        target_names=target_names,
        expected_ddof=args.ensemble_ddof,
    )
    test_shared = _load_audit_ensemble_forecasts(
        test_forecasts_path,
        scaling_stats=scaling_stats,
        expected_profiles=test_profiles,
        target_names=target_names,
        expected_ddof=args.ensemble_ddof,
    )

    methods = list(DEFAULT_UQ_METHODS if args.uq_methods is None else args.uq_methods)
    comparison_dir = Path(args.uq_output_dir) / "uq_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_shared_ensemble_predictions_hdf5(
        cal_shared,
        output_path=comparison_dir / "calibration_ensemble_predictions.h5",
        target_names=target_names,
        ensemble_ddof=args.ensemble_ddof,
    )
    save_shared_ensemble_predictions_hdf5(
        test_shared,
        output_path=comparison_dir / "test_ensemble_predictions.h5",
        target_names=target_names,
        ensemble_ddof=args.ensemble_ddof,
    )

    shared_metadata = {
        "ensemble_source": "saved_audit_forecasts",
        "scaled_h5_path": str(scaled_h5_path),
        "calibration_ensemble_forecasts_audit_h5_path": str(cal_forecasts_path),
        "test_ensemble_forecasts_audit_h5_path": str(test_forecasts_path),
        "target_names": target_names,
        "train_profile_names": manifests["train"],
        "val_profile_names": manifests["val"],
        "calibration_profile_names": cal_profiles,
        "test_profile_names": test_profiles,
        "alpha": float(args.alpha),
        "nominal_coverage": 1.0 - float(args.alpha),
        "sigma_floor": _json_safe(np.asarray(args.sigma_floor)),
        "ensemble_ddof": int(args.ensemble_ddof),
        "ensemble_member_count": int(member_count),
        "config_path": None if config_path is None else str(config_path),
        "reused_saved_ensemble_forecasts": True,
    }
    (comparison_dir / "shared_ensemble_metadata.json").write_text(json.dumps(_json_safe(shared_metadata), indent=2))

    manifest_methods: list[dict[str, Any]] = []
    for method_id in methods:
        info = UQ_METHODS[method_id]
        method_dir = comparison_dir / method_id
        method_dir.mkdir(parents=True, exist_ok=True)
        calibration = _method_calibration_result(method_id, cal_shared, args)
        cal_entries = [
            apply_uq_method(
                forecast,
                method_id=method_id,
                calibration_result=calibration,
                scaling_stats=scaling_stats,
                alpha=args.alpha,
                sigma_floor=np.asarray(args.sigma_floor),
                include_member_predictions=args.save_member_forecasts,
            )
            for forecast in cal_shared
        ]
        test_entries = [
            apply_uq_method(
                forecast,
                method_id=method_id,
                calibration_result=calibration,
                scaling_stats=scaling_stats,
                alpha=args.alpha,
                sigma_floor=np.asarray(args.sigma_floor),
                include_member_predictions=args.save_member_forecasts,
            )
            for forecast in test_shared
        ]
        h5_metadata = {
            "method_id": method_id,
            "method_label": info["label"],
            "alpha": float(args.alpha),
            "nominal_coverage": 1.0 - float(args.alpha),
            "ensemble_member_count": int(member_count),
            "ensemble_ddof": int(args.ensemble_ddof),
            "residual_space": "scaled",
            "temporal_calibration_mode": info["temporal_mode"],
            "uses_ensemble_normalization": bool(info["uses_ensemble_normalization"]),
        }
        calibration_forecasts_path = method_dir / "calibration_forecasts.h5"
        test_method_forecasts_path = method_dir / "test_forecasts.h5"
        save_uq_forecasts_hdf5(cal_entries, output_path=calibration_forecasts_path, metadata=h5_metadata, target_names=target_names)
        save_uq_forecasts_hdf5(test_entries, output_path=test_method_forecasts_path, metadata=h5_metadata, target_names=target_names)
        metrics = compute_uq_coverage_metrics(
            test_entries,
            target_names,
            alpha=args.alpha,
            primary_coverage_type=info["primary_coverage_type"],
            no_conformal_guarantee=(method_id == "raw_ensemble_2sigma"),
        )
        metrics_path = method_dir / "coverage_metrics.json"
        metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2))
        metadata_name = "method_metadata.json" if method_id == "raw_ensemble_2sigma" else "calibration_metadata.json"
        metadata_path = method_dir / metadata_name
        metadata = _calibration_metadata_json(calibration, method_id=method_id, method_dir=method_dir)
        metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2))
        manifest_methods.append(
            {
                "method_id": method_id,
                "method_label": info["label"],
                "temporal_mode": info["temporal_mode"],
                "residual_type": info["residual_type"],
                "normalized": bool(info["uses_ensemble_normalization"]),
                "calibration_metadata_path": str(metadata_path),
                "calibration_forecasts_path": str(calibration_forecasts_path),
                "test_forecasts_path": str(test_method_forecasts_path),
                "metrics_path": str(metrics_path),
            }
        )
        print(
            f"{method_id}: primary coverage={metrics['primary_empirical_coverage']:.4f}, "
            f"mean width={metrics['mean_interval_width_overall']:.6g}"
        )

    manifest = {**shared_metadata, "experiment": Path(args.uq_output_dir).name, "uq_methods": manifest_methods}
    manifest_path = comparison_dir / "uq_methods_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2))
    run_metadata = {"config_path": None if config_path is None else str(config_path), "config": args.__dict__, "uq_methods_manifest": str(manifest_path)}
    (Path(args.uq_output_dir) / "conformal_uq_from_saved_forecasts_metadata.json").write_text(
        json.dumps(_json_safe(run_metadata), indent=2)
    )
    print(f"Saved UQ comparison manifest to: {manifest_path}")


def main() -> None:
    cli_args = parse_args()
    args = _load_cfg(cli_args.config)
    Path(args.uq_output_dir).mkdir(parents=True, exist_ok=True)
    _run_uq_from_saved_forecasts(args, config_path=cli_args.config)


if __name__ == "__main__":
    main()
