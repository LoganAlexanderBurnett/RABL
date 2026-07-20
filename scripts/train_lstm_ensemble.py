"""Train and preserve a reusable profile-bagged LSTM ensemble."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from evaluate_testset_difficulty import evaluate_testset_difficulty
from rabl.machine_learning import build_lstm_dataset
from rabl.machine_learning.bagging_ensemble import (
    ensemble_member_predictions_scaled,
    ensemble_rolling_forecast_and_save,
    plot_ensemble_forecast_profile_grid,
    run_bagging_ensemble,
)
from rabl.machine_learning.dataset_scaling import LSTMDatasetScalerSplitter
from rabl.machine_learning.lstm_pipeline import (
    STATE_DIM,
    TARGET_NAMES,
    ProfileDataset,
    _descale_feature_from_stats,
    _descale_targets_from_stats,
    _load_scaling_stats,
    build_datasets,
)


@dataclass(frozen=True)
class LSTMEnsembleTrainingConfig:
    experiment_id: str
    master_seed: int
    sim_root: str
    config_py_path: str
    output_root: str
    lookback: int
    train_manifest_path: str
    val_manifest_path: str
    cal_manifest_path: str
    test_manifest_path: str
    expected_train_profiles: int
    expected_val_profiles: int
    expected_cal_profiles: int
    expected_test_profiles: int
    n_models: int = 5
    bag_fraction: float = 0.7
    split_mode: str = "profile"
    bag_split_mode: str = "profile"
    scaling_type: str = "standard"
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_lstm: int = 1
    hidden_lstm: int = 64
    lstm_dropout: float = 0.0
    n_fc: int = 1
    hidden_fc: list[int] | tuple[int, ...] = (64,)
    step_lr_step_size: int = 30
    step_lr_gamma: float = 0.5
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = True
    preload_train_to_device: bool = True
    preload_val_to_device: bool = True
    require_gpu: bool = False
    ensemble_use_tqdm: bool = True
    ensemble_forecast_num_workers: int = 4
    test_difficulty_bins: int = 10
    test_difficulty_num_workers: int = 4
    forecast_plot_bins: int = 5
    forecast_plot_profiles_per_bin: int = 2
    forecast_plot_selection_seed: int | None = None
    forecast_plot_metric: str = "scaled_mae"
    resume: bool = False
    overwrite: bool = False

    def validate(self) -> None:
        if self.split_mode != "profile":
            raise ValueError("train_lstm_ensemble requires split_mode='profile'.")
        if self.bag_split_mode != "profile":
            raise ValueError("train_lstm_ensemble requires bag_split_mode='profile'.")
        if self.n_models < 1:
            raise ValueError("n_models must be >= 1.")
        if len(tuple(self.hidden_fc)) != self.n_fc:
            raise ValueError("hidden_fc must contain exactly n_fc values.")
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive.")


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


def _load_cfg(path: Path) -> LSTMEnsembleTrainingConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    cfg = LSTMEnsembleTrainingConfig(**data)
    cfg.validate()
    return cfg


def _manifest_profiles(path: Path, key: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get(key) if isinstance(data, dict) else data
    if not isinstance(profiles, list):
        raise ValueError(f"Manifest {path} must be a list or contain list field {key!r}.")
    names = [str(name) for name in profiles]
    if not names:
        raise ValueError(f"Manifest {path} contains no profiles.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Manifest {path} contains duplicate profiles: {duplicates[:10]}")
    return names


def _load_and_validate_manifests(cfg: LSTMEnsembleTrainingConfig) -> dict[str, list[str]]:
    manifests = {
        "train": _manifest_profiles(Path(cfg.train_manifest_path), "train_profiles"),
        "val": _manifest_profiles(Path(cfg.val_manifest_path), "val_profiles"),
        "cal": _manifest_profiles(Path(cfg.cal_manifest_path), "cal_profiles"),
        "test": _manifest_profiles(Path(cfg.test_manifest_path), "test_profiles"),
    }
    expected = {
        "train": cfg.expected_train_profiles,
        "val": cfg.expected_val_profiles,
        "cal": cfg.expected_cal_profiles,
        "test": cfg.expected_test_profiles,
    }
    for split, names in manifests.items():
        if len(names) != int(expected[split]):
            raise ValueError(f"{split} manifest count mismatch: expected {expected[split]}, got {len(names)}.")
    for left, left_names in manifests.items():
        for right, right_names in manifests.items():
            if left < right:
                overlap = sorted(set(left_names).intersection(right_names))
                if overlap:
                    raise ValueError(f"Manifest profile leakage between {left} and {right}: {overlap[:10]}")
    return {split: sorted(names) for split, names in manifests.items()}


def _discover_profile_csvs(sim_root: Path) -> dict[str, Path]:
    profile_paths: dict[str, Path] = {}
    for batch_dir in sorted(sim_root.glob("batch_????")):
        if not batch_dir.is_dir():
            continue
        for csv_path in sorted(batch_dir.glob(build_lstm_dataset.CSV_PATTERN)):
            if csv_path.stem in profile_paths:
                raise ValueError(f"Duplicate profile stem across batch directories: {csv_path.stem}")
            profile_paths[csv_path.stem] = csv_path
    return profile_paths


def _batch_ids_for_profiles(sim_root: Path, profile_names: list[str]) -> list[str]:
    csvs = _discover_profile_csvs(sim_root)
    missing = sorted(set(profile_names) - set(csvs))
    if missing:
        raise FileNotFoundError(f"Manifest profiles not found under {sim_root}: {missing[:10]}")
    batch_ids = sorted({csvs[name].parent.name.removeprefix("batch_") for name in profile_names})
    if not batch_ids:
        raise ValueError("No batch IDs resolved from manifests.")
    return batch_ids


def _filter_unscaled_dataset(source_h5: Path, output_h5: Path, profile_names: list[str]) -> Path:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(profile_names)
    with h5py.File(source_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        if "files" not in src:
            raise ValueError(f"Unscaled HDF5 is missing files group: {source_h5}")
        missing = sorted(wanted - set(src["files"].keys()))
        if missing:
            raise ValueError(f"Built unscaled HDF5 is missing manifest profiles: {missing[:10]}")
        files = dst.create_group("files")
        for name in sorted(wanted):
            src["files"].copy(name, files)
    return output_h5


def _write_split_manifests(manifests: dict[str, list[str]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, names in manifests.items():
        path = out_dir / f"{split}_manifest.json"
        path.write_text(json.dumps({f"{split}_profiles": sorted(names)}, indent=2), encoding="utf-8")
        paths[split] = path
    return paths


def _save_scaling_stats_json(scaled_h5: Path, output_path: Path) -> None:
    output_path.write_text(json.dumps(_json_safe(_load_scaling_stats(scaled_h5)), indent=2), encoding="utf-8")


def _seed_manifest(master_seed: int, n_models: int, selection_seed: int | None) -> dict[str, Any]:
    sequence = np.random.SeedSequence(int(master_seed))
    children = sequence.spawn(n_models + 3)
    member_seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children[:n_models]]
    if len(set(member_seeds)) != len(member_seeds):
        raise ValueError("Derived duplicate member seeds; choose a different master_seed.")
    bag_seed = int(children[n_models].generate_state(1, dtype=np.uint32)[0])
    derived_selection_seed = int(children[n_models + 1].generate_state(1, dtype=np.uint32)[0])
    misc_seed = int(children[n_models + 2].generate_state(1, dtype=np.uint32)[0])
    return {
        "master_seed": int(master_seed),
        "bag_generation_seed": bag_seed,
        "member_training_seeds": member_seeds,
        "forecast_profile_selection_seed": int(selection_seed) if selection_seed is not None else derived_selection_seed,
        "misc_seed": misc_seed,
    }


def _save_audit_ensemble_forecasts(
    models: list[torch.nn.Module],
    profile_ds: ProfileDataset,
    *,
    scaled_h5_path: Path,
    output_path: Path,
    target_names: list[str],
) -> None:
    scaling_stats = _load_scaling_stats(scaled_h5_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["target_names"] = np.asarray(target_names, dtype="S")
        h5f.attrs["member_count"] = len(models)
        h5f.attrs["ensemble_std_ddof"] = 0
        for profile_name, x_tensor, y_tensor in profile_ds:
            x_np = x_tensor.numpy()
            y_scaled = y_tensor.numpy().astype(np.float32)
            member_scaled, mean_scaled, std_scaled = ensemble_member_predictions_scaled(models, x_np, state_dim=STATE_DIM)
            y_true = _descale_targets_from_stats(scaling_stats, y_scaled)
            mean_physical = _descale_targets_from_stats(scaling_stats, mean_scaled)
            member_physical = _descale_targets_from_stats(scaling_stats, member_scaled)
            two_sigma_physical = 2.0 * np.std(member_physical, axis=0, ddof=0)
            control_scaled = x_np[:, -1, STATE_DIM]
            control = _descale_feature_from_stats(scaling_stats, control_scaled, STATE_DIM)
            grp = h5f.create_group(str(profile_name))
            for name, array in {
                "t": np.arange(y_scaled.shape[0], dtype=np.float32),
                "control": control.astype(np.float32),
                "y_true_scaled": y_scaled,
                "member_predictions_scaled": member_scaled.astype(np.float32),
                "ensemble_mean_scaled": mean_scaled.astype(np.float32),
                "ensemble_std_scaled": std_scaled.astype(np.float32),
                "y_true_physical": y_true.astype(np.float32),
                "ensemble_mean_physical": mean_physical.astype(np.float32),
                "ensemble_lower_2sigma_physical": (mean_physical - two_sigma_physical).astype(np.float32),
                "ensemble_upper_2sigma_physical": (mean_physical + two_sigma_physical).astype(np.float32),
            }.items():
                grp.create_dataset(name, data=array, compression="gzip", chunks=True)


def _ensemble_metrics_from_audit_h5(path: Path, target_names: list[str]) -> dict[str, Any]:
    abs_errs = []
    sq_errs = []
    coverage = []
    widths = []
    spreads = []
    member_mae = []
    with h5py.File(path, "r") as h5f:
        for profile_name in h5f.keys():
            grp = h5f[profile_name]
            truth = grp["y_true_scaled"][...]
            mean = grp["ensemble_mean_scaled"][...]
            std = grp["ensemble_std_scaled"][...]
            members = grp["member_predictions_scaled"][...]
            abs_err = np.abs(truth - mean)
            abs_errs.append(abs_err)
            sq_errs.append((truth - mean) ** 2)
            coverage.append((truth >= mean - 2.0 * std) & (truth <= mean + 2.0 * std))
            widths.append(4.0 * std)
            spreads.append(std)
            member_mae.append(np.mean(np.abs(members - truth[None, :, :]), axis=(1, 2)))
    abs_all = np.concatenate([x.reshape(-1, x.shape[-1]) for x in abs_errs], axis=0)
    sq_all = np.concatenate([x.reshape(-1, x.shape[-1]) for x in sq_errs], axis=0)
    cov_all = np.concatenate([x.reshape(-1, x.shape[-1]) for x in coverage], axis=0)
    width_all = np.concatenate([x.reshape(-1, x.shape[-1]) for x in widths], axis=0)
    spread_all = np.concatenate([x.reshape(-1, x.shape[-1]) for x in spreads], axis=0)
    corr_by_target = {}
    for idx, name in enumerate(target_names):
        corr_by_target[name] = float(np.corrcoef(spread_all[:, idx], abs_all[:, idx])[0, 1]) if np.std(spread_all[:, idx]) > 0 and np.std(abs_all[:, idx]) > 0 else float("nan")
    return {
        "scaled_mae_overall": float(np.mean(abs_all)),
        "scaled_rmse_overall": float(np.sqrt(np.mean(sq_all))),
        "scaled_mae_by_target": {name: float(val) for name, val in zip(target_names, np.mean(abs_all, axis=0))},
        "scaled_rmse_by_target": {name: float(val) for name, val in zip(target_names, np.sqrt(np.mean(sq_all, axis=0)))},
        "raw_2sigma_marginal_coverage_overall": float(np.mean(cov_all)),
        "raw_2sigma_marginal_coverage_by_target": {name: float(val) for name, val in zip(target_names, np.mean(cov_all, axis=0))},
        "mean_interval_width_by_target": {name: float(val) for name, val in zip(target_names, np.mean(width_all, axis=0))},
        "mean_interval_width_by_horizon": np.mean(np.stack(widths, axis=0), axis=(0, 2)).tolist(),
        "spread_abs_error_correlation_by_target": corr_by_target,
        "member_scaled_mae": np.concatenate(member_mae).reshape(-1, len(member_mae[0])).mean(axis=0).tolist(),
        "ensemble_diversity_mean_std": float(np.mean(spread_all)),
    }


def _select_profiles_for_plots(difficulty_csv: Path, *, metric: str, n_bins: int, per_bin: int, seed: int) -> list[str]:
    import csv

    rows = []
    with difficulty_csv.open(newline="") as fp:
        for row in csv.DictReader(fp):
            rows.append(row)
    if not rows:
        return []
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    edges = np.quantile(values, np.linspace(0.0, 1.0, int(n_bins) + 1))
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    manifest_rows = []
    for idx in range(int(n_bins)):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (values >= lo) & (values <= hi if idx == int(n_bins) - 1 else values < hi)
        candidates = [rows[i] for i in np.flatnonzero(mask)]
        if not candidates:
            continue
        take = min(int(per_bin), len(candidates))
        chosen = rng.choice(len(candidates), size=take, replace=False)
        for chosen_idx in chosen:
            profile = str(candidates[int(chosen_idx)]["profile_id"])
            selected.append(profile)
            manifest_rows.append({"bin_index": idx, "profile_id": profile, metric: candidates[int(chosen_idx)][metric]})
    return selected


def run(cfg: LSTMEnsembleTrainingConfig) -> Path:
    if cfg.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("require_gpu=true but CUDA is unavailable.")
    experiment_dir = Path(cfg.output_root) / cfg.experiment_id
    if experiment_dir.exists() and cfg.overwrite:
        shutil.rmtree(experiment_dir)
    if experiment_dir.exists() and not cfg.resume and any(experiment_dir.iterdir()):
        raise FileExistsError(f"Experiment directory exists; set resume=true or overwrite=true: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "resolved_config.json").write_text(json.dumps(_json_safe(cfg.__dict__), indent=2), encoding="utf-8")

    manifests = _load_and_validate_manifests(cfg)
    manifest_dir = experiment_dir / "manifests"
    split_manifest_paths = _write_split_manifests(manifests, manifest_dir)
    all_profiles = sorted({name for names in manifests.values() for name in names})
    batch_ids = _batch_ids_for_profiles(Path(cfg.sim_root), all_profiles)

    seeds = _seed_manifest(cfg.master_seed, cfg.n_models, cfg.forecast_plot_selection_seed)
    (experiment_dir / "seed_manifest.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    reproducibility = {"batch_ids": batch_ids, "manifest_profile_count": len(all_profiles)}
    (experiment_dir / "reproducibility_metadata.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")

    build_config = build_lstm_dataset._validate_config(build_lstm_dataset._load_config(Path(cfg.config_py_path)))
    full_unscaled = build_lstm_dataset.build_dataset(
        Path(cfg.sim_root), experiment_dir / "datasets", build_config["steady_state"], cfg.lookback, batch_ids,
        output_name="unscaled_dataset_all_resolved_batches.h5", verbose=False,
    )
    unscaled_h5 = _filter_unscaled_dataset(full_unscaled, experiment_dir / "datasets" / "unscaled_dataset.h5", all_profiles)
    scaled_h5 = LSTMDatasetScalerSplitter(
        input_path=unscaled_h5,
        scaling_type=cfg.scaling_type,
        train_frac=0.98,
        val_frac=0.01,
        cal_frac=0.0,
        test_frac=0.01,
        output_dir=experiment_dir / "datasets",
        output_name="scaled_dataset.h5",
        seed=cfg.master_seed,
        split_mode="profile",
        test_manifest_path=split_manifest_paths["test"],
        val_manifest_path=split_manifest_paths["val"],
        cal_manifest_path=split_manifest_paths["cal"],
    ).run()
    _save_scaling_stats_json(scaled_h5, experiment_dir / "scaling_statistics.json")
    datasets = build_datasets(scaled_h5, cfg.batch_size, cfg.master_seed)
    if datasets["train_profile_names"] != manifests["train"]:
        raise ValueError("Resolved train split does not exactly match train manifest after filtering/scaling.")

    ensemble_dir = experiment_dir / "ensemble"
    result = run_bagging_ensemble(
        scaled_h5,
        out_dir=ensemble_dir,
        n_models=cfg.n_models,
        bag_fraction=cfg.bag_fraction,
        bag_split_mode="profile",
        seed=int(seeds["bag_generation_seed"]),
        member_seeds=seeds["member_training_seeds"],
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_min_delta=cfg.early_stopping_min_delta,
        learning_rate=cfg.learning_rate,
        step_lr_step_size=cfg.step_lr_step_size,
        step_lr_gamma=cfg.step_lr_gamma,
        n_lstm=cfg.n_lstm,
        lstm_hidden=cfg.hidden_lstm,
        lstm_dropout=cfg.lstm_dropout,
        n_fc=cfg.n_fc,
        fc_hidden=tuple(cfg.hidden_fc),
        prefer_gpu=True,
        preload_train_to_device=cfg.preload_train_to_device,
        preload_val_to_device=cfg.preload_val_to_device,
        restore_best_weights=cfg.restore_best_weights,
        resume=cfg.resume,
        use_tqdm=cfg.ensemble_use_tqdm,
        forecast_num_workers=cfg.ensemble_forecast_num_workers,
        save_member_forecasts=True,
    )
    models = result["models"]
    target_names = list(TARGET_NAMES[:int(datasets["target_shape"][1])])
    forecasts_dir = experiment_dir / "forecasts"
    cal_ds = ProfileDataset(scaled_h5, datasets["cal_profile_names"], "cal")
    test_ds = ProfileDataset(scaled_h5, datasets["test_profile_names"], "test")
    ensemble_rolling_forecast_and_save(models, cal_ds, h5_path=scaled_h5, output_path=forecasts_dir / "calibration_ensemble_forecasts.h5", target_names=target_names, num_workers=cfg.ensemble_forecast_num_workers, save_member_forecasts=True)
    shutil.copy2(result["forecast_output_path"], forecasts_dir / "test_ensemble_forecasts.h5")
    _save_audit_ensemble_forecasts(models, cal_ds, scaled_h5_path=scaled_h5, output_path=forecasts_dir / "calibration_ensemble_forecasts_audit.h5", target_names=target_names)
    _save_audit_ensemble_forecasts(models, test_ds, scaled_h5_path=scaled_h5, output_path=forecasts_dir / "test_ensemble_forecasts_audit.h5", target_names=target_names)

    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics = {
        "calibration": _ensemble_metrics_from_audit_h5(forecasts_dir / "calibration_ensemble_forecasts_audit.h5", target_names),
        "test": _ensemble_metrics_from_audit_h5(forecasts_dir / "test_ensemble_forecasts_audit.h5", target_names),
    }
    (metrics_dir / "ensemble_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")

    difficulty_dir = experiment_dir / "test_difficulty"
    evaluate_testset_difficulty(scaled_h5=scaled_h5, out_dir=difficulty_dir / "no_target_overlays", forecast_h5=forecasts_dir / "test_ensemble_forecasts.h5", n_bins=cfg.test_difficulty_bins, config_path=Path(cfg.config_py_path), include_per_target=False, num_workers=cfg.test_difficulty_num_workers)
    evaluate_testset_difficulty(scaled_h5=scaled_h5, out_dir=difficulty_dir / "with_target_overlays", forecast_h5=forecasts_dir / "test_ensemble_forecasts.h5", n_bins=cfg.test_difficulty_bins, config_path=Path(cfg.config_py_path), include_per_target=True, num_workers=cfg.test_difficulty_num_workers)

    selected = _select_profiles_for_plots(
        difficulty_dir / "no_target_overlays" / "per_profile_metrics_and_difficulty.csv",
        metric=cfg.forecast_plot_metric,
        n_bins=cfg.forecast_plot_bins,
        per_bin=cfg.forecast_plot_profiles_per_bin,
        seed=int(seeds["forecast_profile_selection_seed"]),
    )
    plots_dir = experiment_dir / "forecast_plots"
    plots_dir.mkdir(exist_ok=True)
    (plots_dir / "selected_forecast_profiles.json").write_text(json.dumps({"profiles": selected, "metric": cfg.forecast_plot_metric}, indent=2), encoding="utf-8")
    for profile_name in selected:
        plot_ensemble_forecast_profile_grid(
            forecasts_dir / "test_ensemble_forecasts.h5",
            profile_name=profile_name,
            save_path=plots_dir / f"ensemble_forecast_{profile_name}.png",
            target_names=target_names,
        )
    print(f"Saved reusable LSTM ensemble experiment to: {experiment_dir}")
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and preserve a reusable profile-bagged LSTM ensemble.")
    parser.add_argument("--config", type=Path, required=True, help="Path to ensemble-training JSON config.")
    run(_load_cfg(parser.parse_args().config))


if __name__ == "__main__":
    main()
