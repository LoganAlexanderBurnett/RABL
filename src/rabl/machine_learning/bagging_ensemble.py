from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .lstm_pipeline import (
    STATE_DIM,
    TARGET_NAMES,
    ProfileDataset,
    SampleDataset,
    _count_samples_in_split,
    _descale_feature_from_stats,
    _descale_targets_from_stats,
    _extract_control_series,
    _get_profile_names,
    _get_profile_shapes,
    _load_scaling_stats,
    rolling_forecast,
    train_with_fallback,
)


@dataclass(frozen=True)
class BaggingEnsembleConfig:
    n_models: int
    overlap: float = 0.70
    seed: int = 123
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0
    learning_rate: float = 1e-3
    step_lr_step_size: int = 30
    step_lr_gamma: float = 0.5
    n_lstm: int = 1
    lstm_hidden: int = 64
    lstm_dropout: float = 0.0
    n_fc: int = 1
    fc_hidden: tuple[int, ...] = (64,)
    prefer_gpu: bool = True
    verbose: int = 1

    def validate(self) -> None:
        if self.n_models < 1:
            raise ValueError("n_models must be >= 1.")
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("overlap must be in [0.0, 1.0).")


def _copy_group_shallow(src: h5py.Group, dst_parent: h5py.Group, name: str) -> h5py.Group:
    dst = dst_parent.create_group(name)
    for attr_key, attr_value in src.attrs.items():
        dst.attrs[attr_key] = attr_value
    return dst


def _copy_profile_group(src_profile_group: h5py.Group, dst_profile_group: h5py.Group, sample_limit: int | None = None) -> None:
    for attr_key, attr_value in src_profile_group.attrs.items():
        dst_profile_group.attrs[attr_key] = attr_value

    x_data = src_profile_group["X"]
    y_data = src_profile_group["Y"]
    limit = x_data.shape[0] if sample_limit is None else int(sample_limit)

    if limit < 1:
        raise ValueError("sample_limit must be >= 1 when provided.")
    if limit > x_data.shape[0]:
        raise ValueError("sample_limit cannot exceed number of samples in profile.")

    if limit == x_data.shape[0]:
        src_profile_group.copy("X", dst_profile_group, name="X")
        src_profile_group.copy("Y", dst_profile_group, name="Y")
    else:
        x_ds = dst_profile_group.create_dataset("X", data=x_data[:limit], compression=x_data.compression)
        y_ds = dst_profile_group.create_dataset("Y", data=y_data[:limit], compression=y_data.compression)
        for attr_key, attr_value in x_data.attrs.items():
            x_ds.attrs[attr_key] = attr_value
        for attr_key, attr_value in y_data.attrs.items():
            y_ds.attrs[attr_key] = attr_value
        dst_profile_group.attrs["truncated_samples"] = limit


def _select_profiles_for_bag(
    profile_names: list[str],
    *,
    bag_profile_count: int,
    rng: np.random.Generator,
    shared_pool: list[str],
) -> list[str]:
    """Select unique profile names for one bag (no within-bag replacement)."""
    if bag_profile_count < 1:
        raise ValueError("bag_profile_count must be >= 1.")
    if bag_profile_count > len(profile_names):
        raise ValueError("bag_profile_count cannot exceed total number of train profiles.")

    shared_candidates = list(dict.fromkeys(shared_pool))
    non_shared_candidates = [name for name in profile_names if name not in set(shared_candidates)]

    selected: list[str] = []

    shared_take = min(len(shared_candidates), bag_profile_count)
    if shared_take > 0:
        shared_idx = rng.choice(len(shared_candidates), size=shared_take, replace=False)
        selected.extend(shared_candidates[int(i)] for i in np.atleast_1d(shared_idx))

    remaining = bag_profile_count - len(selected)
    if remaining > 0:
        if remaining > len(non_shared_candidates):
            raise RuntimeError("Unable to complete bag without replacement: insufficient non-shared profiles.")
        extra_idx = rng.choice(len(non_shared_candidates), size=remaining, replace=False)
        selected.extend(non_shared_candidates[int(i)] for i in np.atleast_1d(extra_idx))

    return selected


def create_bagged_training_hdf5(
    input_h5_path: Path,
    output_h5_path: Path,
    *,
    n_models: int,
    overlap: float = 0.70,
    seed: int = 123,
) -> Path:
    """
    Create an HDF5 with train/bag_i subsets and copied val/test/scaling groups.

    - Non-train groups are copied byte-for-byte via h5py copy.
    - Each bag samples train profiles without replacement.
    - `overlap` sets the shared profile fraction across bags; bag size is
      `round(num_train_profiles * (1 - overlap))` (minimum 1 profile).
    """
    if n_models < 1:
        raise ValueError("n_models must be >= 1.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")

    input_h5_path = Path(input_h5_path)
    output_h5_path = Path(output_h5_path)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    with h5py.File(input_h5_path, "r") as src, h5py.File(output_h5_path, "w") as dst:
        for attr_key, attr_value in src.attrs.items():
            dst.attrs[attr_key] = attr_value
        dst.attrs["bagging_n_models"] = n_models
        dst.attrs["bagging_overlap"] = overlap
        dst.attrs["bagging_seed"] = seed

        for group_name in ("val", "test", "scaling"):
            if group_name in src:
                src.copy(group_name, dst)

        train_src = src["train"]
        train_files_src = train_src["files"]
        profile_names = sorted(train_files_src.keys())
        if not profile_names:
            raise ValueError("No profiles found under train/files in source HDF5.")

        profile_sample_counts = {name: int(train_files_src[name]["X"].shape[0]) for name in profile_names}
        total_train_samples = int(sum(profile_sample_counts.values()))

        num_profiles = len(profile_names)
        shared_pool_size = max(1, int(round(num_profiles * overlap)))
        shared_pool = list(rng.choice(profile_names, size=shared_pool_size, replace=False))
        bag_profile_count = max(1, int(round(num_profiles * (1.0 - overlap))))

        train_dst = _copy_group_shallow(train_src, dst, "train")
        train_dst.attrs["bagging_sampling"] = "profile_no_replacement_with_overlap"
        train_dst.attrs["bagging_total_train_samples"] = total_train_samples
        train_dst.attrs["bagging_total_train_profiles"] = num_profiles
        train_dst.attrs["bagging_shared_profile_pool_size"] = shared_pool_size
        train_dst.attrs["bagging_profiles_per_bag"] = bag_profile_count

        for bag_idx in range(n_models):
            bag_group = train_dst.create_group(f"bag_{bag_idx}")
            bag_group.attrs["bag_index"] = bag_idx
            files_group = bag_group.create_group("files")

            selected_profiles = _select_profiles_for_bag(
                profile_names=profile_names,
                bag_profile_count=bag_profile_count,
                rng=rng,
                shared_pool=shared_pool,
            )

            used_names: set[str] = set()
            samples_written = 0
            for profile_name in selected_profiles:
                if profile_name in used_names:
                    raise RuntimeError(f"Duplicate profile '{profile_name}' encountered within bag {bag_idx}.")
                used_names.add(profile_name)

                src_profile = train_files_src[profile_name]
                dst_profile = files_group.create_group(profile_name)
                _copy_profile_group(src_profile, dst_profile)

                samples_written += profile_sample_counts[profile_name]

            bag_group.attrs["num_profile_draws"] = len(selected_profiles)
            bag_group.attrs["num_unique_source_profiles"] = len(used_names)
            bag_group.attrs["num_samples"] = samples_written

    return output_h5_path


def build_datasets_for_train_split(
    h5_path: Path,
    *,
    train_split: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    train_profiles = _get_profile_names(h5_path, train_split)
    val_profiles = _get_profile_names(h5_path, "val")
    test_profiles = _get_profile_names(h5_path, "test")

    if not train_profiles:
        raise ValueError(f"No training profiles found in HDF5 split '{train_split}'.")

    x_shape, y_shape = _get_profile_shapes(h5_path, train_split, train_profiles[0])
    train_num_samples = _count_samples_in_split(h5_path, train_split, train_profiles)
    val_num_samples = _count_samples_in_split(h5_path, "val", val_profiles)
    test_num_samples = _count_samples_in_split(h5_path, "test", test_profiles)

    train_steps = max(1, ceil(train_num_samples / batch_size))
    val_steps = max(1, ceil(val_num_samples / batch_size))

    train_ds = SampleDataset(h5_path, train_profiles, train_split, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True)

    val_sample_ds = SampleDataset(h5_path, val_profiles, "val")
    val_sample_loader = DataLoader(val_sample_ds, batch_size=batch_size, pin_memory=True)

    val_profile_ds = ProfileDataset(h5_path, val_profiles, "val")
    test_profile_ds = ProfileDataset(h5_path, test_profiles, "test")

    return {
        "train": train_loader,
        "val_samples": val_sample_loader,
        "val_profile_ds": val_profile_ds,
        "test_profile_ds": test_profile_ds,
        "train_profile_names": train_profiles,
        "val_profile_names": val_profiles,
        "test_profile_names": test_profiles,
        "sample_shape": x_shape,
        "target_shape": y_shape,
        "batch_size": batch_size,
        "seed": seed,
        "train_num_samples": train_num_samples,
        "val_num_samples": val_num_samples,
        "test_num_samples": test_num_samples,
        "train_steps_per_epoch": train_steps,
        "val_steps": val_steps,
        "h5_path": Path(h5_path),
    }


def _save_ensemble_rolling_forecasts_hdf5(
    forecasts: list[dict[str, Any]],
    *,
    output_path: Path,
    target_names: list[str],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    column_names = (
        ["t", "u(t)"]
        + [f"x_true(t)_{name}" for name in target_names]
        + [f"x_mean(t)_{name}" for name in target_names]
        + [f"x_2sigma(t)_{name}" for name in target_names]
    )
    column_attr = np.array(column_names, dtype="S")

    with h5py.File(output_path, "w") as h5f:
        for entry in forecasts:
            group = h5f.create_group(entry["profile"])
            group.create_dataset("data", data=entry["table"].astype(np.float32))
            group.attrs["columns"] = column_attr


def ensemble_rolling_forecast_and_save(
    models: list[torch.nn.Module],
    profile_ds: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    h5_path: Path,
    output_path: Path,
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
    target_names: list[str] | None = None,
) -> None:
    if target_names is None:
        target_names = list(TARGET_NAMES)

    scaling_stats = _load_scaling_stats(h5_path)
    forecasts: list[dict[str, Any]] = []

    for profile_name, x_profile, y_profile in profile_ds:
        x_np = x_profile.numpy()
        y_true = _descale_targets_from_stats(scaling_stats, y_profile.numpy())

        per_model_preds: list[np.ndarray] = []
        for model in models:
            y_pred = rolling_forecast(model, x_np, state_dim=state_dim)
            y_pred = _descale_targets_from_stats(scaling_stats, y_pred)
            per_model_preds.append(y_pred)

        pred_stack = np.stack(per_model_preds, axis=0)
        y_mean = np.mean(pred_stack, axis=0)
        y_two_sigma = 2.0 * np.std(pred_stack, axis=0, ddof=0)

        t_series = np.arange(y_mean.shape[0], dtype=np.float32)
        u_series = _extract_control_series(x_np, state_dim=state_dim, control_channel=control_channel)
        control_idx = state_dim + control_channel
        u_series = _descale_feature_from_stats(scaling_stats, u_series, control_idx)

        table = np.column_stack([t_series, u_series, y_true, y_mean, y_two_sigma]).astype(np.float32)
        forecasts.append({"profile": str(profile_name), "table": table})

    _save_ensemble_rolling_forecasts_hdf5(
        forecasts,
        output_path=output_path,
        target_names=target_names,
    )


def _decode_columns(columns_attr: np.ndarray | list[Any]) -> list[str]:
    decoded: list[str] = []
    for item in columns_attr:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def plot_ensemble_forecast_profile_grid(
    forecast_h5_path: Path,
    *,
    profile_name: str,
    save_path: Path | None = None,
    control_name: str = "drumAngleDeg",
    target_names: list[str] | None = None,
    close_figure: bool = True,
) -> plt.Figure:
    """
    Plot one ensemble forecast profile in a 2x7 grid.

    Grid layout mirrors the existing pipeline visualization style:
      - subplot [0,0]: control variable u(t)
      - remaining 13 subplots: state targets with ground truth, mean prediction,
        and mean ± 2sigma uncertainty bounds.
    """
    if target_names is None:
        target_names = list(TARGET_NAMES)

    forecast_h5_path = Path(forecast_h5_path)
    with h5py.File(forecast_h5_path, "r") as h5f:
        if profile_name not in h5f:
            raise KeyError(f"Profile '{profile_name}' not found in {forecast_h5_path}.")
        group = h5f[profile_name]
        table = group["data"][...].astype(np.float32)
        columns = _decode_columns(group.attrs.get("columns", []))

    if table.ndim != 2:
        raise ValueError(f"Expected 2D forecast table, got shape {table.shape}.")

    try:
        t_idx = columns.index("t")
        u_idx = columns.index("u(t)")
    except ValueError as exc:
        raise ValueError("Forecast HDF5 columns are missing required 't' or 'u(t)' fields.") from exc

    t_series = table[:, t_idx]
    u_series = table[:, u_idx]

    y_true = np.column_stack([table[:, columns.index(f"x_true(t)_{name}")] for name in target_names])
    y_mean = np.column_stack([table[:, columns.index(f"x_mean(t)_{name}")] for name in target_names])
    y_2sigma = np.column_stack([table[:, columns.index(f"x_2sigma(t)_{name}")] for name in target_names])
    y_upper = y_mean + y_2sigma
    y_lower = y_mean - y_2sigma

    fig, axes = plt.subplots(2, 7, figsize=(26, 8), sharex=True)
    axes_flat = axes.flatten()

    axes_flat[0].plot(t_series, u_series, linewidth=1.5, color="black")
    axes_flat[0].set_title(control_name)
    axes_flat[0].grid(True, alpha=0.3)

    for target_idx, target_name in enumerate(target_names):
        ax = axes_flat[target_idx + 1]
        ax.plot(t_series, y_true[:, target_idx], label="Ground truth", linewidth=1.6, color="C0")
        ax.plot(t_series, y_mean[:, target_idx], label="Mean prediction", linewidth=1.6, color="C3")
        ax.plot(t_series, y_upper[:, target_idx], linestyle="--", linewidth=1.0, color="C1", label="Mean + 2σ")
        ax.plot(t_series, y_lower[:, target_idx], linestyle="--", linewidth=1.0, color="C2", label="Mean - 2σ")
        ax.fill_between(
            t_series,
            y_lower[:, target_idx],
            y_upper[:, target_idx],
            color="C1",
            alpha=0.15,
            linewidth=0,
        )
        ax.set_title(target_name)
        ax.grid(True, alpha=0.3)

    for idx in range(7, 14):
        axes_flat[idx].set_xlabel("Time step")
    axes_flat[0].set_ylabel("u(t)")
    for idx in range(1, 14):
        axes_flat[idx].set_ylabel("State")

    handles, labels = axes_flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Ensemble Rolling Forecast - {profile_name}", y=1.06, fontsize=16)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if close_figure:
        plt.close(fig)

    return fig


def run_bagging_ensemble(
    scaled_h5_path: Path,
    *,
    out_dir: Path,
    n_models: int,
    overlap: float = 0.70,
    seed: int = 123,
    batch_size: int = 64,
    epochs: int = 100,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 0.0,
    learning_rate: float = 1e-3,
    step_lr_step_size: int = 30,
    step_lr_gamma: float = 0.5,
    n_lstm: int = 1,
    lstm_hidden: int = 64,
    lstm_dropout: float = 0.0,
    n_fc: int = 1,
    fc_hidden: tuple[int, ...] = (64,),
    prefer_gpu: bool = True,
    verbose: int = 1,
) -> dict[str, Any]:
    config = BaggingEnsembleConfig(
        n_models=n_models,
        overlap=overlap,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        learning_rate=learning_rate,
        step_lr_step_size=step_lr_step_size,
        step_lr_gamma=step_lr_gamma,
        n_lstm=n_lstm,
        lstm_hidden=lstm_hidden,
        lstm_dropout=lstm_dropout,
        n_fc=n_fc,
        fc_hidden=fc_hidden,
        prefer_gpu=prefer_gpu,
        verbose=verbose,
    )
    config.validate()

    scaled_h5_path = Path(scaled_h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bagged_h5_path = out_dir / "bagged_dataset.h5"
    create_bagged_training_hdf5(
        scaled_h5_path,
        bagged_h5_path,
        n_models=config.n_models,
        overlap=config.overlap,
        seed=config.seed,
    )

    models: list[torch.nn.Module] = []
    histories: list[dict[str, list[float]]] = []
    used_devices: list[str] = []

    for model_idx in range(config.n_models):
        model_dir = out_dir / f"model_{model_idx}"
        model_dir.mkdir(parents=True, exist_ok=True)

        datasets = build_datasets_for_train_split(
            bagged_h5_path,
            train_split=f"train/bag_{model_idx}",
            batch_size=config.batch_size,
            seed=config.seed,
        )

        model, history, used_device = train_with_fallback(
            datasets,
            epochs=config.epochs,
            out_dir=model_dir,
            n_lstm=config.n_lstm,
            lstm_hidden=config.lstm_hidden,
            lstm_dropout=config.lstm_dropout,
            n_fc=config.n_fc,
            fc_hidden=config.fc_hidden,
            learning_rate=config.learning_rate,
            step_lr_step_size=config.step_lr_step_size,
            step_lr_gamma=config.step_lr_gamma,
            verbose=config.verbose,
            prefer_gpu=config.prefer_gpu,
            preload_train_to_device=True,
            deterministic_seed=config.seed,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            restore_best_weights=True,
        )

        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        models.append(model)
        histories.append(history)
        used_devices.append(str(used_device))

    forecast_output_path = out_dir / "rolling_forecasts.h5"
    test_profile_ds = ProfileDataset(bagged_h5_path, _get_profile_names(bagged_h5_path, "test"), "test")
    ensemble_rolling_forecast_and_save(
        models,
        test_profile_ds,
        h5_path=bagged_h5_path,
        output_path=forecast_output_path,
        state_dim=STATE_DIM,
        control_channel=0,
        target_names=list(TARGET_NAMES),
    )

    return {
        "bagged_h5_path": bagged_h5_path,
        "forecast_output_path": forecast_output_path,
        "model_dirs": [out_dir / f"model_{idx}" for idx in range(config.n_models)],
        "used_devices": used_devices,
        "histories": histories,
    }
