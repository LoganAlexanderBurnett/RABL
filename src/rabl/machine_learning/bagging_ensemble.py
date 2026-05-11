from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import importlib.util
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from .branchpoint_finder import finite_difference
from torch.utils.data import DataLoader

from .lstm_pipeline import (
    FORECAST_PLOT_TARGET_ORDER,
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
    bag_fraction: float = 0.70
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
        if not (0.0 < self.bag_fraction <= 1.0):
            raise ValueError("bag_fraction must be in (0.0, 1.0].")


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


def _plot_and_save_bag_venn_diagram(
    bag_sets: list[set[str]],
    *,
    save_path: Path,
    title: str,
    set_labels: tuple[str, str, str] = ("bag_0", "bag_1", "bag_2"),
    weights: dict[str, int] | None = None,
) -> Path | None:
    """Plot and save a Venn diagram for 3 bag sets when supported."""
    if len(bag_sets) != 3:
        return None
    if importlib.util.find_spec("matplotlib_venn") is None:
        return None

    from matplotlib_venn import venn3

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    subsets = _venn_region_counts(bag_sets, weights=weights)

    fig, _ = plt.subplots(figsize=(7, 7))
    venn3(subsets=subsets, set_labels=set_labels)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _venn_region_counts(
    bag_sets: list[set[str]],
    *,
    weights: dict[str, int] | None = None,
) -> dict[str, int]:
    """Return venn3 subset counts for three sets.

    If ``weights`` is provided, each item contributes ``weights[item]`` to its region.
    Otherwise each item contributes 1.
    """
    if len(bag_sets) != 3:
        raise ValueError("venn_region_counts requires exactly three sets.")

    a, b, c = bag_sets
    weighted = weights is not None

    def _sum_items(items: set[str]) -> int:
        if not weighted:
            return int(len(items))
        missing = [item for item in items if item not in weights]
        if missing:
            raise KeyError(f"Missing weights for venn items: {missing[:5]}")
        return int(sum(int(weights[item]) for item in items))

    return {
        "100": _sum_items(a - b - c),
        "010": _sum_items(b - a - c),
        "001": _sum_items(c - a - b),
        "110": _sum_items((a & b) - c),
        "101": _sum_items((a & c) - b),
        "011": _sum_items((b & c) - a),
        "111": _sum_items(a & b & c),
    }


def create_bagged_training_hdf5(
    input_h5_path: Path,
    output_h5_path: Path,
    *,
    n_models: int,
    bag_fraction: float = 0.70,
    seed: int = 123,
    verbose: int = 1,
) -> Path:
    """
    Create an HDF5 with train/bag_i subsets and copied val/test/scaling groups.

    - Non-train groups are copied byte-for-byte via h5py copy.
    - Each bag samples train profiles without replacement.
    - `bag_fraction` controls profiles per bag as
      `round(num_train_profiles * bag_fraction)` (minimum 1 profile).
    """
    if n_models < 1:
        raise ValueError("n_models must be >= 1.")
    if not (0.0 < bag_fraction <= 1.0):
        raise ValueError("bag_fraction must be in (0.0, 1.0].")

    input_h5_path = Path(input_h5_path)
    output_h5_path = Path(output_h5_path)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    with h5py.File(input_h5_path, "r") as src, h5py.File(output_h5_path, "w") as dst:
        for attr_key, attr_value in src.attrs.items():
            dst.attrs[attr_key] = attr_value
        dst.attrs["bagging_n_models"] = n_models
        dst.attrs["bagging_bag_fraction"] = bag_fraction
        dst.attrs["bagging_seed"] = seed

        for group_name in ("val", "test", "scaling"):
            if group_name in src:
                src.copy(group_name, dst)

        train_src = src["train"]
        train_files_src = train_src["files"]
        train_profile_names = sorted(train_files_src.keys())
        if not train_profile_names:
            raise ValueError("No profiles found under train/files in source HDF5.")

        profile_sample_counts = {name: int(train_files_src[name]["X"].shape[0]) for name in train_profile_names}
        total_train_samples = int(sum(profile_sample_counts.values()))

        num_train_profiles = len(train_profile_names)
        bag_profile_count = max(1, int(round(num_train_profiles * bag_fraction)))

        train_dst = _copy_group_shallow(train_src, dst, "train")
        train_dst.attrs["bagging_sampling"] = "profile_no_replacement_subsample_bagging"
        train_dst.attrs["bagging_total_train_samples"] = total_train_samples
        train_dst.attrs["bagging_total_train_profiles"] = num_train_profiles
        train_dst.attrs["bagging_profiles_per_bag"] = bag_profile_count

        if verbose >= 1:
            print(
                "[bagging] configured "
                f"n_models={n_models}, bag_fraction={bag_fraction:.3f}, total_train_profiles={num_train_profiles}, "
                f"total_train_samples={total_train_samples}, "
                f"profiles_per_bag={bag_profile_count}"
            )

        bag_profile_lists = [
            rng.choice(
                train_profile_names,
                size=bag_profile_count,
                replace=False,
            ).tolist()
            for _ in range(n_models)
        ]
        bag_profile_sets = [set(selected_profiles) for selected_profiles in bag_profile_lists]

        venn_profile_plot_path: Path | None = None
        venn_sample_plot_path: Path | None = None
        if n_models == 3:
            venn_profile_plot_path = _plot_and_save_bag_venn_diagram(
                bag_profile_sets,
                save_path=output_h5_path.with_name(f"{output_h5_path.stem}_bag_overlap_profile_venn.png"),
                title="Bag profile overlap (3 estimators)",
            )
            venn_sample_plot_path = _plot_and_save_bag_venn_diagram(
                bag_profile_sets,
                save_path=output_h5_path.with_name(f"{output_h5_path.stem}_bag_overlap_sample_venn.png"),
                title="Bag sample overlap (3 estimators)",
                weights=profile_sample_counts,
            )
            if verbose >= 1:
                if venn_profile_plot_path is None or venn_sample_plot_path is None:
                    print("[bagging] venn diagram not created (requires matplotlib_venn).")
                else:
                    print(f"[bagging] saved profile-overlap venn diagram: {venn_profile_plot_path}")
                    print(f"[bagging] saved sample-overlap venn diagram: {venn_sample_plot_path}")

        shared_profiles_all_bags = set.intersection(*bag_profile_sets) if bag_profile_sets else set()
        profile_frequency: dict[str, int] = {}
        for bag_set in bag_profile_sets:
            for profile_name in bag_set:
                profile_frequency[profile_name] = profile_frequency.get(profile_name, 0) + 1

        for bag_idx, selected_profiles in enumerate(bag_profile_lists):
            bag_group = train_dst.create_group(f"bag_{bag_idx}")
            bag_group.attrs["bag_index"] = bag_idx
            files_group = bag_group.create_group("files")

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

            shared_profiles_in_bag = len(used_names.intersection(shared_profiles_all_bags))
            bag_only_profiles = sum(1 for profile_name in used_names if profile_frequency[profile_name] == 1)

            if verbose >= 1:
                print(
                    f"[bagging] bag_{bag_idx}: profiles={len(selected_profiles)} "
                    f"unique_profiles={len(used_names)} samples={samples_written} "
                    f"shared_across_all_bags={shared_profiles_in_bag} "
                    f"bag_only_profiles={bag_only_profiles}"
                )

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
    with h5py.File(output_path, "w") as h5f:
        for entry in forecasts:
            column_names = (
                ["t", "u(t)"]
                + [f"x_true(t)_{name}" for name in target_names]
                + [f"x_mean(t)_{name}" for name in target_names]
                + [f"x_2sigma(t)_{name}" for name in target_names]
            )
            if "dx_sigma_dt" in entry:
                column_names += [f"x_dsigma_dt(t)_{name}" for name in target_names]
            column_attr = np.array(column_names, dtype="S")

            group = h5f.create_group(entry["profile"])
            table = entry["table"].astype(np.float32)
            if "dx_sigma_dt" in entry:
                table = np.column_stack([table, entry["dx_sigma_dt"].astype(np.float32)]).astype(np.float32)
            group.create_dataset("data", data=table)
            group.attrs["columns"] = column_attr


def _forecast_ensemble_profile(
    profile_name: str,
    x_profile: torch.Tensor,
    y_profile: torch.Tensor,
    *,
    models: list[torch.nn.Module],
    scaling_stats: dict[str, Any],
    state_dim: int,
    control_channel: int,
    derivative_order: int | None,
    derivative_dt: float,
) -> dict[str, Any]:
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
    entry: dict[str, Any] = {"profile": str(profile_name), "table": table}
    if derivative_order is not None:
        entry["dx_sigma_dt"] = finite_difference(y_two_sigma, order=derivative_order, dt=derivative_dt)
    return entry


def ensemble_rolling_forecast_and_save(
    models: list[torch.nn.Module],
    profile_ds: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    h5_path: Path,
    output_path: Path,
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
    target_names: list[str] | None = None,
    derivative_order: int | None = None,
    derivative_dt: float = 1.0,
    num_workers: int = 4,
) -> None:
    if target_names is None:
        target_names = list(TARGET_NAMES)

    scaling_stats = _load_scaling_stats(h5_path)
    workers = max(1, int(num_workers))
    indexed_forecasts: list[tuple[int, dict[str, Any]]] = []

    def _run_one(
        index: int,
        profile_name: str,
        x_profile: torch.Tensor,
        y_profile: torch.Tensor,
    ) -> tuple[int, dict[str, Any]]:
        return index, _forecast_ensemble_profile(
            profile_name,
            x_profile,
            y_profile,
            models=models,
            scaling_stats=scaling_stats,
            state_dim=state_dim,
            control_channel=control_channel,
            derivative_order=derivative_order,
            derivative_dt=derivative_dt,
        )

    if workers <= 1:
        for index, (profile_name, x_profile, y_profile) in enumerate(profile_ds):
            indexed_forecasts.append(_run_one(index, profile_name, x_profile, y_profile))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_one, index, profile_name, x_profile, y_profile)
                for index, (profile_name, x_profile, y_profile) in enumerate(profile_ds)
            ]
            for future in as_completed(futures):
                indexed_forecasts.append(future.result())

    forecasts = [entry for _index, entry in sorted(indexed_forecasts, key=lambda item: item[0])]
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


def _reorder_targets_for_plot(
    target_names: list[str],
    *arrays: np.ndarray | None,
) -> tuple[list[str], list[np.ndarray | None]]:
    index_by_name = {name: idx for idx, name in enumerate(target_names)}
    ordered_names = [name for name in FORECAST_PLOT_TARGET_ORDER if name in index_by_name]
    ordered_indices = [index_by_name[name] for name in ordered_names]
    reordered: list[np.ndarray | None] = []
    for arr in arrays:
        if arr is None:
            reordered.append(None)
        else:
            reordered.append(arr[:, ordered_indices])
    return ordered_names, reordered


def plot_ensemble_forecast_profile_grid(
    forecast_h5_path: Path,
    *,
    profile_name: str,
    save_path: Path | None = None,
    control_name: str = "drumAngleDeg",
    target_names: list[str] | None = None,
    plot_uncertainty_derivative: bool = True,
    close_figure: bool = True,
) -> plt.Figure:
    """
    Plot one ensemble forecast profile in a control+target grid.

    Grid layout mirrors the existing pipeline visualization style:
      - subplot [0,0]: control variable u(t)
      - remaining target subplots: state targets with ground truth, mean prediction,
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
    derivative_cols = [f"x_dsigma_dt(t)_{name}" for name in target_names]
    has_derivative = all(col in columns for col in derivative_cols)
    y_dsigma_dt = (
        np.column_stack([table[:, columns.index(col)] for col in derivative_cols])
        if (plot_uncertainty_derivative and has_derivative)
        else None
    )
    target_names, reordered = _reorder_targets_for_plot(target_names, y_true, y_mean, y_2sigma, y_dsigma_dt)
    y_true, y_mean, y_2sigma, y_dsigma_dt = reordered

    y_upper = y_mean + y_2sigma
    y_lower = y_mean - y_2sigma

    nplots = len(target_names) + 1
    rows, cols = 4, 4
    if nplots > rows * cols:
        raise ValueError(f"Plot requires {nplots} panels but 4x4 supports only 16.")
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()

    axes_flat[0].plot(t_series, u_series, linewidth=1.5, color="black")
    axes_flat[0].set_title(control_name)
    axes_flat[0].grid(True, alpha=0.3)

    derivative_axes: list[plt.Axes] = []

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
        if y_dsigma_dt is not None:
            ax2 = ax.twinx()
            derivative_axes.append(ax2)
            ax2.plot(
                t_series,
                y_dsigma_dt[:, target_idx],
                linewidth=1.2,
                color="C4",
                linestyle=":",
                label="d(x_sigma_scaled)/dt",
            )
            ax2.axhline(0.0, color="0.5", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.set_title(target_name)
        if target_name == "x_steam_out":
            formatter = ax.yaxis.get_major_formatter()
            if hasattr(formatter, "set_useOffset"):
                formatter.set_useOffset(False)
        ax.grid(True, alpha=0.3)

    for idx in range(nplots - cols, nplots):
        axes_flat[idx].set_xlabel("Time (s)")
    axes_flat[0].set_ylabel(control_name)
    for idx, target_name in enumerate(target_names, start=1):
        axes_flat[idx].set_ylabel(target_name)

    handles, labels = axes_flat[1].get_legend_handles_labels()
    if y_dsigma_dt is not None and derivative_axes:
        h2, l2 = derivative_axes[0].get_legend_handles_labels()
        handles = handles + h2
        labels = labels + l2
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
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
    bag_fraction: float = 0.70,
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
    preload_val_to_device: bool = True,
    verbose: int = 1,
    forecast_num_workers: int = 4,
) -> dict[str, Any]:
    config = BaggingEnsembleConfig(
        n_models=n_models,
        bag_fraction=bag_fraction,
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
        bag_fraction=config.bag_fraction,
        seed=config.seed,
        verbose=config.verbose,
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
            preload_val_to_device=preload_val_to_device,
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
        num_workers=forecast_num_workers,
    )

    return {
        "bagged_h5_path": bagged_h5_path,
        "forecast_output_path": forecast_output_path,
        "model_dirs": [out_dir / f"model_{idx}" for idx in range(config.n_models)],
        "used_devices": used_devices,
        "histories": histories,
    }
