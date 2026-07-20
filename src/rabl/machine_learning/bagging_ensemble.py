from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import importlib.util
import json
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
    build_model,
)


@dataclass(frozen=True)
class BaggingEnsembleConfig:
    n_models: int
    bag_fraction: float = 0.70
    bag_split_mode: str = "profile"
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
    use_tqdm: bool = True
    verbose: int = 1
    plot_bag_distributions: bool = True

    def validate(self) -> None:
        if self.n_models < 1:
            raise ValueError("n_models must be >= 1.")
        if not (0.0 < self.bag_fraction <= 1.0):
            raise ValueError("bag_fraction must be in (0.0, 1.0].")
        if self.bag_split_mode not in {"profile", "sample"}:
            raise ValueError("bag_split_mode must be 'profile' or 'sample'.")


def _copy_group_shallow(src: h5py.Group, dst_parent: h5py.Group, name: str) -> h5py.Group:
    dst = dst_parent.create_group(name)
    for attr_key, attr_value in src.attrs.items():
        dst.attrs[attr_key] = attr_value
    return dst


def _copy_profile_group(
    src_profile_group: h5py.Group,
    dst_profile_group: h5py.Group,
    sample_indices: np.ndarray | list[int] | None = None,
) -> None:
    for attr_key, attr_value in src_profile_group.attrs.items():
        dst_profile_group.attrs[attr_key] = attr_value

    x_data = src_profile_group["X"]
    y_data = src_profile_group["Y"]
    if sample_indices is None:
        src_profile_group.copy("X", dst_profile_group, name="X")
        src_profile_group.copy("Y", dst_profile_group, name="Y")
        return

    indices = np.asarray(sample_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("sample_indices must be a non-empty 1D sequence when provided.")
    if np.any(indices < 0) or np.any(indices >= x_data.shape[0]):
        raise ValueError("sample_indices contain out-of-range values for profile.")
    indices = np.unique(indices)

    x_ds = dst_profile_group.create_dataset("X", data=x_data[indices], compression=x_data.compression)
    y_ds = dst_profile_group.create_dataset("Y", data=y_data[indices], compression=y_data.compression)
    for attr_key, attr_value in x_data.attrs.items():
        x_ds.attrs[attr_key] = attr_value
    for attr_key, attr_value in y_data.attrs.items():
        y_ds.attrs[attr_key] = attr_value
    dst_profile_group.attrs["num_samples"] = int(indices.size)
    dst_profile_group.attrs["selected_sample_count"] = int(indices.size)
    dst_profile_group.create_dataset("selected_sample_indices", data=indices, compression="gzip")


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



BAG_DISTRIBUTION_COLUMNS = {
    "theta": "drumAngleDeg",
    "rho": "rho_dollars",
    "n": "n",
    "t": "t",
}


def _descale_target_channel(stats: dict[str, Any], values: np.ndarray, target_idx: int) -> np.ndarray:
    scaling_type = stats["type"]
    y_stats = stats["y"]
    if scaling_type == "standard":
        return values * y_stats["std"][target_idx] + y_stats["mean"][target_idx]
    if scaling_type == "minmax":
        return values * y_stats["span"][target_idx] + y_stats["min"][target_idx]
    raise ValueError(f"Unsupported scaling type: {scaling_type}")


def _collect_bag_distribution_values(
    bag_group: h5py.Group,
    *,
    scaling_stats: dict[str, Any],
    state_dim: int,
    control_channel: int,
    target_names: list[str],
    lookback: int,
) -> dict[str, np.ndarray]:
    rho_idx = target_names.index("rho_dollars")
    n_idx = target_names.index("n")
    files_group = bag_group["files"]
    raw_values: dict[str, list[np.ndarray]] = {key: [] for key in BAG_DISTRIBUTION_COLUMNS}

    for profile_name in sorted(files_group.keys()):
        profile_group = files_group[profile_name]
        x_data = profile_group["X"][...].astype(np.float32)
        y_data = profile_group["Y"][...].astype(np.float32)
        if x_data.ndim != 3:
            raise ValueError(f"Expected 3D X for {bag_group.name}/{profile_name}; got {x_data.shape}.")
        if y_data.ndim != 2:
            raise ValueError(f"Expected 2D Y for {bag_group.name}/{profile_name}; got {y_data.shape}.")
        if y_data.shape[1] <= max(rho_idx, n_idx):
            raise ValueError(
                f"Y target dimension for {bag_group.name}/{profile_name} is too small: {y_data.shape}."
            )
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                f"Mismatched X/Y sample counts for {bag_group.name}/{profile_name}: "
                f"X={x_data.shape[0]}, Y={y_data.shape[0]}."
            )
        control_dim = x_data.shape[2] - state_dim
        if control_dim <= 0:
            raise ValueError(
                f"control_dim <= 0 for {bag_group.name}/{profile_name} "
                f"(features={x_data.shape[2]}, state_dim={state_dim})."
            )
        if not (0 <= control_channel < control_dim):
            raise ValueError(f"control_channel={control_channel} out of range [0, {control_dim - 1}].")
        control_idx = state_dim + control_channel
        theta = _descale_feature_from_stats(scaling_stats, x_data[:, -1, control_idx], control_idx)
        rho = _descale_target_channel(scaling_stats, y_data[:, rho_idx], rho_idx)
        n_values = _descale_target_channel(scaling_stats, y_data[:, n_idx], n_idx)
        t_values = _read_profile_target_times(profile_group, sample_count=x_data.shape[0], lookback=lookback)
        raw_values["theta"].append(theta.astype(np.float32, copy=False))
        raw_values["rho"].append(rho.astype(np.float32, copy=False))
        raw_values["n"].append(n_values.astype(np.float32, copy=False))
        if t_values.size:
            raw_values["t"].append(t_values.astype(np.float32, copy=False))

    concatenated = {
        key: np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)
        for key, arrays in raw_values.items()
    }
    finite_values = {
        key: array[np.isfinite(array)]
        for key, array in concatenated.items()
    }
    return finite_values


def _read_profile_target_times(profile_group: h5py.Group, *, sample_count: int, lookback: int) -> np.ndarray:
    source_file = str(profile_group.attrs.get("source_file", "")).strip()
    if not source_file:
        return np.asarray([], dtype=np.float32)
    source_path = Path(source_file)
    if not source_path.exists():
        return np.asarray([], dtype=np.float32)

    try:
        with source_path.open(newline="") as fp:
            import csv

            reader = csv.DictReader(fp)
            if reader.fieldnames is None or "t" not in reader.fieldnames:
                return np.asarray([], dtype=np.float32)
            times = np.asarray([float(row["t"]) for row in reader], dtype=np.float32)
    except OSError:
        return np.asarray([], dtype=np.float32)
    if times.size == 0:
        return times

    history_rows = max(0, int(profile_group.attrs.get("history_rows_prepended", lookback + 1)))
    target_indices = np.arange(sample_count, dtype=np.int64) + int(lookback) + 1 - history_rows
    if "selected_sample_indices" in profile_group:
        target_indices = profile_group["selected_sample_indices"][...].astype(np.int64) + int(lookback) + 1 - history_rows
    valid = (target_indices >= 0) & (target_indices < times.size)
    return times[target_indices[valid]]


def _plot_bag_distribution_overlap(
    bagged_h5_path: Path,
    *,
    output_path: Path | None = None,
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
    target_names: list[str] | None = None,
) -> Path | None:
    """Plot theta/rho/n/time count overlap across bagged train splits."""
    if target_names is None:
        target_names = list(TARGET_NAMES)
    if "rho_dollars" not in target_names or "n" not in target_names:
        raise ValueError("target_names must include 'rho_dollars' and 'n'.")
    bagged_h5_path = Path(bagged_h5_path)
    if output_path is None:
        output_path = bagged_h5_path.with_name(
            f"{bagged_h5_path.stem}_bag_distribution_overlap_theta_rho_n_t.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scaling_stats = _load_scaling_stats(bagged_h5_path)
    with h5py.File(bagged_h5_path, "r") as h5f:
        train_group = h5f["train"]
        bag_names = sorted(
            (name for name in train_group.keys() if name.startswith("bag_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if not bag_names:
            return None
        lookback = int(h5f.attrs.get("k_lookback", 0))
        bag_values = {
            bag_name: _collect_bag_distribution_values(
                train_group[bag_name],
                scaling_stats=scaling_stats,
                state_dim=state_dim,
                control_channel=control_channel,
                target_names=target_names,
                lookback=lookback,
            )
            for bag_name in bag_names
        }

    if not any(any(values.size for values in bag_data.values()) for bag_data in bag_values.values()):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab10")
    for ax, (summary_name, label) in zip(axes, BAG_DISTRIBUTION_COLUMNS.items(), strict=True):
        all_values = [bag_data[summary_name] for bag_data in bag_values.values() if bag_data[summary_name].size]
        if not all_values:
            ax.set_axis_off()
            continue
        combined = np.concatenate(all_values)
        lo = float(np.min(combined))
        hi = float(np.max(combined))
        if np.isclose(lo, hi):
            pad = max(1.0, abs(lo) * 0.01)
            lo -= pad
            hi += pad
        if summary_name == "t":
            unique_t = np.unique(combined[np.isfinite(combined)])
            if unique_t.size >= 2:
                dt = float(np.median(np.diff(unique_t)))
                if not np.isfinite(dt) or dt <= 0.0:
                    dt = float((hi - lo) / max(1, unique_t.size - 1))
                bins = np.concatenate([[unique_t[0] - dt / 2.0], unique_t + dt / 2.0])
            else:
                bins = np.linspace(lo, hi, 80)
        else:
            bins = np.linspace(lo, hi, 80)
        for bag_idx, (bag_name, bag_data) in enumerate(bag_values.items()):
            values = bag_data[summary_name]
            if not values.size:
                continue
            ax.hist(
                values,
                bins=bins,
                density=False,
                histtype="step",
                linewidth=1.4,
                alpha=0.9,
                color=cmap(bag_idx % 10),
                label=f"{bag_name} (N={values.size:,})",
            )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(4, len(handles)),
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_ensemble_training_curves(histories: list[dict[str, list[float]]], output_path: Path) -> Path | None:
    histories = [history for history in histories if history.get("loss") and history.get("val_loss")]
    if not histories:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("tab10")
    for member_idx, history in enumerate(histories):
        color = cmap(member_idx % 10)
        train_loss = np.asarray(history["loss"], dtype=float)
        val_loss = np.asarray(history["val_loss"], dtype=float)
        train_epochs = np.arange(1, train_loss.size + 1)
        val_epochs = np.arange(1, val_loss.size + 1)
        ax.plot(
            train_epochs,
            train_loss,
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
            label=f"model_{member_idx} train",
        )
        ax.plot(
            val_epochs,
            val_loss,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            label=f"model_{member_idx} val",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Bagging ensemble training and validation loss")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
    bag_split_mode: str = "profile",
    seed: int = 123,
    verbose: int = 1,
) -> Path:
    """
    Create an HDF5 with train/bag_i subsets and copied val/test/scaling groups.

    - Non-train groups are copied byte-for-byte via h5py copy.
    - For ``bag_split_mode='profile'``, each bag samples train profiles without replacement.
    - For ``bag_split_mode='sample'``, each bag samples individual train samples without replacement.
    - `bag_fraction` controls profiles or samples per bag depending on ``bag_split_mode``.
    """
    if n_models < 1:
        raise ValueError("n_models must be >= 1.")
    if not (0.0 < bag_fraction <= 1.0):
        raise ValueError("bag_fraction must be in (0.0, 1.0].")
    if bag_split_mode not in {"profile", "sample"}:
        raise ValueError("bag_split_mode must be 'profile' or 'sample'.")

    input_h5_path = Path(input_h5_path)
    output_h5_path = Path(output_h5_path)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    with h5py.File(input_h5_path, "r") as src, h5py.File(output_h5_path, "w") as dst:
        for attr_key, attr_value in src.attrs.items():
            dst.attrs[attr_key] = attr_value
        dst.attrs["bagging_n_models"] = n_models
        dst.attrs["bagging_bag_fraction"] = bag_fraction
        dst.attrs["bagging_split_mode"] = bag_split_mode
        dst.attrs["bagging_seed"] = seed

        for group_name in ("val", "cal", "test", "scaling"):
            if group_name in src:
                src.copy(group_name, dst)

        train_src = src["train"]
        train_files_src = train_src["files"]
        train_profile_names = sorted(train_files_src.keys())
        if not train_profile_names:
            raise ValueError("No profiles found under train/files in source HDF5.")
        # Calibration profiles are copied for post-hoc UQ only and must never be bagged.
        if "cal" in src and "files" in src["cal"]:
            cal_names = set(src["cal"]["files"].keys())
            overlap = cal_names.intersection(train_profile_names)
            if overlap:
                raise ValueError(f"train/cal profile overlap is invalid for bagging: {sorted(overlap)}")

        profile_sample_counts = {name: int(train_files_src[name]["X"].shape[0]) for name in train_profile_names}
        total_train_samples = int(sum(profile_sample_counts.values()))

        num_train_profiles = len(train_profile_names)
        bag_profile_count = max(1, int(round(num_train_profiles * bag_fraction)))
        bag_sample_count = max(1, int(round(total_train_samples * bag_fraction)))

        train_dst = _copy_group_shallow(train_src, dst, "train")
        train_dst.attrs["bagging_sampling"] = (
            "profile_no_replacement_subsample_bagging"
            if bag_split_mode == "profile"
            else "sample_no_replacement_subsample_bagging"
        )
        train_dst.attrs["bagging_total_train_samples"] = total_train_samples
        train_dst.attrs["bagging_total_train_profiles"] = num_train_profiles
        train_dst.attrs["bagging_profiles_per_bag"] = bag_profile_count if bag_split_mode == "profile" else 0
        train_dst.attrs["bagging_samples_per_bag"] = bag_sample_count if bag_split_mode == "sample" else 0

        if verbose >= 1:
            print(
                "[bagging] configured "
                f"n_models={n_models}, bag_fraction={bag_fraction:.3f}, bag_split_mode={bag_split_mode}, "
                f"total_train_profiles={num_train_profiles}, total_train_samples={total_train_samples}, "
                f"profiles_per_bag={bag_profile_count if bag_split_mode == 'profile' else 'n/a'}, "
                f"samples_per_bag={bag_sample_count if bag_split_mode == 'sample' else 'n/a'}"
            )

        profile_offsets = np.cumsum([0] + [profile_sample_counts[name] for name in train_profile_names])

        bag_sample_indices_by_profile: list[dict[str, np.ndarray]] = []
        if bag_split_mode == "profile":
            bag_profile_lists = [
                rng.choice(
                    train_profile_names,
                    size=bag_profile_count,
                    replace=False,
                ).tolist()
                for _ in range(n_models)
            ]
            for selected_profiles in bag_profile_lists:
                bag_sample_indices_by_profile.append(
                    {
                        profile_name: np.arange(profile_sample_counts[profile_name], dtype=np.int64)
                        for profile_name in selected_profiles
                    }
                )
        else:
            bag_profile_lists = []
            for _ in range(n_models):
                selected_global_indices = np.sort(
                    rng.choice(total_train_samples, size=bag_sample_count, replace=False).astype(np.int64)
                )
                profile_indices = np.searchsorted(profile_offsets[1:], selected_global_indices, side="right")
                selected_by_profile: dict[str, list[int]] = {}
                for global_idx, profile_idx in zip(selected_global_indices, profile_indices, strict=False):
                    profile_name = train_profile_names[int(profile_idx)]
                    local_idx = int(global_idx - profile_offsets[int(profile_idx)])
                    selected_by_profile.setdefault(profile_name, []).append(local_idx)
                bag_profile_lists.append(sorted(selected_by_profile))
                bag_sample_indices_by_profile.append(
                    {
                        profile_name: np.asarray(indices, dtype=np.int64)
                        for profile_name, indices in selected_by_profile.items()
                    }
                )

        bag_profile_sets = [set(selected_profiles) for selected_profiles in bag_profile_lists]

        venn_profile_plot_path: Path | None = None
        venn_sample_plot_path: Path | None = None
        if n_models == 3:
            venn_profile_plot_path = _plot_and_save_bag_venn_diagram(
                bag_profile_sets,
                save_path=output_h5_path.with_name(f"{output_h5_path.stem}_bag_overlap_profile_venn.png"),
                title="Bag profile overlap (3 estimators)",
            )
            if bag_split_mode == "profile":
                venn_sample_plot_path = _plot_and_save_bag_venn_diagram(
                    bag_profile_sets,
                    save_path=output_h5_path.with_name(f"{output_h5_path.stem}_bag_overlap_sample_venn.png"),
                    title="Bag sample overlap (3 estimators)",
                    weights=profile_sample_counts,
                )
            if verbose >= 1:
                if venn_profile_plot_path is None:
                    print("[bagging] profile-overlap venn diagram not created (requires matplotlib_venn).")
                else:
                    print(f"[bagging] saved profile-overlap venn diagram: {venn_profile_plot_path}")
                if bag_split_mode == "profile" and venn_sample_plot_path is not None:
                    print(f"[bagging] saved sample-overlap venn diagram: {venn_sample_plot_path}")

        shared_profiles_all_bags = set.intersection(*bag_profile_sets) if bag_profile_sets else set()
        profile_frequency: dict[str, int] = {}
        for bag_set in bag_profile_sets:
            for profile_name in bag_set:
                profile_frequency[profile_name] = profile_frequency.get(profile_name, 0) + 1

        for bag_idx, selected_profiles in enumerate(bag_profile_lists):
            bag_group = train_dst.create_group(f"bag_{bag_idx}")
            bag_group.attrs["bag_index"] = bag_idx
            bag_group.attrs["bag_split_mode"] = bag_split_mode
            files_group = bag_group.create_group("files")

            used_names: set[str] = set()
            samples_written = 0
            selected_indices_by_profile = bag_sample_indices_by_profile[bag_idx]
            for profile_name in selected_profiles:
                if profile_name in used_names:
                    raise RuntimeError(f"Duplicate profile '{profile_name}' encountered within bag {bag_idx}.")
                used_names.add(profile_name)

                src_profile = train_files_src[profile_name]
                dst_profile = files_group.create_group(profile_name)
                indices = selected_indices_by_profile[profile_name]
                if bag_split_mode == "profile":
                    _copy_profile_group(src_profile, dst_profile)
                else:
                    _copy_profile_group(src_profile, dst_profile, sample_indices=indices)

                samples_written += int(indices.size)

            bag_group.attrs["num_profile_draws"] = len(selected_profiles)
            bag_group.attrs["num_unique_source_profiles"] = len(used_names)
            bag_group.attrs["num_sample_draws"] = samples_written
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
    save_member_forecasts: bool = True,
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
            if save_member_forecasts and "member_predictions" in entry:
                member_predictions = entry["member_predictions"].astype(np.float32)
                member_ds = group.create_dataset("member_predictions", data=member_predictions, compression="gzip")
                member_target_attr = np.array(target_names, dtype="S")
                member_count = int(member_predictions.shape[0])
                member_ds.attrs["target_names"] = member_target_attr
                member_ds.attrs["member_model_count"] = member_count
                group.attrs["member_target_names"] = member_target_attr
                group.attrs["member_model_count"] = member_count



def load_bagged_lstm_ensemble_checkpoints(
    model_paths: list[Path],
    *,
    timesteps: int,
    num_features: int,
    num_targets: int,
    n_lstm: int = 1,
    lstm_hidden: int = 64,
    lstm_dropout: float = 0.0,
    n_fc: int = 1,
    fc_hidden: tuple[int, ...] = (64,),
    device: str | torch.device = "cpu",
) -> list[torch.nn.Module]:
    """Load LSTM ensemble checkpoints with strict architecture validation."""
    if len(model_paths) < 2:
        raise ValueError("joint ensemble checkpoint mode requires at least two model checkpoints.")
    missing_paths = [Path(path) for path in model_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Model checkpoint path(s) not found: {missing_paths}")

    resolved_device = torch.device(device)
    bag_overlap_diagnostics = save_bag_overlap_diagnostics(bagged_h5_path, out_dir / "bag_diagnostics", n_models=config.n_models)

    models: list[torch.nn.Module] = []
    for model_path in model_paths:
        model = build_model(
            timesteps=timesteps,
            num_features=num_features,
            num_targets=num_targets,
            n_lstm=n_lstm,
            lstm_hidden=lstm_hidden,
            lstm_dropout=lstm_dropout,
            n_fc=n_fc,
            fc_hidden=fc_hidden,
        )
        state_dict = torch.load(Path(model_path), map_location=resolved_device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint architecture is incompatible with configured model: {model_path}") from exc
        model.to(resolved_device)
        model.eval()
        models.append(model)
    return models

def ensemble_member_predictions_scaled(
    models: list[torch.nn.Module],
    x_profile: np.ndarray,
    *,
    state_dim: int = STATE_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return autoregressive scaled member forecasts, mean, and spread.

    This is intentionally independent of HDF5/descaling so post-hoc uncertainty
    methods can calibrate in the same scaled space used by the LSTM.
    """
    if len(models) < 1:
        raise ValueError("At least one ensemble model is required.")
    x_np = np.asarray(x_profile, dtype=np.float32)
    predictions = [rolling_forecast(model, x_np, state_dim=state_dim) for model in models]
    stack = np.stack(predictions, axis=0).astype(np.float32, copy=False)
    if stack.ndim != 3:
        raise ValueError(f"Member prediction stack must be (models, steps, targets); got {stack.shape}.")
    if not np.all(np.isfinite(stack)):
        raise ValueError("Ensemble member predictions contain non-finite values.")
    return stack, np.mean(stack, axis=0), np.std(stack, axis=0, ddof=0)


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
    save_member_forecasts: bool = True,
) -> dict[str, Any]:
    x_np = x_profile.numpy()
    y_true = _descale_targets_from_stats(scaling_stats, y_profile.numpy())

    scaled_stack, _scaled_mean, _scaled_std = ensemble_member_predictions_scaled(
        models, x_np, state_dim=state_dim
    )
    pred_stack = _descale_targets_from_stats(scaling_stats, scaled_stack)
    y_mean = np.mean(pred_stack, axis=0)
    y_two_sigma = 2.0 * np.std(pred_stack, axis=0, ddof=0)

    t_series = np.arange(y_mean.shape[0], dtype=np.float32)
    u_series = _extract_control_series(x_np, state_dim=state_dim, control_channel=control_channel)
    control_idx = state_dim + control_channel
    u_series = _descale_feature_from_stats(scaling_stats, u_series, control_idx)

    table = np.column_stack([t_series, u_series, y_true, y_mean, y_two_sigma]).astype(np.float32)
    entry: dict[str, Any] = {"profile": str(profile_name), "table": table}
    if save_member_forecasts:
        entry["member_predictions"] = pred_stack.astype(np.float32)
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
    save_member_forecasts: bool = True,
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
            save_member_forecasts=save_member_forecasts,
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
        save_member_forecasts=save_member_forecasts,
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



def bag_profile_names_from_hdf5(bagged_h5_path: Path, n_models: int | None = None) -> list[list[str]]:
    """Return exact profile membership for each train/bag_i group."""
    bagged_h5_path = Path(bagged_h5_path)
    with h5py.File(bagged_h5_path, "r") as h5f:
        if "train" not in h5f:
            raise ValueError(f"Bagged HDF5 is missing train group: {bagged_h5_path}")
        if n_models is None:
            indices = sorted(
                int(name.removeprefix("bag_")) for name in h5f["train"].keys()
                if str(name).startswith("bag_")
            )
        else:
            indices = list(range(int(n_models)))
        memberships: list[list[str]] = []
        for idx in indices:
            files_path = f"train/bag_{idx}/files"
            if files_path not in h5f:
                raise ValueError(f"Bagged HDF5 is missing required membership group: {files_path}")
            names = sorted(str(name) for name in h5f[files_path].keys())
            if not names:
                raise ValueError(f"Bag {idx} contains no profiles.")
            memberships.append(names)
    return memberships


def save_bag_overlap_diagnostics(bagged_h5_path: Path, output_dir: Path, *, n_models: int | None = None) -> dict[str, Any]:
    """Save profile-overlap matrices, inclusion frequencies, and diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memberships = bag_profile_names_from_hdf5(bagged_h5_path, n_models=n_models)
    bag_sets = [set(names) for names in memberships]
    n_bags = len(bag_sets)
    counts = np.zeros((n_bags, n_bags), dtype=np.int64)
    fractions = np.zeros((n_bags, n_bags), dtype=np.float64)
    jaccard = np.zeros((n_bags, n_bags), dtype=np.float64)
    for i, left in enumerate(bag_sets):
        for j, right in enumerate(bag_sets):
            inter = len(left.intersection(right))
            union = len(left.union(right))
            counts[i, j] = inter
            fractions[i, j] = inter / max(1, len(left))
            jaccard[i, j] = inter / union if union else 0.0

    all_profiles = sorted(set().union(*bag_sets) if bag_sets else set())
    inclusion_frequency = {name: int(sum(name in bag for bag in bag_sets)) for name in all_profiles}
    payload = {
        "pairwise_profile_overlap_counts": counts.tolist(),
        "pairwise_profile_overlap_fractions": fractions.tolist(),
        "jaccard_similarity_matrix": jaccard.tolist(),
        "profile_inclusion_frequency": inclusion_frequency,
        "n_bags": n_bags,
    }
    (output_dir / "bag_overlap_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(jaccard, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xlabel("Bag index")
    ax.set_ylabel("Bag index")
    ax.set_title("Bag profile Jaccard similarity")
    fig.colorbar(im, ax=ax, label="Jaccard similarity")
    fig.tight_layout()
    heatmap_path = output_dir / "bag_jaccard_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(list(inclusion_frequency.values()), bins=np.arange(0.5, n_bags + 1.5, 1.0), color="C0", edgecolor="black")
    ax.set_xlabel("Number of bags containing profile")
    ax.set_ylabel("Profile count")
    ax.set_title("Training profile inclusion frequency")
    fig.tight_layout()
    histogram_path = output_dir / "profile_inclusion_frequency_histogram.png"
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)
    payload["overlap_heatmap_path"] = str(heatmap_path)
    payload["inclusion_frequency_histogram_path"] = str(histogram_path)
    return payload


def _member_metadata_payload(
    *,
    model_idx: int,
    seed: int,
    history: dict[str, list[float]],
    datasets: dict[str, Any],
    used_device: str,
    config: BaggingEnsembleConfig,
) -> dict[str, Any]:
    val_losses = [float(v) for v in history.get("val_loss", [])]
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else None
    best_val = float(np.min(val_losses)) if val_losses else None
    final_val = float(val_losses[-1]) if val_losses else None
    return {
        "member_index": int(model_idx),
        "seed": int(seed),
        "architecture": {
            "n_lstm": config.n_lstm,
            "lstm_hidden": config.lstm_hidden,
            "lstm_dropout": config.lstm_dropout,
            "n_fc": config.n_fc,
            "fc_hidden": list(config.fc_hidden),
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "final_validation_loss": final_val,
        "train_profile_count": len(datasets.get("train_profile_names", [])),
        "train_sample_count": int(datasets.get("train_num_samples", 0)),
        "val_profile_count": len(datasets.get("val_profile_names", [])),
        "val_sample_count": int(datasets.get("val_num_samples", 0)),
        "device": used_device,
    }

def run_bagging_ensemble(
    scaled_h5_path: Path,
    *,
    out_dir: Path,
    n_models: int,
    bag_fraction: float = 0.70,
    bag_split_mode: str = "profile",
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
    preload_train_to_device: bool = True,
    preload_val_to_device: bool = True,
    restore_best_weights: bool = True,
    member_seeds: list[int] | tuple[int, ...] | None = None,
    resume: bool = False,
    use_tqdm: bool = True,
    verbose: int = 1,
    forecast_num_workers: int = 4,
    plot_bag_distributions: bool = True,
    save_member_forecasts: bool = True,
) -> dict[str, Any]:
    config = BaggingEnsembleConfig(
        n_models=n_models,
        bag_fraction=bag_fraction,
        bag_split_mode=bag_split_mode,
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
        use_tqdm=use_tqdm,
        verbose=verbose,
        plot_bag_distributions=plot_bag_distributions,
    )
    config.validate()
    if member_seeds is None:
        resolved_member_seeds = [int(config.seed) + idx for idx in range(config.n_models)]
    else:
        resolved_member_seeds = [int(seed) for seed in member_seeds]
    if len(resolved_member_seeds) != config.n_models:
        raise ValueError("member_seeds must contain exactly n_models values when provided.")
    if len(set(resolved_member_seeds)) != len(resolved_member_seeds):
        raise ValueError("member_seeds must be distinct for ensemble training.")

    scaled_h5_path = Path(scaled_h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bagged_h5_path = out_dir / "bagged_dataset.h5"
    create_bagged_training_hdf5(
        scaled_h5_path,
        bagged_h5_path,
        n_models=config.n_models,
        bag_fraction=config.bag_fraction,
        bag_split_mode=config.bag_split_mode,
        seed=config.seed,
        verbose=config.verbose,
    )
    bag_distribution_overlap_plot: Path | None = None
    if config.plot_bag_distributions:
        bag_distribution_overlap_plot = _plot_bag_distribution_overlap(
            bagged_h5_path,
            state_dim=STATE_DIM,
            control_channel=0,
            target_names=list(TARGET_NAMES),
        )
        if config.verbose >= 1:
            if bag_distribution_overlap_plot is None:
                print("[bagging] bag distribution overlap plot not created (no bag data found).")
            else:
                print(f"[bagging] saved bag distribution overlap plot: {bag_distribution_overlap_plot}")

    bag_overlap_diagnostics = save_bag_overlap_diagnostics(bagged_h5_path, out_dir / "bag_diagnostics", n_models=config.n_models)

    models: list[torch.nn.Module] = []
    histories: list[dict[str, list[float]]] = []
    used_devices: list[str] = []

    for model_idx in range(config.n_models):
        model_dir = out_dir / f"model_{model_idx}"
        model_dir.mkdir(parents=True, exist_ok=True)

        member_seed = resolved_member_seeds[model_idx]
        datasets = build_datasets_for_train_split(
            bagged_h5_path,
            train_split=f"train/bag_{model_idx}",
            batch_size=config.batch_size,
            seed=member_seed,
        )
        model_path = model_dir / "model.pt"
        metadata_path = model_dir / "member_metadata.json"
        history_path = model_dir / "training_history.json"
        if resume and model_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("seed", -1)) == member_seed:
                x_shape = datasets["sample_shape"]
                y_shape = datasets["target_shape"]
                model = load_bagged_lstm_ensemble_checkpoints(
                    [model_path, model_path],
                    timesteps=int(x_shape[1]),
                    num_features=int(x_shape[2]),
                    num_targets=int(y_shape[1]),
                    n_lstm=config.n_lstm,
                    lstm_hidden=config.lstm_hidden,
                    lstm_dropout=config.lstm_dropout,
                    n_fc=config.n_fc,
                    fc_hidden=config.fc_hidden,
                    device="cuda" if config.prefer_gpu and torch.cuda.is_available() else "cpu",
                )[0]
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"loss": [], "val_loss": []}
                used_device = torch.device(next(model.parameters()).device)
                if config.verbose >= 1:
                    print(f"[bagging] resumed member {model_idx} from validated checkpoint: {model_path}")
                models.append(model)
                histories.append(history)
                used_devices.append(str(used_device))
                continue

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
            preload_train_to_device=preload_train_to_device,
            preload_val_to_device=preload_val_to_device,
            deterministic_seed=member_seed,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            restore_best_weights=restore_best_weights,
            use_tqdm=config.use_tqdm,
            save_training_curves=False,
        )

        torch.save(model.state_dict(), model_path)
        metadata = _member_metadata_payload(
            model_idx=model_idx,
            seed=member_seed,
            history=history,
            datasets=datasets,
            used_device=str(used_device),
            config=config,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        models.append(model)
        histories.append(history)
        used_devices.append(str(used_device))

    training_curves_plot = _plot_ensemble_training_curves(histories, out_dir / "ensemble_train_val_curves.png")
    if config.verbose >= 1 and training_curves_plot is not None:
        print(f"[bagging] saved ensemble training curves plot: {training_curves_plot}")

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
        save_member_forecasts=save_member_forecasts,
    )

    return {
        "bagged_h5_path": bagged_h5_path,
        "forecast_output_path": forecast_output_path,
        "model_dirs": [out_dir / f"model_{idx}" for idx in range(config.n_models)],
        "model_paths": [out_dir / f"model_{idx}" / "model.pt" for idx in range(config.n_models)],
        "models": models,
        "bag_profile_names": [_get_profile_names(bagged_h5_path, f"train/bag_{idx}") for idx in range(config.n_models)],
        "used_devices": used_devices,
        "histories": histories,
        "bag_distribution_overlap_plot": bag_distribution_overlap_plot,
        "bag_overlap_diagnostics": bag_overlap_diagnostics,
        "training_curves_plot": training_curves_plot,
        "member_seeds": resolved_member_seeds,
        "save_member_forecasts": bool(save_member_forecasts),
    }
