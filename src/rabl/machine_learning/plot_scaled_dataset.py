from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from rabl.paths import resolve_output_root


def _collect_samples(
    h5f: h5py.File,
    split: str,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    files_group = h5f.get(f"{split}/files")
    if files_group is None:
        raise ValueError(f"Missing '{split}/files' group in dataset.")

    samples = []
    remaining = max_samples
    for file_key in files_group.keys():
        x_data = files_group[file_key]["X"][()]
        x_flat = x_data.reshape(-1, x_data.shape[-1])
        if remaining <= 0:
            break
        if x_flat.shape[0] <= remaining:
            samples.append(x_flat)
            remaining -= x_flat.shape[0]
        else:
            indices = rng.choice(x_flat.shape[0], size=remaining, replace=False)
            samples.append(x_flat[indices])
            remaining = 0

    if not samples:
        raise ValueError("No samples found to plot.")
    return np.concatenate(samples, axis=0)


def _load_split_profiles(
    h5f: h5py.File,
    split: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    files_group = h5f.get(f"{split}/files")
    if files_group is None or len(files_group.keys()) == 0:
        raise ValueError(f"Missing '{split}/files' group in dataset.")
    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    for file_key in sorted(files_group.keys()):
        file_group = files_group[file_key]
        profiles.append((file_group["X"][()], file_group["Y"][()]))
    return profiles


def _reconstruct_profile(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_stats: dict[str, np.ndarray] | None = None,
    y_stats: dict[str, np.ndarray] | None = None,
    scaling_type: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    state_features = y_data.shape[-1]
    x_states = x_data[0, :, :state_features]
    y_states = y_data
    if scaling_type != "none" and x_stats is not None and y_stats is not None:
        x_states = _inverse_scale(scaling_type, x_states, x_stats, indices=slice(0, state_features))
        y_states = _inverse_scale(scaling_type, y_states, y_stats)
    states = np.vstack([x_states, y_states])
    control_window = x_data[0, :, state_features]
    control_tail = x_data[:, -1, state_features]
    if scaling_type != "none" and x_stats is not None:
        control_window = _inverse_scale(scaling_type, control_window, x_stats, index=state_features)
        control_tail = _inverse_scale(scaling_type, control_tail, x_stats, index=state_features)
    control = np.concatenate([control_window, control_tail])
    if control.shape[0] > states.shape[0]:
        control = control[: states.shape[0]]
    elif control.shape[0] < states.shape[0]:
        pad = states.shape[0] - control.shape[0]
        control = np.pad(control, (0, pad), mode="edge")
    return states, control


def _inverse_scale(
    scaling_type: str,
    data: np.ndarray,
    stats: dict[str, np.ndarray],
    index: int | None = None,
    indices: slice | None = None,
) -> np.ndarray:
    if scaling_type == "none":
        return data
    if index is not None:
        if scaling_type == "standard":
            return data * stats["std"][index] + stats["mean"][index]
        return data * stats["span"][index] + stats["min"][index]
    if indices is not None:
        if scaling_type == "standard":
            return data * stats["std"][indices] + stats["mean"][indices]
        return data * stats["span"][indices] + stats["min"][indices]
    if scaling_type == "standard":
        return data * stats["std"] + stats["mean"]
    return data * stats["span"] + stats["min"]


def _load_scaling_stats(h5f: h5py.File) -> tuple[str, dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    scaling_type = h5f.attrs.get("scaling_type", "none")
    if isinstance(scaling_type, bytes):
        scaling_type = scaling_type.decode()
    if scaling_type == "none":
        return scaling_type, None, None

    scaling_group = h5f.get("scaling")
    if scaling_group is None:
        raise ValueError("Dataset missing scaling statistics.")
    if scaling_type == "standard":
        x_stats = {"mean": scaling_group["x_mean"][()], "std": scaling_group["x_std"][()]}
        y_stats = {"mean": scaling_group["y_mean"][()], "std": scaling_group["y_std"][()]}
    else:
        x_stats = {"min": scaling_group["x_min"][()], "span": scaling_group["x_span"][()]}
        y_stats = {"min": scaling_group["y_min"][()], "span": scaling_group["y_span"][()]}
    return scaling_type, x_stats, y_stats


def plot_scaled_features(
    input_path: Path,
    output_path: Path | None = None,
    splits: tuple[str, str, str] = ("train", "val", "test"),
    max_samples: int = 200_000,
    seed: int = 123,
) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    with h5py.File(input_path, "r") as h5f:
        feature_names = h5f.attrs.get("state_feature_names")
        control_name = h5f.attrs.get("control_feature_name")
        if feature_names is None or control_name is None:
            raise ValueError("Dataset missing feature name attributes.")
        feature_labels = [name.decode() if isinstance(name, bytes) else name for name in feature_names]
        feature_labels.append(control_name.decode() if isinstance(control_name, bytes) else control_name)

        data_by_split = {
            split: _collect_samples(h5f, split=split, max_samples=max_samples, seed=seed)
            for split in splits
        }
        scaling_type, x_stats, y_stats = _load_scaling_stats(h5f)
        profiles_by_split: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        descaled_profiles_by_split: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for split in splits:
            split_profiles = _load_split_profiles(h5f, split)
            profiles_by_split[split] = [
                _reconstruct_profile(x_data, y_data) for x_data, y_data in split_profiles
            ]
            descaled_profiles_by_split[split] = [
                _reconstruct_profile(x_data, y_data, x_stats, y_stats, scaling_type)
                for x_data, y_data in split_profiles
            ]

    fig, axes = plt.subplots(3, 5, figsize=(14, 12), sharex=False)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.hist(data_by_split["train"][:, idx], bins=50, color="red", alpha=0.3, label="train")
        ax.hist(data_by_split["val"][:, idx], bins=50, color="yellow", alpha=0.3, label="val")
        ax.hist(data_by_split["test"][:, idx], bins=50, color="green", alpha=0.3, label="test")
        ax.set_title(feature_labels[idx])
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend()

    fig.suptitle("Scaled feature distributions (train/val/test splits)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    if output_path is None:
        output_dir = resolve_output_root() / "datasets"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_all_splits_feature_distributions.png"
    else:
        output_path = Path(output_path)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    profile_fig, profile_axes = plt.subplots(3, 5, figsize=(14, 12), sharex=False)
    profile_axes = profile_axes.flatten()
    for idx, ax in enumerate(profile_axes):
        for split, color in zip(splits, ("red", "yellow", "green"), strict=False):
            for profile_idx, (states, control) in enumerate(profiles_by_split[split]):
                if idx < states.shape[1]:
                    series = states[:, idx]
                else:
                    series = control
                label = split if profile_idx == 0 else "_nolegend_"
                ax.plot(series, color=color, alpha=0.35, label=label)
        ax.set_title(feature_labels[idx])
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend()

    profile_fig.suptitle("Scaled feature profiles by split (time series)", fontsize=14)
    profile_fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    profile_output = output_path.with_name(f"{input_path.stem}_all_splits_profiles.png")
    profile_fig.savefig(profile_output, dpi=200)
    plt.close(profile_fig)

    descaled_fig, descaled_axes = plt.subplots(3, 5, figsize=(14, 12), sharex=False)
    descaled_axes = descaled_axes.flatten()
    for idx, ax in enumerate(descaled_axes):
        for split, color in zip(splits, ("red", "yellow", "green"), strict=False):
            for profile_idx, (states, control) in enumerate(descaled_profiles_by_split[split]):
                if idx < states.shape[1]:
                    series = states[:, idx]
                else:
                    series = control
                label = split if profile_idx == 0 else "_nolegend_"
                ax.plot(series, color=color, alpha=0.35, label=label)
        ax.set_title(feature_labels[idx])
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend()

    descaled_fig.suptitle("Descaled feature profiles by split (time series)", fontsize=14)
    descaled_fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    descaled_output = output_path.with_name(f"{input_path.stem}_all_splits_profiles_descaled.png")
    descaled_fig.savefig(descaled_output, dpi=200)
    plt.close(descaled_fig)
    return output_path


def main() -> None:
    input_path = resolve_output_root() / "datasets" / (
        "lstm_merged_batch_0001-batch_0001_k10_standard_train0.70_val0.15_test0.15.h5"
    )
    output_path = plot_scaled_features(input_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
