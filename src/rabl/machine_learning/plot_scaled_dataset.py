from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


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


def _load_first_profile(
    h5f: h5py.File,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    files_group = h5f.get(f"{split}/files")
    if files_group is None or len(files_group.keys()) == 0:
        raise ValueError(f"Missing '{split}/files' group in dataset.")
    first_key = sorted(files_group.keys())[0]
    file_group = files_group[first_key]
    return file_group["X"][()], file_group["Y"][()]


def _reconstruct_profile(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_features = y_data.shape[-1]
    states = np.vstack([x_data[0, :, :state_features], y_data])
    control_window = x_data[0, :, state_features]
    control_tail = x_data[:, -1, state_features]
    control = np.concatenate([control_window, control_tail])
    if control.shape[0] > states.shape[0]:
        control = control[: states.shape[0]]
    elif control.shape[0] < states.shape[0]:
        pad = states.shape[0] - control.shape[0]
        control = np.pad(control, (0, pad), mode="edge")
    return states, control


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
        profiles_by_split = {}
        for split in splits:
            x_data, y_data = _load_first_profile(h5f, split)
            profiles_by_split[split] = _reconstruct_profile(x_data, y_data)

    fig, axes = plt.subplots(7, 2, figsize=(12, 18), sharex=False)
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
        repo_root = Path(__file__).resolve().parents[3]
        output_dir = repo_root / "outputs" / "datasets"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_all_splits_feature_distributions.png"
    else:
        output_path = Path(output_path)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    profile_fig, profile_axes = plt.subplots(7, 2, figsize=(12, 18), sharex=False)
    profile_axes = profile_axes.flatten()
    for idx, ax in enumerate(profile_axes):
        for split, color in zip(splits, ("red", "yellow", "green"), strict=False):
            states, control = profiles_by_split[split]
            if idx < states.shape[1]:
                series = states[:, idx]
            else:
                series = control
            ax.plot(series, color=color, alpha=0.7, label=split)
        ax.set_title(feature_labels[idx])
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend()

    profile_fig.suptitle("Scaled feature profiles by split (time series)", fontsize=14)
    profile_fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    profile_output = output_path.with_name(f"{input_path.stem}_all_splits_profiles.png")
    profile_fig.savefig(profile_output, dpi=200)
    plt.close(profile_fig)
    return output_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    input_path = repo_root / "outputs" / "datasets" / (
        "lstm_merged_batch_0001-batch_0001_k10_standard_train0.70_val0.15_test0.15.h5"
    )
    output_path = plot_scaled_features(input_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
