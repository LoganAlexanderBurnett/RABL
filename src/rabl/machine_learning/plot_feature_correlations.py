from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _load_features(
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
        raise ValueError("No samples found to compute correlations.")
    return np.concatenate(samples, axis=0)


def plot_feature_correlations(
    input_path: Path,
    output_path: Path | None = None,
    split: str = "train",
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

        data = _load_features(h5f, split=split, max_samples=max_samples, seed=seed)

    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_labels)))
    ax.set_yticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=45, ha="right")
    ax.set_yticklabels(feature_labels)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Feature correlation matrix ({split} split)")
    fig.tight_layout()

    if output_path is None:
        output_dir = input_path.parents[2] / "outputs" / "datasets"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_{split}_correlations.png"
    else:
        output_path = Path(output_path)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
    