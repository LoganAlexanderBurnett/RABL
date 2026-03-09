from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Literal

import h5py
import numpy as np


ScalingType = Literal["standard", "minmax", "none"]
SplitMode = Literal["profile", "sample"]


@dataclass(frozen=True)
class SplitFractions:
    train: float
    val: float
    test: float

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):
            raise ValueError("Train/val/test fractions must sum to 1.0.")
        if any(frac <= 0 for frac in (self.train, self.val, self.test)):
            raise ValueError("Train/val/test fractions must be positive.")


class LSTMDatasetScalerSplitter:
    def __init__(
        self,
        input_path: Path,
        scaling_type: ScalingType = "standard",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        output_dir: Path | None = None,
        output_name: str | None = None,
        seed: int = 123,
        split_mode: SplitMode = "sample",
    ) -> None:
        self.input_path = Path(input_path)
        self.scaling_type = scaling_type
        self.splits = SplitFractions(train=train_frac, val=val_frac, test=test_frac)
        self.output_dir = output_dir
        self.output_name = output_name
        self.seed = seed
        self.split_mode = split_mode

    def run(self) -> Path:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {self.input_path}")
        if self.scaling_type not in ("standard", "minmax", "none"):
            raise ValueError("scaling_type must be 'standard', 'minmax', or 'none'.")
        if self.split_mode not in ("profile", "sample"):
            raise ValueError("split_mode must be 'profile' or 'sample'.")
        self.splits.validate()

        output_path = self._resolve_output_path()

        with h5py.File(self.input_path, "r") as h5f:
            files_group = h5f.get("files")
            if files_group is None:
                raise ValueError("Input dataset is missing the 'files' group.")
            file_keys = sorted(files_group.keys())
            if not file_keys:
                raise ValueError("No per-file datasets found in the input HDF5.")

            rng = np.random.default_rng(self.seed)
            split_payload = self._build_split_payload(files_group, file_keys, rng)
            train_stats = self._compute_stats(files_group, split_payload["train"])

            with h5py.File(output_path, "w") as out_h5f:
                self._write_metadata(h5f, out_h5f, train_stats)
                self._write_split(out_h5f, "train", files_group, split_payload["train"], train_stats)
                self._write_split(out_h5f, "val", files_group, split_payload["val"], train_stats)
                self._write_split(out_h5f, "test", files_group, split_payload["test"], train_stats)

        return output_path

    def _resolve_output_path(self) -> Path:
        base_dir = self.output_dir or self.input_path.parents[2] / "datasets" / "scaled_split"
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        if self.output_name:
            return base_dir / self.output_name

        stem = self.input_path.stem
        name = (
            f"{stem}_{self.scaling_type}_train{self.splits.train:.2f}"
            f"_val{self.splits.val:.2f}_test{self.splits.test:.2f}.h5"
        )
        return base_dir / name

    def _split_keys(self, keys: np.ndarray) -> tuple[list[str], list[str], list[str]]:
        total = len(keys)
        train_end = int(total * self.splits.train)
        val_end = train_end + int(total * self.splits.val)
        train_keys = list(keys[:train_end])
        val_keys = list(keys[train_end:val_end])
        test_keys = list(keys[val_end:])
        return train_keys, val_keys, test_keys

    def _build_split_payload(
        self,
        files_group: h5py.Group,
        file_keys: list[str],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, np.ndarray | None]]:
        if self.split_mode == "profile":
            shuffled = rng.permutation(file_keys)
            train_keys, val_keys, test_keys = self._split_keys(shuffled)
            return {
                "train": {key: None for key in train_keys},
                "val": {key: None for key in val_keys},
                "test": {key: None for key in test_keys},
            }

        sample_entries: list[tuple[str, int]] = []
        for key in file_keys:
            num_samples = int(files_group[key]["X"].shape[0])
            sample_entries.extend((key, idx) for idx in range(num_samples))

        if not sample_entries:
            raise ValueError("No samples found across input profiles.")

        perm_indices = rng.permutation(len(sample_entries))
        shuffled_entries = [sample_entries[idx] for idx in perm_indices]
        train_entries, val_entries, test_entries = self._split_keys(np.asarray(shuffled_entries, dtype=object))

        return {
            "train": self._entries_to_indices(train_entries),
            "val": self._entries_to_indices(val_entries),
            "test": self._entries_to_indices(test_entries),
        }

    @staticmethod
    def _entries_to_indices(entries: list[tuple[str, int]]) -> dict[str, np.ndarray]:
        profile_to_indices: defaultdict[str, list[int]] = defaultdict(list)
        for key, sample_idx in entries:
            profile_to_indices[str(key)].append(int(sample_idx))
        return {
            profile: np.asarray(indices, dtype=np.int64)
            for profile, indices in profile_to_indices.items()
        }

    def _compute_stats(self, files_group: h5py.Group, split_payload: dict[str, np.ndarray | None]) -> dict:
        x_stats = _init_stats(self.scaling_type, features=14)
        y_stats = _init_stats(self.scaling_type, features=13)

        for key, selected_indices in split_payload.items():
            file_group = files_group[key]
            x_data = file_group["X"][()]
            y_data = file_group["Y"][()]
            if selected_indices is not None:
                x_data = x_data[selected_indices]
                y_data = y_data[selected_indices]
            x_flat = x_data.reshape(-1, x_data.shape[-1])
            y_flat = y_data.reshape(-1, y_data.shape[-1])
            x_stats = _update_stats(self.scaling_type, x_stats, x_flat)
            y_stats = _update_stats(self.scaling_type, y_stats, y_flat)

        return {"x": _finalize_stats(self.scaling_type, x_stats), "y": _finalize_stats(self.scaling_type, y_stats)}

    def _write_metadata(self, src_h5f: h5py.File, dst_h5f: h5py.File, stats: dict) -> None:
        for key, value in src_h5f.attrs.items():
            dst_h5f.attrs[key] = value
        dst_h5f.attrs["scaling_type"] = self.scaling_type
        dst_h5f.attrs["train_fraction"] = self.splits.train
        dst_h5f.attrs["val_fraction"] = self.splits.val
        dst_h5f.attrs["test_fraction"] = self.splits.test
        dst_h5f.attrs["split_mode"] = self.split_mode

        scaling_group = dst_h5f.create_group("scaling")
        if self.scaling_type == "standard":
            scaling_group.create_dataset("x_mean", data=stats["x"]["mean"])
            scaling_group.create_dataset("x_std", data=stats["x"]["std"])
            scaling_group.create_dataset("y_mean", data=stats["y"]["mean"])
            scaling_group.create_dataset("y_std", data=stats["y"]["std"])
        elif self.scaling_type == "minmax":
            scaling_group.create_dataset("x_min", data=stats["x"]["min"])
            scaling_group.create_dataset("x_max", data=stats["x"]["max"])
            scaling_group.create_dataset("x_span", data=stats["x"]["span"])
            scaling_group.create_dataset("y_min", data=stats["y"]["min"])
            scaling_group.create_dataset("y_max", data=stats["y"]["max"])
            scaling_group.create_dataset("y_span", data=stats["y"]["span"])

    def _write_split(
        self,
        dst_h5f: h5py.File,
        split_name: str,
        files_group: h5py.Group,
        split_payload: dict[str, np.ndarray | None],
        stats: dict,
    ) -> None:
        split_group = dst_h5f.create_group(split_name)
        files_out = split_group.create_group("files")

        for key, selected_indices in split_payload.items():
            file_group = files_group[key]
            x_data = file_group["X"][()]
            y_data = file_group["Y"][()]
            if selected_indices is not None:
                x_data = x_data[selected_indices]
                y_data = y_data[selected_indices]
            x_scaled = _apply_scaling(self.scaling_type, x_data, stats["x"])
            y_scaled = _apply_scaling(self.scaling_type, y_data, stats["y"])

            out_group = files_out.create_group(key)
            out_group.create_dataset("X", data=x_scaled, compression="gzip")
            out_group.create_dataset("Y", data=y_scaled, compression="gzip")
            for attr_key, attr_value in file_group.attrs.items():
                out_group.attrs[attr_key] = attr_value


def _init_stats(scaling_type: ScalingType, features: int) -> dict:
    if scaling_type == "none":
        return {}
    if scaling_type == "standard":
        return {"count": 0, "mean": np.zeros(features), "M2": np.zeros(features)}
    return {"min": np.full(features, np.inf), "max": np.full(features, -np.inf)}


def _update_stats(scaling_type: ScalingType, stats: dict, batch: np.ndarray) -> dict:
    if scaling_type == "none":
        return stats
    if scaling_type == "standard":
        return _update_running_stats(stats, batch)
    stats["min"] = np.minimum(stats["min"], batch.min(axis=0))
    stats["max"] = np.maximum(stats["max"], batch.max(axis=0))
    return stats


def _finalize_stats(scaling_type: ScalingType, stats: dict) -> dict:
    if scaling_type == "none":
        return {}
    if scaling_type == "standard":
        count = max(stats["count"], 1)
        variance = stats["M2"] / count
        std = np.sqrt(variance)
        std[std == 0] = 1.0
        return {"mean": stats["mean"], "std": std}
    min_vals = stats["min"]
    max_vals = stats["max"]
    span = max_vals - min_vals
    span[span == 0] = 1.0
    return {"min": min_vals, "max": max_vals, "span": span}


def _update_running_stats(stats: dict, batch: np.ndarray) -> dict:
    batch_count = batch.shape[0]
    batch_mean = batch.mean(axis=0)
    batch_M2 = ((batch - batch_mean) ** 2).sum(axis=0)

    delta = batch_mean - stats["mean"]
    total_count = stats["count"] + batch_count
    if total_count == 0:
        return stats

    stats["mean"] += delta * (batch_count / total_count)
    stats["M2"] += batch_M2 + (delta ** 2) * stats["count"] * batch_count / total_count
    stats["count"] = total_count
    return stats


def _apply_scaling(scaling_type: ScalingType, data: np.ndarray, stats: dict) -> np.ndarray:
    if scaling_type == "none":
        return data
    if scaling_type == "standard":
        return (data - stats["mean"]) / stats["std"]
    return (data - stats["min"]) / stats["span"]
