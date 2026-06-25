from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any, Literal
import json

import h5py
import numpy as np


ScalingType = Literal["standard", "minmax", "none"]
SplitMode = Literal["profile", "sample"]


@dataclass(frozen=True)
class SplitFractions:
    train: float
    val: float
    test: float
    cal: float = 0.0

    def validate(self) -> None:
        total = self.train + self.val + self.cal + self.test
        if not np.isclose(total, 1.0):
            raise ValueError("Train/val/cal/test fractions must sum to 1.0.")
        if any(frac <= 0 for frac in (self.train, self.val, self.test)):
            raise ValueError("Train/val/test fractions must be positive.")
        if self.cal < 0:
            raise ValueError("cal fraction must be non-negative.")


class LSTMDatasetScalerSplitter:
    def __init__(
        self,
        input_path: Path,
        scaling_type: ScalingType = "standard",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        cal_frac: float = 0.0,
        test_frac: float = 0.15,
        output_dir: Path | None = None,
        output_name: str | None = None,
        seed: int = 123,
        split_mode: SplitMode = "sample",
        test_manifest_path: Path | None = None,
        val_manifest_path: Path | None = None,
        cal_manifest_path: Path | None = None,
        train_profile_limit_with_manifests: int | None = None,
        test_count: int | None = None,
        save_test_manifest_path: Path | None = None,
        frozen_stats_path: Path | None = None,
        save_stats_path: Path | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.scaling_type = scaling_type
        self.splits = SplitFractions(train=train_frac, val=val_frac, cal=cal_frac, test=test_frac)
        self.output_dir = output_dir
        self.output_name = output_name
        self.seed = seed
        self.split_mode = split_mode
        self.test_manifest_path = Path(test_manifest_path) if test_manifest_path else None
        self.val_manifest_path = Path(val_manifest_path) if val_manifest_path else None
        self.cal_manifest_path = Path(cal_manifest_path) if cal_manifest_path else None
        self.train_profile_limit_with_manifests = train_profile_limit_with_manifests
        self.test_count = test_count
        self.save_test_manifest_path = (
            Path(save_test_manifest_path) if save_test_manifest_path else None
        )
        self.frozen_stats_path = Path(frozen_stats_path) if frozen_stats_path else None
        self.save_stats_path = Path(save_stats_path) if save_stats_path else None

    def run(self) -> Path:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {self.input_path}")
        if self.scaling_type not in ("standard", "minmax", "none"):
            raise ValueError("scaling_type must be 'standard', 'minmax', or 'none'.")
        if self.split_mode not in ("profile", "sample"):
            raise ValueError("split_mode must be 'profile' or 'sample'.")
        if (
            self.test_manifest_path is None
            and self.val_manifest_path is None
            and self.cal_manifest_path is None
            and self.test_count is None
        ):
            self.splits.validate()
        else:
            if self.splits.train <= 0 or self.splits.val <= 0:
                raise ValueError("train_frac and val_frac must be positive when using fixed test splits.")
        if self.train_profile_limit_with_manifests is not None:
            if self.train_profile_limit_with_manifests < 1:
                raise ValueError("train_profile_limit_with_manifests must be >= 1 when provided.")
            if self.test_manifest_path is None or self.val_manifest_path is None:
                raise ValueError(
                    "train_profile_limit_with_manifests can only be used when both "
                    "test_manifest_path and val_manifest_path are provided."
                )

        with h5py.File(self.input_path, "r") as h5f:
            files_group = h5f.get("files")
            if files_group is None:
                raise ValueError("Input dataset is missing the 'files' group.")
            file_keys = sorted(files_group.keys())
            if not file_keys:
                raise ValueError("No per-file datasets found in the input HDF5.")

            rng = np.random.default_rng(self.seed)
            split_payload, split_definition = self._build_split_payload(files_group, file_keys, rng)
            output_path = self._resolve_output_path(split_definition)
            split_summary = self._summarize_split_payload(files_group, split_payload["train"])
            if self.frozen_stats_path is not None:
                print(f"[scale] Loading frozen scaler statistics from {self.frozen_stats_path}")
                train_stats = load_scaler_stats_json(
                    self.frozen_stats_path,
                    scaling_type=self.scaling_type,
                    input_feature_names=_input_feature_names_from_attrs(h5f),
                    target_names=_target_names_from_attrs(h5f),
                )
                current_stats = self._compute_stats(files_group, split_payload["train"])
                _log_stats_comparison(self.scaling_type, train_stats, current_stats)
            else:
                print("[scale] Computing scaler statistics from the current training split.")
                train_stats = self._compute_stats(files_group, split_payload["train"])

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.save_stats_path is not None:
                self.save_stats_path.parent.mkdir(parents=True, exist_ok=True)
                write_scaler_stats_json(
                    self.save_stats_path,
                    scaling_type=self.scaling_type,
                    stats=train_stats,
                    input_feature_names=_input_feature_names_from_attrs(h5f),
                    target_names=_target_names_from_attrs(h5f),
                    train_profile_count=split_summary["train_profile_count"],
                    train_window_count=split_summary["train_window_count"],
                )
                # Immediate reload catches partial writes and schema/feature-order mistakes.
                load_scaler_stats_json(
                    self.save_stats_path,
                    scaling_type=self.scaling_type,
                    input_feature_names=_input_feature_names_from_attrs(h5f),
                    target_names=_target_names_from_attrs(h5f),
                )
                print(f"[scale] Saved scaler statistics to {self.save_stats_path}")

            with h5py.File(output_path, "w") as out_h5f:
                self._write_metadata(h5f, out_h5f, train_stats, split_definition)
                self._write_split(out_h5f, "train", files_group, split_payload["train"], train_stats)
                self._write_split(out_h5f, "val", files_group, split_payload["val"], train_stats)
                if split_payload.get("cal"):
                    self._write_split(out_h5f, "cal", files_group, split_payload["cal"], train_stats)
                self._write_split(out_h5f, "test", files_group, split_payload["test"], train_stats)

        return output_path

    def _resolve_output_path(self, split_definition: dict[str, list[str] | str | int]) -> Path:
        base_dir = self.output_dir or self.input_path.parents[2] / "datasets" / "scaled_split"
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        if self.output_name:
            return base_dir / self.output_name

        stem = self.input_path.stem
        split_strategy = str(split_definition.get("split_strategy", "fractional"))
        if split_strategy == "fractional":
            split_suffix = (
                f"train{self.splits.train:.2f}_val{self.splits.val:.2f}"
                f"_cal{self.splits.cal:.2f}_test{self.splits.test:.2f}"
            )
        else:
            n_train = len(split_definition.get("train_profiles", []))
            n_val = len(split_definition.get("val_profiles", []))
            n_cal = len(split_definition.get("cal_profiles", []))
            n_test = len(split_definition.get("test_profiles", []))
            split_suffix = f"train{n_train}profiles_val{n_val}profiles_cal{n_cal}profiles_test{n_test}profiles"
        name = f"{stem}_{self.scaling_type}_{split_suffix}.h5"
        return base_dir / name

    def _split_keys(self, keys: np.ndarray) -> tuple[list[str], list[str], list[str], list[str]]:
        total = len(keys)
        train_end = int(total * self.splits.train)
        val_end = train_end + int(total * self.splits.val)
        cal_end = val_end + int(total * self.splits.cal)
        train_keys = list(keys[:train_end])
        val_keys = list(keys[train_end:val_end])
        cal_keys = list(keys[val_end:cal_end])
        test_keys = list(keys[cal_end:])
        return train_keys, val_keys, cal_keys, test_keys

    def _build_split_payload(
        self,
        files_group: h5py.Group,
        file_keys: list[str],
        rng: np.random.Generator,
    ) -> tuple[dict[str, dict[str, np.ndarray | None]], dict[str, list[str] | str | int]]:
        if self.test_manifest_path is not None:
            return self._build_split_payload_from_manifests(files_group, file_keys, rng)

        if self.val_manifest_path is not None:
            return self._build_split_payload_from_manifests(files_group, file_keys, rng)

        if self.cal_manifest_path is not None:
            return self._build_split_payload_from_manifests(files_group, file_keys, rng)

        if self.test_count is not None:
            return self._build_split_payload_from_count(files_group, file_keys, rng)

        if self.split_mode == "profile":
            shuffled = rng.permutation(file_keys)
            train_keys, val_keys, cal_keys, test_keys = self._split_keys(shuffled)
            split_definition: dict[str, list[str] | str | int] = {
                "split_strategy": "fractional",
                "train_profiles": train_keys,
                "val_profiles": val_keys,
                "cal_profiles": cal_keys,
                "test_profiles": test_keys,
            }
            return (
                {
                    "train": {key: None for key in train_keys},
                    "val": {key: None for key in val_keys},
                    "cal": {key: None for key in cal_keys},
                    "test": {key: None for key in test_keys},
                },
                split_definition,
            )

        # sample mode semantics:
        # - test split remains profile-disjoint and uses whole, unseen profiles
        # - only train/val are split at sample level
        shuffled_keys = rng.permutation(file_keys)
        test_start = int(len(shuffled_keys) * (1.0 - self.splits.test))
        cal_start = int(len(shuffled_keys) * (1.0 - self.splits.test - self.splits.cal))
        train_val_keys = list(shuffled_keys[:cal_start])
        cal_keys = list(shuffled_keys[cal_start:test_start])
        test_keys = list(shuffled_keys[test_start:])

        sample_entries: list[tuple[str, int]] = []
        for key in train_val_keys:
            num_samples = int(files_group[key]["X"].shape[0])
            sample_entries.extend((key, idx) for idx in range(num_samples))

        if not sample_entries:
            raise ValueError("No train/val samples found across input profiles.")

        perm_indices = rng.permutation(len(sample_entries))
        shuffled_entries = [sample_entries[idx] for idx in perm_indices]

        train_weight = self.splits.train / (self.splits.train + self.splits.val)
        train_end = int(len(shuffled_entries) * train_weight)
        train_entries = shuffled_entries[:train_end]
        val_entries = shuffled_entries[train_end:]

        split_definition = {
            "split_strategy": "fractional",
            "train_profiles": sorted(self._entries_to_indices(train_entries).keys()),
            "val_profiles": sorted(self._entries_to_indices(val_entries).keys()),
            "cal_profiles": sorted(cal_keys),
            "test_profiles": sorted(test_keys),
        }
        return (
            {
                "train": self._entries_to_indices(train_entries),
                "val": self._entries_to_indices(val_entries),
                "cal": {key: None for key in cal_keys},
                "test": {key: None for key in test_keys},
            },
            split_definition,
        )

    def _build_split_payload_from_manifests(
        self,
        files_group: h5py.Group,
        file_keys: list[str],
        rng: np.random.Generator,
    ) -> tuple[dict[str, dict[str, np.ndarray | None]], dict[str, list[str] | str | int]]:
        test_keys: list[str] = []
        val_keys: list[str] = []
        cal_keys: list[str] = []
        meta: dict[str, list[str] | str | int] = {}

        if self.test_manifest_path is not None:
            if not self.test_manifest_path.exists():
                raise FileNotFoundError(f"Test manifest not found: {self.test_manifest_path}")
            data = json.loads(self.test_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Test manifest must be a JSON object.")
            test_profiles = data.get("test_profiles")
            if not isinstance(test_profiles, list):
                raise ValueError("Test manifest must contain list field: test_profiles.")
            test_keys = [str(key) for key in test_profiles]
            self._validate_fixed_test_split(file_keys, test_keys)
            meta["test_manifest_path"] = str(self.test_manifest_path)

        if self.val_manifest_path is not None:
            if not self.val_manifest_path.exists():
                raise FileNotFoundError(f"Val manifest not found: {self.val_manifest_path}")
            data = json.loads(self.val_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Val manifest must be a JSON object.")
            val_profiles = data.get("val_profiles")
            if not isinstance(val_profiles, list):
                raise ValueError("Val manifest must contain list field: val_profiles.")
            val_keys = [str(key) for key in val_profiles]
            self._validate_fixed_test_split(file_keys, val_keys)
            meta["val_manifest_path"] = str(self.val_manifest_path)

        if self.cal_manifest_path is not None:
            if not self.cal_manifest_path.exists():
                raise FileNotFoundError(f"Cal manifest not found: {self.cal_manifest_path}")
            data = json.loads(self.cal_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Cal manifest must be a JSON object.")
            cal_profiles = data.get("cal_profiles")
            if not isinstance(cal_profiles, list):
                raise ValueError("Cal manifest must contain list field: cal_profiles.")
            cal_keys = [str(key) for key in cal_profiles]
            self._validate_fixed_test_split(file_keys, cal_keys)
            meta["cal_manifest_path"] = str(self.cal_manifest_path)

        overlap = sorted(
            set(test_keys).intersection(val_keys)
            | set(test_keys).intersection(cal_keys)
            | set(val_keys).intersection(cal_keys)
        )
        if overlap:
            raise ValueError(f"Fixed split manifests overlap: {overlap[:10]}")

        return self._build_split_with_fixed_splits(
            files_group=files_group,
            file_keys=file_keys,
            test_keys=test_keys,
            val_keys=val_keys,
            cal_keys=cal_keys,
            rng=rng,
            split_strategy="fixed_manifest",
            extra_meta=meta,
        )

    def _build_split_payload_from_count(
        self,
        files_group: h5py.Group,
        file_keys: list[str],
        rng: np.random.Generator,
    ) -> tuple[dict[str, dict[str, np.ndarray | None]], dict[str, list[str] | str | int]]:
        if self.test_count is None:
            raise ValueError("test_count must be provided.")
        if self.test_count < 1:
            raise ValueError("test_count must be a positive integer.")

        total = len(file_keys)
        if self.test_count >= total:
            raise ValueError(
                f"test_count must be less than total profiles ({total}). "
                f"Got test_count={self.test_count}."
            )

        shuffled = list(rng.permutation(file_keys))
        test_keys = sorted(shuffled[: self.test_count])
        extra_meta: dict[str, list[str] | str | int] = {
            "split_seed": int(self.seed),
            "test_count": int(self.test_count),
        }
        if self.save_test_manifest_path is not None:
            payload = {
                "test_profiles": sorted(test_keys),
            }
            self.save_test_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_test_manifest_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            extra_meta["test_manifest_path"] = str(self.save_test_manifest_path)
        return self._build_split_with_fixed_splits(
            files_group=files_group,
            file_keys=file_keys,
            test_keys=test_keys,
            val_keys=[],
            cal_keys=[],
            rng=rng,
            split_strategy="fixed_count",
            extra_meta=extra_meta,
        )

    @staticmethod
    def _validate_fixed_test_split(file_keys: list[str], test_keys: list[str]) -> None:
        file_key_set = set(file_keys)
        unknown_test = sorted(set(test_keys) - file_key_set)
        if unknown_test:
            raise ValueError(f"Manifest test_profiles contain unknown profile keys: {unknown_test[:10]}")
        if not test_keys:
            raise ValueError("Manifest test_profiles is empty.")

    def _build_split_with_fixed_splits(
        self,
        files_group: h5py.Group,
        file_keys: list[str],
        test_keys: list[str],
        val_keys: list[str],
        cal_keys: list[str],
        rng: np.random.Generator,
        *,
        split_strategy: str,
        extra_meta: dict[str, list[str] | str | int] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray | None]], dict[str, list[str] | str | int]]:
        test_set = set(test_keys)
        val_set = set(val_keys)
        cal_set = set(cal_keys)
        train_pool_keys = [key for key in file_keys if key not in test_set and key not in val_set and key not in cal_set]
        if not train_pool_keys:
            raise ValueError("Fixed test selection leaves no profiles for train/val.")

        if not cal_keys and self.splits.cal > 0.0 and len(train_pool_keys) > 1:
            shuffled_cal_pool = list(rng.permutation(train_pool_keys))
            cal_weight = self.splits.cal / (self.splits.train + self.splits.cal)
            n_cal = int(len(shuffled_cal_pool) * cal_weight)
            if n_cal < 1:
                n_cal = 1
            if n_cal >= len(shuffled_cal_pool):
                n_cal = len(shuffled_cal_pool) - 1
            cal_keys = sorted(shuffled_cal_pool[:n_cal])
            cal_set = set(cal_keys)
            train_pool_keys = [key for key in train_pool_keys if key not in cal_set]

        if val_keys:
            train_keys = sorted(train_pool_keys)
            if self.train_profile_limit_with_manifests is not None:
                train_keys = train_keys[: self.train_profile_limit_with_manifests]
            if not train_keys:
                raise ValueError("Fixed test split produced empty train or val split.")
            payload = {
                "train": {key: None for key in train_keys},
                "val": {key: None for key in val_keys},
                "cal": {key: None for key in cal_keys},
                "test": {key: None for key in test_keys},
            }
        elif self.split_mode == "profile":
            shuffled = list(rng.permutation(train_pool_keys))
            train_end = int(len(shuffled) * (self.splits.train / (self.splits.train + self.splits.val)))
            train_keys = shuffled[:train_end]
            val_keys = shuffled[train_end:]
            if not train_keys or not val_keys:
                raise ValueError("Fixed test split produced empty train or val split.")
            payload = {
                "train": {key: None for key in train_keys},
                "val": {key: None for key in val_keys},
                "cal": {key: None for key in cal_keys},
                "test": {key: None for key in test_keys},
            }
        else:
            sample_entries: list[tuple[str, int]] = []
            for key in train_pool_keys:
                num_samples = int(files_group[key]["X"].shape[0])
                sample_entries.extend((key, idx) for idx in range(num_samples))
            if not sample_entries:
                raise ValueError("No train/val samples found after fixed test selection.")
            perm_indices = rng.permutation(len(sample_entries))
            shuffled_entries = [sample_entries[idx] for idx in perm_indices]
            train_weight = self.splits.train / (self.splits.train + self.splits.val)
            train_end = int(len(shuffled_entries) * train_weight)
            train_entries = shuffled_entries[:train_end]
            val_entries = shuffled_entries[train_end:]
            payload = {
                "train": self._entries_to_indices(train_entries),
                "val": self._entries_to_indices(val_entries),
                "cal": {key: None for key in cal_keys},
                "test": {key: None for key in test_keys},
            }

        split_definition: dict[str, list[str] | str | int] = {
            "split_strategy": split_strategy,
            "train_profiles": sorted(payload["train"].keys()),
            "val_profiles": sorted(payload["val"].keys()),
            "cal_profiles": sorted(payload["cal"].keys()),
            "test_profiles": sorted(test_keys),
        }
        if self.train_profile_limit_with_manifests is not None:
            split_definition["train_profile_limit_with_manifests"] = int(
                self.train_profile_limit_with_manifests
            )
        if extra_meta:
            split_definition.update(extra_meta)
        return payload, split_definition


    @staticmethod
    def _summarize_split_payload(
        files_group: h5py.Group,
        split_payload: dict[str, np.ndarray | None],
    ) -> dict[str, int]:
        window_count = 0
        for key, selected_indices in split_payload.items():
            if selected_indices is None:
                window_count += int(files_group[key]["X"].shape[0])
            else:
                window_count += int(len(selected_indices))
        return {
            "train_profile_count": int(len(split_payload)),
            "train_window_count": int(window_count),
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
        x_stats: dict | None = None
        y_stats: dict | None = None

        for key, selected_indices in split_payload.items():
            file_group = files_group[key]
            x_data = file_group["X"][()]
            y_data = file_group["Y"][()]
            if selected_indices is not None:
                x_data = x_data[selected_indices]
                y_data = y_data[selected_indices]
            x_flat = x_data.reshape(-1, x_data.shape[-1])
            y_flat = y_data.reshape(-1, y_data.shape[-1])
            if x_stats is None:
                x_stats = _init_stats(self.scaling_type, features=x_flat.shape[-1])
            if y_stats is None:
                y_stats = _init_stats(self.scaling_type, features=y_flat.shape[-1])
            x_stats = _update_stats(self.scaling_type, x_stats, x_flat)
            y_stats = _update_stats(self.scaling_type, y_stats, y_flat)

        if x_stats is None or y_stats is None:
            raise ValueError("No samples available to compute scaling statistics.")
        return {"x": _finalize_stats(self.scaling_type, x_stats), "y": _finalize_stats(self.scaling_type, y_stats)}

    def _write_metadata(
        self,
        src_h5f: h5py.File,
        dst_h5f: h5py.File,
        stats: dict,
        split_definition: dict[str, list[str] | str | int],
    ) -> None:
        for key, value in src_h5f.attrs.items():
            dst_h5f.attrs[key] = value
        dst_h5f.attrs["scaling_type"] = self.scaling_type
        dst_h5f.attrs["scaling_stats_source"] = "frozen" if self.frozen_stats_path is not None else "computed_train_split"
        dst_h5f.attrs["frozen_scaler_stats_path"] = "" if self.frozen_stats_path is None else str(self.frozen_stats_path)
        dst_h5f.attrs["saved_scaler_stats_path"] = "" if self.save_stats_path is None else str(self.save_stats_path)
        dst_h5f.attrs["train_fraction"] = self.splits.train
        dst_h5f.attrs["val_fraction"] = self.splits.val
        dst_h5f.attrs["cal_fraction"] = self.splits.cal
        dst_h5f.attrs["test_fraction"] = self.splits.test
        dst_h5f.attrs["split_mode"] = self.split_mode
        split_strategy = str(split_definition.get("split_strategy", "fractional"))
        dst_h5f.attrs["split_strategy"] = split_strategy
        if split_strategy == "fixed_manifest":
            dst_h5f.attrs["test_manifest_path"] = str(
                split_definition.get("test_manifest_path", "")
            )
            dst_h5f.attrs["val_manifest_path"] = str(
                split_definition.get("val_manifest_path", "")
            )
            dst_h5f.attrs["cal_manifest_path"] = str(
                split_definition.get("cal_manifest_path", "")
            )
            dst_h5f.attrs["train_profile_limit_with_manifests"] = int(
                split_definition.get("train_profile_limit_with_manifests", 0)
            )
        if split_strategy == "fixed_count":
            dst_h5f.attrs["split_seed"] = int(split_definition.get("split_seed", self.seed))
            dst_h5f.attrs["test_count"] = int(split_definition.get("test_count", 0))

        split_group = dst_h5f.create_group("split_definition")
        split_group.create_dataset(
            "train_profiles",
            data=np.asarray(split_definition.get("train_profiles", []), dtype="S"),
        )
        split_group.create_dataset(
            "val_profiles",
            data=np.asarray(split_definition.get("val_profiles", []), dtype="S"),
        )
        split_group.create_dataset(
            "cal_profiles",
            data=np.asarray(split_definition.get("cal_profiles", []), dtype="S"),
        )
        split_group.create_dataset(
            "test_profiles",
            data=np.asarray(split_definition.get("test_profiles", []), dtype="S"),
        )

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


def _decode_attr_strings(value: Any) -> list[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        return [item.decode() if isinstance(item, bytes) else str(item)]
    return [item.decode() if isinstance(item, bytes) else str(item) for item in arr.tolist()]


def _input_feature_names_from_attrs(h5f: h5py.File) -> list[str]:
    if "state_feature_names" not in h5f.attrs:
        raise ValueError("Input dataset is missing 'state_feature_names' metadata.")
    if "control_feature_name" not in h5f.attrs:
        raise ValueError("Input dataset is missing 'control_feature_name' metadata.")
    state_names = _decode_attr_strings(h5f.attrs["state_feature_names"])
    control_name = _decode_attr_strings(h5f.attrs["control_feature_name"])[0]
    return [*state_names, control_name]


def _target_names_from_attrs(h5f: h5py.File) -> list[str]:
    if "state_feature_names" not in h5f.attrs:
        raise ValueError("Input dataset is missing 'state_feature_names' metadata.")
    return _decode_attr_strings(h5f.attrs["state_feature_names"])


def _as_jsonable_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    return {
        axis: {name: np.asarray(values, dtype=float).tolist() for name, values in axis_stats.items()}
        for axis, axis_stats in stats.items()
    }


def _stats_from_jsonable(payload: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    return {
        axis: {name: np.asarray(values, dtype=float) for name, values in axis_stats.items()}
        for axis, axis_stats in payload.items()
    }


def write_scaler_stats_json(
    path: Path,
    *,
    scaling_type: ScalingType,
    stats: dict[str, dict[str, np.ndarray]],
    input_feature_names: list[str],
    target_names: list[str],
    train_profile_count: int | None = None,
    train_window_count: int | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scaling_type": scaling_type,
        "input_feature_names": list(input_feature_names),
        "target_names": list(target_names),
        "zero_variance_handling": "replace_zero_std_or_span_with_1.0",
        "epsilon": None,
        "clipping": None,
        "train_profile_count": None if train_profile_count is None else int(train_profile_count),
        "train_window_count": None if train_window_count is None else int(train_window_count),
        "stats": _as_jsonable_stats(stats),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not path.exists() or path.stat().st_size <= 0:
        raise OSError(f"Failed to write scaler stats file: {path}")
    return path


def load_scaler_stats_json(
    path: Path,
    *,
    scaling_type: ScalingType,
    input_feature_names: list[str],
    target_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen scaler stats file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Scaler stats file must contain a JSON object: {path}")
    file_scaling_type = str(payload.get("scaling_type", ""))
    if file_scaling_type != scaling_type:
        raise ValueError(
            f"Frozen scaler type mismatch: file has {file_scaling_type!r}, "
            f"configured scaling_type is {scaling_type!r}."
        )
    saved_inputs = [str(name) for name in payload.get("input_feature_names", [])]
    if saved_inputs != list(input_feature_names):
        raise ValueError(
            "Frozen scaler input feature order mismatch. "
            f"saved={saved_inputs}, current={list(input_feature_names)}."
        )
    saved_targets = [str(name) for name in payload.get("target_names", [])]
    if saved_targets != list(target_names):
        raise ValueError(
            "Frozen scaler target order mismatch. "
            f"saved={saved_targets}, current={list(target_names)}."
        )
    stats_payload = payload.get("stats")
    if not isinstance(stats_payload, dict):
        raise ValueError(f"Scaler stats file missing 'stats' object: {path}")
    stats = _stats_from_jsonable(stats_payload)
    _validate_scaler_stats_arrays(stats, scaling_type, len(input_feature_names), len(target_names), source=str(path))
    return stats


def _validate_scaler_stats_arrays(
    stats: dict[str, dict[str, np.ndarray]],
    scaling_type: ScalingType,
    n_input_features: int,
    n_targets: int,
    *,
    source: str,
) -> None:
    required = {
        "none": {"x": (), "y": ()},
        "standard": {"x": ("mean", "std"), "y": ("mean", "std")},
        "minmax": {"x": ("min", "max", "span"), "y": ("min", "max", "span")},
    }.get(scaling_type)
    if required is None:
        raise ValueError(f"Unsupported scaling_type: {scaling_type}")
    expected_lengths = {"x": n_input_features, "y": n_targets}
    for axis, keys in required.items():
        axis_stats = stats.get(axis)
        if not isinstance(axis_stats, dict):
            raise ValueError(f"Scaler stats from {source} missing axis {axis!r}.")
        for key in keys:
            if key not in axis_stats:
                raise ValueError(f"Scaler stats from {source} missing required array {axis}.{key}.")
            arr = np.asarray(axis_stats[key], dtype=float)
            if arr.shape != (expected_lengths[axis],):
                raise ValueError(
                    f"Scaler stats array {axis}.{key} from {source} has shape {arr.shape}; "
                    f"expected {(expected_lengths[axis],)}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Scaler stats array {axis}.{key} from {source} contains non-finite values.")
        if scaling_type == "standard" and keys:
            if np.any(np.asarray(axis_stats["std"], dtype=float) == 0):
                raise ValueError(f"Scaler stats from {source} contain zero standard deviations for axis {axis}.")
        if scaling_type == "minmax" and keys:
            if np.any(np.asarray(axis_stats["span"], dtype=float) == 0):
                raise ValueError(f"Scaler stats from {source} contain zero spans for axis {axis}.")


def _log_stats_comparison(
    scaling_type: ScalingType,
    frozen_stats: dict[str, dict[str, np.ndarray]],
    current_stats: dict[str, dict[str, np.ndarray]],
) -> None:
    if scaling_type == "none":
        print("[scale] Frozen scaling uses type 'none'; no numeric stats to compare.")
        return
    keys = ("mean", "std") if scaling_type == "standard" else ("min", "max", "span")
    parts = []
    for axis in ("x", "y"):
        for key in keys:
            frozen = np.asarray(frozen_stats[axis][key], dtype=float)
            current = np.asarray(current_stats[axis][key], dtype=float)
            parts.append(f"{axis}.{key}_max_abs_delta={float(np.max(np.abs(current - frozen))):.6g}")
    print("[scale] Frozen-vs-current scaler stat deltas: " + ", ".join(parts))


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
