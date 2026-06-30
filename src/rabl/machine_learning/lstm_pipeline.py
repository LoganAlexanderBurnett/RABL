"""
RABL LSTM pipeline utilities (datasets -> training -> rolling forecast -> plotting)

This module merges:
  - inspect_lstm_dataloaders.py (dataset/model/forecast utilities)
  - train_lstm_and_forecast.py (device selection, fallback training, plotting)

Intended usage (example):

    from pathlib import Path
    from rabl.machine_learning.lstm_pipeline import (
        build_datasets,
        inspect_dataset_shapes,
        train_with_fallback,
        rolling_forecast,
        plot_forecast_vs_truth_grid,
        TARGET_NAMES,
    )

    datasets = build_datasets(Path("your_dataset.h5"), batch_size=32, seed=123)
    inspect_dataset_shapes(datasets)
    output_root = resolve_output_root()
    model, history, used_device = train_with_fallback(datasets, epochs=50, out_dir=output_root)

    name, x_prof, y_prof = next(iter(datasets["test_profile_ds"]))
    x_prof = x_prof.numpy()
    y_prof = y_prof.numpy()
    y_pred = rolling_forecast(model, x_prof)

    plot_forecast_vs_truth_grid(
        x_profile=x_prof,
        y_true=y_prof,
        y_pred=y_pred,
        target_names=TARGET_NAMES,
        title=f"Rolling Forecast - {name}",
        save_path=output_root / "rolling_forecast.png",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

from rabl.paths import resolve_output_root
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import random
import gc
import re
import json

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from .branchpoint_finder import finite_difference
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from torch import nn
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader, IterableDataset


def _iter_with_optional_tqdm(
    iterable: Iterable[Any],
    *,
    use_tqdm: bool,
    **kwargs: Any,
) -> Iterable[Any]:
    if use_tqdm:
        return tqdm(iterable, **kwargs)
    return iterable


def _init_io_stats() -> dict[str, float]:
    return {
        "h5_read_s": 0.0,
        "astype_s": 0.0,
        "profiles_read": 0.0,
        "samples_yielded": 0.0,
    }

# --------------------------------------------------------------------------------------
# Defaults / naming
# --------------------------------------------------------------------------------------

STATE_DIM = 15
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100

# The 15 targets you are predicting (y has shape (num_steps, 15))
TARGET_NAMES: list[str] = [
    "TN2",
    "Tm",
    "Thp",
    "Tf",
    "Tsg",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "n",
    "rho_dollars",
    "T_steam_out",
    "x_steam_out",
]

FORECAST_PLOT_TARGET_ORDER: list[str] = [
    "Tf",
    "Tm",
    "Thp",
    "TN2",
    "Tsg",
    "T_steam_out",
    "x_steam_out",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "n",
    "rho_dollars",
]


def _reorder_forecast_plot_targets(
    target_names: list[str],
    *arrays: np.ndarray | None,
) -> tuple[list[str], list[np.ndarray | None]]:
    """Reorder target names/columns for forecast plotting only."""
    index_by_name = {name: idx for idx, name in enumerate(target_names)}
    ordered_names = [name for name in FORECAST_PLOT_TARGET_ORDER if name in index_by_name]
    ordered_indices = [index_by_name[name] for name in ordered_names]
    reordered_arrays: list[np.ndarray | None] = []
    for arr in arrays:
        if arr is None:
            reordered_arrays.append(None)
        elif arr.ndim == 2:
            reordered_arrays.append(arr[:, ordered_indices])
        elif arr.ndim == 3:
            reordered_arrays.append(arr[:, :, ordered_indices])
        else:
            raise ValueError(f"Cannot reorder forecast plot targets for array with shape {arr.shape}.")
    return ordered_names, reordered_arrays


def _disable_y_offset_if_requested(ax: plt.Axes, target_name: str) -> None:
    if target_name != "x_steam_out":
        return
    formatter = ax.yaxis.get_major_formatter()
    if hasattr(formatter, "set_useOffset"):
        formatter.set_useOffset(False)


# --------------------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------------------

def set_global_determinism(seed: int) -> None:
    """
    Configure Python/NumPy/PyTorch RNG state for reproducible runs.
    Uses deterministic CUDA/cuDNN behavior when available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device_prefer_gpu() -> torch.device:
    """
    Prefer GPU if CUDA is available, otherwise use CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}\n")
        return device

    print("No GPU detected by PyTorch. Using CPU.")
    return torch.device("cpu")


def clear_cuda_cache() -> None:
    """
    Clear cached CUDA memory to help free VRAM.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_cuda(model: nn.Module | None, used_device: torch.device | str | None = None) -> None:
    """
    Release model references and aggressively clean CUDA memory.

    This should be called only after caller-side references are dropped as well
    (for example, caller does ``del model`` immediately after calling this helper).
    """
    del model
    gc.collect()

    device_str = "" if used_device is None else str(used_device).lower()
    if "cuda" in device_str and torch.cuda.is_available():
        torch.cuda.synchronize()
        clear_cuda_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        clear_cuda_cache()


def clean_cuda(model: nn.Module | None, used_device: torch.device | str | None = None) -> None:
    """Backward-compatible alias for :func:`cleanup_cuda`."""
    cleanup_cuda(model, used_device)


# --------------------------------------------------------------------------------------
# HDF5 helpers
# --------------------------------------------------------------------------------------

def _get_profile_names(h5_path: Path, split: str) -> list[str]:
    with h5py.File(h5_path, "r") as h5f:
        return sorted(h5f[split]["files"].keys())


def _get_profile_shapes(h5_path: Path, split: str, profile_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with h5py.File(h5_path, "r") as h5f:
        group = h5f[split]["files"][profile_name]
        return tuple(group["X"].shape), tuple(group["Y"].shape)


def _count_samples_in_split(h5_path: Path, split: str, profile_names: list[str]) -> int:
    """
    Count number of (X,Y) samples in a split WITHOUT loading all samples into RAM.
    """
    if not profile_names:
        return 0
    total = 0
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for name in profile_names:
            total += int(files_group[name]["X"].shape[0])
    return total


def _infer_scaling_type(scaling_group: h5py.Group) -> str:
    keys = set(scaling_group.keys())
    if {"x_mean", "x_std", "y_mean", "y_std"}.issubset(keys):
        return "standard"
    if {"x_min", "x_max", "x_span", "y_min", "y_max", "y_span"}.issubset(keys):
        return "minmax"
    raise ValueError(f"Unable to infer scaling type from scaling group keys: {sorted(keys)}")


def _load_scaling_stats(h5_path: Path) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as h5f:
        if "scaling" not in h5f:
            raise KeyError("HDF5 file missing required 'scaling' group.")
        scaling_group = h5f["scaling"]
        scaling_type = _infer_scaling_type(scaling_group)
        if scaling_type == "standard":
            return {
                "type": scaling_type,
                "x": {
                    "mean": scaling_group["x_mean"][...].astype(np.float32),
                    "std": scaling_group["x_std"][...].astype(np.float32),
                },
                "y": {
                    "mean": scaling_group["y_mean"][...].astype(np.float32),
                    "std": scaling_group["y_std"][...].astype(np.float32),
                },
            }
        return {
            "type": scaling_type,
            "x": {
                "min": scaling_group["x_min"][...].astype(np.float32),
                "span": scaling_group["x_span"][...].astype(np.float32),
            },
            "y": {
                "min": scaling_group["y_min"][...].astype(np.float32),
                "span": scaling_group["y_span"][...].astype(np.float32),
            },
        }


def _descale_targets(h5_path: Path, values: np.ndarray) -> np.ndarray:
    stats = _load_scaling_stats(h5_path)
    return _descale_targets_from_stats(stats, values)


def _descale_targets_from_stats(stats: dict[str, Any], values: np.ndarray) -> np.ndarray:
    scaling_type = stats["type"]
    y_stats = stats["y"]
    if scaling_type == "standard":
        return values * y_stats["std"] + y_stats["mean"]
    if scaling_type == "minmax":
        return values * y_stats["span"] + y_stats["min"]
    raise ValueError(f"Unsupported scaling type: {scaling_type}")


def _descale_feature_from_stats(stats: dict[str, Any], values: np.ndarray, feature_idx: int) -> np.ndarray:
    scaling_type = stats["type"]
    x_stats = stats["x"]
    if scaling_type == "standard":
        return values * x_stats["std"][feature_idx] + x_stats["mean"][feature_idx]
    if scaling_type == "minmax":
        return values * x_stats["span"][feature_idx] + x_stats["min"][feature_idx]
    raise ValueError(f"Unsupported scaling type: {scaling_type}")


def _decode_columns(columns_attr: np.ndarray | list[Any]) -> list[str]:
    """Decode HDF5 ``columns`` attrs that may contain raw bytes."""
    decoded: list[str] = []
    for item in columns_attr:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _train_sample_generator(
    h5_path: Path,
    profile_names: list[str],
    split: str,
    seed: int,
    io_stats: dict[str, float] | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for profile_name in profile_names:
            group = files_group[profile_name]
            read_start = perf_counter()
            x_raw = group["X"][...]
            y_raw = group["Y"][...]
            read_elapsed = perf_counter() - read_start

            cast_start = perf_counter()
            x_data = x_raw.astype(np.float32)
            y_data = y_raw.astype(np.float32)
            cast_elapsed = perf_counter() - cast_start

            if io_stats is not None:
                io_stats["h5_read_s"] += read_elapsed
                io_stats["astype_s"] += cast_elapsed
                io_stats["profiles_read"] += 1

            indices = np.arange(x_data.shape[0])
            rng.shuffle(indices)
            for idx in indices:
                if io_stats is not None:
                    io_stats["samples_yielded"] += 1
                yield x_data[idx], y_data[idx]


def _sample_generator(
    h5_path: Path,
    profile_names: list[str],
    split: str,
    io_stats: dict[str, float] | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for profile_name in profile_names:
            group = files_group[profile_name]
            read_start = perf_counter()
            x_raw = group["X"][...]
            y_raw = group["Y"][...]
            read_elapsed = perf_counter() - read_start

            cast_start = perf_counter()
            x_data = x_raw.astype(np.float32)
            y_data = y_raw.astype(np.float32)
            cast_elapsed = perf_counter() - cast_start

            if io_stats is not None:
                io_stats["h5_read_s"] += read_elapsed
                io_stats["astype_s"] += cast_elapsed
                io_stats["profiles_read"] += 1

            for idx in range(x_data.shape[0]):
                if io_stats is not None:
                    io_stats["samples_yielded"] += 1
                yield x_data[idx], y_data[idx]


def _profile_generator(h5_path: Path, profile_names: list[str], split: str) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for profile_name in profile_names:
            group = files_group[profile_name]
            x_data = group["X"][...].astype(np.float32)
            y_data = group["Y"][...].astype(np.float32)
            yield profile_name, x_data, y_data


class SampleDataset(IterableDataset):
    def __init__(self, h5_path: Path, profile_names: list[str], split: str, seed: int | None = None):
        self.h5_path = Path(h5_path)
        self.profile_names = list(profile_names)
        self.split = split
        self.seed = seed
        self.io_stats = _init_io_stats()

    def get_and_reset_io_stats(self) -> dict[str, float]:
        snapshot = dict(self.io_stats)
        self.io_stats = _init_io_stats()
        return snapshot

    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        if self.seed is None:
            generator = _sample_generator(self.h5_path, self.profile_names, self.split, io_stats=self.io_stats)
        else:
            generator = _train_sample_generator(
                self.h5_path,
                self.profile_names,
                self.split,
                self.seed,
                io_stats=self.io_stats,
            )
        for x_data, y_data in generator:
            yield torch.from_numpy(x_data), torch.from_numpy(y_data)


class ProfileDataset(IterableDataset):
    def __init__(self, h5_path: Path, profile_names: list[str], split: str):
        self.h5_path = Path(h5_path)
        self.profile_names = list(profile_names)
        self.split = split

    def __iter__(self) -> Iterable[tuple[str, torch.Tensor, torch.Tensor]]:
        for profile_name, x_data, y_data in _profile_generator(self.h5_path, self.profile_names, self.split):
            yield profile_name, torch.from_numpy(x_data), torch.from_numpy(y_data)


# --------------------------------------------------------------------------------------
# Dataset building
# --------------------------------------------------------------------------------------

def build_datasets(h5_path: Path, batch_size: int, seed: int) -> dict[str, Any]:
    """
    Returns a dict that includes torch DataLoaders plus helpful metadata.
    """
    train_profiles = _get_profile_names(h5_path, "train")
    val_profiles = _get_profile_names(h5_path, "val")
    test_profiles = _get_profile_names(h5_path, "test")

    if not train_profiles:
        raise ValueError("No training profiles found in HDF5.")

    x_shape, y_shape = _get_profile_shapes(h5_path, "train", train_profiles[0])

    # Total sample counts (used for steps_per_epoch and more informative inspection)
    train_num_samples = _count_samples_in_split(h5_path, "train", train_profiles)
    val_num_samples = _count_samples_in_split(h5_path, "val", val_profiles)
    test_num_samples = _count_samples_in_split(h5_path, "test", test_profiles)

    train_steps = max(1, ceil(train_num_samples / batch_size))
    val_steps = max(1, ceil(val_num_samples / batch_size))

    # Train dataset: shuffled sample order per profile
    train_ds = SampleDataset(h5_path, train_profiles, "train", seed=seed)
    # NOTE: keep num_workers=0 while diagnosing HDF5 I/O timing. If num_workers>0,
    # each worker has its own dataset instance and per-dataset counters are not aggregated.
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, num_workers=0)

    # Validation samples dataset (flat)
    val_sample_ds = SampleDataset(h5_path, val_profiles, "val")
    val_sample_loader = DataLoader(val_sample_ds, batch_size=batch_size, pin_memory=True, num_workers=0)

    # Profile datasets: yields entire profile arrays
    val_profile_ds = ProfileDataset(h5_path, val_profiles, "val")
    test_profile_ds = ProfileDataset(h5_path, test_profiles, "test")

    return {
        # datasets
        "train": train_loader,
        "train_ds": train_ds,
        "val_samples": val_sample_loader,
        "val_sample_ds": val_sample_ds,
        "val_profile_ds": val_profile_ds,
        "test_profile_ds": test_profile_ds,
        # metadata
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


# --------------------------------------------------------------------------------------
# Model + training
# --------------------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    def __init__(
        self,
        timesteps: int,
        num_features: int,
        num_targets: int,
        *,
        n_lstm: int = 1,
        lstm_hidden: int = 64,
        lstm_dropout: float = 0.0,
        n_fc: int = 1,
        fc_hidden: tuple[int, ...] = (64,),
    ):
        super().__init__()
        if n_lstm < 1:
            raise ValueError(f"n_lstm must be >= 1 (got {n_lstm}).")
        if n_fc < 1:
            raise ValueError(f"n_fc must be >= 1 (got {n_fc}).")
        if len(fc_hidden) != n_fc:
            raise ValueError("fc_hidden must be a tuple of length n_fc.")

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=n_lstm,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.fc_layers = nn.ModuleList()
        fc_input = lstm_hidden
        for hidden_size in fc_hidden:
            self.fc_layers.append(nn.Linear(fc_input, hidden_size))
            fc_input = hidden_size

        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(fc_hidden[-1], num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        fc_out = last_step
        for fc_layer in self.fc_layers:
            fc_out = self.relu(fc_layer(fc_out))
        return self.output_layer(fc_out)


def build_model(
    timesteps: int,
    num_features: int,
    num_targets: int,
    *,
    n_lstm: int = 1,
    lstm_hidden: int = 64,
    lstm_dropout: float = 0.0,
    n_fc: int = 1,
    fc_hidden: tuple[int, ...] = (64,),
) -> LSTMRegressor:
    model = LSTMRegressor(
        timesteps,
        num_features,
        num_targets,
        n_lstm=n_lstm,
        lstm_hidden=lstm_hidden,
        lstm_dropout=lstm_dropout,
        n_fc=n_fc,
        fc_hidden=fc_hidden,
    )
    fc_desc = ", ".join(str(size) for size in fc_hidden)
    print(
        "LSTMRegressor architecture:\n"
        f"  Input features: {num_features}\n"
        f"  LSTM layers ({n_lstm}): hidden_size={lstm_hidden}, dropout={lstm_dropout}\n"
        f"  FC layers ({n_fc}): [{fc_desc}]\n"
        f"  Output targets: {num_targets}\n"
    )
    return model


def _train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    data_already_on_device: bool = False,
    profiler: torch.profiler.profile | None = None,
    use_tqdm: bool = True,
    verbose: int = 1,
) -> tuple[float, float, float, float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    data_wait_time_s = 0.0
    h2d_time_s = 0.0
    compute_time_s = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    progress = _iter_with_optional_tqdm(
        loader,
        use_tqdm=use_tqdm,
        total=total_batches,
        desc=f"Train {epoch}/{total_epochs}",
        unit="batch",
    )
    fetch_start = perf_counter()
    for x_batch, y_batch in progress:
        data_wait_time_s += perf_counter() - fetch_start

        transfer_start = perf_counter()
        if not data_already_on_device:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        h2d_time_s += perf_counter() - transfer_start

        compute_start = perf_counter()
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        if profiler is not None:
            profiler.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        total_loss += float(loss.item())
        compute_time_s += perf_counter() - compute_start
        num_batches += 1
        if use_tqdm and hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{loss.item():.5e}")
        fetch_start = perf_counter()
    if use_tqdm and hasattr(progress, "close"):
        progress.close()
    max_mem = 0
    if device.type == "cuda":
        max_mem = int(torch.cuda.max_memory_allocated(device))
    return total_loss / max(1, num_batches), data_wait_time_s, h2d_time_s, compute_time_s, max_mem


def _preload_train_batches_to_device(
    loader: DataLoader,
    device: torch.device,
    *,
    use_tqdm: bool = True,
    verbose: int = 1,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], float]:
    """
    Materialize training batches onto `device` once so subsequent epochs avoid per-batch H2D copies.
    Returns (batches_on_device, preload_time_seconds).
    """
    preloaded_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    preload_start = perf_counter()
    preload_iter = _iter_with_optional_tqdm(
        loader,
        use_tqdm=use_tqdm,
        desc="Preloading train batches",
        unit="batch",
    )
    for x_batch, y_batch in preload_iter:
        preloaded_batches.append((x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preload_time_s = perf_counter() - preload_start
    return preloaded_batches, preload_time_s


def _evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    data_already_on_device: bool = False,
    use_tqdm: bool = True,
    verbose: int = 1,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        val_iter = _iter_with_optional_tqdm(
            loader,
            use_tqdm=use_tqdm,
            total=total_batches,
            desc=f"Val {epoch}/{total_epochs}",
            unit="batch",
        )
        for x_batch, y_batch in val_iter:
            if not data_already_on_device:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            total_loss += float(loss.item())
            num_batches += 1
    return total_loss / max(1, num_batches)


def train_model(
    datasets: dict[str, Any],
    *,
    epochs: int = DEFAULT_EPOCHS,
    plot_path: str | Path | None = None,
    training_device: torch.device | None = None,
    n_lstm: int = 1,
    lstm_hidden: int = 64,
    lstm_dropout: float = 0.0,
    n_fc: int = 1,
    fc_hidden: tuple[int, ...] = (64,),
    learning_rate: float = 1e-3,
    step_lr_step_size: int = 30,
    step_lr_gamma: float = 0.5,
    verbose: int = 1,
    preload_train_to_device: bool = False,
    preload_val_to_device: bool = False,
    deterministic_seed: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    restore_best_weights: bool = True,
    enable_torch_profiler: bool = False,
    profiler_wait_steps: int = 1,
    profiler_warmup_steps: int = 1,
    profiler_active_steps: int = 3,
    profiler_repeat: int = 1,
    profiler_row_limit: int = 30,
    use_tqdm: bool = True,
    resume_from_weights: Path | None = None,
    save_training_curves: bool = True,
) -> tuple[nn.Module, dict[str, list[float]], Path | None]:
    """
    Train the LSTM and save a train/val curve plot.

    Args:
        early_stopping_patience: If set to a positive integer, stop training when
            validation loss does not improve by more than `early_stopping_min_delta`
            for this many consecutive epochs.
        early_stopping_min_delta: Minimum decrease in validation loss required to
            count as an improvement.
        restore_best_weights: When early stopping is enabled, restore model weights
            from the best validation-loss epoch before returning.
    """
    timesteps = int(datasets["sample_shape"][1])
    num_features = int(datasets["sample_shape"][2])
    num_targets = int(datasets["target_shape"][1])

    if training_device is None:
        training_device = choose_device_prefer_gpu()

    resolved_seed = int(datasets.get("seed", 0)) if deterministic_seed is None else int(deterministic_seed)
    set_global_determinism(resolved_seed)
    if verbose:
        print(f"Deterministic seed set to: {resolved_seed}")
        train_n = int(datasets.get("train_num_samples", 0))
        val_n = int(datasets.get("val_num_samples", 0))
        test_profile_count = len(datasets.get("test_profile_names", []))
        print(
            "Dataset summary: "
            f"train_samples={train_n:,}, "
            f"val_samples={val_n:,}, "
            f"test_profiles={test_profile_count:,}"
        )

    model = build_model(
        timesteps,
        num_features,
        num_targets,
        n_lstm=n_lstm,
        lstm_hidden=lstm_hidden,
        lstm_dropout=lstm_dropout,
        n_fc=n_fc,
        fc_hidden=fc_hidden,
    ).to(training_device)
    if resume_from_weights is not None:
        state_dict = torch.load(resume_from_weights, map_location=training_device)
        model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_lr_step_size,
        gamma=step_lr_gamma,
    )
    loss_fn = nn.MSELoss()

    history = {
        "loss": [],
        "val_loss": [],
        "lr": [],
        "data_wait_time": [],
        "h2d_time": [],
        "compute_time": [],
        "val_time": [],
    }

    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be > 0 when provided.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")

    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    best_epoch = 0

    train_source: Iterable[tuple[torch.Tensor, torch.Tensor]] = datasets["train"]
    val_source: Iterable[tuple[torch.Tensor, torch.Tensor]] = datasets["val_samples"]
    preloaded_in_gpu = False
    val_preloaded_in_gpu = False
    preload_time_s = 0.0
    val_preload_time_s = 0.0
    if preload_train_to_device:
        if training_device.type != "cuda":
            print("preload_train_to_device=True requested, but training device is CPU. Skipping preload.")
        else:
            train_source, preload_time_s = _preload_train_batches_to_device(
                datasets["train"],
                training_device,
                use_tqdm=use_tqdm,
                verbose=verbose,
            )
            preloaded_in_gpu = True
            print(
                f"Preloaded {len(train_source)} training batches to {training_device} in {preload_time_s:.2f}s."
            )
    if preload_val_to_device:
        if training_device.type != "cuda":
            print("preload_val_to_device=True requested, but training device is CPU. Skipping preload.")
        else:
            val_source, val_preload_time_s = _preload_train_batches_to_device(
                datasets["val_samples"],
                training_device,
                use_tqdm=use_tqdm,
                verbose=verbose,
            )
            val_preloaded_in_gpu = True
            print(
                f"Preloaded {len(val_source)} validation batches to {training_device} in {val_preload_time_s:.2f}s."
            )

    resolved_plot_path = Path(plot_path) if plot_path is not None else resolve_output_root() / "plots" / "lstm_training_curves.png"

    profiler: torch.profiler.profile | None = None
    profiler_trace_dir = resolved_plot_path.parent / "torch_profiler_traces" if enable_torch_profiler else None
    if enable_torch_profiler:
        if min(profiler_wait_steps, profiler_warmup_steps, profiler_active_steps, profiler_repeat) < 1:
            raise ValueError("Profiler wait/warmup/active/repeat steps must be >= 1.")
        assert profiler_trace_dir is not None
        profiler_trace_dir.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if training_device.type == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=profiler_wait_steps,
                warmup=profiler_warmup_steps,
                active=profiler_active_steps,
                repeat=profiler_repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_trace_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        profiler.start()

    try:
        for epoch in range(1, epochs + 1):
            train_loss, data_wait_time_s, h2d_time_s, compute_time_s, max_mem = _train_one_epoch(
                model,
                train_source,
                optimizer,
                loss_fn,
                training_device,
                epoch=epoch,
                total_epochs=epochs,
                data_already_on_device=preloaded_in_gpu,
                profiler=profiler,
                use_tqdm=use_tqdm,
                verbose=verbose,
            )

            if training_device.type =="cuda":
                torch.cuda.synchronize(training_device)

            val_start = perf_counter()
            val_loss = _evaluate(
                model,
                val_source,
                loss_fn,
                training_device,
                epoch=epoch,
                total_epochs=epochs,
                data_already_on_device=val_preloaded_in_gpu,
                use_tqdm=use_tqdm,
                verbose=verbose,
            )

            if training_device.type == "cuda":
                torch.cuda.synchronize(training_device)
            val_time_s = perf_counter() - val_start
            epoch_total_time_s = data_wait_time_s + h2d_time_s + compute_time_s + val_time_s

            scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)
            history["data_wait_time"].append(data_wait_time_s)
            history["h2d_time"].append(h2d_time_s)
            history["compute_time"].append(compute_time_s)
            history["val_time"].append(val_time_s)
            train_io_stats = datasets.get("train_ds").get_and_reset_io_stats() if datasets.get("train_ds") is not None else None
            mem_mb = max_mem / (1024**2)
            io_msg = ""
            if train_io_stats is not None:
                io_total_s = float(train_io_stats["h5_read_s"] + train_io_stats["astype_s"])
                samples = float(train_io_stats["samples_yielded"])
                io_tput = samples / io_total_s if io_total_s > 0 else 0.0
                io_frac = io_total_s / data_wait_time_s if data_wait_time_s > 0 else 0.0
                io_msg = (
                    f" - io_h5: {train_io_stats['h5_read_s']:.2f}s"
                    f" - io_cast: {train_io_stats['astype_s']:.2f}s"
                    f" - io_profiles: {int(train_io_stats['profiles_read'])}"
                    f" - io_samples: {int(samples)}"
                    f" - io_tput: {io_tput:.2f} samp/s"
                    f" - io_frac_of_data_wait: {io_frac:.2%}"
                )

            print(
                f"Epoch {epoch}/{epochs} - loss: {train_loss:.5e} - val_loss: {val_loss:.5e} "
                f"- lr: {current_lr:.3e} - data_wait: {data_wait_time_s:.2f}s "
                f"- h2d: {h2d_time_s:.2f}s - compute: {compute_time_s:.2f}s "
                f"- val_time: {val_time_s:.2f}s - epoch_total: {epoch_total_time_s:.2f}s "
                f"- preloaded: {preloaded_in_gpu} "
                f"- preload_time: {preload_time_s:.2f}s "
                f"- val_preloaded: {val_preloaded_in_gpu} "
                f"- val_preload_time: {val_preload_time_s:.2f}s - max_cuda_mem: {mem_mb:.2f} MB"
                f"{io_msg}"
            )
            print(f"Epoch {epoch}/{epochs} total_time_s: {epoch_total_time_s:.2f}")

            if early_stopping_patience is not None:
                if val_loss < (best_val_loss - early_stopping_min_delta):
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    if restore_best_weights:
                        best_state_dict = {
                            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                        }
                else:
                    epochs_without_improvement += 1
                    if verbose:
                        print(
                            "Early stopping check: "
                            f"{epochs_without_improvement}/{early_stopping_patience} "
                            "epochs without validation improvement."
                        )
                    if epochs_without_improvement >= early_stopping_patience:
                        if verbose:
                            print(
                                "Early stopping triggered at "
                                f"epoch {epoch}; best validation loss was {best_val_loss:.5e} "
                                f"at epoch {best_epoch}."
                            )
                        break
    finally:
        if profiler is not None:
            profiler.stop()
            summary_text = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=profiler_row_limit)
            summary_path = resolved_plot_path.parent / "torch_profiler_summary.txt"
            summary_path.write_text(summary_text + "\n", encoding="utf-8")
            if verbose:
                print(f"Saved PyTorch profiler summary to {summary_path}")
                print(f"Saved PyTorch profiler traces to {profiler_trace_dir}")

    if early_stopping_patience is not None and restore_best_weights and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if verbose:
            print(f"Restored best model weights from epoch {best_epoch}.")

    if save_training_curves:
        resolved_plot_path.parent.mkdir(parents=True, exist_ok=True)

        epochs_range = range(1, len(history["loss"]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, history["loss"], label="Train Loss (MSE)")
        plt.plot(epochs_range, history["val_loss"], label="Val Loss (MSE)")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(resolved_plot_path, dpi=150)
        print(f"Saved training curves to {resolved_plot_path}")

        resolved_lr_plot_path = resolved_plot_path.with_name("lstm_lr_curve.png")
        epochs_range = range(1, len(history["lr"]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, history["lr"], label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule (StepLR)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(resolved_lr_plot_path, dpi=150)
        print(f"Saved learning-rate curve to {resolved_lr_plot_path}")
    else:
        resolved_plot_path = None

    if verbose:
        total_data_wait_s = float(sum(history["data_wait_time"]))
        total_h2d_s = float(sum(history["h2d_time"]))
        total_compute_s = float(sum(history["compute_time"]))
        total_val_s = float(sum(history["val_time"]))
        total_train_epoch_s = total_data_wait_s + total_h2d_s + total_compute_s
        total_wall_estimate_s = preload_time_s + val_preload_time_s + total_train_epoch_s + total_val_s
        print("\nTiming summary (cumulative):")
        print(f"  preload_time: {preload_time_s:.2f}s")
        print(f"  val_preload_time: {val_preload_time_s:.2f}s")
        print(f"  train_data_wait_time: {total_data_wait_s:.2f}s")
        print(f"  train_h2d_time: {total_h2d_s:.2f}s")
        print(f"  train_compute_time: {total_compute_s:.2f}s")
        print(f"  train_epoch_time_total: {total_train_epoch_s:.2f}s")
        print(f"  val_time_total: {total_val_s:.2f}s")
        print(f"  estimated_total_time: {total_wall_estimate_s:.2f}s")

    return model, history, resolved_plot_path


def train_with_fallback(
    datasets: dict[str, Any],
    *,
    epochs: int,
    out_dir: Path,
    n_lstm: int = 1,
    lstm_hidden: int = 64,
    lstm_dropout: float = 0.0,
    n_fc: int = 1,
    fc_hidden: tuple[int, ...] = (64,),
    learning_rate: float = 1e-3,
    step_lr_step_size: int = 30,
    step_lr_gamma: float = 0.5,
    verbose: int = 1,
    prefer_gpu: bool = True,
    preload_train_to_device: bool = False,
    preload_val_to_device: bool = False,
    deterministic_seed: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    restore_best_weights: bool = True,
    enable_torch_profiler: bool = False,
    profiler_wait_steps: int = 1,
    profiler_warmup_steps: int = 1,
    profiler_active_steps: int = 3,
    profiler_repeat: int = 1,
    profiler_row_limit: int = 30,
    use_tqdm: bool = True,
    resume_from_weights: Path | None = None,
    save_training_curves: bool = True,
) -> tuple[nn.Module, dict[str, list[float]], torch.device]:
    """
    Try GPU first (if prefer_gpu). If anything fails, retry on CPU.
    Returns (model, history, used_device).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preferred = choose_device_prefer_gpu() if prefer_gpu else torch.device("cpu")

    # Attempt 1
    try:
        model, history, _curve_path = train_model(
            datasets,
            epochs=epochs,
            plot_path=out_dir / "lstm_train_val_curve.png",
            training_device=preferred,
            n_lstm=n_lstm,
            lstm_hidden=lstm_hidden,
            lstm_dropout=lstm_dropout,
            n_fc=n_fc,
            fc_hidden=fc_hidden,
            learning_rate=learning_rate,
            step_lr_step_size=step_lr_step_size,
            step_lr_gamma=step_lr_gamma,
            verbose=verbose,
            preload_train_to_device=preload_train_to_device,
            preload_val_to_device=preload_val_to_device,
            deterministic_seed=deterministic_seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            restore_best_weights=restore_best_weights,
            enable_torch_profiler=enable_torch_profiler,
            profiler_wait_steps=profiler_wait_steps,
            profiler_warmup_steps=profiler_warmup_steps,
            profiler_active_steps=profiler_active_steps,
            profiler_repeat=profiler_repeat,
            profiler_row_limit=profiler_row_limit,
            use_tqdm=use_tqdm,
            resume_from_weights=resume_from_weights,
            save_training_curves=save_training_curves,
        )
        return model, history, preferred
    except Exception as e:
        print("\nTraining failed on preferred device:", preferred)
        print("Reason:", repr(e))
        print("Retrying on CPU...\n")

    # Attempt 2 (CPU)
    model, history, _curve_path = train_model(
        datasets,
        epochs=epochs,
        plot_path=out_dir / "lstm_train_val_curve.png",
        training_device=torch.device("cpu"),
        n_lstm=n_lstm,
        lstm_hidden=lstm_hidden,
        lstm_dropout=lstm_dropout,
        n_fc=n_fc,
        fc_hidden=fc_hidden,
        learning_rate=learning_rate,
        step_lr_step_size=step_lr_step_size,
        step_lr_gamma=step_lr_gamma,
        verbose=verbose,
        preload_train_to_device=False,
        preload_val_to_device=False,
        deterministic_seed=deterministic_seed,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        restore_best_weights=restore_best_weights,
        enable_torch_profiler=enable_torch_profiler,
        profiler_wait_steps=profiler_wait_steps,
        profiler_warmup_steps=profiler_warmup_steps,
        profiler_active_steps=profiler_active_steps,
        profiler_repeat=profiler_repeat,
        profiler_row_limit=profiler_row_limit,
        use_tqdm=use_tqdm,
        resume_from_weights=resume_from_weights,
        save_training_curves=save_training_curves,
    )
    return model, history, torch.device("cpu")


# --------------------------------------------------------------------------------------
# Rolling forecast
# --------------------------------------------------------------------------------------

def rolling_forecast(model: nn.Module, x_profile: np.ndarray, *, state_dim: int = STATE_DIM) -> np.ndarray:
    """
    Rolling forecast over a single profile.

    Parameters
    ----------
    x_profile:
        (num_steps, timesteps, num_features) numpy array.
        Assumes first `state_dim` are state features and remaining are control features.
    """
    if x_profile.ndim != 3:
        raise ValueError(f"x_profile must be (steps,timesteps,features), got {x_profile.shape}")

    _timesteps, num_features = x_profile.shape[1:]
    if num_features <= state_dim:
        raise ValueError("Expected control features appended to state features.")
    control_dim = num_features - state_dim

    window_states = x_profile[0, :, :state_dim].copy()
    preds: list[np.ndarray] = []

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for step in range(x_profile.shape[0]):
            control_window = x_profile[step, :, state_dim : state_dim + control_dim]
            input_window = np.concatenate([window_states, control_window], axis=1)
            input_tensor = torch.from_numpy(input_window[None, ...]).to(device)
            pred = model(input_tensor).cpu().numpy()[0]
            preds.append(pred)

            # Slide state window forward by 1 (append prediction)
            if step + 1 < x_profile.shape[0]:
                window_states = np.vstack([window_states[1:], pred])

    return np.asarray(preds, dtype=np.float32)


def teacher_forcing_forecast(model: nn.Module, x_profile: np.ndarray) -> np.ndarray:
    """
    One-step-ahead forecast over a single profile using ground-truth history.

    Unlike :func:`rolling_forecast`, this does not feed predictions back into
    later input windows. Each timestep prediction is made from the corresponding
    prebuilt dataset window, whose state-history channels contain ground-truth
    values.
    """
    if x_profile.ndim != 3:
        raise ValueError(f"x_profile must be (steps,timesteps,features), got {x_profile.shape}")

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        input_tensor = torch.from_numpy(x_profile).to(device)
        pred = model(input_tensor).cpu().numpy()
    return np.asarray(pred, dtype=np.float32)


def _extract_control_series(
    x_profile: np.ndarray,
    *,
    state_dim: int,
    control_channel: int,
) -> np.ndarray:
    if x_profile.ndim != 3:
        raise ValueError(f"x_profile must be 3D (steps,timesteps,features). Got {x_profile.shape}")
    _num_steps, _timesteps, num_features = x_profile.shape
    control_dim = num_features - state_dim
    if control_dim <= 0:
        raise ValueError(
            f"control_dim <= 0 (num_features={num_features}, state_dim={state_dim}). "
            "This violates the rolling_forecast assumption."
        )
    if not (0 <= control_channel < control_dim):
        raise ValueError(f"control_channel={control_channel} out of range [0, {control_dim-1}]")
    control_idx = state_dim + control_channel
    return x_profile[:, -1, control_idx].astype(np.float32)


def _assemble_forecast_table(
    t_series: np.ndarray,
    u_series: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    if t_series.ndim != 1 or u_series.ndim != 1:
        raise ValueError("t_series and u_series must be 1D arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must match. Got {y_true.shape} vs {y_pred.shape}")
    if t_series.shape[0] != y_true.shape[0] or u_series.shape[0] != y_true.shape[0]:
        raise ValueError("t_series/u_series length must match number of steps in y_true/y_pred.")
    return np.column_stack([t_series, u_series, y_true, y_pred]).astype(np.float32)


def save_rolling_forecasts_hdf5(
    forecasts: list[dict[str, Any]],
    *,
    output_path: Path,
    target_names: list[str],
    forecast_mode: str = "autoregressive",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    column_names = (
        ["t", "u(t)"]
        + [f"x(t)_{name}" for name in target_names]
        + [f"x^~(t)_{name}" for name in target_names]
    )
    column_attr = np.array(column_names, dtype="S")

    with h5py.File(output_path, "w") as h5f:
        for entry in forecasts:
            group = h5f.create_group(entry["profile"])
            group.create_dataset("data", data=entry["table"])
            group.create_dataset("mae", data=entry["mae"])
            group.create_dataset("rmse", data=entry["rmse"])
            group.create_dataset("mape", data=entry["mape"])
            group.attrs["columns"] = column_attr
            group.attrs["forecast_mode"] = str(forecast_mode)
            group.attrs["plot_title"] = (
                f"{str(forecast_mode).replace('_', ' ').title()} Forecast - {entry['profile']}"
            )


def test_and_save_forecasts(
    model: nn.Module,
    profile_ds: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    out_dir: Path,
    h5_path: Path | None = None,
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
    target_names: list[str] | None = None,
    output_name: str = "rolling_forecasts.h5",
    forecast_mode: str = "autoregressive",
    max_plots: int = 0,
    plot_callback: Callable[..., None] | None = None,
    use_tqdm: bool = True,
    verbose: int = 1,
    num_workers: int = 4,
) -> dict[str, float]:
    forecast_mode = str(forecast_mode).replace("-", "_")
    if forecast_mode not in {"autoregressive", "teacher_forcing"}:
        raise ValueError("forecast_mode must be either 'autoregressive' or 'teacher_forcing'.")
    if target_names is None:
        target_names = list(TARGET_NAMES)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecasts: list[dict[str, Any]] = []
    fetch_times: list[float] = []
    inference_times: list[float] = []
    scaling_stats = _load_scaling_stats(h5_path) if h5_path is not None else None

    try:
        total_profiles = len(profile_ds)
    except TypeError:
        total_profiles = None
    def _forecast_single(profile_name: str, x_profile: torch.Tensor, y_profile: torch.Tensor) -> dict[str, Any]:
        x_np = x_profile.numpy()
        y_np = y_profile.numpy()
        inference_start = perf_counter()
        if forecast_mode == "teacher_forcing":
            y_pred = teacher_forcing_forecast(model, x_np)
        else:
            y_pred = rolling_forecast(model, x_np, state_dim=state_dim)
        inference_time = perf_counter() - inference_start

        y_true = y_np
        y_pred_out = y_pred
        if scaling_stats is not None:
            y_true = _descale_targets_from_stats(scaling_stats, y_true)
            y_pred_out = _descale_targets_from_stats(scaling_stats, y_pred_out)

        t_series = np.arange(y_pred_out.shape[0], dtype=np.float32)
        u_series = _extract_control_series(x_np, state_dim=state_dim, control_channel=control_channel)
        if scaling_stats is not None:
            control_idx = state_dim + control_channel
            u_series = _descale_feature_from_stats(scaling_stats, u_series, control_idx)
        table = _assemble_forecast_table(t_series, u_series, y_true, y_pred_out)
        abs_error = np.abs(y_true - y_pred_out)
        mae = np.mean(abs_error, axis=0).astype(np.float32)
        rmse = np.sqrt(np.mean((y_true - y_pred_out) ** 2, axis=0)).astype(np.float32)
        denominator = np.where(np.abs(y_true) > 1e-8, np.abs(y_true), np.nan)
        mape = np.nanmean(abs_error / denominator, axis=0)
        mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) * 100.0
        return {
            "profile": str(profile_name),
            "table": table,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "x_np": x_np,
            "y_true": y_true,
            "y_pred_out": y_pred_out,
            "inference_time": inference_time,
        }

    progress = _iter_with_optional_tqdm(
        use_tqdm=use_tqdm,
        iterable=range(total_profiles or 0),
        total=total_profiles,
        desc="Forecast profiles",
        unit="profile",
    )
    if num_workers <= 1:
        iterator = profile_ds
        for profile_name, x_profile, y_profile in iterator:
            rec = _forecast_single(profile_name, x_profile, y_profile)
            inference_times.append(rec.pop("inference_time"))
            forecasts.append({k: rec[k] for k in ("profile", "table", "mae", "rmse", "mape")})
            if use_tqdm and hasattr(progress, "update"):
                progress.update(1)
            if plot_callback is not None and len(forecasts) <= max_plots:
                save_path = out_dir / f"rolling_forecast_{profile_name}.png"
                x_plot = rec["x_np"]
                if scaling_stats is not None:
                    control_idx = state_dim + control_channel
                    x_plot = x_plot.copy()
                    x_plot[:, :, control_idx] = _descale_feature_from_stats(
                        scaling_stats, x_plot[:, :, control_idx], control_idx
                    )
                plot_callback(
                    x_profile=x_plot,
                    y_true=rec["y_true"],
                    y_pred=rec["y_pred_out"],
                    title=f"{forecast_mode.replace('_', ' ').title()} Forecast - {profile_name}",
                    save_path=save_path,
                )
    else:
        with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
            futures = [
                executor.submit(_forecast_single, profile_name, x_profile, y_profile)
                for profile_name, x_profile, y_profile in profile_ds
            ]
            for fut in as_completed(futures):
                rec = fut.result()
                inference_times.append(rec.pop("inference_time"))
                forecasts.append({k: rec[k] for k in ("profile", "table", "mae", "rmse", "mape")})
                if use_tqdm and hasattr(progress, "update"):
                    progress.update(1)
                if plot_callback is not None and len(forecasts) <= max_plots:
                    profile_name = str(rec["profile"])
                    save_path = out_dir / f"rolling_forecast_{profile_name}.png"
                    x_plot = rec["x_np"]
                    if scaling_stats is not None:
                        control_idx = state_dim + control_channel
                        x_plot = x_plot.copy()
                        x_plot[:, :, control_idx] = _descale_feature_from_stats(
                            scaling_stats, x_plot[:, :, control_idx], control_idx
                        )
                    plot_callback(
                        x_profile=x_plot,
                        y_true=rec["y_true"],
                        y_pred=rec["y_pred_out"],
                        title=f"{forecast_mode.replace('_', ' ').title()} Forecast - {profile_name}",
                        save_path=save_path,
                    )
                if use_tqdm and hasattr(progress, "set_postfix"):
                    progress.set_postfix(mae_avg=f"{float(np.mean(rec['mae'])):.6e}")
    if use_tqdm and hasattr(progress, "close"):
        progress.close()

    save_start = perf_counter()
    save_rolling_forecasts_hdf5(
        forecasts,
        output_path=out_dir / output_name,
        target_names=target_names,
        forecast_mode=forecast_mode,
    )
    save_time_s = perf_counter() - save_start

    avg_fetch = float(np.mean(fetch_times)) if fetch_times else 0.0
    avg_inference = float(np.mean(inference_times)) if inference_times else 0.0
    total_fetch = float(np.sum(fetch_times)) if fetch_times else 0.0
    total_inference = float(np.sum(inference_times)) if inference_times else 0.0
    total_test = total_fetch + total_inference
    print(
        "\nTesting timing summary:"
        f" total_fetch: {total_fetch:.4f}s - "
        f" total_test: {total_test:.4f}s - "
        f" avg_fetch_profile: {avg_fetch:.4f}s - "
        f" avg_inference_profile: {avg_inference:.4f}s - "
        f" save_time: {save_time_s:.4f}s\n"
    )

    return {
        "avg_fetch_profile_s": avg_fetch,
        "avg_inference_profile_s": avg_inference,
        "total_fetch_s": total_fetch,
        "total_test_s": total_test,
        "save_time_s": save_time_s,
        "forecast_mode": forecast_mode,
    }


# --------------------------------------------------------------------------------------
# Inspection utilities
# --------------------------------------------------------------------------------------

def _try_cardinality(count: int) -> int | None:
    if count < 0:
        return None
    return int(count)


def inspect_dataset_shapes(datasets: dict[str, Any]) -> None:
    """
    More informative inspection than the original version:
      - split sizes (#profiles, #samples)
      - batch size and inferred #batches
      - first batch shapes
      - first profile shapes
      - loader cardinality derived from precomputed counts
    """
    print("Dataset summary:")
    print(f"  H5 path:         {datasets.get('h5_path', 'UNKNOWN')}")
    print(f"  Batch size:      {datasets.get('batch_size', 'UNKNOWN')}")
    print(f"  Seed:            {datasets.get('seed', 'UNKNOWN')}")
    print("")
    print(f"  Train profiles:  {len(datasets['train_profile_names'])}")
    print(f"  Val profiles:    {len(datasets['val_profile_names'])}")
    print(f"  Test profiles:   {len(datasets['test_profile_names'])}")
    print("")
    print(f"  Sample X shape:  {datasets['sample_shape']}   (per-sample)")
    print(f"  Sample Y shape:  {datasets['target_shape']}   (per-sample)")
    print("")

    # Sample counts + steps
    train_n = int(datasets.get("train_num_samples", 0))
    val_n = int(datasets.get("val_num_samples", 0))
    test_n = int(datasets.get("test_num_samples", 0))
    bs = int(datasets.get("batch_size", 1))

    print(f"  Train samples:   {train_n:,}  (~{ceil(train_n/bs):,} batches)")
    print(f"  Val samples:     {val_n:,}  (~{ceil(val_n/bs):,} batches)")
    print(f"  Test samples:    {test_n:,}  (profiles used for forecasting)")
    print("")

    # Known cardinality from precomputed counts
    train_card = _try_cardinality(int(datasets.get("train_steps_per_epoch", 0)))
    val_card = _try_cardinality(int(datasets.get("val_steps", 0)))
    val_prof_card = _try_cardinality(len(datasets["val_profile_names"]))
    test_prof_card = _try_cardinality(len(datasets["test_profile_names"]))

    print("  loader cardinality (derived from precomputed counts):")
    print(f"    train:          {train_card}")
    print(f"    val_samples:    {val_card}")
    print(f"    val_profile_ds: {val_prof_card}")
    print(f"    test_profile_ds:{test_prof_card}")
    print("")

    # Batch inspection
    train_batch = next(iter(datasets["train"]))
    print(f"Train batch X shape: {train_batch[0].shape}")
    print(f"Train batch Y shape: {train_batch[1].shape}")
    print("")

    # Profile inspection
    for split_name, dataset_key in (("val", "val_profile_ds"), ("test", "test_profile_ds")):
        profile_name, x_profile, y_profile = next(iter(datasets[dataset_key]))
        print(f"{split_name.capitalize()} profile: {profile_name}")
        print(f"  X profile shape: {x_profile.shape}")
        print(f"  Y profile shape: {y_profile.shape}\n")


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def plot_forecast_vs_truth_grid(
    *,
    x_profile: np.ndarray,          # (num_steps, timesteps, num_features)
    y_true: np.ndarray,             # (num_steps, num_targets)
    y_pred: np.ndarray,             # (num_steps, num_targets)
    target_names: list[str],
    title: str,
    save_path: Path | None = None,
    control_name: str = "drumAngleDeg",
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
    close_figure: bool = True,
) -> plt.Figure:
    """
    Grid includes control + all targets.
      - [0] control profile across all forecast steps
      - remaining target plots: each target (truth + pred)

    Control profile is extracted from x_profile windows as:
        control_series[step] = x_profile[step, -1, state_dim + control_channel]
    i.e. the last control value in the window for each forecast step.
    """
    # validations
    if x_profile.ndim != 3:
        raise ValueError(f"x_profile must be 3D (steps,timesteps,features). Got {x_profile.shape}")
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(f"y_true/y_pred must be 2D (steps,targets). Got {y_true.shape}, {y_pred.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must match. Got {y_true.shape} vs {y_pred.shape}")

    num_steps, _timesteps, num_features = x_profile.shape
    y_steps, num_targets = y_true.shape

    if num_steps != y_steps:
        raise ValueError(f"x_profile steps != y_true steps: {num_steps} vs {y_steps}")
    if len(target_names) != num_targets:
        raise ValueError(f"target_names must have length {num_targets}, got {len(target_names)}")

    target_names, reordered = _reorder_forecast_plot_targets(target_names, y_true, y_pred)
    y_true, y_pred = reordered
    num_targets = len(target_names)

    control_dim = num_features - state_dim
    if control_dim <= 0:
        raise ValueError(
            f"control_dim <= 0 (num_features={num_features}, state_dim={state_dim}). "
            "This violates the rolling_forecast assumption."
        )
    if not (0 <= control_channel < control_dim):
        raise ValueError(f"control_channel={control_channel} out of range [0, {control_dim-1}]")

    control_idx = state_dim + control_channel
    control_series = x_profile[:, -1, control_idx]  # (num_steps,)

    nplots = num_targets + 1
    rows, cols = 4, 4
    if nplots > rows * cols:
        raise ValueError(f"Plot requires {nplots} panels but 4x4 supports only 16.")
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
    axes = np.atleast_1d(axes).ravel()

    # Control plot (top-left)
    ax0 = axes[0]
    ax0.plot(control_series, label=control_name)
    ax0.set_title(control_name)
    ax0.set_xlabel("Forecast step")
    ax0.set_ylabel(control_name)
    ax0.grid(True)
    ax0.legend(loc="best")

    # Target plots
    for i in range(num_targets):
        ax = axes[i + 1]
        name = target_names[i]
        ax.plot(y_true[:, i], label="truth", color='black')
        ax.plot(y_pred[:, i], label="pred", color='blue')
        ax.set_title(name)
        ax.set_xlabel("Forecast step")
        ax.set_ylabel(name)
        _disable_y_offset_if_requested(ax, name)
        ax.grid(True)
        ax.legend(fontsize=7, loc="best")

    abs_error = np.abs(y_true - y_pred)
    mae_all = float(np.mean(abs_error))
    rmse_all = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denominator = np.where(np.abs(y_true) > 1e-8, np.abs(y_true), np.nan)
    mape_all = float(np.nanmean(abs_error / denominator) * 100.0)
    if not np.isfinite(mape_all):
        mape_all = 0.0

    fig.suptitle(
        f"{title} | MAE(all dims) = {mae_all:.6e} | RMSE(all dims) = {rmse_all:.6e} | MAPE(all dims) = {mape_all:.3f}%",
        y=0.98,
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved forecast plot to: {save_path}")

    if close_figure:
        plt.close(fig)

    return fig


def _build_profile_control_tensor(
    u_series: np.ndarray,
    *,
    state_dim: int,
    control_channel: int,
) -> np.ndarray:
    """Build a minimal x_profile tensor containing only the requested control channel."""
    control_idx = state_dim + control_channel
    x_profile = np.zeros((u_series.shape[0], 1, control_idx + 1), dtype=np.float32)
    x_profile[:, 0, control_idx] = u_series
    return x_profile


def _resolve_forecast_columns(
    *,
    profile_name: str,
    columns: list[str],
    target_names: list[str],
) -> tuple[str, list[str], list[str], list[str]]:
    """Resolve and validate forecast schema columns for a single profile."""
    column_set = set(columns)
    required_base = {"t", "u(t)"}
    missing_base = sorted(required_base - column_set)
    if missing_base:
        raise ValueError(
            f"Profile '{profile_name}' is missing required base columns {missing_base}; "
            f"available columns: {columns}."
        )

    single_truth = [f"x(t)_{name}" for name in target_names]
    single_pred = [f"x^~(t)_{name}" for name in target_names]
    if set(single_truth).issubset(column_set) and set(single_pred).issubset(column_set):
        return "single", single_truth, single_pred, []

    ensemble_truth = [f"x_true(t)_{name}" for name in target_names]
    ensemble_mean = [f"x_mean(t)_{name}" for name in target_names]
    ensemble_sigma = [f"x_2sigma(t)_{name}" for name in target_names]

    missing_truth = sorted(set(ensemble_truth) - column_set)
    missing_mean = sorted(set(ensemble_mean) - column_set)
    if not missing_truth and not missing_mean:
        missing_sigma = sorted(set(ensemble_sigma) - column_set)
        if missing_sigma and len(missing_sigma) != len(target_names):
            raise ValueError(
                f"Profile '{profile_name}' has partial ensemble uncertainty columns. "
                f"Expected all or none of {ensemble_sigma}; missing {missing_sigma}."
            )
        sigma_columns = [] if missing_sigma else ensemble_sigma
        return "ensemble", ensemble_truth, ensemble_mean, sigma_columns

    raise ValueError(
        f"Profile '{profile_name}' columns do not match a supported forecast schema. "
        f"Expected single-model keys {single_truth + single_pred} or ensemble keys "
        f"{ensemble_truth + ensemble_mean} (optionally {ensemble_sigma})."
    )


def _plot_ensemble_forecast_vs_truth_grid(
    *,
    x_profile: np.ndarray,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_2sigma: np.ndarray | None,
    y_dsigma_dt: np.ndarray | None,
    y_members: np.ndarray | None,
    target_names: list[str],
    title: str,
    control_name: str,
    state_dim: int,
    control_channel: int,
    t_series: np.ndarray | None = None,
) -> plt.Figure:
    """Plot ensemble forecasts in the same style as bagging ensemble profile plotting."""
    if x_profile.ndim != 3:
        raise ValueError(f"x_profile must be 3D (steps,timesteps,features). Got {x_profile.shape}")
    if y_true.ndim != 2 or y_mean.ndim != 2:
        raise ValueError(f"y_true/y_mean must be 2D (steps,targets). Got {y_true.shape}, {y_mean.shape}")
    if y_true.shape != y_mean.shape:
        raise ValueError(f"y_true and y_mean must match. Got {y_true.shape} vs {y_mean.shape}")
    if y_2sigma is not None and y_2sigma.shape != y_mean.shape:
        raise ValueError(f"y_2sigma must match y_mean shape. Got {y_2sigma.shape} vs {y_mean.shape}")
    if y_dsigma_dt is not None and y_dsigma_dt.shape != y_mean.shape:
        raise ValueError(f"y_dsigma_dt must match y_mean shape. Got {y_dsigma_dt.shape} vs {y_mean.shape}")
    if y_members is not None:
        if y_members.ndim != 3:
            raise ValueError(f"y_members must be 3D (members,steps,targets). Got {y_members.shape}")
        if y_members.shape[1:] != y_mean.shape:
            raise ValueError(f"y_members steps/targets must match y_mean. Got {y_members.shape} vs {y_mean.shape}")

    num_steps, num_targets = y_true.shape
    if len(target_names) != num_targets:
        raise ValueError(f"target_names must have length {num_targets}, got {len(target_names)}")

    target_names, reordered = _reorder_forecast_plot_targets(
        target_names,
        y_true,
        y_mean,
        y_2sigma,
        y_dsigma_dt,
        y_members,
    )
    y_true, y_mean, y_2sigma, y_dsigma_dt, y_members = reordered
    num_targets = len(target_names)

    if t_series is None:
        t_series = np.arange(num_steps, dtype=np.float32)
    else:
        t_series = np.asarray(t_series, dtype=np.float32)
        if t_series.ndim != 1 or t_series.shape[0] != num_steps:
            raise ValueError(f"t_series must be 1D with length {num_steps}, got {t_series.shape}")
    control_series = _extract_control_series(
        x_profile,
        state_dim=state_dim,
        control_channel=control_channel,
    )

    nplots = num_targets + 1
    rows, cols = 4, 4
    if nplots > rows * cols:
        raise ValueError(f"Plot requires {nplots} panels but 4x4 supports only 16.")
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()

    axes_flat[0].plot(t_series, control_series, linewidth=1.5, color="black")
    axes_flat[0].set_title(control_name)
    axes_flat[0].grid(True, alpha=0.3)

    derivative_axes: list[plt.Axes] = []
    derivative_styles = _build_uncertainty_derivative_styles(target_names)
    for target_idx, target_name in enumerate(target_names):
        ax = axes_flat[target_idx + 1]
        ax.plot(t_series, y_true[:, target_idx], label="Ground truth", linewidth=1.6, color="C0")
        if y_members is not None:
            for member_idx in range(y_members.shape[0]):
                ax.plot(
                    t_series,
                    y_members[member_idx, :, target_idx],
                    linewidth=0.7,
                    color="0.35",
                    alpha=0.22,
                    label="Member forecasts" if member_idx == 0 else "_nolegend_",
                )
        ax.plot(t_series, y_mean[:, target_idx], label="Mean prediction", linewidth=1.8, color="C3")
        if y_2sigma is not None:
            y_upper = y_mean[:, target_idx] + y_2sigma[:, target_idx]
            y_lower = y_mean[:, target_idx] - y_2sigma[:, target_idx]
            ax.plot(t_series, y_upper, linestyle="--", linewidth=1.0, color="C1", label="Mean + 2σ")
            ax.plot(t_series, y_lower, linestyle="--", linewidth=1.0, color="C2", label="Mean - 2σ")
            ax.fill_between(t_series, y_lower, y_upper, color="C1", alpha=0.15, linewidth=0)
        if y_dsigma_dt is not None:
            ax2 = ax.twinx()
            derivative_axes.append(ax2)
            dstyle = derivative_styles[target_idx]
            ax2.plot(
                t_series,
                y_dsigma_dt[:, target_idx],
                linewidth=1.2,
                color=dstyle["color"],
                linestyle=dstyle["linestyle"],
                label="d(x_sigma_scaled)/dt",
            )
            ax2.axhline(0.0, color="0.5", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.set_title(target_name)
        _disable_y_offset_if_requested(ax, target_name)
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
    legend_cols = 5 if (y_dsigma_dt is not None or y_members is not None) else (4 if y_2sigma is not None else 2)
    fig.suptitle(title, y=0.985, fontsize=16)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=legend_cols,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def _plot_uncertainty_derivative_only_grid(
    *,
    t_series: np.ndarray,
    y_dsigma_dt: np.ndarray,
    target_names: list[str],
    title: str,
) -> plt.Figure:
    if y_dsigma_dt.ndim != 2:
        raise ValueError(f"y_dsigma_dt must be 2D (steps,targets). Got {y_dsigma_dt.shape}")
    num_steps, num_targets = y_dsigma_dt.shape
    if len(target_names) != num_targets:
        raise ValueError(f"target_names must have length {num_targets}, got {len(target_names)}")
    if t_series.ndim != 1 or t_series.shape[0] != num_steps:
        raise ValueError(f"t_series must be 1D with length {num_steps}, got {t_series.shape}")

    fig, ax = plt.subplots(figsize=(14, 7))
    derivative_styles = _build_uncertainty_derivative_styles(target_names)
    for target_idx, target_name in enumerate(target_names):
        dstyle = derivative_styles[target_idx]
        ax.plot(
            t_series,
            y_dsigma_dt[:, target_idx],
            linewidth=1.2,
            color=dstyle["color"],
            linestyle=dstyle["linestyle"],
            label=target_name,
        )
    ax.axhline(0.0, color="0.5", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("d(x_sigma_scaled)/dt")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", ncol=min(4, max(1, len(target_names))), frameon=False)
    fig.tight_layout()
    return fig


def _physical_group_for_target_name(target_name: str) -> str:
    name = target_name.strip().lower()
    if name == "n" or "power" in name:
        return "power"
    if "reactivity" in name or name in {"rho", "rho_dollars"}:
        return "reactivity"
    if "q_to_steam" in name or "steam" in name:
        return "q_to_steam"
    if re.fullmatch(r"c[\[_]?\d+\]?", name):
        return "concentration"
    if name.startswith("t") or "temp" in name:
        return "temperature"
    return "other"


def _build_uncertainty_derivative_styles(target_names: list[str]) -> list[dict[str, str]]:
    group_base_colors = {
        "temperature": "red",
        "concentration": "green",
        "power": "deeppink",
        "reactivity": "black",
        "q_to_steam": "gray",
        "other": "C4",
    }

    groups = [_physical_group_for_target_name(name) for name in target_names]
    group_counts: dict[str, int] = {}
    group_indices: dict[str, int] = {}
    for group in groups:
        group_counts[group] = group_counts.get(group, 0) + 1

    styles: list[dict[str, str]] = []
    for group in groups:
        idx_in_group = group_indices.get(group, 0)
        group_indices[group] = idx_in_group + 1
        shade_idx = 0 if group_counts[group] <= 1 else idx_in_group
        color = _vary_group_color(group_base_colors[group], shade_idx=shade_idx, n_shades=group_counts[group])
        styles.append({"color": color, "linestyle": "-"})
    return styles


def _vary_group_color(base_color: str, *, shade_idx: int, n_shades: int) -> str:
    if n_shades <= 1:
        return base_color
    rgb = np.asarray(mcolors.to_rgb(base_color), dtype=np.float64)
    center = 0.5 * (n_shades - 1)
    offset = (shade_idx - center) / max(center, 1.0)  # [-1, 1]
    if offset >= 0:
        # Lighter for later traces.
        mix = 0.22 * offset
        out = rgb * (1.0 - mix) + np.ones(3, dtype=np.float64) * mix
    else:
        # Slightly darker for earlier traces.
        mix = 0.18 * (-offset)
        out = rgb * (1.0 - mix)
    return mcolors.to_hex(np.clip(out, 0.0, 1.0))


def save_forecast_profiles_pdf(
    *,
    forecast_h5_path: Path,
    output_pdf_path: Path,
    target_names: list[str] | None = None,
    control_name: str = "drumAngleDeg",
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
    mode: str = "auto",
    include_uncertainty_derivative: bool = False,
    derivative_order: int = 4,
    derivative_dt: float = 1.0,
    include_ensemble_members: bool = True,
) -> None:
    """
    Render one 3x5 forecast-vs-truth page per profile from a forecast HDF5 file.

    Supported per-profile HDF5 schemas are:
      * Single-model schema from :func:`test_and_save_forecasts`:
        ``[t, u(t), x(t)_{target}..., x^~(t)_{target}...]``
      * Ensemble schema from :func:`ensemble_rolling_forecast_and_save`:
        ``[t, u(t), x_true(t)_{target}..., x_mean(t)_{target}..., [x_2sigma(t)_{target}...]]``

    By default (``mode='auto'``), schema detection uses ``group.attrs['columns']``.

    When ``include_uncertainty_derivative=True``, derivative overlays are interpreted
    as scaled-space uncertainty derivatives (``d(x_sigma_scaled)/dt``).

    When ``include_ensemble_members=True`` and a profile contains a
    ``member_predictions`` dataset, each ensemble member trajectory is plotted as
    a thin transparent line behind the ensemble mean.
    """
    if target_names is None:
        target_names = list(TARGET_NAMES)
    if mode not in {"auto", "single", "ensemble"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: auto, single, ensemble.")

    forecast_h5_path = Path(forecast_h5_path)
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(forecast_h5_path, "r") as h5f, PdfPages(output_pdf_path) as pdf:
        profile_names = sorted(h5f.keys())
        for i, profile_name in enumerate(profile_names, start=1):
            if i % 10 == 0:
                print(f"Plotted {i}/{len(profile_names)}")
            group = h5f[profile_name]
            if "data" not in group:
                raise KeyError(f"Profile '{profile_name}' is missing required dataset 'data'.")
            profile_title = str(group.attrs.get("plot_title", f"Rolling Forecast - {profile_name}"))

            table = group["data"][...].astype(np.float32)
            if table.ndim != 2 or table.shape[1] < 4:
                raise ValueError(
                    f"Profile '{profile_name}' has invalid table shape {table.shape}; expected 2D with >=4 columns."
                )

            columns = _decode_columns(group.attrs.get("columns", []))
            if len(columns) != table.shape[1]:
                raise ValueError(
                    f"Profile '{profile_name}' has mismatched metadata: data has {table.shape[1]} columns but "
                    f"attrs['columns'] has {len(columns)} entries."
                )

            detected_mode, truth_cols, pred_cols, sigma_cols = _resolve_forecast_columns(
                profile_name=profile_name,
                columns=columns,
                target_names=target_names,
            )
            use_mode = mode if mode != "auto" else detected_mode
            if mode != "auto" and use_mode != detected_mode:
                raise ValueError(
                    f"Profile '{profile_name}' schema is '{detected_mode}', but mode='{mode}' was requested."
                )

            u_series = table[:, columns.index("u(t)")]
            t_series = table[:, columns.index("t")]
            y_true = np.column_stack([table[:, columns.index(col)] for col in truth_cols])
            y_pred_or_mean = np.column_stack([table[:, columns.index(col)] for col in pred_cols])
            y_2sigma = None if not sigma_cols else np.column_stack([table[:, columns.index(col)] for col in sigma_cols])
            y_members = None
            if include_ensemble_members and use_mode == "ensemble" and "member_predictions" in group:
                y_members = group["member_predictions"][...].astype(np.float32)
                if y_members.ndim != 3:
                    raise ValueError(
                        f"Profile '{profile_name}' member_predictions must be 3D "
                        f"(members,steps,targets); got {y_members.shape}."
                    )
                member_target_names = _decode_columns(
                    group.attrs.get(
                        "member_target_names",
                        group["member_predictions"].attrs.get("target_names", []),
                    )
                )
                if member_target_names and member_target_names != target_names:
                    missing_member_targets = [name for name in target_names if name not in member_target_names]
                    if missing_member_targets:
                        raise ValueError(
                            f"Profile '{profile_name}' member_predictions are missing targets "
                            f"{missing_member_targets}."
                        )
                    member_indices = [member_target_names.index(name) for name in target_names]
                    y_members = y_members[:, :, member_indices]
                if y_members.shape[1:] != y_pred_or_mean.shape:
                    raise ValueError(
                        f"Profile '{profile_name}' member_predictions shape {y_members.shape} "
                        f"does not match mean forecast shape {y_pred_or_mean.shape}."
                    )

            y_dsigma_dt = None
            if include_uncertainty_derivative and y_2sigma is not None:
                dsigma_cols = [f"x_dsigma_dt(t)_{name}" for name in target_names]
                if all(col in columns for col in dsigma_cols):
                    y_dsigma_dt = np.column_stack([table[:, columns.index(col)] for col in dsigma_cols])
                else:
                    y_dsigma_dt = finite_difference(y_2sigma, order=derivative_order, dt=derivative_dt)

            x_profile = _build_profile_control_tensor(
                u_series,
                state_dim=state_dim,
                control_channel=control_channel,
            )

            if use_mode == "single":
                fig = plot_forecast_vs_truth_grid(
                    x_profile=x_profile,
                    y_true=y_true,
                    y_pred=y_pred_or_mean,
                    target_names=target_names,
                    title=profile_title,
                    save_path=None,
                    control_name=control_name,
                    state_dim=state_dim,
                    control_channel=control_channel,
                    close_figure=False,
                )
            else:
                fig = _plot_ensemble_forecast_vs_truth_grid(
                    x_profile=x_profile,
                    y_true=y_true,
                    y_mean=y_pred_or_mean,
                    y_2sigma=y_2sigma,
                    y_dsigma_dt=y_dsigma_dt,
                    y_members=y_members,
                    target_names=target_names,
                    title=profile_title,
                    control_name=control_name,
                    state_dim=state_dim,
                    control_channel=control_channel,
                    t_series=t_series,
                )
            pdf.savefig(fig, orientation="landscape")
            plt.close(fig)
            if include_uncertainty_derivative and y_dsigma_dt is not None:
                fig_deriv = _plot_uncertainty_derivative_only_grid(
                    t_series=t_series,
                    y_dsigma_dt=y_dsigma_dt,
                    target_names=target_names,
                    title=f"Scaled Uncertainty Derivative - {profile_name}",
                )
                pdf.savefig(fig_deriv, orientation="landscape")
                plt.close(fig_deriv)

    print(f"Saved forecast PDF to: {output_pdf_path}")


def compute_and_save_rolling_forecast_metrics(
    forecast_h5_path: Path,
    output_json_path: Path,
    scaled_h5: Path | None = None,
) -> dict[str, Any]:
    """
    Compute scaled forecast metrics from rolling_forecasts.h5 for either:
      - single-model schema: x(t)_*, x^~(t)_*
      - ensemble schema: x_true(t)_*, x_mean(t)_*, x_2sigma(t)_*

    Scaled MAE/RMSE metrics use physical/descaled forecast values divided by
    fixed per-target scaling factors from ``scaled_h5``. Evaluation-set ranges
    are not used for normalization.
    """
    forecast_h5_path = Path(forecast_h5_path)
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if scaled_h5 is None:
        raise ValueError(
            "compute_and_save_rolling_forecast_metrics requires scaled_h5 so scaled MAE/RMSE can use "
            "fixed target scaling factors; refusing to fall back to evaluation-set ranges."
        )
    scaling_stats = _load_scaling_stats(scaled_h5)

    def _target_scale_for(target_name_list: list[str]) -> np.ndarray:
        target_indices = [TARGET_NAMES.index(name) for name in target_name_list]
        y_stats = scaling_stats["y"]
        if scaling_stats["type"] == "standard":
            scale = np.asarray(y_stats["std"], dtype=float)[target_indices]
        elif scaling_stats["type"] == "minmax":
            scale = np.asarray(y_stats["span"], dtype=float)[target_indices]
        else:
            raise ValueError(f"Unsupported scaling type: {scaling_stats['type']}")
        if scale.shape[-1] != len(target_name_list):
            raise ValueError(
                f"Target scale length mismatch: expected {len(target_name_list)} values, got shape {scale.shape}."
            )
        if not np.all(np.isfinite(scale)) or not np.all(scale > 0.0):
            raise ValueError(f"Target scales must all be finite and positive. Got: {scale!r}")
        return scale

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_2sigma_all: list[np.ndarray] = []
    horizon_rmse_chunks: list[np.ndarray] = []
    horizon_mae_chunks: list[np.ndarray] = []
    schema_mode: str | None = None
    target_names: list[str] | None = None

    with h5py.File(forecast_h5_path, "r") as h5f:
        for profile_name in sorted(h5f.keys()):
            group = h5f[profile_name]
            table = group["data"][...].astype(np.float64)
            cols = _decode_columns(group.attrs.get("columns", []))
            mode, truth_cols, pred_cols, sigma_cols = _resolve_forecast_columns(
                profile_name=profile_name,
                columns=cols,
                target_names=list(TARGET_NAMES),
            )
            if schema_mode is None:
                schema_mode = mode
                target_names = [c.split(")_", 1)[1] for c in truth_cols]
            y_true = np.column_stack([table[:, cols.index(c)] for c in truth_cols])
            y_pred = np.column_stack([table[:, cols.index(c)] for c in pred_cols])
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            err = y_pred - y_true
            profile_target_names = [c.split(")_", 1)[1] for c in truth_cols]
            horizon_err = err / _target_scale_for(profile_target_names)
            horizon_rmse_chunks.append(np.sqrt(np.mean(horizon_err**2, axis=1)))
            horizon_mae_chunks.append(np.mean(np.abs(horizon_err), axis=1))
            if sigma_cols:
                y_2sigma = np.column_stack([table[:, cols.index(c)] for c in sigma_cols])
                y_2sigma_all.append(y_2sigma)

    if not y_true_all or schema_mode is None or target_names is None:
        raise ValueError(f"No forecast profiles found in {forecast_h5_path}")

    y_true_cat = np.concatenate(y_true_all, axis=0)
    y_pred_cat = np.concatenate(y_pred_all, axis=0)
    err = y_pred_cat - y_true_cat
    abs_err = np.abs(err)

    rmse_per_target = np.sqrt(np.mean(err**2, axis=0))
    mae_per_target = np.mean(abs_err, axis=0)
    bias_per_target = np.mean(err, axis=0)
    target_scale = _target_scale_for(target_names)
    err_scaled = err / target_scale
    scaled_mae_per_target = np.nanmean(np.abs(err_scaled), axis=0)
    scaled_rmse_per_target = np.sqrt(np.nanmean(err_scaled**2, axis=0))
    scaled_mae = float(np.nanmean(scaled_mae_per_target))
    scaled_rmse = float(np.nanmean(scaled_rmse_per_target))
    if not np.isclose(scaled_mae, float(np.nanmean(scaled_mae_per_target)), rtol=1e-12, atol=1e-12):
        raise AssertionError("Aggregate scaled_mae does not equal the mean of per-target scaled MAE values.")
    if not np.isclose(scaled_rmse, float(np.nanmean(scaled_rmse_per_target)), rtol=1e-12, atol=1e-12):
        raise AssertionError("Aggregate scaled_rmse does not equal the mean of per-target scaled RMSE values.")

    ss_res = np.sum((y_true_cat - y_pred_cat) ** 2, axis=0)
    y_mean = np.mean(y_true_cat, axis=0)
    ss_tot = np.sum((y_true_cat - y_mean) ** 2, axis=0)
    r2_per_target = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    horizon_rmse = np.mean(np.stack(horizon_rmse_chunks, axis=0), axis=0)
    horizon_mae = np.mean(np.stack(horizon_mae_chunks, axis=0), axis=0)

    result: dict[str, Any] = {
        "schema_mode": schema_mode,
        "targets": target_names,
        "scaled_mae": scaled_mae,
        "scaled_rmse": scaled_rmse,
        "per_target_rmse": {n: float(v) for n, v in zip(target_names, rmse_per_target, strict=True)},
        "per_target_mae": {n: float(v) for n, v in zip(target_names, mae_per_target, strict=True)},
        "per_target_scaled_mae": {n: float(v) for n, v in zip(target_names, scaled_mae_per_target, strict=True)},
        "per_target_scaled_rmse": {n: float(v) for n, v in zip(target_names, scaled_rmse_per_target, strict=True)},
        "per_target_horizon_mean_scaled_mae": {
            n: float(v) for n, v in zip(target_names, scaled_mae_per_target, strict=True)
        },
        "per_target_bias": {n: float(v) for n, v in zip(target_names, bias_per_target, strict=True)},
        "per_target_r2": {n: float(v) for n, v in zip(target_names, r2_per_target, strict=True)},
        "horizon_mean_scaled_rmse": horizon_rmse.tolist(),
        "horizon_mean_scaled_mae": horizon_mae.tolist(),
        "horizon_error_target_space": "scaled",
        "scaled_metric_target_space": "scaled",
    }

    if schema_mode == "ensemble" and y_2sigma_all:
        sigma95 = np.concatenate(y_2sigma_all, axis=0)
        half_width = sigma95
        cov95 = float(np.mean(abs_err <= half_width))
        result["empirical_coverage_95"] = cov95
        result["calibration_error_95"] = float(abs(cov95 - 0.95))
        result["interval_width_95_mean"] = float(np.mean(2.0 * half_width))
        cov95_per_target = np.mean(abs_err <= half_width, axis=0)
        width95_per_target = np.mean(2.0 * half_width, axis=0)
        result["per_target_empirical_coverage_95"] = {
            n: float(v) for n, v in zip(target_names, cov95_per_target, strict=True)
        }
        result["per_target_calibration_error_95"] = {
            n: float(abs(v - 0.95)) for n, v in zip(target_names, cov95_per_target, strict=True)
        }
        result["per_target_interval_width_95_mean"] = {
            n: float(v) for n, v in zip(target_names, width95_per_target, strict=True)
        }

    output_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------------------
# Optional "pipeline" class for clean imports
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class LSTMPipelineConfig:
    h5_path: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int = 123
    target_names: list[str] = None  # set in __post_init__
    state_dim: int = STATE_DIM
    control_name: str = "drumAngleDeg"
    control_channel: int = 0
    n_lstm: int = 1
    lstm_hidden: int = 64
    lstm_dropout: float = 0.0
    n_fc: int = 1
    fc_hidden: tuple[int, ...] = (64,)
    learning_rate: float = 1e-3
    use_tqdm: bool = True

    def __post_init__(self) -> None:
        if self.target_names is None:
            self.target_names = list(TARGET_NAMES)
        self.fc_hidden = tuple(self.fc_hidden)
        if self.n_lstm < 1:
            raise ValueError(f"n_lstm must be >= 1 (got {self.n_lstm}).")
        if self.n_fc < 1:
            raise ValueError(f"n_fc must be >= 1 (got {self.n_fc}).")
        if len(self.fc_hidden) != self.n_fc:
            raise ValueError("fc_hidden must be a tuple of length n_fc.")


class LSTMPipeline:
    """
    Convenience wrapper around the core functions for interactive use.
    """
    def __init__(self, config: LSTMPipelineConfig):
        self.config = config
        self.datasets: dict[str, Any] | None = None

    def build(self) -> dict[str, Any]:
        self.datasets = build_datasets(self.config.h5_path, self.config.batch_size, self.config.seed)
        return self.datasets

    def inspect(self) -> None:
        if self.datasets is None:
            self.build()
        inspect_dataset_shapes(self.datasets)

    def train(
        self,
        *,
        epochs: int = DEFAULT_EPOCHS,
        out_dir: Path | None = None,
        prefer_gpu: bool = True,
        preload_train_to_device: bool = False,
        deterministic_seed: int | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        restore_best_weights: bool = True,
        step_lr_step_size: int = 30,
        step_lr_gamma: float = 0.5,
        verbose: int = 1,
        enable_torch_profiler: bool = False,
        profiler_wait_steps: int = 1,
        profiler_warmup_steps: int = 1,
        profiler_active_steps: int = 3,
        profiler_repeat: int = 1,
        profiler_row_limit: int = 30,
        use_tqdm: bool | None = None,
    ):

        if self.datasets is None:
            self.build()
        resolved_use_tqdm = self.config.use_tqdm if use_tqdm is None else bool(use_tqdm)
        return train_with_fallback(
            self.datasets,
            epochs=epochs,
            out_dir=resolve_output_root() if out_dir is None else out_dir,
            n_lstm=self.config.n_lstm,
            lstm_hidden=self.config.lstm_hidden,
            lstm_dropout=self.config.lstm_dropout,
            n_fc=self.config.n_fc,
            fc_hidden=self.config.fc_hidden,
            learning_rate=self.config.learning_rate,
            step_lr_step_size=step_lr_step_size,
            step_lr_gamma=step_lr_gamma,
            verbose=verbose,
            prefer_gpu=prefer_gpu,
            preload_train_to_device=preload_train_to_device,
            deterministic_seed=self.config.seed if deterministic_seed is None else deterministic_seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            restore_best_weights=restore_best_weights,
            enable_torch_profiler=enable_torch_profiler,
            profiler_wait_steps=profiler_wait_steps,
            profiler_warmup_steps=profiler_warmup_steps,
            profiler_active_steps=profiler_active_steps,
            profiler_repeat=profiler_repeat,
            profiler_row_limit=profiler_row_limit,
            use_tqdm=resolved_use_tqdm,
        )

    def sample_test_profile(self):
        if self.datasets is None:
            self.build()
        profile_name, x_profile, y_profile = next(iter(self.datasets["test_profile_ds"]))
        return profile_name, x_profile.numpy(), y_profile.numpy()

    def forecast(self, model: nn.Module, x_profile: np.ndarray) -> np.ndarray:
        return rolling_forecast(model, x_profile, state_dim=self.config.state_dim)

    def plot(self, *, x_profile: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path | None = None):
        plot_forecast_vs_truth_grid(
            x_profile=x_profile,
            y_true=y_true,
            y_pred=y_pred,
            target_names=self.config.target_names,
            title=title,
            save_path=save_path,
            control_name=self.config.control_name,
            state_dim=self.config.state_dim,
            control_channel=self.config.control_channel,
        )

    @staticmethod
    def save_model_pt(model: nn.Module, output_path: Path) -> Path:
        """Save model weights to a PyTorch .pt file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        return output_path


# --------------------------------------------------------------------------------------
# Script entrypoint example
# --------------------------------------------------------------------------------------
def main() -> None:
    # Example local run: edit the dataset path
    output_root = resolve_output_root()
    h5_path = output_root / "datasets" / "lstm_merged_batch_0001-batch_0001_k10_standard_train0.70_val0.15_test0.15.h5"
    batch_size = 64
    epochs = 20
    seed = 123

    out_dir = output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(h5_path=h5_path, batch_size=batch_size, seed=seed)
    inspect_dataset_shapes(datasets)

    model, history, used_device = train_with_fallback(
        datasets,
        epochs=epochs,
        out_dir=out_dir,
    )
    print(f"\nFinished training using device: {used_device}\n")

    profile_name, x_profile, y_profile = next(iter(datasets["test_profile_ds"]))
    name = profile_name
    x_np = x_profile.numpy()
    y_np = y_profile.numpy()

    y_pred = rolling_forecast(model, x_np)

    plot_forecast_vs_truth_grid(
        x_profile=x_np,
        y_true=y_np,
        y_pred=y_pred,
        target_names=TARGET_NAMES,
        title=f"Rolling Forecast - {name}",
        save_path=out_dir / f"rolling_forecast_{name}.png",
    )

    test_and_save_forecasts(
        model,
        datasets["test_profile_ds"],
        out_dir=out_dir,
        h5_path=datasets["h5_path"],
        state_dim=STATE_DIM,
        control_channel=0,
        target_names=TARGET_NAMES,
    )


if __name__ == "__main__":
    main()
