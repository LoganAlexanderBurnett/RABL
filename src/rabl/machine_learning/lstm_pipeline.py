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
    model, history, used_device = train_with_fallback(datasets, epochs=50, out_dir=Path("outputs"))

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
        save_path=Path("outputs/rolling_forecast.png"),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

# --------------------------------------------------------------------------------------
# Defaults / naming
# --------------------------------------------------------------------------------------

STATE_DIM = 13
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100

# The 13 targets you are predicting (y has shape (num_steps, 13))
TARGET_NAMES: list[str] = [
    "TN2",
    "Tm",
    "Thp",
    "Tf",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "n",
    "rho_dollars",
    "Q_to_steam",
]


# --------------------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------------------

def choose_device_prefer_gpu() -> torch.device:
    """
    Prefer GPU if CUDA is available, otherwise use CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        return device

    print("No GPU detected by PyTorch. Using CPU.")
    return torch.device("cpu")


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
                "y": {
                    "mean": scaling_group["y_mean"][...].astype(np.float32),
                    "std": scaling_group["y_std"][...].astype(np.float32),
                },
            }
        return {
            "type": scaling_type,
            "y": {
                "min": scaling_group["y_min"][...].astype(np.float32),
                "span": scaling_group["y_span"][...].astype(np.float32),
            },
        }


def _descale_targets(h5_path: Path, values: np.ndarray) -> np.ndarray:
    stats = _load_scaling_stats(h5_path)
    scaling_type = stats["type"]
    y_stats = stats["y"]
    if scaling_type == "standard":
        return values * y_stats["std"] + y_stats["mean"]
    if scaling_type == "minmax":
        return values * y_stats["span"] + y_stats["min"]
    raise ValueError(f"Unsupported scaling type: {scaling_type}")


def _train_sample_generator(
    h5_path: Path, profile_names: list[str], split: str, seed: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for profile_name in profile_names:
            group = files_group[profile_name]
            x_data = group["X"][...].astype(np.float32)
            y_data = group["Y"][...].astype(np.float32)
            indices = np.arange(x_data.shape[0])
            rng.shuffle(indices)
            for idx in indices:
                yield x_data[idx], y_data[idx]


def _sample_generator(h5_path: Path, profile_names: list[str], split: str) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for profile_name in profile_names:
            group = files_group[profile_name]
            x_data = group["X"][...].astype(np.float32)
            y_data = group["Y"][...].astype(np.float32)
            for idx in range(x_data.shape[0]):
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

    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        if self.seed is None:
            generator = _sample_generator(self.h5_path, self.profile_names, self.split)
        else:
            generator = _train_sample_generator(self.h5_path, self.profile_names, self.split, self.seed)
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True)

    # Validation samples dataset (flat)
    val_sample_ds = SampleDataset(h5_path, val_profiles, "val")
    val_sample_loader = DataLoader(val_sample_ds, batch_size=batch_size, pin_memory=True)

    # Profile datasets: yields entire profile arrays
    val_profile_ds = ProfileDataset(h5_path, val_profiles, "val")
    test_profile_ds = ProfileDataset(h5_path, test_profiles, "test")

    return {
        # datasets
        "train": train_loader,
        "val_samples": val_sample_loader,
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
    def __init__(self, timesteps: int, num_features: int, num_targets: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.fc2(self.relu(self.fc1(last_step)))


def build_model(timesteps: int, num_features: int, num_targets: int) -> LSTMRegressor:
    return LSTMRegressor(timesteps, num_features, num_targets)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    data_time_s = 0.0
    compute_time_s = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data_iter = iter(loader)
    while True:
        fetch_start = perf_counter()
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            break
        data_time_s += perf_counter() - fetch_start

        compute_start = perf_counter()
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize(device)

        total_loss += float(loss.item())
        compute_time_s += perf_counter() - compute_start       
        num_batches += 1
    max_mem = 0
    if device.type == "cuda":
        max_mem = int(torch.cuda.max_memory_allocated(device))
    return total_loss / max(1, num_batches), data_time_s, compute_time_s, max_mem


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
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
    verbose: int = 1,
) -> tuple[nn.Module, dict[str, list[float]], Path]:
    """
    Train the LSTM and save a train/val curve plot.
    """
    timesteps = int(datasets["sample_shape"][1])
    num_features = int(datasets["sample_shape"][2])
    num_targets = int(datasets["target_shape"][1])

    if training_device is None:
        training_device = choose_device_prefer_gpu()

    model = build_model(timesteps, num_features, num_targets).to(training_device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    history = {"loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        train_loss, data_time_s, compute_time_s, max_mem = _train_one_epoch(
            model,
            datasets["train"],
            optimizer,
            loss_fn,
            training_device,
        )

        if training_device.type =="cuda":
            torch.cuda.synchronize(training_device)
        
        val_start = perf_counter()
        val_loss = _evaluate(model, datasets["val_samples"], loss_fn, training_device)

        if training_device.type == "cuda":
            torch.cuda.synchronize(training_device)
        val_time_s = perf_counter() - val_start
        
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose:
            mem_mb = max_mem / (1024**2)
            print(
                f"Epoch {epoch}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f} "
                f"- data_time: {data_time_s:.2f}s - compute_time: {compute_time_s:.2f}s "
                f"- val_time: {val_time_s:.2f}s "
                f"- max_cuda_mem: {mem_mb:.2f} MB"
            )

    resolved_plot_path = Path(plot_path) if plot_path is not None else None
    if resolved_plot_path is None:
        resolved_plot_path = Path("outputs") / "plots" / "lstm_training_curves.png"
    resolved_plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history["loss"], label="Train Loss (MSE)")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolved_plot_path, dpi=150)
    print(f"Saved training curves to {resolved_plot_path}")

    return model, history, resolved_plot_path


def train_with_fallback(
    datasets: dict[str, Any],
    *,
    epochs: int,
    out_dir: Path,
    prefer_gpu: bool = True,
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
            group.create_dataset("mse", data=entry["mse"])
            group.attrs["columns"] = column_attr


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
    max_plots: int = 0,
    plot_callback: Callable[..., None] | None = None,
) -> dict[str, float]:
    if target_names is None:
        target_names = list(TARGET_NAMES)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecasts: list[dict[str, Any]] = []
    fetch_times: list[float] = []
    inference_times: list[float] = []

    profile_iter = iter(profile_ds)
    while True:
        fetch_start = perf_counter()
        try:
            profile_name, x_profile, y_profile = next(profile_iter)
        except StopIteration:
            break
        fetch_times.append(perf_counter() - fetch_start)

        x_np = x_profile.numpy()
        y_np = y_profile.numpy()
        inference_start = perf_counter()
        y_pred = rolling_forecast(model, x_np, state_dim=state_dim)
        inference_times.append(perf_counter() - inference_start)

        y_true = y_np
        y_pred_out = y_pred
        if h5_path is not None:
            y_true = _descale_targets(h5_path, y_true)
            y_pred_out = _descale_targets(h5_path, y_pred_out)

        t_series = np.arange(y_pred_out.shape[0], dtype=np.float32)
        u_series = _extract_control_series(x_np, state_dim=state_dim, control_channel=control_channel)
        table = _assemble_forecast_table(t_series, u_series, y_true, y_pred_out)
        mse = float(np.mean((y_true - y_pred_out) ** 2))

        forecasts.append(
            {
                "profile": str(profile_name),
                "table": table,
                "mse": mse,
            }
        )
        if plot_callback is not None and len(forecasts) <= max_plots:
            save_path = out_dir / f"rolling_forecast_{profile_name}.png"
            plot_callback(
                x_profile=x_np,
                y_true=y_true,
                y_pred=y_pred_out,
                title=f"Rolling Forecast - {profile_name}",
                save_path=save_path,
            )

    save_start = perf_counter()
    save_rolling_forecasts_hdf5(
        forecasts,
        output_path=out_dir / output_name,
        target_names=target_names,
    )
    save_time_s = perf_counter() - save_start

    avg_fetch = float(np.mean(fetch_times)) if fetch_times else 0.0
    avg_inference = float(np.mean(inference_times)) if inference_times else 0.0
    total_fetch = float(np.sum(fetch_times)) if fetch_times else 0.0
    total_inference = float(np.sum(inference_times)) if inference_times else 0.0
    total_test = total_fetch + total_inference
    print(
        "Testing timing summary:"
        f" avg_fetch_profile: {avg_fetch:.4f}s"
        f" avg_inference_profile: {avg_inference:.4f}s"
        f" total_fetch: {total_fetch:.4f}s"
        f" total_test: {total_test:.4f}s"
        f" save_time: {save_time_s:.4f}s"
    )

    return {
        "avg_fetch_profile_s": avg_fetch,
        "avg_inference_profile_s": avg_inference,
        "total_fetch_s": total_fetch,
        "total_test_s": total_test,
        "save_time_s": save_time_s,
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
        print(f"  Y profile shape: {y_profile.shape}")


# --------------------------------------------------------------------------------------
# Plotting (2 x 7 grid)
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
) -> None:
    """
    2x7 grid (14 plots total):
      - [0] control profile across all forecast steps
      - remaining 13 plots: each target (truth + pred)

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

    fig, axes = plt.subplots(2, 7, figsize=(26, 7))
    axes = axes.ravel()

    # Control plot (top-left)
    ax0 = axes[0]
    ax0.plot(control_series, label=control_name)
    ax0.set_title(control_name)
    ax0.set_xlabel("Forecast step")
    ax0.set_ylabel(control_name)
    ax0.grid(True)
    ax0.legend(loc="best")

    # 13 target plots
    for i in range(num_targets):
        ax = axes[i + 1]
        name = target_names[i]
        ax.plot(y_true[:, i], label="truth")
        ax.plot(y_pred[:, i], "--", label="pred")
        ax.set_title(name)
        ax.set_xlabel("Forecast step")
        ax.set_ylabel(name)
        ax.grid(True)
        ax.legend(fontsize=7, loc="best")

    mse_all = float(np.mean((y_true - y_pred) ** 2))
    fig.suptitle(f"{title} | MSE(all dims) = {mse_all:.6e}", y=1.02, fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved forecast plot to: {save_path}")

    plt.close(fig)


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

    def __post_init__(self) -> None:
        if self.target_names is None:
            self.target_names = list(TARGET_NAMES)


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

    def train(self, *, epochs: int = DEFAULT_EPOCHS, out_dir: Path = Path("outputs"), prefer_gpu: bool = True):
        if self.datasets is None:
            self.build()
        return train_with_fallback(self.datasets, epochs=epochs, out_dir=out_dir, prefer_gpu=prefer_gpu)

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


# --------------------------------------------------------------------------------------
# Script entrypoint example
# --------------------------------------------------------------------------------------
def main() -> None:
    # Example local run: edit the dataset path
    h5_path = Path(r"../outputs/datasets/lstm_merged_batch_0001-batch_0001_k10_standard_train0.70_val0.15_test0.15.h5")
    batch_size = 64
    epochs = 20
    seed = 123

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(h5_path=h5_path, batch_size=batch_size, seed=seed)
    inspect_dataset_shapes(datasets)

    model, history, used_device = train_with_fallback(datasets, epochs=epochs, out_dir=out_dir)
    print(f"\nFinished training using device: {used_device}")

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
