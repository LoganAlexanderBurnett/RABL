"""Notebook-friendly wrapper for running the LSTM inspection script."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rabl.machine_learning.inspect_lstm_dataloaders import main as inspect_main


def run(
    h5_path: str | Path,
    batch_size: int = 64,
    epochs: int = 100,
    seed: int = 7,
    plot_path: Optional[str | Path] = None,
    demo_rolling: bool = False,
) -> None:
    """Run the inspection/training entry point with explicit arguments.

    Args:
        h5_path: Path to the merged LSTM HDF5 dataset.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        seed: Random seed for training shuffle.
        plot_path: Optional path to save the training curves.
        demo_rolling: Whether to run the rolling-forecast demo after training.
    """
    argv = ["--h5", str(h5_path), "--batch-size", str(batch_size), "--epochs", str(epochs), "--seed", str(seed)]
    if plot_path is not None:
        argv.extend(["--plot-path", str(plot_path)])
    if demo_rolling:
        argv.append("--demo-rolling")

    inspect_main(argv)


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be imported and used in notebooks. "
        "Call run(h5_path, ...) instead."
    )
