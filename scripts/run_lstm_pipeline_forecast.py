"""Train LSTM pipeline and plot rolling forecasts for test profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from rabl.machine_learning.lstm_pipeline import (
    LSTMPipeline,
    LSTMPipelineConfig,
    test_and_save_forecasts,
    clear_cuda_cache
)


repo_root = Path(__file__).resolve().parents[1]
DEFAULT_H5_PATH = repo_root / "outputs" / "datasets" / "lstm_toy_batch_0001-batch_0001_k3_standard_train0.70_val0.15_test0.15.h5"
DEFAULT_OUT_DIR = repo_root / "outputs" / "ml_results" / "toy_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LSTM pipeline and plot rolling forecasts for test profiles.",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=DEFAULT_H5_PATH,
        help="Path to the scaled LSTM dataset (.h5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used when building datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to save forecast plots.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=5,
        help="Number of test profiles to plot.",
    )
    parser.add_argument(
        "--n-lstm",
        type=int,
        default=3,
        help="Number of stacked LSTM layers.",
    )
    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=64,
        help="Hidden units for the LSTM layers (shared across all layers).",
    )
    parser.add_argument(
        "--lstm-dropout",
        type=float,
        default=0.0,
        help="Dropout between stacked LSTM layers (ignored if n_lstm=1).",
    )
    parser.add_argument(
        "--n-fc",
        type=int,
        default=1,
        help="Number of intermediate fully connected layers.",
    )
    parser.add_argument(
        "--fc-hidden",
        type=int,
        nargs="+",
        default=(64,),
        help="Hidden units per fully connected layer (space-separated list).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = LSTMPipelineConfig(
        h5_path=args.h5_path,
        batch_size=args.batch_size,
        seed=args.seed,
        n_lstm=args.n_lstm,
        lstm_hidden=args.lstm_hidden,
        lstm_dropout=args.lstm_dropout,
        n_fc=args.n_fc,
        fc_hidden=tuple(args.fc_hidden),
        learning_rate=args.learning_rate,
    )
    pipeline = LSTMPipeline(config)

    datasets = pipeline.build()
    pipeline.inspect()

    model, _history, used_device = pipeline.train(epochs=args.epochs, out_dir=args.out_dir)
    print(f"Finished training using device: {used_device}")

    test_and_save_forecasts(
        model,
        datasets["test_profile_ds"],
        out_dir=args.out_dir,
        state_dim=pipeline.config.state_dim,
        control_channel=pipeline.config.control_channel,
        target_names=pipeline.config.target_names,
        max_plots=args.max_plots,
        plot_callback=pipeline.plot,
        h5_path=args.h5_path
    )

    if "cuda" in str(used_device):
        print(f"Clearing {used_device}...")
        del model
        clear_cuda_cache()


if __name__ == "__main__":
    main()
