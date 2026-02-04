"""Train LSTM pipeline and plot rolling forecasts for test profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from rabl.machine_learning.lstm_pipeline import (
    LSTMPipeline,
    LSTMPipelineConfig,
    test_and_save_forecasts,
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
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
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
        default=10,
        help="Number of test profiles to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = LSTMPipelineConfig(
        h5_path=args.h5_path,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    pipeline = LSTMPipeline(config)

    datasets = pipeline.build()
    pipeline.inspect()

    model, _history, used_device = pipeline.train(epochs=args.epochs, out_dir=args.out_dir)
    print(f"Finished training using device: {used_device}")

    forecast_results: list[dict[str, object]] = []

    for index, (profile_name, x_profile, y_profile) in enumerate(datasets["test_profile_ds"]):
        name = profile_name
        x_np = x_profile.numpy()
        y_np = y_profile.numpy()

        y_pred = pipeline.forecast(model, x_np)

        forecast_results.append(
            {
                "profile_name": name,
                "y_pred": y_pred,
                "y_true": y_np,
            }
        )

        if index < args.max_plots:
            save_path = args.out_dir / f"rolling_forecast_{name}.png"
            pipeline.plot(
                x_profile=x_np,
                y_true=y_np,
                y_pred=y_pred,
                title=f"Rolling Forecast - {name}",
                save_path=save_path,
            )

    print(f"Generated rolling forecasts for {len(forecast_results)} test profiles.")

    test_and_save_forecasts(
        model,
        datasets["test_profile_ds"],
        out_dir=args.out_dir,
        state_dim=pipeline.config.state_dim,
        control_channel=pipeline.config.control_channel,
        target_names=pipeline.config.target_names,
    )


if __name__ == "__main__":
    main()
