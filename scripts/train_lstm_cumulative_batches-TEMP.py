from pathlib import Path
import argparse

from rabl.machine_learning.lstm_pipeline import (
    LSTMPipeline,
    LSTMPipelineConfig,
    test_and_save_forecasts,
    clear_cuda_cache,
    compute_and_save_rolling_forecast_metrics,
)


def format_cumulative_batch_window(end_batch: int, start_batch: int = 1) -> str:
    return "-".join(f"{b:04d}" for b in range(start_batch, end_batch + 1))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--end-batch", type=int, required=True)
    parser.add_argument("--start-batch", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-plots", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--n-lstm", type=int, default=1)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-dropout", type=float, default=0.0)

    parser.add_argument("--n-fc", type=int, default=1)
    parser.add_argument("--fc-hidden", type=int, nargs="+", default=[384])

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--preload-training-data", action="store_true")

    args = parser.parse_args()

    batch_window = format_cumulative_batch_window(
        end_batch=args.end_batch,
        start_batch=args.start_batch,
    )

    h5_path = Path(
        f"outputs/datasets/scaled_split/"
        f"lstm_merged_batches_{batch_window}_k12_minmax_train0.70_val0.15_test0.15.h5"
    )

    out_dir = Path(
        f"outputs/ml_results/training_playground/"
        f"Batch{batch_window}_k12_minmax"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Training cumulative batch window: {batch_window}")
    print("h5_path:", h5_path)
    print("h5_path exists:", h5_path.exists())
    print("out_dir:", out_dir)
    print("=" * 80)

    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find HDF5 file: {h5_path}")

    config = LSTMPipelineConfig(
        h5_path=h5_path,
        batch_size=args.batch_size,
        seed=args.seed,
        n_lstm=args.n_lstm,
        lstm_hidden=args.lstm_hidden,
        lstm_dropout=args.lstm_dropout,
        n_fc=args.n_fc,
        fc_hidden=tuple(args.fc_hidden),
        learning_rate=args.learning_rate,
        use_tqdm=False,
    )

    pipeline = LSTMPipeline(config)

    datasets = pipeline.build()
    pipeline.inspect()

    model, history, used_device = pipeline.train(
        epochs=args.epochs,
        out_dir=out_dir,
        preload_train_to_device=args.preload_training_data,
        early_stopping_min_delta=1e-7,
        early_stopping_patience=10,
        restore_best_weights=True,
        step_lr_step_size=30,
        step_lr_gamma=0.5,
    )

    print(f"Finished training using device: {used_device}")

    test_and_save_forecasts(
        model,
        datasets["test_profile_ds"],
        out_dir=out_dir,
        state_dim=pipeline.config.state_dim,
        control_channel=pipeline.config.control_channel,
        target_names=pipeline.config.target_names,
        max_plots=args.max_plots,
        plot_callback=pipeline.plot,
        h5_path=h5_path,
        num_workers=args.num_workers,
    )

    pt_path = out_dir / "model.pt"
    pipeline.save_model_pt(model, pt_path)
    print(f"Saved model weights to: {pt_path}")

    if "cuda" in str(used_device).lower():
        print(f"Clearing {used_device}...")
        del model
        clear_cuda_cache()

    forecast_h5_path = out_dir / "rolling_forecasts.h5"
    output_json_path = out_dir / "rolling_forecasts.json"

    compute_and_save_rolling_forecast_metrics(
        forecast_h5_path=forecast_h5_path,
        output_json_path=output_json_path,
    )

    print(f"Finished cumulative training workflow for: {batch_window}")


if __name__ == "__main__":
    main()
    
