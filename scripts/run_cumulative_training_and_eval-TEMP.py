from pathlib import Path
import subprocess
import sys


def format_cumulative_batch_window(start_batch: int, end_batch: int) -> str:
    """
    Example:
        start_batch=1, end_batch=4
        -> 0001-0002-0003-0004
    """
    return "-".join(f"{b:04d}" for b in range(start_batch, end_batch + 1))


def run_command(cmd: list[str], cwd: Path) -> None:
    """
    Run a command and stop immediately if it fails.
    """
    print("\n" + "=" * 100)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 100 + "\n")

    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}:\n"
            f"{' '.join(cmd)}"
        )


def main():
    repo_root = Path(__file__).resolve().parents[1]

    start_batch = 1
    first_end_batch = 4
    final_end_batch = 11

    for end_batch in range(first_end_batch, final_end_batch + 1):
        batch_window = format_cumulative_batch_window(
            start_batch=start_batch,
            end_batch=end_batch,
        )

        scaled_h5 = (
            repo_root
            / "outputs"
            / "datasets"
            / "scaled_split"
            / f"lstm_merged_batches_{batch_window}_k12_minmax_train0.70_val0.15_test0.15.h5"
        )

        out_dir = (
            repo_root
            / "outputs"
            / "ml_results"
            / "training_playground"
            / f"Batch{batch_window}_k12_minmax"
        )

        model_path = out_dir / "model.pt"

        print("\n" + "#" * 100)
        print(f"Processing cumulative batch window: {batch_window}")
        print("#" * 100)
        print(f"scaled_h5:   {scaled_h5}")
        print(f"out_dir:     {out_dir}")
        print(f"model_path:  {model_path}")

        if not scaled_h5.exists():
            raise FileNotFoundError(f"Missing HDF5 file: {scaled_h5}")

        out_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "scripts/train_lstm_cumulative_batches.py",
            "--start-batch",
            str(start_batch),
            "--end-batch",
            str(end_batch),
            "--preload-training-data",
        ]

        run_command(train_cmd, cwd=repo_root)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Training finished, but model was not found at: {model_path}"
            )

        eval_cmd = [
            sys.executable,
            "scripts/evaluate_testset_difficulty.py",
            "--scaled-h5",
            str(scaled_h5),
            "--model-path",
            str(model_path),
            "--out-dir",
            str(out_dir),
            "--dt",
            "0.4",
            "--n-bins",
            "10",
            "--include-per-target",
        ]

        run_command(eval_cmd, cwd=repo_root)

    print("\nAll cumulative batch windows completed successfully.")


if __name__ == "__main__":
    main()
    