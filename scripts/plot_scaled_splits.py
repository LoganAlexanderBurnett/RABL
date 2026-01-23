from pathlib import Path

from machine_learning import plot_scaled_features


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "outputs" / "datasets" / (
        "lstm_merged_batch_0001-batch_0001_k10_none_train0.70_val0.15_test0.15.h5"
    )

    for split in ("train", "val", "test"):
        output_path = plot_scaled_features(input_path, split=split)
        print(f"Saved {split} plot to {output_path}")


if __name__ == "__main__":
    main()
