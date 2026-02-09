from pathlib import Path

from rabl.machine_learning import plot_scaled_features


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "outputs" / "datasets" / (
        "lstm_merged_batches_0001_k5_standard_train0.70_val0.15_test0.15.h5"
    )

    output_path = plot_scaled_features(input_path)
    print(f"Saved combined split plot to {output_path}")


if __name__ == "__main__":
    main()
