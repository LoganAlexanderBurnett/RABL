from pathlib import Path

from rabl.machine_learning import plot_feature_correlations


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "outputs" / "datasets" / (
        "lstm_merged_batch_0001-batch_0001_k10_standard_train0.70_val0.15_test0.15.h5"
    )
    output_path = plot_feature_correlations(input_path)
    print(f"Saved correlation plot to {output_path}")


if __name__ == "__main__":
    main()
