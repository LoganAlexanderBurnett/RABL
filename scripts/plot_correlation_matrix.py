from pathlib import Path

from rabl.paths import resolve_output_root
from rabl.machine_learning import plot_feature_correlations


def main() -> None:
    input_path = resolve_output_root() / "datasets" / (
        "lstm_merged_batch_0001-batch_0001_k10_standard_train0.70_val0.15_test0.15.h5"
    )
    output_path = plot_feature_correlations(input_path)
    print(f"Saved correlation plot to {output_path}")


if __name__ == "__main__":
    main()
