from pathlib import Path

from rabl.machine_learning import LSTMDatasetScalerSplitter


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "outputs" / "datasets" / "lstm_merged_batch_0001-batch_0001_k10.h5"

    splitter = LSTMDatasetScalerSplitter(
        input_path=input_path,
        scaling_type="none",
    )
    output_path = splitter.run()
    print(f"Saved scaled dataset to {output_path}")


if __name__ == "__main__":
    main()
