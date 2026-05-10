from pathlib import Path

from rabl.paths import resolve_output_root
from rabl.machine_learning import LSTMDatasetScalerSplitter


def main() -> None:
    input_path = resolve_output_root() / "datasets" / "lstm_toy_batch_0001-batch_0001_k3.h5"

    splitter = LSTMDatasetScalerSplitter(
        input_path=input_path,
        scaling_type="standard",
    )
    output_path = splitter.run()
    print(f"Saved scaled dataset to {output_path}")


if __name__ == "__main__":
    main()
