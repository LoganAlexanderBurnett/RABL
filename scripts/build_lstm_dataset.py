import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np


STATE_COLUMNS = (
    "TN2",
    "Tm",
    "Thp",
    "Tf",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "n",
    "rho_dollars",
    "Q_to_steam",
)
CONTROL_COLUMN = "drumAngleDeg"
CSV_PATTERN = "results_drum_profile_*.csv"


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"Missing config file: {config_path}")

    spec = importlib.util.spec_from_file_location("lstm_dataset_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load config file: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return {
        "k_lookback": getattr(config_module, "K_LOOKBACK", None),
    }


def _validate_config(config: dict) -> dict:
    if not isinstance(config["k_lookback"], int) or config["k_lookback"] < 1:
        raise SystemExit("K_LOOKBACK must be a positive integer in config.py.")
    return config


def _collect_csv_files(sim_root: Path) -> list[Path]:
    if not sim_root.exists():
        raise SystemExit(f"Simulation directory not found: {sim_root}")
    return sorted(sim_root.rglob(CSV_PATTERN))


def _read_csv_columns(csv_path: Path, columns: tuple[str, ...]) -> np.ndarray:
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        missing = [col for col in columns if col not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing columns in {csv_path.name}: {missing}")

        rows = []
        for row in reader:
            rows.append([float(row[col]) for col in columns])

    if not rows:
        raise SystemExit(f"No data rows found in {csv_path}")

    return np.asarray(rows, dtype=float)


def _build_sequences(states: np.ndarray, control: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    num_steps = states.shape[0]
    if control.shape[0] != num_steps:
        raise ValueError("State/control arrays must have the same length.")
    if num_steps < k + 2:
        return np.empty((0, k + 1, states.shape[1] + 1)), np.empty((0, states.shape[1]))

    x_list = []
    y_list = []
    for t in range(k, num_steps - 1):
        state_window = states[t - k : t + 1]
        control_window = control[t - k + 1 : t + 2]
        merged = np.concatenate([state_window, control_window], axis=1)
        x_list.append(merged)
        y_list.append(states[t + 1])

    return np.asarray(x_list, dtype=float), np.asarray(y_list, dtype=float)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sim_root = repo_root / "outputs" / "sim"
    output_dir = repo_root / "outputs" / "datasets"
    config_path = Path(__file__).resolve().parent / "config.py"

    config = _validate_config(_load_config(config_path))
    k = config["k_lookback"]

    csv_files = _collect_csv_files(sim_root)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {sim_root} matching {CSV_PATTERN}.")

    batch_dirs = sorted(p for p in sim_root.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    if not batch_dirs:
        raise SystemExit(f"No batch directories found in {sim_root}.")
    batch_numbers = [int(path.name.split("_", maxsplit=1)[1]) for path in batch_dirs]
    batch_i = min(batch_numbers)
    batch_f = max(batch_numbers)

    state_cols = STATE_COLUMNS
    control_col = CONTROL_COLUMN
    all_x = []
    all_y = []
    for csv_path in csv_files:
        state_data = _read_csv_columns(csv_path, state_cols)
        control_data = _read_csv_columns(csv_path, (control_col,))
        x_seq, y_seq = _build_sequences(state_data, control_data, k)
        if x_seq.size:
            all_x.append(x_seq)
            all_y.append(y_seq)

    if not all_x:
        raise SystemExit("No samples generated; check lookback size or input CSV lengths.")

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lstm_merged_batch_{batch_i:04d}-batch_{batch_f:04d}_k{k}.h5"

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("X", data=x, compression="gzip")
        h5f.create_dataset("Y", data=y, compression="gzip")
        h5f.attrs["k_lookback"] = k
        h5f.attrs["state_feature_names"] = np.asarray(state_cols, dtype="S")
        h5f.attrs["control_feature_name"] = control_col
        h5f.attrs["source_files"] = np.asarray([str(p) for p in csv_files], dtype="S")

    print(f"Found {len(csv_files)} CSV files under {sim_root}.")
    print(f"Generated {x.shape[0]} samples.")
    print(f"X shape: {x.shape}, Y shape: {y.shape}")
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
