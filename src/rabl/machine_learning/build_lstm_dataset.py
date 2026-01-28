import argparse
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
        "steady_state": getattr(config_module, "STEADY_STATE", None),
    }


def _validate_config(config: dict) -> dict:
    if not isinstance(config["steady_state"], dict):
        raise SystemExit("STEADY_STATE must be a dictionary in config.py.")
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


def _steady_state_rows(steady_state: dict, k: int) -> tuple[np.ndarray, np.ndarray]:
    missing = [key for key in STATE_COLUMNS + (CONTROL_COLUMN,) if key not in steady_state]
    if missing:
        raise SystemExit(f"STEADY_STATE is missing keys: {missing}")

    state_row = np.asarray([steady_state[key] for key in STATE_COLUMNS], dtype=float)
    control_row = np.asarray([steady_state[CONTROL_COLUMN]], dtype=float)
    state_pad = np.repeat(state_row[None, :], k + 1, axis=0)
    control_pad = np.repeat(control_row[None, :], k + 1, axis=0)
    return state_pad, control_pad


def _validate_lookback(k: int) -> int:
    if not isinstance(k, int) or k < 1:
        raise SystemExit("Lookback must be a positive integer.")
    return k


def build_dataset(sim_root: Path, output_dir: Path, steady_state: dict, k: int) -> Path:
    k = _validate_lookback(k)

    csv_files = _collect_csv_files(sim_root)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {sim_root} matching {CSV_PATTERN}.")

    batch_dirs = sorted(p for p in sim_root.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    if not batch_dirs:
        raise SystemExit(f"No batch directories found in {sim_root}.")
    batch_numbers = [int(path.name.split("_", maxsplit=1)[1]) for path in batch_dirs]
    batch_i = min(batch_numbers)
    batch_f = max(batch_numbers)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lstm_merged_batch_{batch_i:04d}-batch_{batch_f:04d}_k{k}.h5"

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["k_lookback"] = k
        h5f.attrs["state_feature_names"] = np.asarray(STATE_COLUMNS, dtype="S")
        h5f.attrs["control_feature_name"] = CONTROL_COLUMN
        files_group = h5f.create_group("files")

        total_samples = 0
        state_pad, control_pad = _steady_state_rows(steady_state, k)
        for csv_path in csv_files:
            state_data = _read_csv_columns(csv_path, STATE_COLUMNS)
            control_data = _read_csv_columns(csv_path, (CONTROL_COLUMN,))
            padded_states = np.vstack([state_pad, state_data])
            padded_control = np.vstack([control_pad, control_data])
            x_seq, y_seq = _build_sequences(padded_states, padded_control, k)
            if not x_seq.size:
                continue

            file_group = files_group.create_group(csv_path.stem)
            file_group.create_dataset("X", data=x_seq, compression="gzip")
            file_group.create_dataset("Y", data=y_seq, compression="gzip")
            file_group.attrs["source_file"] = str(csv_path)
            file_group.attrs["num_samples"] = x_seq.shape[0]
            total_samples += x_seq.shape[0]
            print(f"{csv_path.name}: {x_seq.shape[0]} samples")

    if total_samples == 0:
        raise SystemExit("No samples generated; check lookback size or input CSV lengths.")

    print(f"Found {len(csv_files)} CSV files under {sim_root}.")
    print(f"Generated {total_samples} samples across {len(csv_files)} files.")
    print(f"Saved dataset to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an LSTM-ready dataset from simulation outputs.")
    parser.add_argument("--lookback", type=int, required=True, help="Number of past timesteps to include.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    sim_root = repo_root / "outputs" / "sim_profiles"
    output_dir = repo_root / "outputs" / "datasets"
    config_path = repo_root / "scripts" / "config.py"

    config = _validate_config(_load_config(config_path))
    build_dataset(sim_root, output_dir, config["steady_state"], args.lookback)


if __name__ == "__main__":
    main()
