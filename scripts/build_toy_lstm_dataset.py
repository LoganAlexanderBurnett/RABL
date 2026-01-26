import csv
from pathlib import Path

import h5py
import numpy as np

from scripts import build_lstm_dataset

TOY_SCALE_FACTORS = (
    2.0,
    3.0,
    1.5,
    0.5,
    -1.0,
    0.1,
    4.0,
    -0.5,
    2.5,
    0.75,
    1.25,
    -2.0,
    0.25,
)

DRUM_PROFILE_PATTERN = "drum_profile_*.csv"
DRUM_ANGLE_COLUMN = "Drum_Angle(deg)"
TIME_COLUMN = "Time(s)"
RESULTS_PATTERN = "results_drum_profile_*.csv"


def _read_drum_profile(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        missing = [col for col in (TIME_COLUMN, DRUM_ANGLE_COLUMN) if col not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing columns in {csv_path.name}: {missing}")

        times = []
        angles = []
        for row in reader:
            times.append(float(row[TIME_COLUMN]))
            angles.append(float(row[DRUM_ANGLE_COLUMN]))

    if not times:
        raise SystemExit(f"No data rows found in {csv_path}")

    return np.asarray(times, dtype=float), np.asarray(angles, dtype=float)


def _write_results_csv(
    output_path: Path,
    times: np.ndarray,
    control: np.ndarray,
    state_columns: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["t", *state_columns, build_lstm_dataset.CONTROL_COLUMN]
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for t_val, u_val in zip(times, control, strict=True):
            row = {"t": t_val, build_lstm_dataset.CONTROL_COLUMN: u_val}
            for col_name, factor in zip(state_columns, TOY_SCALE_FACTORS, strict=True):
                row[col_name] = u_val * factor
            writer.writerow(row)


def _generate_toy_results(variography_dir: Path, toy_dir: Path) -> list[Path]:
    csv_paths = sorted(variography_dir.glob(DRUM_PROFILE_PATTERN))
    if not csv_paths:
        raise SystemExit(f"No drum profiles found in {variography_dir}")

    result_paths = []
    for csv_path in csv_paths:
        times, angles = _read_drum_profile(csv_path)
        result_name = RESULTS_PATTERN.replace("*", csv_path.stem.replace("drum_profile_", ""))
        result_path = toy_dir / result_name
        _write_results_csv(result_path, times, angles, build_lstm_dataset.STATE_COLUMNS)
        result_paths.append(result_path)

    return result_paths


def _build_h5_dataset(sim_root: Path, output_dir: Path) -> Path:
    config = build_lstm_dataset._validate_config(
        build_lstm_dataset._load_config(Path(__file__).resolve().parent / "config.py")
    )
    k = config["k_lookback"]
    steady_state = config["steady_state"]

    csv_files = build_lstm_dataset._collect_csv_files(sim_root)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {sim_root} matching {build_lstm_dataset.CSV_PATTERN}.")

    batch_dirs = sorted(p for p in sim_root.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    if not batch_dirs:
        raise SystemExit(f"No batch directories found in {sim_root}.")
    batch_numbers = [int(path.name.split("_", maxsplit=1)[1]) for path in batch_dirs]
    batch_i = min(batch_numbers)
    batch_f = max(batch_numbers)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lstm_toy_batch_{batch_i:04d}-batch_{batch_f:04d}_k{k}.h5"

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["k_lookback"] = k
        h5f.attrs["state_feature_names"] = np.asarray(build_lstm_dataset.STATE_COLUMNS, dtype="S")
        h5f.attrs["control_feature_name"] = build_lstm_dataset.CONTROL_COLUMN
        files_group = h5f.create_group("files")

        total_samples = 0
        state_pad, control_pad = build_lstm_dataset._steady_state_rows(steady_state, k)
        for csv_path in csv_files:
            state_data = build_lstm_dataset._read_csv_columns(csv_path, build_lstm_dataset.STATE_COLUMNS)
            control_data = build_lstm_dataset._read_csv_columns(csv_path, (build_lstm_dataset.CONTROL_COLUMN,))
            padded_states = np.vstack([state_pad, state_data])
            padded_control = np.vstack([control_pad, control_data])
            x_seq, y_seq = build_lstm_dataset._build_sequences(padded_states, padded_control, k)
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
    print(f"Saved toy dataset to {output_path}")
    return output_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    variography_dir = repo_root / "outputs" / "variography" / "batch_0001"
    toy_root = repo_root / "outputs" / "toy_data"
    toy_batch_dir = toy_root / "batch_0001"
    output_dir = repo_root / "outputs" / "datasets"

    if len(TOY_SCALE_FACTORS) != len(build_lstm_dataset.STATE_COLUMNS):
        raise SystemExit("TOY_SCALE_FACTORS must have 13 entries to match STATE_COLUMNS.")

    generated = _generate_toy_results(variography_dir, toy_batch_dir)
    print(f"Generated {len(generated)} toy results CSVs in {toy_batch_dir}")

    _build_h5_dataset(toy_root, output_dir)


if __name__ == "__main__":
    main()
