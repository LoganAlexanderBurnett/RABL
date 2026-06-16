import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from rabl.paths import resolve_output_root


STATE_COLUMNS = (
    "TN2",
    "Tm",
    "Thp",
    "Tf",
    "Tsg",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "n",
    "rho_dollars",
    "T_steam_out",
    "x_steam_out",
)
CONTROL_COLUMN = "drumAngleDeg"
TIME_COLUMN = "t"
CSV_PATTERN = "results_drum_profile_*.csv"
BRANCH_LINEAGE_FILENAME = "branched_results_lineage.json"
_BRANCH_TIME_ATOL = 1e-6
_BRANCH_TIME_RTOL = 1e-8


def _branch_time_tolerance(time_value: float) -> float:
    return max(_BRANCH_TIME_ATOL, abs(float(time_value)) * _BRANCH_TIME_RTOL)


@dataclass(frozen=True)
class BranchLineageEntry:
    result_stem: str
    root_id: str
    profile_id: str
    parent_profile_id: str | None
    branch_time: float | None
    source_stem: str | None = None


@dataclass(frozen=True)
class ProfileData:
    csv_path: Path
    stem: str
    t: np.ndarray
    states: np.ndarray
    control: np.ndarray
    lineage: BranchLineageEntry | None = None


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


def _collect_csv_files(batch_dirs: list[Path]) -> list[Path]:
    csv_files: list[Path] = []
    seen_paths: set[Path] = set()
    stems_seen: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for batch_dir in batch_dirs:
        for csv_path in sorted(batch_dir.glob(CSV_PATTERN)):
            resolved = csv_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            stem = csv_path.stem
            if stem in stems_seen and stems_seen[stem] != resolved:
                duplicate_stems.add(stem)
            else:
                stems_seen[stem] = resolved
            csv_files.append(csv_path)
    if duplicate_stems:
        duplicates = ", ".join(sorted(duplicate_stems))
        print(f"Warning: duplicate result stems detected across batches: {duplicates}")
    return sorted(csv_files)


def _resolve_batch_dirs(sim_root: Path, batch_ids: list[str]) -> list[Path]:
    if not sim_root.exists():
        raise SystemExit(f"Simulation directory not found: {sim_root}")
    if not batch_ids:
        raise SystemExit("At least one batch id must be provided.")

    batch_dirs = []
    for batch_id in batch_ids:
        if not isinstance(batch_id, str):
            raise SystemExit(f"Invalid batch id: {batch_id}")
        batch_id = batch_id.strip()
        if not (batch_id.isdigit() and len(batch_id) == 4):
            raise SystemExit(f"Invalid batch id: {batch_id}")
        batch_dir = sim_root / f"batch_{batch_id}"
        if not batch_dir.is_dir():
            raise SystemExit(f"Batch directory not found: {batch_dir}")
        batch_dirs.append(batch_dir)
    return batch_dirs


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


def _read_profile_data(csv_path: Path, lineage: BranchLineageEntry | None = None) -> ProfileData:
    data = _read_csv_columns(csv_path, (TIME_COLUMN,) + STATE_COLUMNS + (CONTROL_COLUMN,))
    t = data[:, 0]
    if np.any(np.diff(t) < -_BRANCH_TIME_ATOL):
        raise SystemExit(f"Time column is not non-decreasing in {csv_path}")
    return ProfileData(
        csv_path=csv_path,
        stem=csv_path.stem,
        t=t,
        states=data[:, 1 : 1 + len(STATE_COLUMNS)],
        control=data[:, 1 + len(STATE_COLUMNS) : 1 + len(STATE_COLUMNS) + 1],
        lineage=lineage,
    )


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


def _normalize_lineage_entries(raw: Any, lineage_path: Path) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, dict)]
    if isinstance(raw, dict):
        for key in ("profiles", "entries", "lineage"):
            value = raw.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        # Also accept a mapping keyed by result stem.
        entries: list[dict[str, Any]] = []
        for result_stem, value in raw.items():
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("result_stem", result_stem)
                entries.append(entry)
        if entries:
            return entries
    raise SystemExit(f"Invalid branch lineage JSON structure: {lineage_path}")


def _load_branch_lineage_from_batch_summary(batch_dir: Path) -> dict[str, BranchLineageEntry]:
    summary_path = batch_dir / "batch_summary.csv"
    branched_results_dir = batch_dir / "branched_results"
    if not summary_path.exists() or not branched_results_dir.exists():
        return {}

    with summary_path.open(newline="") as fp:
        rows = [row for row in csv.DictReader(fp) if str(row.get("status", "")).upper() == "OK"]
    if not rows:
        return {}
    if not any(str(row.get("parent_profile_id", "")).strip() for row in rows):
        return {}

    rows_by_source = {str(row.get("result_csv_out", "")).removesuffix(".csv"): row for row in rows}
    source_stems = sorted(path.stem for path in branched_results_dir.glob("results_*.csv"))
    copied_stems = sorted(path.stem for path in batch_dir.glob(CSV_PATTERN))
    if not source_stems or len(source_stems) != len(copied_stems):
        raise SystemExit(
            f"Cannot infer branch lineage from {summary_path}: found {len(source_stems)} branched results "
            f"and {len(copied_stems)} copied dataset CSVs."
        )

    mapping: dict[str, BranchLineageEntry] = {}
    for source_stem, copied_stem in zip(source_stems, copied_stems, strict=True):
        row = rows_by_source.get(source_stem)
        if row is None:
            raise SystemExit(f"Cannot infer branch lineage for {source_stem}: missing batch_summary row in {summary_path}")
        root_id = str(row.get("root_id", "")).strip()
        profile_id = str(row.get("profile_id", "")).strip()
        if not root_id or not profile_id:
            raise SystemExit(f"Invalid branch lineage row in {summary_path}: {row}")
        parent = str(row.get("parent_profile_id", "")).strip() or None
        branch_time_raw = row.get("branch_time_s", "")
        branch_time = None if branch_time_raw in (None, "") else float(branch_time_raw)
        if parent is not None and branch_time is None:
            raise SystemExit(f"Missing branch_time_s for {root_id}/{profile_id} in {summary_path}")
        mapping[copied_stem] = BranchLineageEntry(
            result_stem=copied_stem,
            source_stem=source_stem,
            root_id=root_id,
            profile_id=profile_id,
            parent_profile_id=parent,
            branch_time=branch_time,
        )
    return mapping


def _load_branch_lineage(batch_dirs: list[Path]) -> dict[Path, dict[str, BranchLineageEntry]]:
    lineage_by_batch: dict[Path, dict[str, BranchLineageEntry]] = {}
    for batch_dir in batch_dirs:
        lineage_path = batch_dir / BRANCH_LINEAGE_FILENAME
        if not lineage_path.exists():
            lineage_by_batch[batch_dir.resolve()] = _load_branch_lineage_from_batch_summary(batch_dir)
            continue
        raw = json.loads(lineage_path.read_text(encoding="utf-8"))
        entries = _normalize_lineage_entries(raw, lineage_path)
        mapping: dict[str, BranchLineageEntry] = {}
        for entry in entries:
            result_stem = str(entry.get("result_stem") or entry.get("copied_result_stem") or "").strip()
            if not result_stem:
                raise SystemExit(f"Branch lineage entry missing result_stem in {lineage_path}: {entry}")
            root_id = str(entry.get("root_id") or entry.get("root_group_name") or "").strip()
            profile_id = str(entry.get("profile_id") or "").strip()
            if not root_id or not profile_id:
                raise SystemExit(f"Branch lineage entry missing root/profile id in {lineage_path}: {entry}")
            parent = str(entry.get("parent_profile_id", "")).strip() or None
            branch_time_raw = entry.get("branch_time", entry.get("branch_time_s"))
            branch_time = None if branch_time_raw in (None, "") else float(branch_time_raw)
            if parent is not None and branch_time is None:
                raise SystemExit(f"Branch lineage entry missing branch_time for {root_id}/{profile_id} in {lineage_path}")
            mapping[result_stem] = BranchLineageEntry(
                result_stem=result_stem,
                source_stem=str(entry.get("source_stem") or entry.get("source_result_stem") or "").strip() or None,
                root_id=root_id,
                profile_id=profile_id,
                parent_profile_id=parent,
                branch_time=branch_time,
            )
        lineage_by_batch[batch_dir.resolve()] = mapping
    return lineage_by_batch


def _lineage_for_csv(
    csv_path: Path,
    lineage_by_batch: dict[Path, dict[str, BranchLineageEntry]],
) -> BranchLineageEntry | None:
    mapping = lineage_by_batch.get(csv_path.parent.resolve(), {})
    return mapping.get(csv_path.stem)


def _profile_lineage_key(profile: ProfileData) -> tuple[str, str]:
    if profile.lineage is None:
        raise ValueError(f"Profile {profile.stem} has no branch lineage metadata.")
    return (profile.lineage.root_id, profile.lineage.profile_id)


def _collect_pre_branch_history(
    profile: ProfileData,
    *,
    cutoff_time: float,
    rows_needed: int,
    by_lineage_key: dict[tuple[str, str], ProfileData],
    steady_state_rows: tuple[np.ndarray, np.ndarray],
    stack: tuple[tuple[str, str], ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the last ``rows_needed`` physical rows before ``cutoff_time``.

    If ``profile`` does not contain enough rows before ``cutoff_time``, this
    function walks up its recorded parent lineage. Equilibrium rows are used
    only after the traversal reaches a true root profile and still lacks enough
    pre-history.
    """
    if rows_needed < 1:
        return (
            np.empty((0,), dtype=float),
            np.empty((0, profile.states.shape[1]), dtype=float),
            np.empty((0, 1), dtype=float),
        )

    mask = profile.t < float(cutoff_time) - _branch_time_tolerance(float(cutoff_time))
    own_t = profile.t[mask]
    own_states = profile.states[mask]
    own_control = profile.control[mask]
    if own_t.size >= rows_needed:
        return own_t[-rows_needed:], own_states[-rows_needed:], own_control[-rows_needed:]

    missing = rows_needed - own_t.size
    lineage = profile.lineage
    if lineage is None or lineage.parent_profile_id is None:
        state_pad, control_pad = steady_state_rows
        pad_t = np.full((missing,), np.nan, dtype=float)
        return (
            np.concatenate([pad_t, own_t]),
            np.vstack([state_pad[:missing], own_states]),
            np.vstack([control_pad[:missing], own_control]),
        )

    key = _profile_lineage_key(profile)
    if key in stack:
        cycle = " -> ".join(f"{root}/{pid}" for root, pid in stack + (key,))
        raise SystemExit(f"Cycle detected in branch lineage while building dataset: {cycle}")
    parent_key = (lineage.root_id, lineage.parent_profile_id)
    parent = by_lineage_key.get(parent_key)
    if parent is None:
        raise SystemExit(
            f"Missing parent metadata/result for branched profile {lineage.root_id}/{lineage.profile_id}: "
            f"parent_profile_id={lineage.parent_profile_id}"
        )
    if lineage.branch_time is None:
        raise SystemExit(f"Missing branch_time for branched profile {lineage.root_id}/{lineage.profile_id}")

    parent_t, parent_states, parent_control = _collect_pre_branch_history(
        parent,
        cutoff_time=float(lineage.branch_time),
        rows_needed=missing,
        by_lineage_key=by_lineage_key,
        steady_state_rows=steady_state_rows,
        stack=stack + (key,),
    )
    return (
        np.concatenate([parent_t, own_t]),
        np.vstack([parent_states, own_states]),
        np.vstack([parent_control, own_control]),
    )


def _build_branch_sequences(
    profile: ProfileData,
    *,
    k: int,
    by_lineage_key: dict[tuple[str, str], ProfileData],
    steady_state_rows: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    lineage = profile.lineage
    if lineage is None or lineage.parent_profile_id is None:
        raise ValueError("_build_branch_sequences requires a non-root lineage entry.")
    if lineage.branch_time is None:
        raise SystemExit(f"Missing branch_time for branched profile {lineage.root_id}/{lineage.profile_id}")
    parent = by_lineage_key.get((lineage.root_id, lineage.parent_profile_id))
    if parent is None:
        raise SystemExit(
            f"Missing parent metadata/result for branched profile {lineage.root_id}/{lineage.profile_id}: "
            f"parent_profile_id={lineage.parent_profile_id}"
        )
    branch_time = float(lineage.branch_time)
    branch_time_tol = _branch_time_tolerance(branch_time)
    first_t = float(profile.t[0])
    if first_t + branch_time_tol < branch_time:
        raise SystemExit(
            f"Branched result {profile.csv_path} starts before branch_time: "
            f"first_t={first_t}, branch_time={lineage.branch_time}"
        )

    hist_t, hist_states, hist_control = _collect_pre_branch_history(
        parent,
        cutoff_time=float(lineage.branch_time),
        rows_needed=k,
        by_lineage_key=by_lineage_key,
        steady_state_rows=steady_state_rows,
    )
    finite_hist_t = hist_t[np.isfinite(hist_t)]
    if finite_hist_t.size and finite_hist_t[-1] >= first_t - _branch_time_tolerance(first_t):
        raise SystemExit(
            f"Branch history for {lineage.root_id}/{lineage.profile_id} overlaps branch continuation: "
            f"last_history_t={finite_hist_t[-1]}, first_branch_t={first_t}"
        )

    states = np.vstack([hist_states, profile.states])
    control = np.vstack([hist_control, profile.control])
    x_seq, y_seq = _build_sequences(states, control, k)
    return x_seq, y_seq, int(hist_states.shape[0])


def build_dataset(
    sim_root: Path,
    output_dir: Path,
    steady_state: dict,
    k: int,
    batch_ids: list[str],
    *,
    verbose: bool = False,
) -> Path:
    k = _validate_lookback(k)

    batch_dirs = _resolve_batch_dirs(sim_root, batch_ids)
    csv_files = _collect_csv_files(batch_dirs)
    if not csv_files:
        raise SystemExit(
            f"No CSV files found in requested batches under {sim_root} matching {CSV_PATTERN}."
        )
    formatted_batches = "-".join(batch_ids)

    lineage_by_batch = _load_branch_lineage(batch_dirs)
    profiles: list[ProfileData] = []
    for csv_path in csv_files:
        profiles.append(_read_profile_data(csv_path, _lineage_for_csv(csv_path, lineage_by_batch)))

    by_lineage_key: dict[tuple[str, str], ProfileData] = {}
    for profile in profiles:
        if profile.lineage is None:
            continue
        key = _profile_lineage_key(profile)
        if key in by_lineage_key:
            raise SystemExit(f"Duplicate branch lineage key found while building dataset: {key}")
        by_lineage_key[key] = profile

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lstm_merged_batches_{formatted_batches}_k{k}.h5"

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["k_lookback"] = k
        h5f.attrs["state_feature_names"] = np.asarray(STATE_COLUMNS, dtype="S")
        h5f.attrs["control_feature_name"] = CONTROL_COLUMN
        files_group = h5f.create_group("files")

        total_samples = 0
        state_pad, control_pad = _steady_state_rows(steady_state, k)
        branch_state_pad = state_pad[:k]
        branch_control_pad = control_pad[:k]
        for profile in profiles:
            lineage = profile.lineage
            if lineage is not None and lineage.parent_profile_id is not None:
                x_seq, y_seq, history_rows = _build_branch_sequences(
                    profile,
                    k=k,
                    by_lineage_key=by_lineage_key,
                    steady_state_rows=(branch_state_pad, branch_control_pad),
                )
            else:
                padded_states = np.vstack([state_pad, profile.states])
                padded_control = np.vstack([control_pad, profile.control])
                x_seq, y_seq = _build_sequences(padded_states, padded_control, k)
                history_rows = state_pad.shape[0]
            if not x_seq.size:
                continue

            file_group = files_group.create_group(profile.stem)
            file_group.create_dataset("X", data=x_seq, compression="gzip")
            file_group.create_dataset("Y", data=y_seq, compression="gzip")
            file_group.attrs["source_file"] = str(profile.csv_path)
            file_group.attrs["num_samples"] = x_seq.shape[0]
            file_group.attrs["history_rows_prepended"] = int(history_rows)
            if lineage is not None:
                file_group.attrs["branch_root_id"] = lineage.root_id
                file_group.attrs["branch_profile_id"] = lineage.profile_id
                file_group.attrs["branch_parent_profile_id"] = lineage.parent_profile_id or ""
                file_group.attrs["branch_time"] = np.nan if lineage.branch_time is None else float(lineage.branch_time)
                if lineage.source_stem:
                    file_group.attrs["branch_source_stem"] = lineage.source_stem
            total_samples += x_seq.shape[0]
            if verbose:
                print(f"{profile.csv_path.name}: {x_seq.shape[0]} samples")

    if total_samples == 0:
        raise SystemExit("No samples generated; check lookback size or input CSV lengths.")

    print(f"Found {len(csv_files)} CSV files in batches {formatted_batches}.")
    print(f"Generated {total_samples} samples across {len(csv_files)} files.")
    print(f"Saved dataset to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an LSTM-ready dataset from simulation outputs.")
    parser.add_argument("--lookback", type=int, required=True, help="Number of past timesteps to include.")
    parser.add_argument(
        "--batches",
        type=str,
        nargs="+",
        required=True,
        help="Batch IDs to include (e.g., --batches 0001 0002 0004).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output_root = resolve_output_root()
    sim_root = output_root / "sim_profiles"
    output_dir = output_root / "datasets"
    config_path = repo_root / "scripts" / "config.py"

    config = _validate_config(_load_config(config_path))
    build_dataset(sim_root, output_dir, config["steady_state"], args.lookback, args.batches)


if __name__ == "__main__":
    main()
