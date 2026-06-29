#!/usr/bin/env python3
"""Validate newly added branch samples between two LSTM dataset HDF5 files.

The script treats every ``(X[i], Y[i])`` pair present in ``after`` but not in
``before`` as newly added data, then runs branch-specific sanity checks on the
profiles that contain those samples.  It is intended for the unscaled datasets
written by ``src/rabl/machine_learning/build_lstm_dataset.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


BRANCH_TIME_ATOL = 1e-5
BRANCH_TIME_RTOL = 1e-7


@dataclass(frozen=True)
class SampleRef:
    profile: str
    index: int


@dataclass
class ProfileArrays:
    name: str
    x: np.ndarray
    y: np.ndarray
    attrs: dict[str, Any]

    @property
    def is_branch_child(self) -> bool:
        return bool(str(self.attrs.get("branch_parent_profile_id", "")).strip())


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_decode(v) for v in value.tolist()]
    return value


def _normalise_attr_dict(attrs: h5py.AttributeManager) -> dict[str, Any]:
    return {str(k): _decode(v) for k, v in attrs.items()}


def _h5_string_attr(h5f: h5py.File, key: str, default: Any = None) -> Any:
    return _decode(h5f.attrs[key]) if key in h5f.attrs else default


def _feature_names(h5f: h5py.File) -> tuple[list[str], str, int]:
    if "state_feature_names" not in h5f.attrs:
        raise RuntimeError("Dataset is missing state_feature_names metadata.")
    state_names = [str(v) for v in _decode(h5f.attrs["state_feature_names"])]
    control_name = str(_h5_string_attr(h5f, "control_feature_name", "drumAngleDeg"))
    lookback = int(_h5_string_attr(h5f, "k_lookback", 0))
    if lookback < 1:
        raise RuntimeError("Dataset is missing a valid k_lookback attribute.")
    return state_names, control_name, lookback


def _sample_digest(x_row: np.ndarray, y_row: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=24)
    x = np.ascontiguousarray(x_row)
    y = np.ascontiguousarray(y_row)
    h.update(str(x.shape).encode())
    h.update(str(x.dtype).encode())
    h.update(x.view(np.uint8))
    h.update(str(y.shape).encode())
    h.update(str(y.dtype).encode())
    h.update(y.view(np.uint8))
    return h.hexdigest()


def _load_profiles(path: Path) -> tuple[dict[str, ProfileArrays], Counter[str], dict[str, list[SampleRef]], tuple[list[str], str, int]]:
    profiles: dict[str, ProfileArrays] = {}
    counts: Counter[str] = Counter()
    refs: dict[str, list[SampleRef]] = defaultdict(list)
    with h5py.File(path, "r") as h5f:
        if "files" not in h5f:
            raise RuntimeError(f"{path} is missing the top-level 'files' group.")
        features = _feature_names(h5f)
        for name, group in h5f["files"].items():
            if "X" not in group or "Y" not in group:
                raise RuntimeError(f"{path}: files/{name} is missing X or Y.")
            x = group["X"][()]
            y = group["Y"][()]
            profiles[str(name)] = ProfileArrays(str(name), x, y, _normalise_attr_dict(group.attrs))
            for idx in range(x.shape[0]):
                digest = _sample_digest(x[idx], y[idx])
                counts[digest] += 1
                refs[digest].append(SampleRef(str(name), idx))
    return profiles, counts, refs, features


def _branch_time_tolerance(time_value: float) -> float:
    return max(BRANCH_TIME_ATOL, abs(float(time_value)) * BRANCH_TIME_RTOL)


def _read_csv(path: Path, columns: Iterable[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = list(columns)
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        missing = [col for col in columns if col not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"{path} is missing required columns: {missing}")
        rows = [[float(row[col]) for col in columns] for row in reader]
    if not rows:
        raise RuntimeError(f"{path} contains no data rows.")
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1:-1], data[:, -1:]


def _source_path(profile: ProfileArrays, sim_profiles_dir: Path | None = None) -> Path | None:
    raw = str(profile.attrs.get("source_file", "")).strip()
    if not raw:
        return None
    source = Path(raw)
    if sim_profiles_dir is None:
        return source

    # Dataset files often contain absolute source paths from the machine that
    # built them.  Allow callers to relocate those paths under a local
    # sim_profiles directory by preserving the batch_XXXX-relative suffix.
    for idx, part in enumerate(source.parts):
        if part.startswith("batch_"):
            return sim_profiles_dir.joinpath(*source.parts[idx:])

    candidate = sim_profiles_dir / source.name
    if candidate.exists():
        return candidate
    matches = list(sim_profiles_dir.glob(f"batch_*/{source.name}"))
    if len(matches) == 1:
        return matches[0]
    return candidate


def _lineage_key(profile: ProfileArrays, sim_profiles_dir: Path | None = None) -> tuple[str, str, str] | None:
    root = str(profile.attrs.get("branch_root_id", "")).strip()
    pid = str(profile.attrs.get("branch_profile_id", "")).strip()
    source = _source_path(profile, sim_profiles_dir)
    if not root or not pid or source is None:
        return None
    return (str(source.parent.resolve()), root, pid)


def _collect_parent_history(
    profile: ProfileArrays,
    profiles_by_key: dict[tuple[str, str, str], ProfileArrays],
    state_names: list[str],
    control_name: str,
    cutoff_time: float,
    rows_needed: int,
    sim_profiles_dir: Path | None = None,
    stack: tuple[tuple[str, str, str], ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _source_path(profile, sim_profiles_dir)
    if source is None:
        raise RuntimeError(f"{profile.name}: missing source_file attr.")
    t, states, control = _read_csv(source, ["t", *state_names, control_name])
    mask = t < float(cutoff_time) - _branch_time_tolerance(cutoff_time)
    own_t, own_states, own_control = t[mask], states[mask], control[mask]
    if own_t.size >= rows_needed:
        return own_t[-rows_needed:], own_states[-rows_needed:], own_control[-rows_needed:]

    missing = rows_needed - int(own_t.size)
    parent_id = str(profile.attrs.get("branch_parent_profile_id", "")).strip()
    if not parent_id:
        raise RuntimeError(
            f"{profile.name}: only {own_t.size} pre-branch rows available and no parent lineage to fill {missing} rows."
        )
    key = _lineage_key(profile, sim_profiles_dir)
    if key is None:
        raise RuntimeError(f"{profile.name}: missing lineage attrs needed to find parent.")
    if key in stack:
        raise RuntimeError(f"{profile.name}: cycle detected in branch lineage.")
    parent_key = (key[0], key[1], parent_id)
    parent = profiles_by_key.get(parent_key)
    if parent is None:
        raise RuntimeError(f"{profile.name}: parent profile not found for lineage key {parent_key}.")
    parent_branch_time_raw = profile.attrs.get("branch_time")
    if parent_branch_time_raw is None or (isinstance(parent_branch_time_raw, float) and math.isnan(parent_branch_time_raw)):
        raise RuntimeError(f"{profile.name}: missing branch_time attr.")
    parent_t, parent_states, parent_control = _collect_parent_history(
        parent,
        profiles_by_key,
        state_names,
        control_name,
        float(parent_branch_time_raw),
        missing,
        sim_profiles_dir,
        stack + (key,),
    )
    return np.concatenate([parent_t, own_t]), np.vstack([parent_states, own_states]), np.vstack([parent_control, own_control])


def _allclose(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    return bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))


def _validate_branch_profile(
    profile: ProfileArrays,
    profiles_by_key: dict[tuple[str, str, str], ProfileArrays],
    state_names: list[str],
    control_name: str,
    lookback: int,
    atol: float,
    rtol: float,
    sim_profiles_dir: Path | None = None,
) -> list[str]:
    problems: list[str] = []
    prefix = f"{profile.name}: "
    if profile.x.ndim != 3 or profile.y.ndim != 2:
        return [prefix + f"unexpected X/Y ranks: X{profile.x.shape}, Y{profile.y.shape}"]
    if profile.x.shape[1] != lookback + 1:
        problems.append(prefix + f"X timestep dimension {profile.x.shape[1]} != lookback+1 ({lookback + 1}).")
    if profile.x.shape[2] != len(state_names) + 1:
        problems.append(prefix + f"X feature dimension {profile.x.shape[2]} != state_count+1 ({len(state_names) + 1}).")
    if profile.y.shape[1] != len(state_names):
        problems.append(prefix + f"Y feature dimension {profile.y.shape[1]} != state_count ({len(state_names)}).")
    if profile.x.shape[0] != profile.y.shape[0]:
        problems.append(prefix + f"X/Y sample count mismatch: {profile.x.shape[0]} vs {profile.y.shape[0]}.")
    if not np.isfinite(profile.x).all() or not np.isfinite(profile.y).all():
        problems.append(prefix + "X or Y contains NaN/inf.")

    history_rows = int(profile.attrs.get("history_rows_prepended", -1))
    if history_rows != lookback:
        problems.append(prefix + f"history_rows_prepended={history_rows}, expected lookback={lookback} for branch child.")

    branch_time_raw = profile.attrs.get("branch_time")
    if branch_time_raw is None or (isinstance(branch_time_raw, float) and math.isnan(branch_time_raw)):
        problems.append(prefix + "missing branch_time attr.")
        return problems
    branch_time = float(branch_time_raw)
    source = _source_path(profile, sim_profiles_dir)
    if source is None or not source.exists():
        problems.append(prefix + f"source_file is missing or unreadable: {source}")
        return problems

    try:
        child_t, child_states, child_control = _read_csv(source, ["t", *state_names, control_name])
    except Exception as exc:  # noqa: BLE001 - script should aggregate validation problems.
        problems.append(prefix + f"could not read child source CSV: {exc}")
        return problems

    if child_t.size < 2:
        problems.append(prefix + f"child CSV has {child_t.size} rows; need at least 2 to create a branch target.")
        return problems
    first_t = float(child_t[0])
    if first_t + _branch_time_tolerance(branch_time) < branch_time:
        problems.append(prefix + f"child CSV starts before branch_time: first_t={first_t}, branch_time={branch_time}.")
    if profile.x.shape[0] != child_t.size - 1:
        problems.append(prefix + f"num_samples={profile.x.shape[0]}, expected child_csv_rows-1={child_t.size - 1}.")
    attr_num = int(profile.attrs.get("num_samples", profile.x.shape[0]))
    if attr_num != profile.x.shape[0]:
        problems.append(prefix + f"num_samples attr={attr_num}, but X has {profile.x.shape[0]} rows.")

    parent_id = str(profile.attrs.get("branch_parent_profile_id", "")).strip()
    key = _lineage_key(profile, sim_profiles_dir)
    if key is None or not parent_id:
        problems.append(prefix + "missing lineage attrs required for parent-history validation.")
        return problems
    parent = profiles_by_key.get((key[0], key[1], parent_id))
    if parent is None:
        problems.append(prefix + f"parent profile {parent_id!r} not found in after dataset.")
        return problems

    try:
        hist_t, hist_states, hist_control = _collect_parent_history(
            parent, profiles_by_key, state_names, control_name, branch_time, lookback, sim_profiles_dir
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(prefix + f"could not collect parent history: {exc}")
        return problems

    if hist_t.size != lookback:
        problems.append(prefix + f"collected {hist_t.size} history rows, expected {lookback}.")
        return problems
    finite_hist_t = hist_t[np.isfinite(hist_t)]
    if finite_hist_t.size and finite_hist_t[-1] >= first_t - _branch_time_tolerance(first_t):
        problems.append(prefix + f"history overlaps branch continuation: last_history_t={finite_hist_t[-1]}, first_child_t={first_t}.")

    expected_states = np.vstack([hist_states, child_states[:1]])
    expected_y0 = child_states[1]
    expected_control = np.vstack([hist_control, child_control[:2]])[1:]

    if profile.x.shape[0] > 0:
        x0 = profile.x[0]
        y0 = profile.y[0]
        if not _allclose(x0[:, : len(state_names)], expected_states, atol, rtol):
            problems.append(prefix + "first X state window does not equal parent history plus first child state.")
        if not _allclose(x0[:, len(state_names) : len(state_names) + 1], expected_control, atol, rtol):
            problems.append(prefix + "first X control window does not match expected one-step-shifted controls.")
        if not _allclose(y0, expected_y0, atol, rtol):
            problems.append(prefix + "first Y target does not equal second child state row.")

    return problems




def _plot_added_samples(
    profiles: dict[str, ProfileArrays],
    added_refs: list[SampleRef],
    state_names: list[str],
    control_name: str,
    output_path: Path,
    plot_count: int,
) -> Path | None:
    """Plot up to ``plot_count`` added samples in a 4x4 feature layout."""
    if plot_count < 1 or not added_refs:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = added_refs[:plot_count]
    feature_labels = [*state_names, control_name]
    n_axes = 16
    if len(feature_labels) > n_axes:
        feature_labels = feature_labels[:n_axes]
    elif len(feature_labels) < n_axes:
        feature_labels = [*feature_labels, *[""] * (n_axes - len(feature_labels))]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    flat_axes = axes.ravel()
    for sample_num, ref in enumerate(selected, start=1):
        profile = profiles[ref.profile]
        x_row = profile.x[ref.index]
        steps = np.arange(x_row.shape[0])
        label = f"{sample_num}: {ref.profile}[{ref.index}]"
        for feature_idx, ax in enumerate(flat_axes):
            if feature_idx >= x_row.shape[1] or not feature_labels[feature_idx]:
                ax.set_visible(False)
                continue
            ax.plot(steps, x_row[:, feature_idx], alpha=0.8, linewidth=1.0, label=label)

    for feature_idx, ax in enumerate(flat_axes):
        if not ax.get_visible():
            continue
        ax.set_title(feature_labels[feature_idx])
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("window step")
    flat_axes[0].legend(fontsize="xx-small", loc="best")
    fig.suptitle(f"First {len(selected)} newly added samples: X windows ({selected[0].profile} ...)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Dataset HDF5 before data addition.")
    parser.add_argument("after", type=Path, help="Dataset HDF5 after data addition.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for numeric comparisons.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for numeric comparisons.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--plot-out", type=Path, default=None, help="Output path for a 4x4 preview plot of newly added samples.")
    parser.add_argument("--plot-count", type=int, default=10, help="Maximum number of newly added samples to overlay in the preview plot.")
    parser.add_argument("--no-plot", action="store_true", help="Disable preview plot generation.")
    parser.add_argument(
        "--sim-profiles-dir",
        type=Path,
        default=None,
        help=(
            "Local sim_profiles directory to use when source_file attrs in the HDF5 point "
            "to another machine or stale absolute path. Batch-relative suffixes are preserved."
        ),
    )
    args = parser.parse_args()
    sim_profiles_dir = args.sim_profiles_dir.resolve() if args.sim_profiles_dir is not None else None

    before_profiles, before_counts, _before_refs, before_features = _load_profiles(args.before)
    after_profiles, after_counts, after_refs, after_features = _load_profiles(args.after)
    if before_features != after_features:
        raise RuntimeError(f"Dataset metadata differs: before={before_features}, after={after_features}")
    state_names, control_name, lookback = after_features

    added_counts = after_counts - before_counts
    removed_counts = before_counts - after_counts
    added_refs: list[SampleRef] = []
    for digest, count in added_counts.items():
        added_refs.extend(after_refs[digest][:count])
    added_by_profile: dict[str, list[int]] = defaultdict(list)
    for ref in added_refs:
        added_by_profile[ref.profile].append(ref.index)

    profiles_by_key: dict[tuple[str, str, str], ProfileArrays] = {}
    for profile in after_profiles.values():
        key = _lineage_key(profile, sim_profiles_dir)
        if key is not None:
            profiles_by_key[key] = profile

    problems: list[str] = []
    new_profile_names = sorted(set(after_profiles) - set(before_profiles))
    added_profile_names = sorted(added_by_profile)
    branch_added = [name for name in added_profile_names if after_profiles[name].is_branch_child]
    nonbranch_added = [name for name in added_profile_names if not after_profiles[name].is_branch_child]

    for name in branch_added:
        problems.extend(
            _validate_branch_profile(
                after_profiles[name], profiles_by_key, state_names, control_name, lookback, args.atol, args.rtol, sim_profiles_dir
            )
        )

    for name in nonbranch_added:
        problems.append(f"{name}: added samples are not marked as branch children; branch_parent_profile_id is empty/missing.")

    for name, indices in sorted(added_by_profile.items()):
        prof = after_profiles[name]
        if any(i < 0 or i >= prof.x.shape[0] for i in indices):
            problems.append(f"{name}: added sample index outside X/Y bounds.")
        if not np.isfinite(prof.x[indices]).all() or not np.isfinite(prof.y[indices]).all():
            problems.append(f"{name}: at least one added sample contains NaN/inf.")

    plot_path: Path | None = None
    if not args.no_plot:
        resolved_plot_out = args.plot_out or args.after.with_name(f"{args.after.stem}_added_samples_preview.png")
        plot_path = _plot_added_samples(after_profiles, added_refs, state_names, control_name, resolved_plot_out, args.plot_count)

    report = {
        "before": str(args.before),
        "after": str(args.after),
        "lookback": lookback,
        "state_feature_names": state_names,
        "control_feature_name": control_name,
        "sim_profiles_dir": None if sim_profiles_dir is None else str(sim_profiles_dir),
        "before_profile_count": len(before_profiles),
        "after_profile_count": len(after_profiles),
        "new_profile_count": len(new_profile_names),
        "new_profiles": new_profile_names,
        "added_sample_count": int(sum(added_counts.values())),
        "removed_sample_count": int(sum(removed_counts.values())),
        "added_profiles": {name: len(indices) for name, indices in sorted(added_by_profile.items())},
        "branch_added_profile_count": len(branch_added),
        "nonbranch_added_profile_count": len(nonbranch_added),
        "sample_plot": None if plot_path is None else str(plot_path),
        "plot_count": 0 if plot_path is None else min(max(args.plot_count, 0), len(added_refs)),
        "problems": problems,
    }

    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if removed_counts:
        print(f"ERROR: {sum(removed_counts.values())} samples from the before dataset are absent from after.", file=sys.stderr)
    if problems:
        print(f"ERROR: detected {len(problems)} branch-sample validation problem(s).", file=sys.stderr)
    return 1 if removed_counts or problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
