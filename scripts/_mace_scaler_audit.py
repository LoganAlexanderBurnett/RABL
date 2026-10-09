"""Precision-explicit, read-only checks of the MACE min-max dataset scaler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def decode_names(value: Any) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v)
            for v in np.atleast_1d(value).tolist()]


def compare_arrays(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Never apply a tolerance. Report native differences and explicit rounding."""
    a, b = np.asarray(actual), np.asarray(reference)
    report = {"actual_dtype": str(a.dtype), "reference_dtype": str(b.dtype),
              "actual_shape": list(a.shape), "reference_shape": list(b.shape)}
    if a.shape != b.shape:
        return {**report, "status": "shape_mismatch", "match": False}
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return {**report, "status": "nonfinite", "match": False}
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    report.update(max_absolute_difference=float(delta.max(initial=0)),
                  unequal_count_native=int(np.count_nonzero(a != b)))
    if a.dtype == b.dtype:
        exact = bool(np.array_equal(a, b))
        return {**report, "status": "exact_match" if exact else "mismatch",
                "match": exact, "conversion": None}
    # For this study all numeric source arrays are floating point. Round both
    # explicitly to the lower stored precision; retain the native delta above.
    if a.dtype.kind != "f" or b.dtype.kind != "f":
        raise ValueError("Precision conversion comparisons require floating arrays.")
    dtype = a.dtype if a.dtype.itemsize <= b.dtype.itemsize else b.dtype
    ac, bc = a.astype(dtype), b.astype(dtype)
    exact = bool(np.array_equal(ac, bc))
    report.update(conversion=f"both arrays -> {dtype}",
                  unequal_count_after_conversion=int(np.count_nonzero(ac != bc)),
                  max_absolute_difference_after_conversion=float(
                      np.abs(ac.astype(np.float64) - bc.astype(np.float64)).max(initial=0)),
                  match=exact, status="precision_converted_match" if exact else "mismatch")
    return report


def read_scaler(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    with h5py.File(path, "r") as f:
        if "scaling" not in f or "y_span" not in f["scaling"]:
            raise ValueError(f"A min-max scaling group is required: {path}")
        if "scaling_type" in f.attrs and str(f.attrs["scaling_type"]) != "minmax":
            raise ValueError(f"Expected minmax scaling: {path}")
        stats = {axis: {key: f[f"scaling/{axis}_{key}"][()]
                        for key in ("min", "max", "span")
                        if f"{axis}_{key}" in f["scaling"]} for axis in ("x", "y")}
        for axis in stats:
            for key in ("min", "span"):
                if key not in stats[axis] or stats[axis][key].ndim != 1:
                    raise ValueError(f"Missing or invalid {axis}_{key} in {path}")
            if not all(np.all(np.isfinite(a)) for a in stats[axis].values()):
                raise ValueError(f"Nonfinite scaler in {path}")
            if np.any(stats[axis]["span"] <= 0):
                raise ValueError(f"Nonpositive spans in {path}")
        states = decode_names(f.attrs.get("state_feature_names", []))
        control = decode_names(f.attrs.get("control_feature_name", []))
        metadata = {"target_names": states, "input_feature_names": states + control,
                    "attributes": {str(k): np.asarray(v).tolist() for k, v in f.attrs.items()
                                   if np.asarray(v).dtype.kind != "S"}}
    return stats, metadata


def profile_group(f: h5py.File, split: str, name: str) -> h5py.Group:
    for path in (f"{split}/files/{name}", f"files/{name}"):
        if path in f:
            return f[path]
    raise ValueError(f"Missing raw profile {split}/{name} in {f.filename}")


def verify_scaler(
    *, external_path: Path, training_path: Path, raw_path: Path,
    reference_json_path: Path, manifests: dict[str, list[str]], target_names: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    external, emeta = read_scaler(external_path)
    training, tmeta = read_scaler(training_path)
    payload = json.loads(reference_json_path.read_text())
    if payload.get("type", payload.get("scaling_type")) != "minmax":
        raise ValueError("Reference scaler JSON must be minmax.")
    reference = payload.get("stats", payload)
    # train_lstm_ensemble._save_scaling_stats_json writes the float32 arrays
    # loaded by lstm_pipeline, not the native HDF5 metadata precision.
    json_precision = np.float32
    comparisons: dict[str, Any] = {}
    for axis in ("x", "y"):
        for key in ("min", "span"):
            value = np.asarray(reference[axis][key], dtype=np.float64)
            if not np.array_equal(value, value.astype(json_precision).astype(np.float64)):
                raise ValueError("Reference JSON is not the float32 run scaler export; "
                                 "use the experiment's scaling_statistics.json.")
            comparisons[f"external_vs_run_json/{axis}_{key}"] = compare_arrays(
                external[axis][key], value.astype(json_precision))
        for key in training[axis]:
            comparisons[f"external_vs_training_h5/{axis}_{key}"] = compare_arrays(
                external[axis].get(key, np.array([], dtype=float)), training[axis][key])
    order_ok = (emeta["target_names"] == tmeta["target_names"] == target_names
                and emeta["input_feature_names"] == tmeta["input_feature_names"]
                and len(emeta["input_feature_names"]) == len(external["x"]["min"]))
    report: dict[str, Any] = {
        "external_metadata": emeta, "training_metadata": tmeta,
        "reference_json_numeric_source_precision": "float32 (run scaler export)",
        "feature_order_match": order_ok, "metadata_comparisons": comparisons,
        "sample_checks": [], "train_only_extrema_comparisons": {},
        "train_only_scope": "All X windows and Y rows of raw training profiles only; "
                            "includes prepared steady-state history, matching scaler fitting.",
    }
    with h5py.File(raw_path, "r") as raw, h5py.File(external_path, "r") as scaled:
        raw_states = decode_names(raw.attrs.get("state_feature_names", []))
        raw_inputs = raw_states + decode_names(raw.attrs.get("control_feature_name", []))
        report["raw_feature_order_match"] = (
            raw_states == target_names and raw_inputs == emeta["input_feature_names"])
        minima, maxima = {}, {}
        # Independent recomputation; do not call the implementation's fit helper.
        for index, name in enumerate(manifests["train"]):
            group = profile_group(raw, "train", name)
            for axis, dataset in (("x", "X"), ("y", "Y")):
                a = group[dataset][()]
                if not np.all(np.isfinite(a)):
                    raise ValueError(f"Nonfinite raw training {name}/{dataset}")
                flat = a.reshape(-1, a.shape[-1]).astype(np.float64)
                lo, hi = flat.min(axis=0), flat.max(axis=0)
                minima[axis] = np.minimum(minima.get(axis, lo), lo)
                maxima[axis] = np.maximum(maxima.get(axis, hi), hi)
            if (index + 1) % 500 == 0:
                print(f"Scaler check: raw training profiles {index + 1}/{len(manifests['train'])}", flush=True)
        for axis in ("x", "y"):
            span = maxima[axis] - minima[axis]
            span[span == 0] = 1.0  # reproduce the documented constant-feature convention
            fitted = {"min": minima[axis], "max": maxima[axis], "span": span}
            for source, stats in (("external", external), ("training", training)):
                for key, saved in stats[axis].items():
                    report["train_only_extrema_comparisons"][f"{source}/{axis}_{key}"] = (
                        compare_arrays(saved, fitted[key]))
        # Check actual raw -> stored scaled values, not descaled forecast truth.
        # Deterministic profiles from every split and first/middle/last rows.
        selected_targets = list(dict.fromkeys([target_names[0], "x_steam_out"]))
        if "x_steam_out" not in target_names:
            selected_targets = [target_names[0], target_names[-1]]
        for split, names in manifests.items():
            for name in dict.fromkeys([names[0], names[len(names)//2], names[-1]]):
                rgroup, sgroup = profile_group(raw, split, name), scaled[f"{split}/files/{name}"]
                for axis, dataset in (("x", "X"), ("y", "Y")):
                    r, s = rgroup[dataset][()], sgroup[dataset][()]
                    if r.shape != s.shape:
                        raise ValueError(f"Raw/scaled shape mismatch: {name}/{dataset}")
                    if axis == "y":
                        cols = [target_names.index(n) for n in selected_targets]
                        labels = selected_targets
                    else:
                        cols = [len(emeta["input_feature_names"]) - 1]
                        labels = [emeta["input_feature_names"][-1]]
                    for row in dict.fromkeys([0, len(r)//2, len(r)-1]):
                        for col, label in zip(cols, labels):
                            index_tuple = (row, col) if axis == "y" else (row, r.shape[1]-1, col)
                            physical = np.asarray(r[index_tuple])
                            actual = np.asarray(s[index_tuple])
                            expected = (physical - external[axis]["min"][col]) / external[axis]["span"][col]
                            check = compare_arrays(actual, np.asarray(expected))
                            report["sample_checks"].append({
                                "split": split, "profile": name, "dataset": dataset,
                                "index": list(index_tuple), "feature": label,
                                "physical_value": float(physical), "stored_scaled_value": float(actual),
                                "recomputed_scaled_value": float(expected), **check})
    report["match"] = bool(order_ok and report["raw_feature_order_match"]
                           and all(x["match"] for x in comparisons.values())
                           and all(x["match"] for x in report["train_only_extrema_comparisons"].values())
                           and all(x["match"] for x in report["sample_checks"]))
    return report, external
