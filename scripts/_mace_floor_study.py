"""Fixed-forecast MACE floor study. No training or inference dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from _mace_scaler_audit import compare_arrays, decode_names


FLOORS = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
METHODS = {
    "horizon": "ensemble_conformal_target_horizon",
    "trajectory": "ensemble_conformal_target_trajectory",
}


def write_json(path: Path, data: Any) -> None:
    def convert(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(data, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "sha256": digest.hexdigest()}


def load_manifests(paths: dict[str, Path], counts: dict[str, int]) -> dict[str, list[str]]:
    result = {}
    for split, path in paths.items():
        names = json.loads(path.read_text())[f"{split}_profiles"]
        if not names or len(names) != len(set(names)) or len(names) != counts[split]:
            raise ValueError(f"Invalid fixed {split} profile manifest: {path}")
        result[split] = [str(n) for n in names]
    for left in result:
        for right in result:
            if left < right and set(result[left]) & set(result[right]):
                raise ValueError(f"Overlapping {left}/{right} profiles")
    return result


def load_forecasts(
    path: Path, profiles: list[str], *, scaled_path: Path, split: str,
    target_names: list[str], member_count: int, steps: int, stats: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shape = (len(profiles), steps, len(target_names))
    arrays = {key: np.empty(shape, dtype=np.float32) for key in ("truth", "mean", "spread")}
    arrays["truth_physical"] = np.empty(shape, dtype=np.float32)
    checks = []
    with h5py.File(path, "r") as f, h5py.File(scaled_path, "r") as scaled:
        if set(f.keys()) != set(profiles):
            raise ValueError(f"Forecast profiles differ from the fixed {split} manifest: {path}")
        if decode_names(f.attrs.get("target_names", [])) != target_names:
            raise ValueError(f"Forecast target order mismatch: {path}")
        count = int(f.attrs.get("member_count", f.attrs.get("ensemble_member_count", -1)))
        ddof = int(f.attrs.get("ensemble_std_ddof", f.attrs.get("ensemble_ddof", -1)))
        if count != member_count or ddof != 0:
            raise ValueError(f"Expected {member_count} members and ddof=0: {path}")
        for i, name in enumerate(profiles):
            group = f[name]
            checks_i = {"profile": name, "precision_conversions": {}}
            for key, aliases in (
                ("truth", ("y_true_scaled",)),
                ("mean", ("ensemble_mean_scaled", "mean_scaled")),
                ("spread", ("ensemble_std_scaled", "spread_scaled")),
            ):
                dataset = next((a for a in aliases if a in group), None)
                if dataset is None:
                    raise ValueError(f"Missing {aliases} in {name}")
                native = group[dataset][()]
                if native.shape != shape[1:] or not np.all(np.isfinite(native)):
                    raise ValueError(f"Invalid forecast {name}/{dataset}")
                converted = native.astype(np.float32)
                arrays[key][i] = converted
                if native.dtype != converted.dtype:
                    checks_i["precision_conversions"][dataset] = compare_arrays(native, converted)
            if np.any(arrays["spread"][i] < 0):
                raise ValueError(f"Negative ensemble spread: {name}")
            members = group["member_predictions_scaled"][()]
            if members.shape != (member_count, steps, len(target_names)) or not np.all(np.isfinite(members)):
                raise ValueError(f"Invalid saved member forecasts: {name}")
            # Check the saved aggregation, but retain the original mean/spread.
            members32 = members.astype(np.float32)
            checks_i["member_mean"] = compare_arrays(arrays["mean"][i], members32.mean(axis=0))
            checks_i["member_spread"] = compare_arrays(arrays["spread"][i], members32.std(axis=0, ddof=0))
            if members.dtype != np.dtype("float32"):
                checks_i["precision_conversions"]["member_predictions_scaled"] = compare_arrays(members, members32)
            source = scaled[f"{split}/files/{name}"]
            checks_i["truth_vs_scaled_dataset"] = compare_arrays(group["y_true_scaled"][()], source["Y"][()])
            t = group["t"][()]
            if not np.array_equal(t, np.arange(steps, dtype=t.dtype)):
                raise ValueError(f"Expected the original indexed forecast time grid: {name}")
            control_key = "control" if "control" in group else "u"
            x_control = source["X"][:, -1, -1].astype(np.float32)
            expected_control = (x_control * np.float32(stats["x"]["span"][-1])
                                + np.float32(stats["x"]["min"][-1]))
            checks_i["control_vs_scaled_dataset"] = compare_arrays(group[control_key][()], expected_control)
            expected_physical = (arrays["truth"][i] * stats["y"]["span"].astype(np.float32)
                                 + stats["y"]["min"].astype(np.float32))
            physical_key = "y_true_physical" if "y_true_physical" in group else None
            if physical_key:
                physical = group[physical_key][()]
                checks_i["physical_truth_vs_run_descaling"] = compare_arrays(physical, expected_physical)
                arrays["truth_physical"][i] = physical.astype(np.float32)
            else:
                arrays["truth_physical"][i] = expected_physical
                checks_i["physical_truth_source"] = "reconstructed with float32 run scaler"
            if not checks_i["truth_vs_scaled_dataset"]["match"] or not checks_i["control_vs_scaled_dataset"]["match"]:
                raise ValueError(f"Forecast/data mismatch: {name}; {checks_i}")
            checks.append(checks_i)
            if (i + 1) % 500 == 0:
                print(f"Read saved {split} forecasts {i + 1}/{len(profiles)}", flush=True)
    return arrays, {"profiles": checks, "native_forecasts_retained_after_explicit_float32_conversion": True,
                    "aggregation_check_note": "Differences are reported exactly; original saved aggregates are retained."}


def calibrate(cal: dict[str, np.ndarray], *, alpha: float, floor: float | None) -> dict[str, np.ndarray]:
    """Recompute scores and both order statistics for each common additive floor."""
    n = len(cal["truth"])
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    k = ceil((n + 1) * (1 - alpha))
    if k > n:
        raise ValueError("Calibration size too small for a finite conformal quantile.")
    scores = np.abs(cal["truth"] - cal["mean"])
    if floor is not None:
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("The study requires a positive finite common floor.")
        scores = scores / (cal["spread"] + np.float32(floor))
    if not np.all(np.isfinite(scores)):
        raise ValueError("Nonfinite calibration scores")
    maxima = np.max(scores, axis=1)  # one maximum over time per profile and target
    return {"horizon": np.partition(scores, k - 1, axis=0)[k - 1],
            "trajectory": np.partition(maxima, k - 1, axis=0)[k - 1],
            "trajectory_scores": maxima, "rank": np.asarray(k, dtype=np.int32)}


def bounds(test: dict[str, np.ndarray], q: np.ndarray, floor: float | None) -> tuple[np.ndarray, np.ndarray]:
    scale = np.float32(1) if floor is None else test["spread"] + np.float32(floor)
    half = q * scale
    return test["mean"] - half, test["mean"] + half


def coverage(mask: np.ndarray) -> dict[str, Any]:
    ht = mask.mean(axis=0)
    traj = mask.all(axis=1).mean(axis=0)
    return {"pointwise": float(mask.mean()), "pointwise_by_target": mask.mean(axis=(0, 1)),
            "horizon_target": ht, "trajectory_by_target": traj,
            "mean_targetwise_trajectory": float(traj.mean()),
            "complete_multivariate_trajectory": float(mask.all(axis=(1, 2)).mean())}


def evaluate(test: dict[str, np.ndarray], q: np.ndarray, *, floor: float | None,
             alpha: float, span: np.ndarray) -> dict[str, Any]:
    lower, upper = bounds(test, q, floor)
    truth = test["truth"]
    # One inclusive scaled float32 endpoint convention for ALL coverage summaries.
    covered = (lower <= truth) & (truth <= upper)
    lo64, hi64, y64 = lower.astype(np.float64), upper.astype(np.float64), truth.astype(np.float64)
    widths = hi64 - lo64
    scores = widths + (2 / alpha) * (np.maximum(lo64 - y64, 0) + np.maximum(y64 - hi64, 0))
    return {"coverage": coverage(covered), "mask": covered, "lower": lower, "upper": upper,
            "width_scaled": widths, "width_physical": widths * span.astype(np.float64),
            "score_scaled": scores, "mean_scaled_width": float(widths.mean()),
            "mean_scaled_score": float(scores.mean()),
            "mean_physical_width_by_target": widths.mean(axis=(0, 1)) * span,
            "mean_physical_score_by_target": scores.mean(axis=(0, 1)) * span}


def distribution(values: np.ndarray, *, histogram: bool = False) -> dict[str, Any]:
    v = np.asarray(values).reshape(-1)
    levels = (0, .01, .05, .25, .5, .75, .9, .95, .99, 1)
    result = {"count": int(v.size), "mean": float(v.mean(dtype=np.float64)),
              **{f"p{100*p:g}": float(x) for p, x in zip(levels, np.quantile(v, levels))}}
    if histogram:
        edges = np.linspace(0, 1, 21)
        counts, _ = np.histogram(v, bins=edges)
        result.update(histogram_edges=edges.tolist(), histogram_counts=counts.tolist())
    return result


def wilson(successes: np.ndarray | int, n: int) -> tuple[np.ndarray, np.ndarray]:
    # Pointwise binomial intervals are valid at a fixed horizon/target; trajectory
    # intervals use profiles. Never treat all times as independent observations.
    p = np.asarray(successes, dtype=np.float64) / n
    z = 1.959963984540054
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return center - half, center + half


def legacy_metrics(cov: dict[str, Any], physical_cov: dict[str, Any], names: list[str], mode: str) -> dict[str, Any]:
    """Mirror the saved metrics' mixed-space coverage convention for comparison."""
    pc = physical_cov
    return {"marginal_pointwise_coverage_overall": cov["pointwise"],
            "marginal_pointwise_coverage_by_target": dict(zip(names, pc["pointwise_by_target"].tolist())),
            "marginal_pointwise_coverage_by_horizon": pc["horizon_target"].mean(axis=1).tolist(),
            "marginal_pointwise_coverage_by_horizon_target": pc["horizon_target"].tolist(),
            "targetwise_trajectory_coverage": dict(zip(names, pc["trajectory_by_target"].tolist())),
            "mean_targetwise_trajectory_coverage": pc["mean_targetwise_trajectory"],
            "complete_multivariate_trajectory_coverage": pc["complete_multivariate_trajectory"],
            "primary_empirical_coverage": (pc["mean_targetwise_trajectory"] if mode == "trajectory"
                                            else float(pc["horizon_target"].mean()))}


def reconcile_saved(
    results: dict[str, dict[str, Any]], quantiles: dict[str, np.ndarray], *,
    saved_dir: Path, test: dict[str, np.ndarray], stats: dict[str, Any],
    profiles: list[str], target_names: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    span32, min32 = stats["y"]["span"].astype(np.float32), stats["y"]["min"].astype(np.float32)
    for mode, result in results.items():
        method_id = METHODS[mode] if mode in METHODS else mode
        temporal = "horizon" if method_id.endswith("horizon") else "trajectory"
        method_dir = saved_dir / method_id
        saved_cal = json.loads((method_dir / "calibration_metadata.json").read_text())
        saved_metrics = json.loads((method_dir / "coverage_metrics.json").read_text())
        qkey = "q_by_target" if temporal == "trajectory" else "q_by_horizon_target"
        qjson = np.asarray(saved_cal[qkey], dtype=np.float64)
        qsaved = qjson.astype(np.float32)
        entry: dict[str, Any] = {"quantiles": compare_arrays(quantiles[mode], qsaved),
                                "saved_quantile_json_source_precision": "float32",
                                "json_quantile_conversion": compare_arrays(qjson, qsaved)}
        physical_mask = ((result["lower"] * span32 + min32 <= test["truth_physical"])
                         & (test["truth_physical"] <= result["upper"] * span32 + min32))
        pcover = coverage(physical_mask)
        entry["canonical_scaled_coverage"] = result["coverage"]
        entry["legacy_float32_physical_coverage"] = pcover
        changed = result["mask"] != physical_mask
        entry["coverage_disagreement_count"] = int(changed.sum())
        entry["coverage_disagreement_count_by_target"] = dict(zip(target_names, changed.sum(axis=(0, 1)).tolist()))
        entry["first_disagreements"] = [
            {"profile": profiles[i], "horizon_index": int(t), "target": target_names[j],
             "scaled_truth": float(test["truth"][i,t,j]),
             "scaled_lower": float(result["lower"][i,t,j]), "scaled_upper": float(result["upper"][i,t,j]),
             "physical_truth": float(test["truth_physical"][i,t,j])}
            for i, t, j in np.argwhere(changed)[:20]]
        mirrored = legacy_metrics(result["coverage"], pcover, target_names, temporal)
        entry["saved_metric_comparisons"] = {}
        entry["saved_coverage_count_comparisons"] = {}
        n, horizons, targets = result["mask"].shape
        denominators = {
            "marginal_pointwise_coverage_overall": n*horizons*targets,
            "marginal_pointwise_coverage_by_target": n*horizons,
            "marginal_pointwise_coverage_by_horizon": n*targets,
            "marginal_pointwise_coverage_by_horizon_target": n,
            "targetwise_trajectory_coverage": n,
            "mean_targetwise_trajectory_coverage": n*targets,
            "complete_multivariate_trajectory_coverage": n,
            "primary_empirical_coverage": n*targets if temporal == "trajectory" else n*horizons*targets,
        }
        mirrored["mean_scaled_interval_width_overall"] = result["mean_scaled_width"]
        # Preserve the historical float32 miss subtraction solely to explain
        # differences from the new float64 metric arithmetic.
        miss32 = (np.maximum(result["lower"] - test["truth"], 0)
                  + np.maximum(test["truth"] - result["upper"], 0))
        mirrored["mean_scaled_interval_score_overall"] = float(
            (result["width_scaled"] + (2 / float(saved_cal["alpha"])) * miss32).mean())
        entry["canonical_scaled_interval_score"] = result["mean_scaled_score"]
        for key, value in mirrored.items():
            old = saved_metrics[key]
            if isinstance(value, dict):
                a = np.asarray([value[n] for n in target_names], dtype=float)
                b = np.asarray([old[n] for n in target_names], dtype=float)
            else:
                a, b = np.asarray(value, dtype=float), np.asarray(old, dtype=float)
            entry["saved_metric_comparisons"][key] = compare_arrays(a, b)
            if key in denominators:
                denom = denominators[key]
                regenerated_counts = np.rint(a*denom).astype(np.int64)
                saved_counts = np.rint(b*denom).astype(np.int64)
                entry["saved_coverage_count_comparisons"][key] = {
                    "denominator": denom,
                    "conversion": "coverage fraction * known event count, rounded to integer covered events",
                    "match": bool(np.array_equal(regenerated_counts, saved_counts)),
                    "unequal_count": int(np.count_nonzero(regenerated_counts != saved_counts)),
                    "maximum_covered_event_count_difference": int(np.abs(regenerated_counts-saved_counts).max(initial=0)),
                }
        forecasts_path = method_dir / "test_forecasts.h5"
        entry["saved_interval_arrays"] = {"status": "not_available", "path": str(forecasts_path)}
        if forecasts_path.exists():
            checks = []
            with h5py.File(forecasts_path, "r") as f:
                if set(f.keys()) != set(profiles) or decode_names(f.attrs.get("target_names", [])) != target_names:
                    raise ValueError(f"Saved UQ forecast identity mismatch: {forecasts_path}")
                for i, name in enumerate(profiles):
                    checks.append({"profile": name, **{
                        key: compare_arrays(result[key][i], f[name][f"{key}_scaled"][()])
                        for key in ("lower", "upper")}})
            entry["saved_interval_arrays"] = {"status": "checked", "profiles": checks}
        report[method_id] = entry
    return report
