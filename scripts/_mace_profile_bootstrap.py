"""Paired resampling of complete profiles, conditional on frozen calibration."""
from __future__ import annotations

import hashlib
from typing import Any

import h5py
import numpy as np

from _mace_floor_study import write_csv

FIELDS = ("width_scaled", "score_scaled", "pointwise_coverage", "trajectory_coverage", "joint_coverage")


def profile_statistics(result: dict[str, Any]) -> dict[str, np.ndarray]:
    """Sufficient statistics preserve each profile's time/target dependence."""
    return {
        "width_scaled": result["width_scaled"].mean(axis=1),
        "score_scaled": result["score_scaled"].mean(axis=1),
        "pointwise_coverage": result["mask"].mean(axis=1),
        "trajectory_coverage": result["mask"].all(axis=1).astype(np.float64),
        "joint_coverage": result["mask"].all(axis=(1, 2)).astype(np.float64)[:, None],
    }


def resample_statistics(statistics: dict[str, dict[str, np.ndarray]], *,
                        replicates: int, seed: int, batch_size: int) -> tuple[dict, str]:
    """Use one common sequence of draws for every method, scope and metric."""
    if replicates < 2 or batch_size < 1 or seed < 0:
        raise ValueError("Require >=2 replicates, positive batch size, and nonnegative seed.")
    n = next(iter(statistics.values()))[FIELDS[0]].shape[0]
    if n < 2:
        raise ValueError("Need at least two independent test profiles.")
    columns, slices, offset = [], {}, 0
    for method, st in statistics.items():
        slices[method] = {}
        for field in FIELDS:
            a = np.asarray(st[field], dtype=np.float64)
            if a.ndim != 2 or len(a) != n or not np.all(np.isfinite(a)):
                raise ValueError(f"Invalid profile statistics: {method}/{field}")
            slices[method][field] = slice(offset, offset + a.shape[1])
            offset += a.shape[1]
            columns.append(a)
    matrix = np.column_stack(columns)
    samples = np.empty((replicates, matrix.shape[1]), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    digest = hashlib.sha256()
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates-start)
        indices = rng.integers(0, n, size=(size, n), dtype=np.int64)
        digest.update(indices.astype("<i8", copy=False).tobytes())
        counts = np.asarray([np.bincount(row, minlength=n) for row in indices], dtype=np.float64)
        # Accumulate in a fixed column order. einsum avoids BLAS/thread-count
        # differences and bounds memory without expanding full trajectories.
        samples[start:start+size] = np.einsum("bn,nk->bk", counts, matrix, optimize=False)/n
        if start % 1000 == 0:
            print(f"Paired profile bootstrap: {start+size}/{replicates}", flush=True)
    return {method: {field: samples[:, sl] for field, sl in fields.items()}
            for method, fields in slices.items()}, digest.hexdigest()


def safe_reduction(mace: np.ndarray, absolute: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(mace), np.shape(absolute)), np.nan)
    np.divide(mace, absolute, out=out, where=np.asarray(absolute)>0)
    return 1-out


def interval_row(estimate: float, samples: np.ndarray, confidence: float = .95) -> dict:
    finite = samples[np.isfinite(samples)]
    tail = (1-confidence)/2
    bounds = np.quantile(finite, [tail, 1-tail]) if finite.size else (None, None)
    return {"estimate": float(estimate) if np.isfinite(estimate) else None,
            "ci_lower": None if bounds[0] is None else float(bounds[0]),
            "ci_upper": None if bounds[1] is None else float(bounds[1]),
            "valid_replicates": int(finite.size), "undefined_replicates": int(samples.size-finite.size)}


def scoped(st: dict[str, np.ndarray], *, index: int | None, span: np.ndarray, mode: str) -> dict:
    def value(field):
        a = st[field]
        return a.mean(axis=-1) if index is None else a[..., index]
    scale = 1. if index is None else float(span[index])
    objective = (st["joint_coverage"][..., 0] if mode=="joint" and index is None
                 else value("pointwise_coverage" if mode=="horizon" else "trajectory_coverage"))
    return {"mean_full_width": value("width_scaled")*scale,
            "mean_interval_score": value("score_scaled")*scale,
            "objective_coverage": objective, "pointwise_coverage": value("pointwise_coverage"),
            "targetwise_trajectory_coverage": value("trajectory_coverage"),
            "joint_trajectory_coverage": st["joint_coverage"][..., 0]}


def bootstrap_outputs(out, statistics, samples, *, names, span, modes, target_units):
    comparisons, methods = [], []
    points = {method: {field: a.mean(axis=0) for field,a in st.items()} for method,st in statistics.items()}
    with h5py.File(out/"bootstrap_distributions.h5", "w") as f:
        f.attrs["scope_names"] = np.asarray(["aggregate"]+names, dtype="S")
        for mode in modes:
            for index in [None]+list(range(len(names))):
                scope = "aggregate" if index is None else names[index]
                units = "scaled" if index is None else target_units[index]
                objective = ("pointwise" if mode=="horizon" else
                             "all-target trajectory" if mode=="joint" and index is None else "targetwise trajectory")
                bootstrap, point = {}, {}
                for method in ("absolute", "mace"):
                    key = f"{mode}/{method}"
                    bootstrap[method] = scoped(samples[key], index=index, span=span, mode=mode)
                    point[method] = scoped(points[key], index=index, span=span, mode=mode)
                    for metric, draws in bootstrap[method].items():
                        # Joint coverage repeats only in aggregate; per-target
                        # joint scope means per-target trajectory diagnostics.
                        if index is not None and metric=="joint_trajectory_coverage":
                            continue
                        unit = units if metric.startswith("mean_") else "fraction"
                        methods.append({"mode":mode, "method":method, "scope":scope,
                                        "coverage_objective":objective, "metric":metric, "units":unit,
                                        **interval_row(point[method][metric], draws)})
                        f.create_dataset(f"methods/{mode}/{method}/{scope}/{metric}", data=draws, compression="gzip")
                for metric in point["mace"]:
                    if index is not None and metric=="joint_trajectory_coverage":
                        continue
                    a,b = bootstrap["mace"][metric], bootstrap["absolute"][metric]
                    ap,bp = point["mace"][metric], point["absolute"][metric]
                    calculations = [(metric+"_difference", ap-bp, a-b,
                                     units if metric.startswith("mean_") else "fraction")]
                    if metric in ("mean_full_width", "mean_interval_score"):
                        calculations.append((metric+"_relative_reduction", safe_reduction(ap,bp).item(),
                                             safe_reduction(a,b), "fraction"))
                    for label, estimate, draws, unit in calculations:
                        comparisons.append({"mode":mode, "scope":scope, "coverage_objective":objective,
                                            "metric":label, "units":unit,
                                            "comparison_definition":"1_minus_mace_over_absolute" if label.endswith("relative_reduction") else "mace_minus_absolute",
                                            **interval_row(estimate, draws)})
                        f.create_dataset(f"comparisons/{mode}/{scope}/{label}", data=draws, compression="gzip")
    write_csv(out/"bootstrap_method_metrics.csv", methods)
    write_csv(out/"bootstrap_comparisons.csv", comparisons)
    return comparisons
