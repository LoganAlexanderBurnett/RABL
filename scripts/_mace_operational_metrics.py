"""Phase-aware sufficient statistics and one paired whole-profile bootstrap."""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

from _mace_floor_study import calibrate, evaluate, write_csv, write_json
from _mace_operational_profiles import FAMILIES, PHASES, TIME, phase_masks
from _mace_profile_bootstrap import interval_row
from run_lstm_floor_sensitivity import config_target_unit
from run_mace_profile_validation import joint_calibration

MODES = ("horizon", "target_trajectory", "joint_trajectory")
SOURCES = ("original", "operational")
KINDS = ("absolute", "mace")
FIELDS = ("width_sum", "score_sum", "hit_sum", "target_success", "joint_success", "point_count", "profile_count")


def quantiles(cal, *, alpha, floor):
    result = {}
    for kind in KINDS:
        q = calibrate(cal, alpha=alpha, floor=floor if kind == "mace" else None)
        joint, scores = joint_calibration(q["trajectory_scores"], rank=int(q["rank"]))
        result[kind] = dict(horizon=q["horizon"], target_trajectory=q["trajectory"],
                            joint_trajectory=joint, target_maxima=q["trajectory_scores"],
                            joint_maxima=scores, rank=q["rank"])
    return result


def sufficient(result, mask):
    """Sums/counts avoid equal-profile averaging when phase lengths differ."""
    m = np.asarray(mask, bool)
    if m.shape != result["mask"].shape[:2]:
        raise ValueError("Phase mask must have one entry per profile/time")
    count = m.sum(axis=1).astype(np.float64)[:, None]
    valid = count > 0
    success = (result["mask"] | ~m[:, :, None]).all(axis=1) & valid
    return dict(width_sum=np.where(m[:, :, None], result["width_scaled"], 0).sum(axis=1),
                score_sum=np.where(m[:, :, None], result["score_scaled"], 0).sum(axis=1),
                hit_sum=(result["mask"] & m[:, :, None]).sum(axis=1).astype(np.float64),
                target_success=success.astype(np.float64),
                joint_success=success.all(axis=1).astype(np.float64)[:, None],
                point_count=count, profile_count=valid.astype(np.float64))


def divide(a, b):
    out = np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan, dtype=np.float64)
    np.divide(a, b, out=out, where=np.asarray(b) > 0)
    return out


def summarize(st, weights=None):
    matrix = np.column_stack([st[f] for f in FIELDS])
    sums = matrix.sum(axis=0) if weights is None else weights @ matrix
    j = st["width_sum"].shape[1]
    width, score, hit, target = [sums[..., i*j:(i+1)*j] for i in range(4)]
    joint, points, profiles = [sums[..., 4*j+i:4*j+i+1] for i in range(3)]
    return dict(mean_full_width=divide(width, points), mean_interval_score=divide(score, points),
                pointwise_coverage=divide(hit, points), target_trajectory_coverage=divide(target, profiles),
                joint_trajectory_coverage=divide(joint, profiles))


def scoped(values, *, index, mode, span):
    scale = 1. if index is None else float(span[index])
    result = {}
    for metric, value in values.items():
        if metric == "joint_trajectory_coverage":
            if index is None:
                result[metric] = value[..., 0]
            continue
        a = value.mean(axis=-1) if index is None else value[..., index]
        result[metric] = a*scale if metric.startswith("mean_") else a
    objective = ("pointwise_coverage" if mode == "horizon" else
                 "joint_trajectory_coverage" if mode == "joint_trajectory" and index is None
                 else "target_trajectory_coverage")
    result["objective_coverage"] = result[objective]
    return result


def draw_weights(path: Path, *, n, replicates, seed, batch_size):
    """Common profile IDs for every method/phase/family; batch-invariant PCG64."""
    if n < 2 or replicates < 2 or batch_size < 1:
        raise ValueError("Require two test profiles/replicates and a positive batch size")
    weights = np.empty((replicates, n), dtype=np.float64)
    rng, digest = np.random.default_rng(seed), hashlib.sha256()
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("test_profile_indices", shape=(replicates, n), dtype="i8",
                              chunks=(min(batch_size, replicates), n), compression="gzip")
        for start in range(0, replicates, batch_size):
            size = min(batch_size, replicates-start)
            indices = rng.integers(0, n, size=(size, n), dtype=np.int64)
            digest.update(indices.astype("<i8", copy=False).tobytes())
            ds[start:start+size] = indices
            weights[start:start+size] = np.asarray([np.bincount(row, minlength=n) for row in indices])
        f.attrs.update(seed=seed, algorithm="PCG64 whole-profile draws with replacement",
                       draws_sha256=digest.hexdigest())
    return weights, digest.hexdigest()


def comparison_metrics(a, b):
    result = {metric+"_difference": a[metric]-b[metric] for metric in a}
    ratio = divide(a["mean_full_width"], b["mean_full_width"])
    result.update(width_ratio=ratio, width_relative_reduction=1-ratio)
    return result


def add_rows(rows, context, point, samples, *, units):
    for metric, estimate in point.items():
        unit = (units if metric.startswith("mean_") else "fraction")
        rows.append({**context, "metric": metric, "units": unit,
                     **interval_row(float(estimate), np.asarray(samples[metric]))})


def accuracy_statistics(test, masks, stats):
    pred_physical = test["mean"].astype(np.float64)*stats["y"]["span"]+stats["y"]["min"]
    error = (pred_physical-test["truth_physical"])/stats["y"]["span"]
    return {phase: dict(absolute_sum=np.where(m[:, :, None], abs(error), 0).sum(axis=1),
                        square_sum=np.where(m[:, :, None], error**2, 0).sum(axis=1),
                        count=m.sum(axis=1).astype(np.float64)[:, None]) for phase, m in masks.items()}


def summarize_accuracy(st, weights=None):
    matrix = np.column_stack([st["absolute_sum"], st["square_sum"], st["count"]])
    sums = matrix.sum(axis=0) if weights is None else weights @ matrix
    j = st["absolute_sum"].shape[1]
    count = sums[..., -1:]
    return dict(mae=divide(sums[..., :j], count), mse=divide(sums[..., j:2*j], count))


def evaluate_study(out, *, calibrations, test, profiles, names, stats, alpha, floor,
                   settling_seconds, replicates, seed, batch_size, calibration_profiles=None):
    """Save calibrated quantiles, endpoints, profile statistics, and paired CIs."""
    if (out/"evaluation_metadata.json").exists() or (out/"intervals_and_statistics.h5").exists():
        raise FileExistsError("Evaluation already started; choose a new evaluation output directory")
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(completed=False, alpha=alpha, common_floor=floor,
                    calibration_fixed_in_bootstrap=True, bootstrap_replicates=replicates,
                    bootstrap_seed=seed, bootstrap_batch_size=batch_size,
                    coverage_convention="Inclusive float32 scaled endpoints/truth for every coverage metric",
                    original_calibration_scope="Transfer diagnostic; original stochastic profiles are not exchangeable with operational test profiles",
                    operational_calibration_scope="Independent calibration/test draws from the same prespecified uniform B/C/D/F mixture",
                    phase_coverage_scope="Full trajectory objectives apply only to all-time pooled mixture. Phase/family coverage is a diagnostic; no conditional conformal guarantee.",
                    accuracy_truth="Native physical simulator truth, float64; aggregate errors divided by original Y spans",
                    point_metrics="Pooled sums/counts over supported time points; aggregate equal target weights",
                    trajectory_metrics="Each supported profile counts once; zero-support phase profiles excluded",
                    bootstrap_scope="Paired ordinary whole-profile percentile intervals, conditional on frozen calibration and models; no simultaneous/multiple-comparison guarantee",
                    test_profile_names=[p["id"] for p in profiles], target_names=names)
    write_json(out/"evaluation_metadata.json", metadata)
    qs = {source: quantiles(calibrations[source], alpha=alpha, floor=floor) for source in SOURCES}
    with h5py.File(out/"calibration_quantiles.h5", "w") as f:
        f.attrs.update(alpha=alpha, common_floor=floor, target_names=np.asarray(names, dtype="S"))
        for source, kinds in qs.items():
            if calibration_profiles is not None:
                f.require_group(source).attrs["calibration_profile_names"] = np.asarray(calibration_profiles[source], dtype="S")
            for kind, values in kinds.items():
                for key, value in values.items():
                    f.create_dataset(f"{source}/{kind}/{key}", data=value)
    masks = {phase: np.stack([phase_masks(p["moves"], settling_seconds=settling_seconds)[phase]
                              for p in profiles]) for phase in PHASES}
    families = np.asarray([p["family"] for p in profiles])
    groups = {"pooled": np.ones(len(profiles), bool), **{family: families == family for family in FAMILIES}}
    weights, digest = draw_weights(out/"paired_bootstrap_draws.h5", n=len(profiles),
                                  replicates=replicates, seed=seed, batch_size=batch_size)
    metadata["bootstrap_draws_sha256"] = digest
    with h5py.File(out/"paired_bootstrap_draws.h5", "a") as f:
        f.attrs["test_profile_names"] = np.asarray(metadata["test_profile_names"], dtype="S")
    methods, comparisons, accuracy, horizons, support = [], [], [], [], []
    span = stats["y"]["span"].astype(np.float64)
    accuracy_stats = accuracy_statistics(test, masks, stats)
    for phase in PHASES:
        for family, selected in groups.items():
            st = {k: v[selected] for k, v in accuracy_stats[phase].items()}
            npoints, nprofiles = int(st["count"].sum()), int(np.count_nonzero(st["count"]))
            support.append(dict(phase=phase, family=family, test_profiles=int(selected.sum()),
                                supported_profiles=nprofiles, points_per_target=npoints))
            if not npoints:
                continue
            point, draws = summarize_accuracy(st), summarize_accuracy(st, weights[:, selected])
            for index in [None]+list(range(len(names))):
                scale = 1 if index is None else span[index]
                def extract(v):
                    # Aggregate RMSE is sqrt(equal-target mean normalized MSE).
                    mean = {k: a.mean(axis=-1) if index is None else a[..., index] for k, a in v.items()}
                    return dict(mae=mean["mae"]*scale, rmse=np.sqrt(mean["mse"])*scale)
                context = dict(phase=phase, family=family, scope="aggregate" if index is None else names[index])
                add_rows(accuracy, context, extract(point), extract(draws),
                         units="scaled" if index is None else config_target_unit(names[index]))
                # Accuracy has physical/scaled units, unlike coverage fractions.
                for row in accuracy[-2:]:
                    row["units"] = "scaled" if index is None else config_target_unit(names[index])
    with h5py.File(out/"intervals_and_statistics.h5", "w") as f:
        f.attrs.update(coverage_convention=metadata["coverage_convention"], target_names=np.asarray(names, dtype="S"),
                       test_profile_names=np.asarray(metadata["test_profile_names"], dtype="S"))
        for phase, m in masks.items():
            f.create_dataset("phase_masks/"+phase, data=m, compression="gzip")
            for key, value in accuracy_stats[phase].items():
                f.create_dataset(f"accuracy/{phase}/{key}", data=value, compression="gzip")
        for mode in MODES:
            mode_stats = {}
            for source in SOURCES:
                for kind in KINDS:
                    key = source+"/"+kind
                    result = evaluate(test, qs[source][kind][mode], floor=floor if kind == "mace" else None,
                                      alpha=alpha, span=span)
                    path = f"{mode}/{key}"
                    for field in ("lower", "upper", "mask"):
                        f.create_dataset(f"intervals/{path}/{field}", data=result[field], compression="gzip")
                    mode_stats[key] = {phase: sufficient(result, m) for phase, m in masks.items()}
                    for phase, st in mode_stats[key].items():
                        for field, value in st.items():
                            f.create_dataset(f"statistics/{path}/{phase}/{field}", data=value, compression="gzip")
                    if mode == "horizon":
                        for family, selected in groups.items():
                            if not selected.any():
                                continue
                            for t, row in enumerate(result["mask"][selected].mean(axis=0)):
                                for j, name in enumerate(names):
                                    horizons.append(dict(calibration=source, method=kind, family=family,
                                                         time_s=float(TIME[t]), target=name, coverage=float(row[j]),
                                                         test_profiles=int(selected.sum())))
                    del result
            for phase in PHASES:
                for family, selected in groups.items():
                    if not masks[phase][selected].any():
                        continue
                    print(f"Paired bootstrap summaries: {mode}, {phase}, {family}", flush=True)
                    point, draws = {}, {}
                    for key, phases in mode_stats.items():
                        st = {k: v[selected] for k, v in phases[phase].items()}
                        point[key], draws[key] = summarize(st), summarize(st, weights[:, selected])
                    for index in [None]+list(range(len(names))):
                        scope = "aggregate" if index is None else names[index]
                        units = "scaled" if index is None else config_target_unit(scope)
                        objective = ("pointwise" if mode == "horizon" else "joint trajectory" if mode == "joint_trajectory" and index is None else "target trajectory")
                        context = dict(mode=mode, phase=phase, family=family, scope=scope,
                                       coverage_objective=objective, full_objective=phase == "all",
                                       primary_scope=(phase == "all" and family == "pooled"
                                                      and (mode != "joint_trajectory" or index is None)),
                                       nominal_reference=1-alpha)
                        p = {key: scoped(v, index=index, mode=mode, span=span) for key, v in point.items()}
                        b = {key: scoped(v, index=index, mode=mode, span=span) for key, v in draws.items()}
                        for key in point:
                            source, kind = key.split("/")
                            add_rows(methods, {**context, "calibration": source, "method": kind}, p[key], b[key], units=units)
                        pairs = [(s+"/mace", s+"/absolute", s+":mace_minus_absolute") for s in SOURCES]
                        pairs += [("operational/"+k, "original/"+k, k+":operational_minus_original") for k in KINDS]
                        for a, c, label in pairs:
                            add_rows(comparisons, {**context, "comparison": label},
                                     comparison_metrics(p[a], p[c]), comparison_metrics(b[a], b[c]), units=units)
    write_csv(out/"accuracy_metrics.csv", accuracy)
    write_csv(out/"uq_method_metrics.csv", methods)
    write_csv(out/"paired_comparisons.csv", comparisons)
    write_csv(out/"horizon_target_coverage.csv", horizons)
    write_csv(out/"phase_support.csv", support)
    # Regenerate coverage from saved endpoints, exactly, with the same convention.
    with h5py.File(out/"intervals_and_statistics.h5", "r") as f:
        for mode in MODES:
            for source in SOURCES:
                for kind in KINDS:
                    g = f[f"intervals/{mode}/{source}/{kind}"]
                    regenerated = (g["lower"][()] <= test["truth"]) & (test["truth"] <= g["upper"][()])
                    if not np.array_equal(regenerated, g["mask"][()]):
                        raise ValueError("Saved endpoint/mask disagreement")
    metadata.update(completed=True, saved_endpoint_coverage_exact=True,
                    accuracy_rows=len(accuracy), method_rows=len(methods), comparison_rows=len(comparisons))
    write_json(out/"evaluation_metadata.json", metadata)
    return metadata
