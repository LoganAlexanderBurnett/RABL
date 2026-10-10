"""Independent coordinate checks and instrumented replay of frozen LSTM models."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from _mace_floor_study import evaluate, fingerprint
from _mace_scaler_audit import compare_arrays


class Checks:
    """Exact comparisons; numeric differences are never hidden behind a tolerance."""

    def __init__(self):
        self.rows = []

    def add(self, name, actual, expected, *, category="invariant", labels=None, **context):
        a, b = np.asarray(actual), np.asarray(expected)
        row = {**context, "check": name, "category": category, **compare_arrays(a, b)}
        if a.shape == b.shape and a.size and labels is not None and a.shape[-1] == len(labels):
            d = np.abs(a.astype(np.float64)-b.astype(np.float64)).reshape(-1, len(labels))
            row["max_absolute_difference_by_feature"] = dict(zip(labels, d.max(axis=0).tolist()))
        self.rows.append(row)
        return row

    def summary(self):
        result = {}
        for category in ("invariant", "coordinate", "replay", "diagnostic"):
            rows = [r for r in self.rows if r["category"] == category]
            result[category] = {"count": len(rows), "mismatches": sum(not r["match"] for r in rows)}
        return result


def selected_profiles(names, count):
    """Evenly spaced manifest positions, chosen without using performance."""
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= len(names):
        raise ValueError("profiles_per_split must be a positive integer within each split size")
    positions = np.linspace(0, len(names)-1, int(count), dtype=int)
    return [names[i] for i in positions]


def raw_to_scaled(raw, stats, axis):
    return (np.asarray(raw)-stats[axis]["min"])/stats[axis]["span"]


def target_to_input(values, stats):
    """Map output Y coordinates to input X coordinates using native scaler values.

    The affine form preserves the exact identity when the frames are identical.
    Cast only after the coordinate mapping, at the model input boundary.
    """
    n = len(stats["y"]["min"])
    factor = stats["y"]["span"]/stats["x"]["span"][:n]
    offset = (stats["y"]["min"]-stats["x"]["min"][:n])/stats["x"]["span"][:n]
    return (np.asarray(values, dtype=np.float64)*factor+offset).astype(np.float32)


def independent_rollout(model, x):
    """Reproduce the historical direct-feedback recurrence without its helper.

    This is a reproduction check, NOT a corrected-coordinate forecast method.
    The separate input-frame checks identify semantic coordinate mismatches.
    """
    import torch
    n = model.output_layer.out_features
    x = np.asarray(x, dtype=np.float32)
    history = x[0, :, :n].copy()
    result = np.empty((len(x), n), dtype=np.float32)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for t in range(len(x)):
            frame = np.empty(x.shape[1:], dtype=np.float32)
            frame[:, :n] = history
            frame[:, n:] = x[t, :, n:]
            result[t] = model(torch.from_numpy(frame[None]).to(device)).detach().cpu().numpy()[0]
            history[:-1] = history[1:].copy()
            history[-1] = result[t]
    return result


def observed_ensemble(models, x, stats, checks, *, split, profile, production, target_names=None):
    """Observe actual production calls, including own-member feedback and controls."""
    n = len(stats["y"]["min"])
    handles, traces = [], []
    steps = len(x)
    selected = sorted(set([0, steps//2, steps-1]))
    labels = None if target_names is None else list(target_names)+["drumAngleDeg"]
    try:
        for member, model in enumerate(models):
            trace = {"inputs": [], "predictions": []}
            traces.append(trace)

            def pre(_model, args, trace=trace, member=member):
                if not trace["inputs"]:
                    print(f"Production replay {split}/{profile}, member {member+1}/{len(models)}", flush=True)
                value = args[0].detach().cpu().numpy()
                if value.shape != (1, *x.shape[1:]):
                    raise ValueError("Unexpected model input shape")
                trace["inputs"].append(value[0].copy())

            def post(_model, _args, value, trace=trace):
                trace["predictions"].append(value.detach().cpu().numpy()[0].copy())

            handles.extend([model.register_forward_pre_hook(pre), model.register_forward_hook(post)])
        stack, mean, spread = production(models, x, state_dim=n)
    finally:
        for handle in handles:
            handle.remove()
    snapshots = []
    for member, trace in enumerate(traces):
        if len(trace["inputs"]) != steps or len(trace["predictions"]) != steps:
            raise ValueError("Production did not call every member once per horizon")
        observed = np.asarray(trace["inputs"], dtype=np.float32)
        predictions = np.asarray(trace["predictions"], dtype=np.float32)
        historical = np.empty_like(observed)
        required = np.empty_like(observed)
        direct, mapped = x[0, :, :n].copy(), x[0, :, :n].copy()
        for t in range(steps):
            historical[t, :, :n] = direct
            required[t, :, :n] = mapped
            historical[t, :, n:] = required[t, :, n:] = x[t, :, n:]
            direct = np.concatenate([direct[1:], predictions[t:t+1]], axis=0)
            mapped = np.concatenate([mapped[1:], target_to_input(predictions[t:t+1], stats)], axis=0)
        context = {"split": split, "profile": profile, "member": member}
        checks.add("actual_model_inputs_vs_raw_reconstructed_history_and_own_feedback",
                   observed, historical, labels=labels, **context)
        checks.add("actual_feedback_vs_required_input_coordinates", observed, required,
                   category="coordinate", labels=labels, **context)
        checks.add("actual_member_outputs_vs_production_stack", predictions, stack[member], **context)
        snapshots.append({"horizon_indices": np.asarray(selected),
                          "actual_model_inputs": observed[selected],
                          "required_input_coordinates": required[selected]})
    return stack, mean, spread, snapshots


def source_trace(functions):
    result = []
    for label, fn in functions.items():
        file = Path(inspect.getsourcefile(fn)).resolve()
        source, line = inspect.getsourcelines(fn)
        result.append({"role": label, "function": fn.__qualname__, "first_line": line,
                       "line_count": len(source), **fingerprint(file)})
    return result


def coverage_effect(saved, replay, truth, *, q, floor, span, names):
    """Frozen-q replay diagnostics, not a recalibrated method evaluation."""
    def result(values):
        return evaluate({"truth": truth[None], "mean": values[0][None], "spread": values[1][None]},
                        q, floor=floor, alpha=.05, span=span)
    a, b = result(saved), result(replay)
    rows = []
    for j, name in enumerate(names):
        lo, hi = b["lower"][0, :, j], b["upper"][0, :, j]
        rows.append({"target": name,
                     "saved_covered_points": int(a["mask"][0, :, j].sum()),
                     "replayed_covered_points": int(b["mask"][0, :, j].sum()),
                     "changed_coverage_events": int(np.count_nonzero(a["mask"][0, :, j] != b["mask"][0, :, j])),
                     "saved_target_trajectory_covered": bool(a["mask"][0, :, j].all()),
                     "replayed_target_trajectory_covered": bool(b["mask"][0, :, j].all()),
                     "max_scaled_lower_difference": float(np.max(np.abs(lo.astype(float)-a["lower"][0, :, j]))),
                     "max_scaled_upper_difference": float(np.max(np.abs(hi.astype(float)-a["upper"][0, :, j]))),
                     "mean_scaled_width_difference": float(b["width_scaled"][0, :, j].mean()-a["width_scaled"][0, :, j].mean()),
                     "mean_physical_width_difference": float(b["width_physical"][0, :, j].mean()-a["width_physical"][0, :, j].mean()),
                     "mean_scaled_interval_score_difference": float(b["score_scaled"][0, :, j].mean()-a["score_scaled"][0, :, j].mean())})
    return rows


def read_alias(group, *keys):
    for key in keys:
        if key in group:
            return group[key][()]
    raise KeyError(f"Missing {keys} in {group.name}")


def audit_status(summary):
    if summary["invariant"]["mismatches"]:
        return "audit_failed"
    if summary["coordinate"]["mismatches"]:
        return "input_output_coordinate_mismatch"
    if summary["replay"]["mismatches"]:
        return "replay_review_required"
    return "passed"
