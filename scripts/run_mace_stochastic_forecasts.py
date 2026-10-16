"""Reforecast the fixed stochastic test split with the operational Y-to-X recurrence.

Reuse a hash-validated original calibration archive when compatible; otherwise
recompute original calibration with the same corrected recurrence as the test.
Never re-simulate, refit scalers, or train any network.
The exported HDF5 layout matches the historical MACE floor/profile readers, but
the archived historical predictions and outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np

from _mace_floor_study import fingerprint, write_json
from _mace_operational_forecasts import FEEDBACK, corrected_rollout, frozen_inputs, load_forecast, same_hashes
from _mace_original_calibration_cache import validate_cache
from _mace_scaler_audit import profile_group, read_scaler
from run_mace_scaler_use_audit import production_modules, runtime_settings

ROOT = Path(__file__).resolve().parents[1]
CAL_NAME = "calibration_ensemble_forecasts_audit.h5"
TEST_NAME = "test_ensemble_forecasts_audit.h5"


def write_profile(file, name, members, truth, control, stats):
    """Write a corrected ensemble profile in the fixed-study reader's schema.

    Physical truth is the exact float32 descaling of the frozen scaled Y array,
    matching the historical reader's documented precision convention.
    """
    members = np.asarray(members, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    control = np.asarray(control, dtype=np.float32)
    if (members.shape != (10, 501, 15) or truth.shape != (501, 15)
            or control.shape != (501,)
            or not all(np.all(np.isfinite(x)) for x in (members, truth, control))):
        raise ValueError(f"Invalid corrected forecast shape/values for {name}")
    y_min = np.asarray(stats["y"]["min"], dtype=np.float32)
    y_span = np.asarray(stats["y"]["span"], dtype=np.float32)
    if y_min.shape != (15,) or y_span.shape != (15,) or np.any(y_span <= 0):
        raise ValueError("Invalid frozen output scaler")
    mean = members.mean(axis=0)
    spread = members.std(axis=0, ddof=0)
    true_physical = truth * y_span + y_min
    mean_physical = mean * y_span + y_min
    physical_members = members * y_span + y_min
    two_sigma = np.float32(2) * physical_members.std(axis=0, ddof=0)
    g = file.create_group(name)
    for key, data in {
        "t": np.arange(501, dtype=np.float32),
        "control": control,
        "y_true_scaled": truth,
        "member_predictions_scaled": members,
        "ensemble_mean_scaled": mean,
        "ensemble_std_scaled": spread,
        "y_true_physical": true_physical,
        "ensemble_mean_physical": mean_physical,
        "ensemble_lower_2sigma_physical": mean_physical - two_sigma,
        "ensemble_upper_2sigma_physical": mean_physical + two_sigma,
    }.items():
        g.create_dataset(key, data=np.asarray(data, dtype=np.float32), compression="gzip")


def configure(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"floor_study_dir", "operational_study_dir", "output_dir",
               "device", "torch_threads", "input_overrides"}
    required = {"floor_study_dir", "operational_study_dir", "output_dir"}
    if required - config.keys() or config.keys() - allowed:
        raise ValueError("Missing or unknown corrected stochastic forecast config keys")
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError("Run from the repository root to resolve fixed study paths")
    if not isinstance(config.get("input_overrides", {}), dict):
        raise ValueError("input_overrides must be an object")
    config.setdefault("input_overrides", {})
    config.setdefault("device", "auto")
    config.setdefault("torch_threads", 1)
    config["reuse_original_calibration_dir"] = config["operational_study_dir"]
    out = Path(config["output_dir"]).resolve()
    if out.exists():
        raise FileExistsError(f"Existing outputs are protected: {out}")
    original = Path(config["operational_study_dir"]).resolve()
    floor = Path(config["floor_study_dir"]).resolve()
    if out == original or out == floor or original in out.parents or floor in out.parents:
        raise ValueError("Corrected output must be separate from source studies")
    return config, out


def checked_calibration_cache(config, prior, hashes, *, root, runtime):
    """Reuse only a fully verified cache; reforecast after compatible-code drift.

    A stale cache must never be treated as verified. Code/runtime drift permits
    recomputing *both* original calibration and test under the current frozen
    ensemble, scaler, and common corrected recurrence. Failed identity, hash, or
    content verification is not swallowed as an optional cache miss.
    """
    try:
        return validate_cache(config, prior, hashes, root=root, runtime=runtime), None
    except ValueError as exc:
        reason = str(exc)
        expected = ("Cache prediction/scaling source changed:",
                    "Cache recurrence source is not a reviewed compatible version",
                    "Cache runtime/precision settings differ")
        if not reason.startswith(expected):
            raise
        print(f"Cached calibration not reusable: {reason}", flush=True)
        print("Regenerating corrected original calibration from frozen models/data.", flush=True)
        return None, reason


def run(config_path):
    config, out = configure(config_path)
    prior, paths, hashes = frozen_inputs(config, ROOT)
    device, runtime = runtime_settings(config["device"], config["torch_threads"])
    cache, cache_invalid_reason = checked_calibration_cache(
        config, prior, hashes, root=ROOT, runtime=runtime)
    names = prior["target_names"]
    profiles = prior["profile_names"]
    if (len(names) != 15 or len(profiles["cal"]) != 1000
            or len(profiles["test"]) != 2000 or prior["steps"] != 501):
        raise ValueError("Expected original 15-target, 501-step, 1000/2000 fixed split")
    stats, scaler_meta = read_scaler(paths["external_scaled_h5_path"])
    if scaler_meta["target_names"] != names or len(stats["x"]["min"]) != 16:
        raise ValueError("Frozen scaler feature/target order mismatch")
    cache_path = Path(cache["input_records"]["forecasts"]["path"]) if cache else None
    if cache is not None:
        # Verify all saved calibration members, means, spreads, and coordinates.
        _arrays, reports = load_forecast(
            cache_path, names=names, profile_names=profiles["cal"],
            manifest_identity=cache["original_manifest_identity"], stats=stats)
        if len(reports) != len(profiles["cal"]):
            raise ValueError("Incomplete corrected original calibration")
        del _arrays, reports

    pipeline, bag, _cp, _ds, _uq = production_modules()
    architecture = []
    for i in range(10):
        m = json.loads(paths[f"member_metadata_{i}"].read_text(encoding="utf-8"))
        if m["member_index"] != i:
            raise ValueError(f"Checkpoint member metadata mismatch: {i}")
        architecture.append(m["architecture"])
    if any(a != architecture[0] for a in architecture):
        raise ValueError("Inconsistent frozen ensemble architecture")
    models = bag.load_bagged_lstm_ensemble_checkpoints(
        [paths[f"checkpoint_{i}"] for i in range(10)],
        timesteps=13, num_features=16, num_targets=15,
        device=device, **architecture[0])

    metadata = {
        "completed": False, "feedback": FEEDBACK,
        "no_training": True, "no_simulation": True, "no_scaler_refitting": True,
        "original_calibration_reused": cache is not None,
        "original_calibration_regenerated": cache is None,
        "original_calibration_recompute_reason": cache_invalid_reason,
        "original_calibration_identity": cache["original_manifest_identity"] if cache else None,
        "original_calibration_provenance": cache["input_records"] if cache else None,
        "input_hashes": hashes, "target_names": names,
        "profile_names": {"cal": profiles["cal"], "test": profiles["test"]},
        "runtime": runtime, "output_hashes": {},
        "source_hashes": {
            "scripts/run_mace_stochastic_forecasts.py": fingerprint(Path(__file__))["sha256"],
            "scripts/_mace_operational_forecasts.py": fingerprint(ROOT / "scripts/_mace_operational_forecasts.py")["sha256"],
            "scripts/_mace_scaler_use.py": fingerprint(ROOT / "scripts/_mace_scaler_use.py")["sha256"],
            "src/rabl/machine_learning/lstm_pipeline.py": fingerprint(ROOT / "src/rabl/machine_learning/lstm_pipeline.py")["sha256"],
            "src/rabl/machine_learning/bagging_ensemble.py": fingerprint(ROOT / "src/rabl/machine_learning/bagging_ensemble.py")["sha256"],
        },
        "physical_truth_representation": "float32 inverse transform of frozen scaled Y",
        "time_representation": "indexed 0..500; physical dt=0.4 s",
    }
    out.mkdir(parents=True)
    write_json(out / "forecast_metadata.json", metadata)
    output_paths = {CAL_NAME: out / CAL_NAME, TEST_NAME: out / TEST_NAME}
    cal_context = h5py.File(cache_path, "r") if cache is not None else nullcontext(None)
    with h5py.File(paths["external_scaled_h5_path"], "r") as scaled, cal_context as cal_source:
        def make_file(path):
            f = h5py.File(path, "w")
            f.attrs["target_names"] = np.asarray(names, dtype="S")
            f.attrs["member_count"] = 10
            f.attrs["ensemble_std_ddof"] = 0
            f.attrs["feedback"] = FEEDBACK
            f.attrs["completed"] = False
            return f

        with make_file(output_paths[CAL_NAME]) as dest:
            for n, name in enumerate(profiles["cal"]):
                source = profile_group(scaled, "cal", name)
                truth = np.asarray(source["Y"][()], dtype=np.float32)
                x = np.asarray(source["X"][()], dtype=np.float32)
                if x.shape != (501, 13, 16) or truth.shape != (501, 15):
                    raise ValueError(f"Original calibration profile dimensions changed: {name}")
                if cache is not None:
                    g = cal_source[name]
                    if not np.array_equal(truth, g["truth"][()]):
                        raise ValueError(f"Corrected calibration truth does not match fixed dataset: {name}")
                    members = g["member_predictions_scaled"][()]
                else:
                    members = np.stack([corrected_rollout(m, x, stats) for m in models])
                control = x[:, -1, -1] * np.float32(stats["x"]["span"][-1]) + np.float32(stats["x"]["min"][-1])
                write_profile(dest, name, members, truth, control, stats)
                if (n + 1) % 100 == 0:
                    action = "Reused" if cache is not None else "Recomputed"
                    print(f"{action} corrected calibration: {n + 1}/1000", flush=True)
            dest.attrs["completed"] = True
        with make_file(output_paths[TEST_NAME]) as dest:
            for n, name in enumerate(profiles["test"]):
                source = profile_group(scaled, "test", name)
                x = np.asarray(source["X"][()], dtype=np.float32)
                truth = np.asarray(source["Y"][()], dtype=np.float32)
                if x.shape != (501, 13, 16) or truth.shape != (501, 15):
                    raise ValueError(f"Original test profile dimensions changed: {name}")
                members = np.stack([corrected_rollout(m, x, stats) for m in models])
                control = x[:, -1, -1] * np.float32(stats["x"]["span"][-1]) + np.float32(stats["x"]["min"][-1])
                write_profile(dest, name, members, truth, control, stats)
                if (n + 1) % 100 == 0:
                    print(f"Corrected stochastic test: {n + 1}/2000", flush=True)
            dest.attrs["completed"] = True
    same_hashes(paths, hashes)
    if cache is not None and fingerprint(cache_path)["sha256"] != cache["input_records"]["forecasts"]["sha256"]:
        raise ValueError("Corrected original calibration cache changed during inference")
    for name, path in output_paths.items():
        with h5py.File(path, "r") as f:
            split = "cal" if name == CAL_NAME else "test"
            if (not f.attrs["completed"] or f.attrs["feedback"] != FEEDBACK
                    or set(f.keys()) != set(profiles[split])):
                raise ValueError(f"Incomplete/mismatched output: {name}")
        metadata["output_hashes"][name] = fingerprint(path)
    metadata["completed"] = True
    write_json(out / "forecast_metadata.json", metadata)
    print(f"Corrected stochastic forecasts saved: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
