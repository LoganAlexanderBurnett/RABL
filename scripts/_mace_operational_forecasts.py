"""Frozen-member forecasts with explicit native Y-output to X-state feedback."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from _mace_floor_study import fingerprint, write_json
from _mace_operational_profiles import TIME, control
from _mace_scaler_audit import compare_arrays, profile_group, read_scaler
from _mace_scaler_use import raw_to_scaled, target_to_input
from run_mace_profile_validation import validate_prior_study
from run_mace_scaler_use_audit import production_modules, runtime_settings

FEEDBACK = "native-Y-to-X-affine-float32-v1"


def corrected_rollout(model, x, stats):
    """Own-member recurrence; future measured states are never consumed.

    Deliberately separate from the historical helper audited on this branch.
    Outputs remain Y-scaled; only recurrent input states are converted to X.
    """
    import torch
    x = np.asarray(x, dtype=np.float32)
    n = len(stats["y"]["min"])
    if x.ndim != 3 or x.shape[2] != n+1 or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite (steps,window,states+one control) inputs")
    history = x[0, :, :n].copy()
    predictions = np.empty((len(x), n), dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for t in range(len(x)):
            frame = np.concatenate([history, x[t, :, n:]], axis=1)
            pred = model(torch.from_numpy(frame[None]).to(device)).detach().cpu().numpy()[0]
            if pred.shape != (n,) or not np.all(np.isfinite(pred)):
                raise ValueError("Invalid frozen member prediction")
            predictions[t] = pred
            if t+1 < len(x):
                history = np.vstack([history[1:], target_to_input(pred, stats)])
    return predictions


def ensemble(models, x, stats):
    if len(models) != 10:
        raise ValueError("Requires exactly ten frozen members")
    members = np.stack([corrected_rollout(model, x, stats) for model in models])
    return members, members.mean(axis=0), members.std(axis=0, ddof=0)


def operational_input(angle, seed_x, stats):
    """Original prepared history; shifted controls match dataset builder timing."""
    n = len(stats["y"]["min"])
    window = seed_x.shape[0]
    angle = np.asarray(angle, dtype=np.float64)
    raw = np.zeros((len(angle), window, n+1), dtype=np.float64)
    raw[0, :, :n] = seed_x[:, :n]
    controls = np.concatenate([seed_x[:window-1, n], angle])
    for t in range(len(angle)):
        raw[t, :, n] = controls[t:t+window]
    # Subsequent true-state windows are intentionally unused by recursive inference.
    return raw_to_scaled(raw, stats, "x").astype(np.float32)


def read_simulation(path, *, names, moves):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = ["t", "drumAngleDeg"]+names
    if not rows or any(k not in rows[0] for k in required):
        raise ValueError(f"Missing simulation columns: {path}")
    a = np.asarray([[float(row[k]) for k in required] for row in rows], dtype=np.float64)
    if a.shape != (501, len(required)) or not np.all(np.isfinite(a)):
        raise ValueError(f"Requires 501 finite rows on the 0–200 s grid: {path}")
    time_delta = float(np.max(np.abs(a[:, 0]-TIME)))
    expected = control(moves)[0]
    differences = np.abs(a[:, 1]-expected)
    angle_delta = float(differences.max())
    raw_result, storage = None, None
    match = re.fullmatch(r"results_drum_profile_(\d{5})", Path(path).stem)
    if match:
        raw = Path(path).with_name("run_"+match.group(1)+".mat")
        if raw.is_file():
            # readTrajectory's Python floats and the CSV's float64 parser do not
            # restore precision lost in the native Dymola trajectory matrix.
            data = loadmat(raw, variable_names=["data_2"])
            if "data_2" not in data or data["data_2"].dtype.kind != "f":
                raise ValueError(f"Missing numeric Dymola data_2 matrix: {raw}")
            storage = data["data_2"].dtype.name
            raw_result = {**fingerprint(raw), "trajectory_dtype": storage}
    bounds = np.full(len(TIME), 1e-8, dtype=np.float64)
    policy = "solver_input_float64_bound"
    quantized_reference_delta = None
    observed_vs_quantized_reference = None
    if storage == "float32":
        def half_ulp32(values):
            v = np.asarray(values, np.float32)
            above = np.nextafter(v, np.float32(np.inf)).astype(float)-v.astype(float)
            below = v.astype(float)-np.nextafter(v, np.float32(-np.inf)).astype(float)
            return np.maximum(above, below)/2
        # A resampled signal can have both angle rounding and timestamp rounding
        # errors. Bound these in degrees, rather than declaring the parsed CSV
        # float64 and requiring an impossible 1e-8-degree match. Bracketing sample
        # times are at most one output interval away; maneuver corners are on
        # this same grid and are stored as events by the existing runner/model.
        max_rate = max(abs(m["speed_deg_s"]) for m in moves)
        angle_rounding = float(half_ulp32(np.max(np.abs(expected))))
        time_rounding = half_ulp32(np.minimum(TIME+.4, 200.))
        arithmetic = 16*np.finfo(float).eps*np.maximum(1., np.abs(expected))
        bounds = angle_rounding+max_rate*time_rounding+arithmetic
        policy = "verified_float32_angle_and_time_rounding_plus_linear_resampling"
        reference = np.interp(a[:, 0], TIME.astype(np.float32).astype(float),
                              expected.astype(np.float32).astype(float))
        quantized_reference_delta = float(np.max(np.abs(reference-expected)))
        observed_vs_quantized_reference = float(np.max(np.abs(a[:, 1]-reference)))
    # This is an analytic command-versus-extracted-solver-signal diagnostic.
    # Native saved forecast/scaler array comparisons elsewhere remain exact.
    if time_delta > 1e-8 or np.any(differences > bounds):
        raise ValueError(f"Simulation grid/applied angle disagrees with input: {path}: {time_delta}, {angle_delta}")
    return a[:, 2:], a[:, 1], dict(time_max_difference_seconds=time_delta,
                                  angle_max_difference_degrees=angle_delta,
                                  time_check_bound_seconds=1e-8,
                                  angle_check_policy=policy,
                                  angle_check_bound_min_degrees=float(bounds.min()),
                                  angle_check_bound_max_degrees=float(bounds.max()),
                                  angle_max_fraction_of_pointwise_bound=float(np.max(differences/bounds)),
                                  raw_dymola_result=raw_result,
                                  regular_grid_float32_reference_max_difference_degrees=quantized_reference_delta,
                                  observed_vs_float32_reference_max_difference_degrees=observed_vs_quantized_reference)


def same_hashes(paths, hashes):
    for key, path in paths.items():
        if fingerprint(path)["sha256"] != hashes[key]["sha256"]:
            raise ValueError(f"Frozen scientific input changed: {key}")


def simulation_hashes(out, profiles, metadata):
    for p in profiles:
        path = out/"simulation"/p["split"]/(p["id"]+".csv")
        if fingerprint(path)["sha256"] != metadata["result_hashes"][p["id"]]["sha256"]:
            raise ValueError(f"Simulation CSV changed: {p['id']}")
        raw = metadata.get("result_validation", {}).get(p["id"], {}).get("raw_dymola_result")
        if raw is not None and fingerprint(Path(raw["path"]))["sha256"] != raw["sha256"]:
            raise ValueError(f"Native Dymola MAT changed: {p['id']}")


def frozen_inputs(config, root):
    study = Path(config["floor_study_dir"])
    prior = validate_prior_study(study)
    overrides = config.get("input_overrides", {})
    keys = ("external_scaled_h5_path", "training_scaled_h5_path", "raw_h5_path", "reference_scaler_json_path")
    if set(overrides)-set(keys)-{"ensemble_experiment_dir"}:
        raise ValueError("Unknown input relocation")
    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else root/p
    paths = {k: resolve(overrides.get(k, prior["config"][k])) for k in keys}
    exp = resolve(overrides.get("ensemble_experiment_dir", prior["config"]["ensemble_experiment_dir"]))
    for i in range(10):
        paths[f"checkpoint_{i}"] = exp/f"ensemble/model_{i}/model.pt"
        paths[f"member_metadata_{i}"] = exp/f"ensemble/model_{i}/member_metadata.json"
    hashes = {key: fingerprint(path) for key, path in paths.items()}
    for key in (*keys, *(f"checkpoint_{i}" for i in range(10))):
        if hashes[key]["sha256"] != prior["input_hashes"][key]["sha256"]:
            raise ValueError(f"Scientific input differs from completed floor study: {key}")
    paths["prior_metadata"] = study/"study_metadata.json"
    paths["prior_scaler_verification"] = study/"scaler_verification.json"
    hashes.update({k: fingerprint(paths[k]) for k in ("prior_metadata", "prior_scaler_verification")})
    return prior, paths, hashes


def forecast_stage(config, manifest, *, root):
    out = Path(config["output_dir"])
    folder = out/"forecasts"
    if folder.exists():
        raise FileExistsError("Forecast stage already started; retain outputs and choose a new output_dir")
    simmeta = json.loads((out/"simulation_metadata.json").read_text())
    if not simmeta.get("completed") or simmeta["manifest_identity"] != manifest["identity"]:
        raise ValueError("Requires completed simulations for this manifest")
    print("Hashing original datasets and ten checkpoints", flush=True)
    prior, paths, hashes = frozen_inputs(config, root)
    if prior["steps"] != 501 or len(prior["profile_names"]["cal"]) != 1000:
        raise ValueError("Requires the original 1,000 calibration profiles, with 501 time steps")
    names = prior["target_names"]
    if {p["id"] for p in manifest["profiles"]} & {name for ids in prior["profile_names"].values() for name in ids}:
        raise ValueError("Operational IDs overlap original fixed profile IDs")
    native, scaling_metadata = read_scaler(paths["external_scaled_h5_path"])
    if scaling_metadata["target_names"] != names or len(native["x"]["min"]) != len(names)+1:
        raise ValueError("Scaler feature order mismatch")
    simulation_hashes(out, manifest["profiles"], simmeta)
    pipeline, bag, _cp, _ds, _uq = production_modules()
    device, runtime = runtime_settings(config["device"], config["torch_threads"])
    cache = None
    if config.get('reuse_original_calibration_dir'):
        from _mace_original_calibration_cache import validate_cache
        cache = validate_cache(config, prior, hashes, root=root, runtime=runtime)
    architecture = []
    for i in range(10):
        m = json.loads(paths[f"member_metadata_{i}"].read_text())
        if m["member_index"] != i:
            raise ValueError("Member identity mismatch")
        architecture.append(m["architecture"])
    if any(a != architecture[0] for a in architecture):
        raise ValueError("Member architecture mismatch")
    with h5py.File(paths["raw_h5_path"], "r") as raw:
        seed_x = profile_group(raw, "cal", prior["profile_names"]["cal"][0])["X"][0]
    if (seed_x.ndim != 2 or seed_x.shape[1] != len(names)+1
        or not np.array_equal(seed_x[:, :-1], np.repeat(seed_x[:1, :-1], len(seed_x), axis=0))
        or not np.array_equal(seed_x[:, -1], np.full(len(seed_x), 45.))):
        raise ValueError("Original initial history is not the expected prepared 45-degree steady state")
    models = bag.load_bagged_lstm_ensemble_checkpoints([paths[f"checkpoint_{i}"] for i in range(10)],
                timesteps=len(seed_x), num_features=len(names)+1, num_targets=len(names),
                device=device, **architecture[0])
    folder.mkdir()
    metadata = dict(completed=False, feedback=FEEDBACK, runtime=runtime, input_hashes=hashes,
                    target_names=names, manifest_identity=manifest["identity"], member_count=10,
                    original_calibration_profile_names=prior["profile_names"]["cal"],
                    original_calibration="Same original profiles, corrected own-member rollout, newly recalculated quantiles",
                    no_training=True, no_scaler_refitting=True, historical_forecasts_replaced=False,
                    feedback_y_to_x_factor=native["y"]["span"]/native["x"]["span"][:len(names)],
                    feedback_y_to_x_offset=(native["y"]["min"]-native["x"]["min"][:len(names)])/native["x"]["span"][:len(names)],
                    initial_history_source=prior["profile_names"]["cal"][0], initial_history_physical=seed_x)
    metadata['original_calibration_manifest_identity'] = cache['original_manifest_identity'] if cache else manifest['identity']
    write_json(folder/"forecast_metadata.json", metadata)
    checks, results = [], {}
    def save(f, name, x, truth_physical, truth_scaled, angle):
        members, mean, spread = ensemble(models, x, native)
        g = f.create_group(name)
        for key, value in dict(member_predictions_scaled=members, mean=mean, spread=spread,
                               truth=truth_scaled, truth_physical=truth_physical,
                               control_physical=angle, time_seconds=TIME).items():
            g.create_dataset(key, data=value, compression="gzip")
        return g
    def open_forecast(path):
        f = h5py.File(path, "w")
        f.attrs.update(completed=False, feedback=FEEDBACK, member_count=10, ensemble_std_ddof=0,
                       manifest_identity=manifest["identity"], target_names=np.asarray(names, dtype="S"))
        return f
    if cache:
        from _mace_original_calibration_cache import copy_cache
        _arrays, reports = load_forecast(Path(cache['input_records']['forecasts']['path']), names=names,
                                        profile_names=prior['profile_names']['cal'],
                                        manifest_identity=cache['original_manifest_identity'], stats=native)
        del _arrays
        checks.extend(dict(profile=r['profile'], **{k:v for k,v in r.items() if k != 'profile'}) for r in reports)
        metadata['original_calibration_reuse'] = copy_cache(cache, folder/'original_cal.h5')
        write_json(folder/'forecast_metadata.json', metadata)
        print('Reused byte-identical corrected original calibration: 1000 profiles; zero original inference runs', flush=True)
    else:
        with h5py.File(paths["external_scaled_h5_path"], "r") as scaled, h5py.File(paths["raw_h5_path"], "r") as raw:
            with open_forecast(folder/"original_cal.h5") as f:
                for i, name in enumerate(prior["profile_names"]["cal"]):
                    r, s = profile_group(raw, "cal", name), profile_group(scaled, "cal", name)
                    x, truth = s["X"][()], s["Y"][()]
                    physical_y = r["Y"][()]
                    for field, axis, values in (("X", "x", x), ("Y", "y", truth)):
                        report = compare_arrays(values, raw_to_scaled(r[field][()], native, axis))
                        checks.append(dict(profile=name, field=field, **report))
                        if not report["match"]:
                            raise ValueError(f"Original raw/scaled arrays disagree: {name}/{field}")
                    if x.shape != (501, len(seed_x), len(names)+1) or truth.shape != (501, len(names)):
                        raise ValueError(f"Original profile shape mismatch: {name}")
                    for field, value in (("X", x), ("Y", truth)):
                        checks.append(dict(profile=name, field=field+"_float32_boundary", **compare_arrays(value.astype(np.float32), value)))
                    save(f, name, x.astype(np.float32), physical_y, truth.astype(np.float32), r["X"][()][:, -1, -1])
                    if i % 20 == 0:
                        print(f"Corrected original-calibration forecasts: {i+1}/1000", flush=True)
                f.attrs["completed"] = True
    for split in ("cal", "test"):
        chosen = [p for p in manifest["profiles"] if p["split"] == split]
        with open_forecast(folder/f"operational_{split}.h5") as f:
            for i, p in enumerate(chosen):
                path = out/"simulation"/split/(p["id"]+".csv")
                y, angle, report = read_simulation(path, names=names, moves=p["moves"])
                checks.append(dict(profile=p["id"], simulation_input_checks=report))
                x = operational_input(angle, seed_x, native)
                truth = raw_to_scaled(y, native, "y").astype(np.float32)
                g = save(f, p["id"], x, y, truth, angle)
                g.attrs.update(family=p["family"], split=split)
                if i % 20 == 0:
                    print(f"Operational {split} forecasts: {i+1}/{len(chosen)}", flush=True)
            f.attrs["completed"] = True
    same_hashes(paths, hashes)
    simulation_hashes(out, manifest["profiles"], simmeta)
    for filename in ("original_cal.h5", "operational_cal.h5", "operational_test.h5"):
        results[filename] = fingerprint(folder/filename)
    write_json(folder/"precision_and_input_checks.json", checks)
    metadata.update(completed=True, inputs_unchanged=True, output_hashes=results)
    write_json(folder/"forecast_metadata.json", metadata)
    return metadata


def load_forecast(path, *, names, profile_names, manifest_identity, stats):
    arrays = {key: [] for key in ("truth", "truth_physical", "mean", "spread")}
    checks = []
    with h5py.File(path, "r") as f:
        if (not f.attrs.get("completed") or f.attrs.get("feedback") != FEEDBACK
            or f.attrs.get("manifest_identity") != manifest_identity or int(f.attrs.get("member_count", 0)) != 10
            or int(f.attrs.get("ensemble_std_ddof", -1)) != 0
            or list(f.attrs["target_names"].astype(str)) != names or set(f.keys()) != set(profile_names)):
            raise ValueError(f"Forecast identity/feedback provenance mismatch: {path}")
        for name in profile_names:
            g = f[name]
            members = g["member_predictions_scaled"][()]
            for field, reference in (("mean", members.mean(axis=0)), ("spread", members.std(axis=0, ddof=0)),
                                     ("truth", raw_to_scaled(g["truth_physical"][()], stats, "y")),
                                     ("time_seconds", TIME)):
                report = compare_arrays(g[field][()], reference)
                checks.append(dict(profile=name, field=field, **report))
                if not report["match"]:
                    raise ValueError(f"Stored forecast array mismatch: {name}/{field}")
            if members.shape != (10, 501, len(names)) or members.dtype != np.float32 or not np.all(np.isfinite(members)):
                raise ValueError("Invalid stored member predictions")
            for key in arrays:
                value = g[key][()]
                if value.shape != (501, len(names)) or not np.all(np.isfinite(value)):
                    raise ValueError(f"Invalid forecast field: {name}/{key}")
                arrays[key].append(value)
    return {k: np.stack(v).astype(np.float64 if k == "truth_physical" else np.float32) for k, v in arrays.items()}, checks
