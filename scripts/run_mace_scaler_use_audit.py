"""Read-only scaler-use audit with frozen-checkpoint, whole-profile replays."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from math import ceil
from pathlib import Path

import h5py
import numpy as np

from _mace_floor_study import fingerprint, write_csv, write_json
from _mace_scaler_audit import decode_names, profile_group, read_scaler, verify_scaler
from _mace_scaler_use import (Checks, audit_status, coverage_effect, independent_rollout,
                             observed_ensemble, raw_to_scaled, read_alias,
                             selected_profiles, source_trace)
from run_mace_profile_validation import validate_prior_study

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 1e-3
METHODS = {"absolute_conformal_target_horizon": ("absolute", "horizon"),
           "absolute_conformal_target_trajectory": ("absolute", "trajectory"),
           "ensemble_conformal_target_horizon": ("mace", "horizon"),
           "ensemble_conformal_target_trajectory": ("mace", "trajectory")}


def production_modules():
    # Audit the current checkout, not a stale installed copy of RABL.
    sys.path.insert(0, str(ROOT/"src"))
    from rabl.machine_learning import bagging_ensemble as bag, conformal_prediction as cp
    from rabl.machine_learning import dataset_scaling as ds, lstm_pipeline as pipeline
    import run_lstm_conformal_prediction as uq
    for module in (bag, cp, ds, pipeline):
        if ROOT/"src" not in Path(module.__file__).resolve().parents:
            raise ValueError(f"Loaded a RABL module outside this checkout: {module.__file__}")
    return pipeline, bag, cp, ds, uq


def runtime_settings(device, threads):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Audit device must be cpu, cuda, or auto")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    torch.set_num_threads(int(threads))
    # Preserve existing backend precision choices; record them rather than
    # silently switching kernels/TF32 and calling resulting drift a scaler fix.
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "h5py": h5py.__version__, "torch": torch.__version__,
              "device": str(resolved), "torch_threads": torch.get_num_threads(),
              "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
              "cudnn_enabled": torch.backends.cudnn.enabled,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "mkldnn_enabled": torch.backends.mkldnn.enabled,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if resolved.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(resolved)
        result["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        result["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    return resolved, result


def run(config, *, config_path=None):
    required = {"floor_study_dir", "output_dir"}
    allowed = required | {"profiles_per_split", "device", "torch_threads", "input_overrides",
                          "recompute_train_extrema"}
    if required-config.keys() or config.keys()-allowed:
        raise ValueError("Missing required or unknown configuration keys")
    if not isinstance(config.get("recompute_train_extrema", True), bool):
        raise ValueError("recompute_train_extrema must be boolean")
    study, out = Path(config["floor_study_dir"]), Path(config["output_dir"])
    if out.exists():
        raise FileExistsError(f"Choose a new output_dir; already exists: {out}")
    if study.resolve() == out.resolve() or study.resolve() in out.resolve().parents:
        raise ValueError("Audit outputs must be outside the prior study")
    prior = validate_prior_study(study)
    names, profiles = prior["target_names"], prior["profile_names"]
    chosen = {s: selected_profiles(profiles[s], config.get("profiles_per_split", 3)) for s in ("cal", "test")}
    keys = ("external_scaled_h5_path", "training_scaled_h5_path", "raw_h5_path",
            "reference_scaler_json_path", "calibration_forecasts_h5_path", "test_forecasts_h5_path")
    overrides = config.get("input_overrides", {})
    if overrides.keys()-set(keys)-{"ensemble_experiment_dir", "saved_uq_comparison_dir"}:
        raise ValueError("Unknown input override; relocations must preserve original file bytes")
    paths = {k: Path(overrides.get(k, prior["config"][k])) for k in keys}
    exp = Path(overrides.get("ensemble_experiment_dir", prior["config"]["ensemble_experiment_dir"]))
    saved_uq = Path(overrides.get("saved_uq_comparison_dir", prior["config"]["saved_uq_comparison_dir"]))
    for member in range(10):
        paths[f"checkpoint_{member}"] = exp/f"ensemble/model_{member}/model.pt"
        paths[f"member_metadata_{member}"] = exp/f"ensemble/model_{member}/member_metadata.json"
    paths["prior_metadata"] = study/"study_metadata.json"
    paths["prior_scaler_verification"] = study/"scaler_verification.json"
    paths["prior_quantiles"] = study/"recalibration.h5"
    for method in METHODS:
        paths[method+"/calibration"] = saved_uq/method/"calibration_metadata.json"
        for split, filename in (("cal", "calibration_forecasts.h5"), ("test", "test_forecasts.h5")):
            paths[method+"/"+split] = saved_uq/method/filename
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(out.resolve() == p.resolve() or out.resolve() in p.resolve().parents for p in paths.values()):
        raise ValueError("Output directory contains an existing scientific input")
    out.mkdir(parents=True)
    metadata = {"completed": False, "status": "running", "config": config, "config_path": config_path,
                "selected_profiles": chosen, "profile_selection": "Evenly spaced fixed manifest positions; no performance selection",
                "alpha": .05, "floor": FLOOR, "member_count": 10, "target_names": names,
                "no_training": True, "historical_forecasts_replaced": False,
                "replay_scope": "Historical direct-feedback recurrence; coordinate mapping checked separately; no corrected-method study",
                "coverage_convention": prior["coverage_convention"],
                "replay_coverage_scope": "Numerical effects on selected profiles at frozen full-calibration quantiles, not a new calibrated evaluation"}
    write_json(out/"audit_metadata.json", metadata)
    try:
        print("Hashing frozen datasets, forecasts, checkpoints and prior study", flush=True)
        hashes = {key: fingerprint(path) for key, path in paths.items()}
        metadata["input_hashes"] = hashes
        for key in (*keys, *(f"checkpoint_{i}" for i in range(10))):
            if hashes[key]["sha256"] != prior["input_hashes"][key]["sha256"]:
                raise ValueError(f"Scientific input differs from completed floor study: {key}")
        metadata["runtime"] = None
        write_json(out/"audit_metadata.json", metadata)
        if config.get("recompute_train_extrema", True):
            report, native = verify_scaler(external_path=paths[keys[0]], training_path=paths[keys[1]],
                raw_path=paths[keys[2]], reference_json_path=paths[keys[3]], manifests=profiles, target_names=names)
            write_json(out/"scaler_verification.json", report)
            if not report["match"]:
                raise ValueError("Independent train-only extrema/raw-to-scaled checks failed")
        else:
            native, _ = read_scaler(paths[keys[0]])
            metadata["train_extrema_scope"] = "Reused completed independent audit; original raw/scaler/dataset hashes match"
        pipeline, bag, cp, ds, uq = production_modules()
        device, metadata["runtime"] = runtime_settings(config.get("device", "auto"), config.get("torch_threads", 1))
        trace = source_trace({"input_scaling": ds._apply_scaling, "profile_loading": pipeline._profile_generator,
            "scaler_loading": pipeline._load_scaling_stats, "checkpoint_loading": bag.load_bagged_lstm_ensemble_checkpoints,
            "recursive_rollout": pipeline.rolling_forecast, "ensemble_aggregation": bag.ensemble_member_predictions_scaled,
            "saved_forecast_loading": uq._load_audit_ensemble_forecasts, "interval_application": cp.apply_uq_method,
            "target_descaling": pipeline._descale_targets_from_stats, "control_descaling": pipeline._descale_feature_from_stats})
        trace.extend({"role": "audit_source", **fingerprint(Path(__file__).with_name(p))} for p in
                     ("run_mace_scaler_use_audit.py", "_mace_scaler_use.py", "_mace_scaler_audit.py", "_mace_floor_study.py"))
        metadata["source_trace"] = trace
        write_json(out/"source_trace.json", trace)
        try:
            metadata["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            metadata["code_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError):
            metadata["code_commit"] = None
        checks = Checks()
        run_scaler = pipeline._load_scaling_stats(paths[keys[0]])
        training_scaler = pipeline._load_scaling_stats(paths[keys[1]])
        for axis in ("x", "y"):
            for key in ("min", "span"):
                checks.add(f"loader_native_precision/{axis}_{key}", run_scaler[axis][key], native[axis][key])
                checks.add(f"loader_external_vs_training/{axis}_{key}", run_scaler[axis][key], training_scaler[axis][key])
        n = len(names)
        if len(native["x"]["min"]) != n+1:
            raise ValueError("Expected one appended control input")
        for key in ("min", "span"):
            checks.add(f"output_target_vs_input_state_scaler/{key}", native["y"][key], native["x"][key][:n],
                       category="coordinate", labels=names)
        write_json(out/"feedback_coordinates.json", {
            "target_names": names, "x_state_min": native["x"]["min"][:n], "y_min": native["y"]["min"],
            "x_state_span": native["x"]["span"][:n], "y_span": native["y"]["span"],
            "Y_to_X_factor": native["y"]["span"]/native["x"]["span"][:n],
            "Y_to_X_offset": (native["y"]["min"]-native["x"]["min"][:n])/native["x"]["span"][:n],
            "current_rollout": "Appends Y-scaled member output directly to X-scaled state window",
            "required_mapping": "X_scaled = Y_scaled * (Y_span / X_span) + (Y_min - X_min) / X_span; native scaler arithmetic then float32 input"})
        architectures = []
        for i in range(10):
            member = json.loads(paths[f"member_metadata_{i}"].read_text())
            if member["member_index"] != i:
                raise ValueError("Checkpoint member order differs from metadata")
            architectures.append(member["architecture"])
        if any(a != architectures[0] for a in architectures):
            raise ValueError("Members have different architectures")
        metadata["member_metadata"] = [json.loads(paths[f"member_metadata_{i}"].read_text()) for i in range(10)]
        with h5py.File(paths[keys[0]]) as f:
            shape = f[f"cal/files/{chosen['cal'][0]}/X"].shape
        if len(shape) != 3 or shape[0] != prior["steps"] or shape[-1] != n+1:
            raise ValueError("Unexpected model input dimensions")
        metadata["actual_input_window_length"] = shape[1]
        models = bag.load_bagged_lstm_ensemble_checkpoints([paths[f"checkpoint_{i}"] for i in range(10)],
            timesteps=shape[1], num_features=shape[2], num_targets=n, device=device, **architectures[0])
        quantiles = {}
        with h5py.File(paths["prior_quantiles"]) as f:
            if decode_names(f.attrs["target_names"]) != names or decode_names(f.attrs["calibration_profile_names"]) != profiles["cal"]:
                raise ValueError("Prior quantile identities differ")
            rank = ceil((len(profiles["cal"])+1)*.95)
            if int(f.attrs["quantile_rank"]) != rank or rank > len(profiles["cal"]):
                raise ValueError("Prior full-calibration quantile rank differs")
            metadata["quantile_rank"] = rank
            for method, (kind, mode) in METHODS.items():
                q = f[f"{'absolute' if kind == 'absolute' else 'floor_0.001'}/{mode}"][()]
                saved = json.loads(paths[method+"/calibration"].read_text())
                qkey = "q_by_target" if mode == "trajectory" else "q_by_horizon_target"
                if saved["alpha"] != .05 or saved["calibration_profile_names"] != profiles["cal"]:
                    raise ValueError("Saved UQ calibration identity/objective differs")
                checks.add(method+"/quantile_json_precision", np.asarray(saved[qkey], dtype=np.float64),
                           np.asarray(saved[qkey], dtype=np.float32))
                checks.add(method+"/full_calibration_quantile", q, np.asarray(saved[qkey], dtype=np.float32))
                if kind == "mace":
                    floor = np.broadcast_to(np.asarray(saved["sigma_floor"], dtype=np.float64), (n,))
                    checks.add(method+"/common_floor", floor, np.full(n, np.float32(FLOOR)))
                quantiles[method] = {qkey: q, "sigma_floor": FLOOR, "alpha": .05}
        effects = []
        with h5py.File(out/"replays.h5", "w") as dest, h5py.File(paths[keys[2]]) as raw, h5py.File(paths[keys[0]]) as external, h5py.File(paths[keys[1]]) as training:
            dest.attrs["target_names"] = np.asarray(names, dtype="S")
            dest.attrs["replay_method"] = "Historical direct-feedback recurrence; audit only"
            for split, forecast_key in (("cal", keys[4]), ("test", keys[5])):
                stored = uq._load_audit_ensemble_forecasts(paths[forecast_key], scaling_stats=run_scaler,
                    expected_profiles=chosen[split], target_names=names, expected_ddof=0)
                ext_ds = pipeline.ProfileDataset(paths[keys[0]], chosen[split], split)
                train_ds = pipeline.ProfileDataset(paths[keys[1]], chosen[split], split)
                with h5py.File(paths[forecast_key]) as original:
                    if set(original.keys()) != set(profiles[split]) or int(original.attrs.get("member_count", -1)) != 10:
                        raise ValueError("Frozen forecast profile set/member count differs")
                    for entry, ext_item, train_item in zip(stored, ext_ds, train_ds):
                        profile, x_tensor, y_tensor = ext_item
                        if profile != entry["profile"] or train_item[0] != profile:
                            raise ValueError("Profile loader order differs")
                        context = {"split": split, "profile": profile}
                        checks.add("saved_mean_vs_member_aggregation", entry["mean_scaled"],
                                   entry["member_predictions_scaled"].mean(axis=0), labels=names, **context)
                        checks.add("saved_std_vs_member_aggregation", entry["spread_scaled"],
                                   entry["member_predictions_scaled"].std(axis=0, ddof=0), labels=names, **context)
                        rg = profile_group(raw, split, profile)
                        reconstructed = {}
                        for axis, field in (("x", "X"), ("y", "Y")):
                            physical = rg[field][()]
                            expected = raw_to_scaled(physical, native, axis)
                            checks.add("production_scaling_vs_independent_formula/"+field,
                                       ds._apply_scaling("minmax", physical, native[axis]), expected, **context)
                            for label, f in (("external", external), ("training", training)):
                                checks.add(label+"_stored_vs_raw_formula/"+field,
                                           f[f"{split}/files/{profile}/{field}"][()], expected, **context)
                            reconstructed[axis] = expected.astype(np.float32)
                        checks.add("external_runtime_X_vs_raw_formula_float32", x_tensor.numpy(), reconstructed["x"], **context)
                        checks.add("training_runtime_X_vs_external", train_item[1].numpy(), x_tensor.numpy(), **context)
                        checks.add("external_runtime_Y_vs_raw_formula_float32", y_tensor.numpy(), reconstructed["y"], **context)
                        checks.add("training_runtime_Y_vs_external", train_item[2].numpy(), y_tensor.numpy(), **context)
                        checks.add("saved_truth_vs_raw_formula_float32", entry["y_true_scaled"], reconstructed["y"], **context)
                        x = reconstructed["x"]
                        if x.shape != shape or reconstructed["y"].shape != (prior["steps"], n):
                            raise ValueError("Selected profile shape differs from fixed run")
                        members, mean, spread, snapshots = observed_ensemble(models, x, native, checks,
                            split=split, profile=profile, production=bag.ensemble_member_predictions_scaled, target_names=names)
                        independent = []
                        for i, model in enumerate(models):
                            print(f"Independent replay {split}/{profile}, member {i+1}/10", flush=True)
                            independent.append(independent_rollout(model, x))
                        independent = np.stack(independent)
                        checks.add("production_vs_independent_member_replay", members, independent, labels=names, **context)
                        checks.add("production_ensemble_mean", mean, independent.mean(axis=0), labels=names, **context)
                        checks.add("production_ensemble_population_std", spread, independent.std(axis=0, ddof=0), labels=names, **context)
                        checks.add("replayed_members_vs_saved", members, original[profile]["member_predictions_scaled"][()], category="replay", labels=names, **context)
                        for field, value, aliases in (("mean", mean, ("ensemble_mean_scaled", "mean_scaled")),
                                                       ("spread", spread, ("ensemble_std_scaled", "spread_scaled"))):
                            checks.add("replayed_"+field+"_vs_saved", value, read_alias(original[profile], *aliases), category="replay", labels=names, **context)
                        for field, value in (("y_true", entry["y_true_scaled"]), ("y_pred", entry["mean_scaled"])):
                            expected = value*run_scaler["y"]["span"]+run_scaler["y"]["min"]
                            checks.add("saved_physical_"+field+"_vs_external_scaler", entry[field], expected, labels=names, **context)
                            checks.add("production_descaling/"+field, pipeline._descale_targets_from_stats(run_scaler, value), expected, labels=names, **context)
                        control = x[:, -1, n]
                        expected_control = control*run_scaler["x"]["span"][n]+run_scaler["x"]["min"][n]
                        checks.add("saved_control_vs_external_scaler", entry["u"], expected_control, **context)
                        checks.add("production_control_descaling", pipeline._descale_feature_from_stats(run_scaler, control, n), expected_control, **context)
                        original_members = entry["member_predictions_scaled"]
                        physical_members = original_members*run_scaler["y"]["span"]+run_scaler["y"]["min"]
                        physical_spread = physical_members.std(axis=0, ddof=0)
                        checks.add("physical_member_std_vs_scaled_std_times_span", physical_spread,
                                   entry["spread_scaled"]*run_scaler["y"]["span"], category="diagnostic", labels=names, **context)
                        for key, sign in (("ensemble_lower_2sigma_physical", -1), ("ensemble_upper_2sigma_physical", 1)):
                            if key in original[profile]:
                                checks.add("saved_"+key, original[profile][key][()],
                                           entry["y_pred"]+sign*2.*physical_spread, labels=names, **context)
                        checks.add("raw_control_vs_float32_roundtrip", rg["X"][:, -1, n], expected_control,
                                   category="diagnostic", **context)
                        g = dest.create_group(split+"/"+profile)
                        for field, value in (("member_predictions_scaled", members), ("mean_scaled", mean), ("spread_scaled", spread)):
                            g.create_dataset(field, data=value, compression="gzip")
                        for i, snap in enumerate(snapshots):
                            for field, value in snap.items():
                                g.create_dataset(f"model_{i}/{field}", data=value, compression="gzip")
                        for method, (kind, mode) in METHODS.items():
                            actual = cp.apply_uq_method(entry, method_id=method, calibration_result=quantiles[method],
                                scaling_stats=run_scaler, alpha=.05, sigma_floor=FLOOR, include_member_predictions=False)
                            scale = entry["spread_scaled"]+np.float32(FLOOR) if kind == "mace" else np.ones_like(entry["mean_scaled"])
                            q = next(value for key, value in quantiles[method].items() if key.startswith("q_by_"))
                            lower = entry["mean_scaled"]-q*scale
                            upper = entry["mean_scaled"]+q*scale
                            expected_arrays = {"lower_scaled": lower, "upper_scaled": upper,
                                "lower": lower*run_scaler["y"]["span"]+run_scaler["y"]["min"],
                                "upper": upper*run_scaler["y"]["span"]+run_scaler["y"]["min"]}
                            expected_arrays["interval_width"] = expected_arrays["upper"]-expected_arrays["lower"]
                            checks.add(method+"/physical_endpoint_width_vs_canonical_scaled_width",
                                       expected_arrays["upper"].astype(float)-expected_arrays["lower"].astype(float),
                                       (upper.astype(float)-lower.astype(float))*native["y"]["span"],
                                       category="diagnostic", labels=names, **context)
                            if kind == "mace":
                                expected_arrays["effective_scale_scaled"] = scale
                                expected_arrays["normalized_abs_error"] = np.abs(entry["y_true_scaled"]-entry["mean_scaled"])/scale
                            with h5py.File(paths[method+"/"+split]) as saved:
                                if decode_names(saved.attrs["target_names"]) != names:
                                    raise ValueError("Saved interval target order differs")
                                for field, expected in expected_arrays.items():
                                    checks.add(method+"/production/"+field, actual[field], expected, labels=names, **context)
                                    checks.add(method+"/saved/"+field, saved[profile][field][()], expected, labels=names, **context)
                            changed = coverage_effect((entry["mean_scaled"], entry["spread_scaled"]), (mean, spread), entry["y_true_scaled"],
                                q=q, floor=FLOOR if kind == "mace" else None, span=native["y"]["span"], names=names)
                            effects.extend({**context, "method": method, **r} for r in changed)
                        write_json(out/"comparisons.json", checks.rows)
        write_csv(out/"replay_coverage_effects.csv", effects)
        metadata["check_summary"] = checks.summary()
        metadata["status"] = audit_status(metadata["check_summary"])
        metadata["passed"] = metadata["status"] == "passed"
        metadata["input_sizes_and_mtimes_unchanged"] = all(p.stat().st_size == hashes[k]["bytes"] and p.stat().st_mtime_ns == hashes[k]["mtime_ns"] for k, p in paths.items())
        metadata["sources_unchanged"] = all(fingerprint(Path(s["path"]))["sha256"] == s["sha256"] for s in trace)
        if not metadata["input_sizes_and_mtimes_unchanged"] or not metadata["sources_unchanged"]:
            raise ValueError("Frozen inputs or executing source changed during audit")
        metadata["completed"] = True
        write_json(out/"audit_metadata.json", metadata)
        summary = metadata["check_summary"]
        (out/"audit_summary.md").write_text(
            "# Frozen-model scaler-use audit\n\n"
            f"Status: **{metadata['status']}**. Completed: true.\n\n"
            + "\n".join(f"- {k}: {v['count']} checks, {v['mismatches']} mismatches." for k, v in summary.items())
            + "\n\nCoordinate mismatches concern the X-input versus Y-output frames. Historical replay can be exact while this check fails. "
            "The audit does not repair feedback, retrain models, replace forecasts, or evaluate a corrected method. "
            "Replay drift is reported without tolerances and requires review of the saved execution environment. "
            "Diagnostic raw-control roundtrip differences are expected to expose float32 quantization. "
            "Inspect comparisons.json, feedback_coordinates.json, replays.h5 and replay_coverage_effects.csv.\n", encoding="utf-8")
        print(f"Audit completed: {metadata['status']}; outputs: {out}", flush=True)
        return out
    except Exception as exc:
        metadata.update(status="execution_failed", error=f"{type(exc).__name__}: {exc}", completed=False, passed=False)
        write_json(out/"audit_metadata.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    out = run(json.loads(args.config.read_text()), config_path=args.config)
    if not json.loads((out/"audit_metadata.json").read_text())["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
