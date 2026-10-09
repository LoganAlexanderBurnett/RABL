"""Run the seven-floor MACE study on fixed saved forecasts and verify scaling.

Run from the repository root: python scripts/run_lstm_floor_sensitivity.py --config ...
Only NumPy and h5py are required; optional plots use matplotlib. No checkpoint
is deserialized and no model is trained or used for inference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from _mace_floor_study import (
    FLOORS, METHODS, calibrate, distribution, evaluate, fingerprint,
    load_forecasts, load_manifests, reconcile_saved, wilson, write_csv, write_json,
)
from _mace_scaler_audit import decode_names, verify_scaler


def check_dataset_splits(path: Path, manifests: dict[str, list[str]]) -> None:
    with h5py.File(path, "r") as f:
        if str(f.attrs.get("split_mode", "")) != "profile":
            raise ValueError(f"Dataset must have profile-based splits: {path}")
        for split, names in manifests.items():
            if set(f[f"{split}/files"].keys()) != set(names):
                raise ValueError(f"Dataset split differs from fixed manifest: {path}/{split}")
            if "split_definition" in f:
                if set(decode_names(f[f"split_definition/{split}_profiles"][()])) != set(names):
                    raise ValueError(f"Stored split definition mismatch: {path}/{split}")


def validate_saved_calibrations(path: Path, profiles: list[str], alpha: float, floor: float) -> None:
    for mode in METHODS:
        for prefix in ("ensemble", "absolute"):
            method = f"{prefix}_conformal_target_{mode}"
            data = json.loads((path / method / "calibration_metadata.json").read_text())
            if data["calibration_profile_names"] != profiles or float(data["alpha"]) != alpha:
                raise ValueError(f"Saved calibration identity/alpha mismatch: {method}")
            if prefix == "ensemble" and not np.all(np.asarray(data["sigma_floor"], dtype=np.float32) == np.float32(floor)):
                raise ValueError(f"Saved method is not the current common floor: {method}")


def plot_results(out: Path, aggregate: list[dict[str, Any]], targets: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for mode, color in (("horizon", "tab:blue"), ("trajectory", "tab:orange")):
        rows = [r for r in aggregate if r["method"] == "mace" and r["mode"] == mode]
        axes[0,0].semilogx([r["floor"] for r in rows], [r["width_ratio_vs_absolute"] for r in rows], "o-", color=color, label=mode)
        steam = [r for r in targets if r["method"] == "mace" and r["mode"] == mode and r["target"] == "x_steam_out"]
        if steam:
            axes[0,1].semilogx([r["floor"] for r in steam], [r["objective_coverage"] for r in steam], "o-", color=color, label=mode)
            axes[1,0].semilogx([r["floor"] for r in steam], [r["width_physical_p50"] for r in steam], "o-", color=color, label=mode)
            axes[1,1].semilogx([r["floor"] for r in steam], [r["width_ratio_vs_absolute"] for r in steam], "o-", color=color, label=mode)
    for ax in axes.flat:
        ax.axvline(1e-3, color="gray", linestyle=":")
        ax.set_xlabel("Common additive floor (scaled)")
        ax.grid(alpha=.25)
        ax.legend()
    axes[0,0].axhline(1, color="black", linestyle="--")
    axes[0,0].set_ylabel("Aggregate width / absolute CP")
    axes[0,1].axhline(.95, color="black", linestyle="--")
    axes[0,1].set_ylabel("Steam-quality coverage")
    axes[1,0].set_ylabel("Median full width (quality fraction)")
    axes[1,1].axhline(1, color="black", linestyle="--")
    axes[1,1].set_ylabel("Steam-quality width / absolute CP")
    fig.savefig(out / "floor_sensitivity.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "floor_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def run(config: dict[str, Any], *, config_path: Path | None = None) -> Path:
    required = {"output_dir", "ensemble_experiment_dir", "external_scaled_h5_path",
                "training_scaled_h5_path", "raw_h5_path", "reference_scaler_json_path",
                "calibration_forecasts_h5_path", "test_forecasts_h5_path",
                "train_manifest_path", "val_manifest_path", "cal_manifest_path", "test_manifest_path",
                "saved_uq_comparison_dir"}
    allowed = required | {"plots", "expected_profile_counts", "expected_steps", "expected_member_count", "alpha"}
    if required - config.keys() or config.keys() - allowed:
        raise ValueError(f"Missing config keys: {sorted(required - config.keys())}; "
                         f"unknown keys: {sorted(config.keys() - allowed)}")
    paths = {k: Path(config[k]) for k in required}
    missing = [str(p) for k, p in paths.items() if k != "output_dir" and not p.exists()]
    if missing:
        raise FileNotFoundError("Required local study inputs are missing:\n" + "\n".join(missing))
    out = paths["output_dir"]
    # Never overwrite a previous study or place outputs within an input directory.
    if out.exists():
        raise FileExistsError(f"Choose a new output_dir; already exists: {out}")
    for key, path in paths.items():
        if key != "output_dir" and path.is_dir() and (path.resolve() == out.resolve() or path.resolve() in out.resolve().parents):
            raise ValueError(f"output_dir must be outside input directories: {path}")
    expected = config.get("expected_profile_counts", {"train":5000, "val":1000, "cal":1000, "test":2000})
    manifests = load_manifests({s: paths[f"{s}_manifest_path"] for s in ("train","val","cal","test")}, expected)
    steps, members = int(config.get("expected_steps", 501)), int(config.get("expected_member_count", 10))
    alpha = float(config.get("alpha", .05))
    if alpha != .05 or steps < 1 or members < 2:
        raise ValueError("This fixed study requires alpha=.05, positive steps and >=2 members.")
    with h5py.File(paths["calibration_forecasts_h5_path"], "r") as f:
        names = decode_names(f.attrs["target_names"])
    if not names or "x_steam_out" not in names:
        raise ValueError("Saved forecast targets must include x_steam_out.")
    model_paths = [paths["ensemble_experiment_dir"] / f"model_{i}/model.pt" for i in range(members)]
    if any(not p.is_file() for p in model_paths):
        raise FileNotFoundError("All original ensemble checkpoints are required for provenance hashing.")
    for path in (paths["external_scaled_h5_path"], paths["training_scaled_h5_path"]):
        check_dataset_splits(path, manifests)
    validate_saved_calibrations(paths["saved_uq_comparison_dir"], manifests["cal"], alpha, 1e-3)
    out.mkdir(parents=True)
    file_inputs = {k: p for k, p in paths.items() if p.is_file()}
    file_inputs.update({f"checkpoint_{i}": p for i,p in enumerate(model_paths)})
    for p in sorted(paths["ensemble_experiment_dir"].glob("*.json")):
        file_inputs[f"ensemble_metadata/{p.name}"] = p
    source_paths = [Path(__file__), Path(__file__).with_name("_mace_floor_study.py"),
                    Path(__file__).with_name("_mace_scaler_audit.py")]
    # Hash all scientific inputs, including large HDF5 files, once before use.
    # Metadata/scores from the current reference run are included as well.
    for p in sorted(paths["saved_uq_comparison_dir"].glob("*/calibration_metadata.json")):
        file_inputs[f"saved/{p.parent.name}/calibration"] = p
    for p in sorted(paths["saved_uq_comparison_dir"].glob("*/coverage_metrics.json")):
        file_inputs[f"saved/{p.parent.name}/metrics"] = p
    for p in sorted(paths["saved_uq_comparison_dir"].glob("*/test_forecasts.h5")):
        file_inputs[f"saved/{p.parent.name}/forecasts"] = p
    print("Hashing fixed inputs (large HDF5 files may take a few minutes)", flush=True)
    hashes = {k: fingerprint(p) for k,p in file_inputs.items()}
    metadata = {"config": config, "config_path": config_path, "floors": list(FLOORS),
                "floor_operation": "ensemble_std_scaled + common_floor",
                "nominal_coverage": 1-alpha, "target_names": names, "input_hashes": hashes,
                "profile_names": manifests, "member_count": members, "steps": steps,
                "coverage_convention": "Inclusive comparisons of float32 scaled truth and endpoints, for every coverage summary",
                "calibration_arithmetic": "float32 residuals, spread, additive floor and order-statistic values, matching existing implementation",
                "metric_arithmetic": "float64 endpoint subtraction and interval score; physical widths/scores = scaled values * native target span",
                "no_training_or_inference": True, "test_used_to_select_floor": False,
                "study_source_hashes": [fingerprint(p) for p in source_paths],
                "note": "Descriptive fixed-grid sensitivity study; no floor selection or automatic declaration of stability."}
    try:
        metadata["code_commit"] = subprocess.check_output(["git","rev-parse","HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        metadata["code_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata["code_commit"] = None
    write_json(out / "study_metadata.json", metadata)
    verification, stats = verify_scaler(
        external_path=paths["external_scaled_h5_path"], training_path=paths["training_scaled_h5_path"],
        raw_path=paths["raw_h5_path"], reference_json_path=paths["reference_scaler_json_path"],
        manifests=manifests, target_names=names)
    write_json(out / "scaler_verification.json", verification)
    if not verification["match"]:
        raise ValueError(f"Scaler/data checks failed; inspect {out / 'scaler_verification.json'}. No sweep was run.")
    cal, cal_checks = load_forecasts(paths["calibration_forecasts_h5_path"], manifests["cal"],
        scaled_path=paths["external_scaled_h5_path"], split="cal", target_names=names,
        member_count=members, steps=steps, stats=stats)
    test, test_checks = load_forecasts(paths["test_forecasts_h5_path"], manifests["test"],
        scaled_path=paths["external_scaled_h5_path"], split="test", target_names=names,
        member_count=members, steps=steps, stats=stats)
    write_json(out / "forecast_verification.json", {"cal":cal_checks, "test":test_checks})
    abs_q = calibrate(cal, alpha=alpha, floor=None)
    baseline = {mode: evaluate(test, abs_q[mode], floor=None, alpha=alpha, span=stats["y"]["span"])
                for mode in METHODS}
    aggregates, target_rows, horizon_rows, distributions = [], [], [], []
    current_results, current_quantiles = {}, {}
    with h5py.File(out / "recalibration.h5", "w") as f:
        f.attrs["target_names"] = np.asarray(names, dtype="S")
        f.attrs["calibration_profile_names"] = np.asarray(manifests["cal"], dtype="S")
        f.attrs["alpha"] = alpha
        f.attrs["quantile_rank"] = int(abs_q["rank"])
        for label, floor in [("absolute", None)] + [(f"floor_{tau:g}", tau) for tau in FLOORS]:
            q = abs_q if floor is None else calibrate(cal, alpha=alpha, floor=floor)
            group = f.create_group(label)
            if floor is not None:
                group.attrs["requested_common_floor"] = floor
                group.attrs["effective_float32_common_floor"] = np.float32(floor)
            for key in ("horizon", "trajectory", "trajectory_scores"):
                group.create_dataset(key, data=q[key], compression="gzip")
            for mode in METHODS:
                result = baseline[mode] if floor is None else evaluate(test, q[mode], floor=floor, alpha=alpha, span=stats["y"]["span"])
                cov, ref = result["coverage"], baseline[mode]
                aggregates.append({"method":"absolute" if floor is None else "mace", "mode":mode, "floor":floor,
                    "objective_coverage": cov["pointwise"] if mode=="horizon" else cov["mean_targetwise_trajectory"],
                    "worst_horizon_target_coverage": float(cov["horizon_target"].min()),
                    "worst_target_trajectory_coverage": float(cov["trajectory_by_target"].min()),
                    "pointwise_coverage":cov["pointwise"], "mean_targetwise_trajectory_coverage":cov["mean_targetwise_trajectory"],
                    "complete_multivariate_trajectory_coverage":cov["complete_multivariate_trajectory"],
                    "mean_scaled_full_width":result["mean_scaled_width"],
                    "width_ratio_vs_absolute":result["mean_scaled_width"]/ref["mean_scaled_width"],
                    "mean_scaled_interval_score":result["mean_scaled_score"],
                    "interval_score_ratio_vs_absolute":result["mean_scaled_score"]/ref["mean_scaled_score"]})
                hlo, hhi = wilson(result["mask"].sum(axis=0), len(manifests["test"]))
                tlo, thi = wilson(result["mask"].all(axis=1).sum(axis=0), len(manifests["test"]))
                for j, name in enumerate(names):
                    dist = distribution(result["width_physical"][:,:,j])
                    target_rows.append({"method":"absolute" if floor is None else "mace", "mode":mode, "floor":floor,
                        "target":name, "physical_unit": config_target_unit(name),
                        "additive_floor_physical":None if floor is None else float(np.float32(floor)*stats["y"]["span"][j]),
                        "objective_coverage": float(cov["pointwise_by_target"][j] if mode=="horizon" else cov["trajectory_by_target"][j]),
                        "minimum_horizon_coverage": float(cov["horizon_target"][:,j].min()),
                        "trajectory_coverage_ci95_lower":float(tlo[j]), "trajectory_coverage_ci95_upper":float(thi[j]),
                        "mean_physical_full_width": float(result["mean_physical_width_by_target"][j]),
                        "width_ratio_vs_absolute":float(result["mean_physical_width_by_target"][j]/ref["mean_physical_width_by_target"][j]),
                        "mean_physical_interval_score":float(result["mean_physical_score_by_target"][j]),
                        "interval_score_ratio_vs_absolute":float(result["mean_physical_score_by_target"][j]/ref["mean_physical_score_by_target"][j]),
                        "quantile_units":"scaled residual" if floor is None else "dimensionless normalized residual",
                        "quantile_min":float(np.min(q[mode][...,j])), "quantile_median":float(np.median(q[mode][...,j])),
                        "quantile_max":float(np.max(q[mode][...,j])),
                        **{f"width_physical_{key}":value for key,value in dist.items() if key.startswith("p")}})
                    for h in range(steps):
                        horizon_rows.append({"method":"absolute" if floor is None else "mace", "mode":mode,
                            "floor":floor, "target":name, "horizon_index":h,
                            "coverage":float(cov["horizon_target"][h,j]),
                            "ci95_lower":float(hlo[h,j]), "ci95_upper":float(hhi[h,j]),
                            "mean_physical_full_width":float(result["width_physical"][:,h,j].mean()),
                            "width_ratio_vs_absolute":float(result["width_physical"][:,h,j].mean()/ref["width_physical"][:,h,j].mean()) if ref["width_physical"][:,h,j].mean()>0 else None})
                if floor == 1e-3:
                    current_results[mode] = result
                    current_quantiles[mode] = q[mode]
                print(f"{label}, {mode}: coverage={aggregates[-1]['objective_coverage']:.6f}, width ratio={aggregates[-1]['width_ratio_vs_absolute']:.6f}", flush=True)
            if floor is not None:
                for split, arrays in (("cal",cal),("test",test)):
                    fraction = np.float32(floor)/(arrays["spread"]+np.float32(floor))
                    for j,name in enumerate(names):
                        distributions.append({"floor":floor, "split":split, "target":name,
                                              "distribution":distribution(fraction[:,:,j], histogram=True)})
    write_csv(out / "aggregate_metrics.csv", aggregates)
    write_csv(out / "target_metrics.csv", target_rows)
    write_csv(out / "horizon_target_metrics.csv", horizon_rows)
    write_json(out / "floor_fraction_distributions.json", distributions)
    for mode in METHODS:
        key = f"absolute_conformal_target_{mode}"
        current_results[key], current_quantiles[key] = baseline[mode], abs_q[mode]
    reconciliation = reconcile_saved(current_results, current_quantiles,
        saved_dir=paths["saved_uq_comparison_dir"], test=test, stats=stats,
        profiles=manifests["test"], target_names=names)
    write_json(out / "saved_run_reconciliation.json", reconciliation)
    # Surface failures without accepting any broad numerical tolerance. Metric
    # differences are recorded, since reduction order can differ by a few ULPs.
    if any(not entry["quantiles"]["match"] for entry in reconciliation.values()):
        raise ValueError("Current-floor/absolute quantiles differ from saved run; inspect reconciliation before interpreting the sweep.")
    if any(not check["match"] for entry in reconciliation.values()
           for check in entry["saved_coverage_count_comparisons"].values()):
        raise ValueError("Regenerated legacy coverage counts disagree with the saved run; inspect reconciliation before interpreting the sweep.")
    metadata["input_file_sizes_and_mtimes_unchanged"] = all(
        p.stat().st_size == hashes[k]["bytes"] and p.stat().st_mtime_ns == hashes[k]["mtime_ns"]
        for k,p in file_inputs.items())
    if not metadata["input_file_sizes_and_mtimes_unchanged"]:
        raise ValueError("Fixed study inputs changed during execution.")
    array_mismatches = [method for method, entry in reconciliation.items()
                        if entry["saved_interval_arrays"]["status"] == "checked"
                        and any(not check[key]["match"] for check in entry["saved_interval_arrays"]["profiles"]
                                for key in ("lower", "upper"))]
    if array_mismatches:
        raise ValueError(f"Saved interval arrays disagree: {array_mismatches}. Inspect reconciliation.")
    if config.get("plots", True):
        plot_results(out, aggregates, target_rows)
    metadata["completed"] = True
    write_json(out / "study_metadata.json", metadata)
    print(f"Study saved to {out}", flush=True)
    return out


def config_target_unit(name: str) -> str:
    if name.startswith("T"):
        return "K"
    if name == "x_steam_out":
        return "quality fraction"
    if name == "rho_dollars":
        return "dollars"
    # These state variables retain their simulator definitions; do not invent
    # concentration/density units from their names.
    return "simulator-native units"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), config_path=args.config)


if __name__ == "__main__":
    main()
