"""Generate, simulate, forecast and evaluate prespecified B/C/D/F operations.

Run from the repository root; see mace_operational_validation_workflow.md.
No training, scaler fitting, performance-based profile selection or floor tuning.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import ceil
from pathlib import Path

import h5py
import numpy as np

from _mace_floor_study import fingerprint, write_json
from _mace_operational_forecasts import (forecast_stage, frozen_inputs, load_forecast,
                                         read_simulation, same_hashes)
from _mace_operational_metrics import evaluate_study
from _mace_operational_profiles import generate, load_manifest, plot_profiles
from _mace_scaler_audit import read_scaler
from run_mace_profile_validation import validate_prior_study

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(operational_calibration_count=1000, operational_test_count=2000,
                profile_seed=20261010, early_settling_seconds=20., alpha=.05, common_floor=.001,
                device="auto", torch_threads=1, bootstrap_replicates=10000,
                bootstrap_seed=20261011, bootstrap_batch_size=64, plots=True,
                package_mo="src/modelica/MicroreactorPK/package.mo", input_overrides={})


def configuration(payload):
    required = {"floor_study_dir", "output_dir"}
    if required-set(payload) or set(payload)-required-set(DEFAULTS):
        raise ValueError("Missing required or unknown configuration keys")
    c = {**DEFAULTS, **payload}
    for key in ("operational_calibration_count", "operational_test_count", "profile_seed", "torch_threads",
                "bootstrap_replicates", "bootstrap_seed", "bootstrap_batch_size"):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < (0 if "seed" in key else 1):
            raise ValueError(f"Invalid integer: {key}")
    if (c["alpha"] != .05 or c["common_floor"] != .001 or c["bootstrap_replicates"] < 2
        or c["operational_test_count"] < 2
        or ceil((c["operational_calibration_count"]+1)*.95) > c["operational_calibration_count"]):
        raise ValueError("Requires alpha=.05, common floor=.001, finite operational quantiles and bootstrap")
    if not isinstance(c["plots"], bool) or not np.isfinite(c["early_settling_seconds"]) or not 0 < c["early_settling_seconds"] < 200:
        raise ValueError("Invalid plot/settling settings")
    if c["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    for key in ("floor_study_dir", "output_dir", "package_mo"):
        p = Path(c[key])
        c[key] = str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    if Path(c["floor_study_dir"]) == Path(c["output_dir"]) or Path(c["floor_study_dir"]) in Path(c["output_dir"]).parents:
        raise ValueError("Operational output must be separate from the historical floor study")
    return c


def source_hashes():
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in
             ("_mace_operational_profiles.py", "_mace_operational_forecasts.py", "_mace_operational_metrics.py",
              "_mace_floor_study.py", "_mace_profile_bootstrap.py", "_mace_scaler_audit.py", "_mace_scaler_use.py",
              "run_mace_profile_validation.py", "run_mace_scaler_use_audit.py", "run_lstm_floor_sensitivity.py"))]
    paths += [ROOT/"src/rabl/machine_learning"/name for name in ("lstm_pipeline.py", "bagging_ensemble.py")]
    return {str(p.relative_to(ROOT)): fingerprint(p)["sha256"] for p in paths}


def git_state():
    try:
        return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip(),
                    dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


def generated_inputs(c):
    out = Path(c["output_dir"])
    run = json.loads((out/"run_configuration.json").read_text())
    if run["configuration"] != c or run["source_hashes"] != source_hashes():
        raise ValueError("Configuration/executing source changed after generation; use a new output_dir")
    manifest = load_manifest(out)
    if manifest["profile_count"] != dict(cal=c["operational_calibration_count"], test=c["operational_test_count"]):
        raise ValueError("Operational split size mismatch")
    return manifest


def simulate_stage(c, manifest):
    out, package = Path(c["output_dir"]), Path(c["package_mo"])
    if package.name != "package.mo" or not package.is_file():
        raise FileNotFoundError(f"MicroreactorPK package.mo required: {package}")
    modelica_hashes = {str(p.relative_to(package.parent)): fingerprint(p)["sha256"]
                       for p in sorted(package.parent.rglob("*.mo"))}
    runner_source = ROOT/"src/rabl/interface/pymola.py"
    identity = dict(manifest_identity=manifest["identity"], modelica_source_hashes=modelica_hashes,
                    runner_source_sha256=fingerprint(runner_source)["sha256"], output_interval=.4,
                    model_name="MicroreactorPK.Experiments.RunOneProfile")
    metapath = out/"simulation_metadata.json"
    if metapath.exists():
        previous = json.loads(metapath.read_text())
        if previous.get("completed"):
            raise FileExistsError("Simulation already completed; proceed to forecast")
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Cannot resume simulation with different inputs/source")
        for name, old in previous.get("result_hashes", {}).items():
            profile = next(p for p in manifest["profiles"] if p["id"] == name)
            path = out/"simulation"/profile["split"]/(name+".csv")
            if fingerprint(path)["sha256"] != old["sha256"]:
                raise ValueError(f"Completed simulation CSV changed before resumption: {name}")
    elif any((out/"simulation").rglob("results_*.csv")):
        raise FileExistsError("Existing simulation CSVs have no matching run provenance")
    metadata = dict(**identity, completed=False, result_hashes={}, config=c)
    write_json(metapath, metadata)
    sys.path.insert(0, str(ROOT/"src"))
    from rabl.interface.pymola import BatchConfig, DymolaBatchRunner
    names = validate_prior_study(Path(c["floor_study_dir"]))["target_names"]
    for split in ("cal", "test"):
        results = out/"simulation"/split
        results.mkdir(parents=True, exist_ok=True)
        cfg = BatchConfig(package_mo=str(package), model_name=identity["model_name"],
                          profiles_dir=str(out/"profiles"/split), out_dir=str(results), profile_mode="flat_mat",
                          table_name="profile", output_interval=.4, canonical_output_interval=.4,
                          skip_existing=True)
        runner = DymolaBatchRunner(cfg)
        try:
            runner.start()
            runner.run_all()
        finally:
            runner.close()
        for p in manifest["profiles"]:
            if p["split"] != split:
                continue
            path = results/(p["id"]+".csv")
            read_simulation(path, names=names, moves=p["moves"])
            metadata["result_hashes"][p["id"]] = fingerprint(path)
        write_json(metapath, metadata)
    if modelica_hashes != {str(p.relative_to(package.parent)): fingerprint(p)["sha256"]
                           for p in sorted(package.parent.rglob("*.mo"))}:
        raise ValueError("Modelica source changed during simulation")
    if fingerprint(runner_source)["sha256"] != identity["runner_source_sha256"]:
        raise ValueError("Simulation runner changed during execution")
    metadata["completed"] = True
    write_json(metapath, metadata)
    return metadata


def evaluate_stage(c, manifest):
    out = Path(c["output_dir"])
    if (out/"evaluation").exists():
        raise FileExistsError("Evaluation stage already started; retain outputs and choose a new output_dir")
    folder = out/"forecasts"
    meta = json.loads((folder/"forecast_metadata.json").read_text())
    if not meta.get("completed") or not meta.get("inputs_unchanged") or meta["manifest_identity"] != manifest["identity"]:
        raise ValueError("Requires completed frozen corrected forecasts")
    prior, paths, hashes = frozen_inputs(c, ROOT)
    if {k: v["sha256"] for k, v in hashes.items()} != {k: v["sha256"] for k, v in meta["input_hashes"].items()}:
        raise ValueError("Frozen input hashes changed since forecasting")
    for filename, old in meta["output_hashes"].items():
        if fingerprint(folder/filename)["sha256"] != old["sha256"]:
            raise ValueError(f"Frozen forecast changed: {filename}")
    stats, _ = read_scaler(paths["external_scaled_h5_path"])
    names = meta["target_names"]
    cals, checks, calibration_profiles = {}, [], {}
    for source in ("original", "operational"):
        ids = (prior["profile_names"]["cal"] if source == "original" else
               [p["id"] for p in manifest["profiles"] if p["split"] == "cal"])
        calibration_profiles[source] = ids
        cals[source], reports = load_forecast(folder/f"{source}_cal.h5", names=names, profile_names=ids,
                                             manifest_identity=manifest["identity"], stats=stats)
        checks.extend(dict(split=source+"_cal", **r) for r in reports)
    test_profiles = [p for p in manifest["profiles"] if p["split"] == "test"]
    test, reports = load_forecast(folder/"operational_test.h5", names=names, profile_names=[p["id"] for p in test_profiles],
                                   manifest_identity=manifest["identity"], stats=stats)
    checks.extend(dict(split="operational_test", **r) for r in reports)
    reportdir = out/"evaluation"
    metadata = evaluate_study(reportdir, calibrations=cals, test=test, profiles=test_profiles, names=names,
                             stats=stats, alpha=c["alpha"], floor=c["common_floor"],
                             settling_seconds=c["early_settling_seconds"], replicates=c["bootstrap_replicates"],
                             seed=c["bootstrap_seed"], batch_size=c["bootstrap_batch_size"],
                             calibration_profiles=calibration_profiles)
    write_json(reportdir/"stored_array_checks.json", checks)
    same_hashes(paths, hashes)
    for filename, old in meta["output_hashes"].items():
        if fingerprint(folder/filename)["sha256"] != old["sha256"]:
            raise ValueError(f"Frozen forecast changed during evaluation: {filename}")
    metadata.update(input_hashes=hashes, forecast_hashes=meta["output_hashes"],
                    manifest_identity=manifest["identity"], source_hashes=source_hashes(), git=git_state())
    write_json(reportdir/"evaluation_metadata.json", metadata)
    (reportdir/"README.md").write_text(
        "# Operational-profile evaluation\n\n"
        "Completed with corrected own-member Y-to-X recurrent feedback, frozen checkpoints and scaler. "
        "Both calibration arms use newly calibrated quantiles for that same corrected model; original profile identities are unchanged.\n\n"
        "Inspect accuracy_metrics.csv, uq_method_metrics.csv and paired_comparisons.csv by mode, phase, family and scope. "
        "Widths are full widths. Ratios are first-arm width / reference-arm width; reductions are one minus that ratio. "
        "Difference signs follow the comparison label. Target values use physical units; aggregate values use equally weighted scaled targets. "
        "MAE and RMSE use native simulator truth. Coverage and interval score use one float32 scaled endpoint/truth convention.\n\n"
        "Original calibration is a nonexchangeable transfer diagnostic. Operational calibration/test are independent draws from the same prespecified mixture. "
        "Only full-time pooled-mixture objectives are conformal objectives; phase/family subsets and joint-mode per-target coverage are diagnostics. "
        "Horizon coverage is marginal at each horizon/target, target trajectory is all-time for each target, joint trajectory is all-time/all-target. "
        "Phase trajectory coverage means every supported point of that phase, not complete-trajectory coverage.\n\n"
        "Bootstrap intervals are paired whole-test-profile 95% percentile intervals conditional on calibration, not a new calibration bootstrap. "
        "They are descriptive, not simultaneous confidence bounds or evidence of exchangeability. "
        "Zero-support phase profiles are excluded; inspect phase_support.csv and valid/undefined replicate counts. "
        "No failed simulations may be dropped. Individual horizon coverage in horizon_target_coverage.csv is a diagnostic without bootstrap intervals.\n",
        encoding="utf-8")
    return metadata


def run(payload, stage):
    c = configuration(payload)
    out = Path(c["output_dir"])
    stages = ("generate", "simulate", "forecast", "evaluate") if stage == "all" else (stage,)
    for current in stages:
        print(f"Operational validation stage: {current}", flush=True)
        sources = source_hashes()
        started = False
        try:
            if current == "generate":
                manifest = generate(out, calibration_count=c["operational_calibration_count"],
                                    test_count=c["operational_test_count"], seed=c["profile_seed"],
                                    settling_seconds=c["early_settling_seconds"])
                write_json(out/"run_configuration.json", dict(configuration=c, source_hashes=sources, git=git_state()))
                if c["plots"]:
                    plot_profiles(out, manifest)
            else:
                manifest = generated_inputs(c)
                meta_path = {"simulate": out/"simulation_metadata.json",
                             "forecast": out/"forecasts/forecast_metadata.json",
                             "evaluate": out/"evaluation/evaluation_metadata.json"}[current]
                # Do not turn a completed earlier stage into a failure when a
                # caller accidentally requests it again.
                completed_before = meta_path.exists() and json.loads(meta_path.read_text()).get("completed", False)
                started = not completed_before
                if current == "simulate":
                    simulate_stage(c, manifest)
                elif current == "forecast":
                    forecast_stage(c, manifest, root=ROOT)
                elif current == "evaluate":
                    evaluate_stage(c, manifest)
            load_manifest(out)
            if source_hashes() != sources:
                raise ValueError("Executing source changed during stage")
        except Exception as exc:
            if started and meta_path.exists():
                state = json.loads(meta_path.read_text())
                state.update(completed=False, error=f"{type(exc).__name__}: {exc}")
                write_json(meta_path, state)
            raise
        print(f"Completed {current}: {out}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--stage", required=True, choices=("generate", "simulate", "forecast", "evaluate", "all"))
    args = p.parse_args()
    run(json.loads(args.config.read_text()), args.stage)


if __name__ == "__main__":
    main()
