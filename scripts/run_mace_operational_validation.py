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
                                         read_simulation, same_hashes, simulation_hashes)
from _mace_operational_metrics import evaluate_study
from _mace_operational_profiles import RECIPE, WIDE_RECIPE, generate, load_manifest, plot_profiles
from _mace_scaler_audit import read_scaler
from run_mace_profile_validation import validate_prior_study

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(operational_calibration_count=1000, operational_test_count=2000,
                profile_seed=20261010, early_settling_seconds=20., alpha=.05, common_floor=.001,
                device="auto", torch_threads=1, bootstrap_replicates=10000,
                bootstrap_seed=20261011, bootstrap_batch_size=64, plots=True,
                package_mo="src/modelica/MicroreactorPK/package.mo", input_overrides={},
                profile_recipe="local", reuse_original_calibration_dir=None)

# Only these two reviewed pre-fix files may change during recovery of an
# interrupted be406101 simulation. Include LF and Git-for-Windows CRLF bytes.
PRE_FIX_SOURCE_HASHES = {
    "scripts/run_mace_operational_validation.py": {
        "955c22f62988cc46add08234787dd05a29fbe02284ca257aff6f70397a0deab5",
        "96faa98189335fa6ee095c6f3153559c5e778e92ae431c405108ff35850c11a8"},
    "scripts/_mace_operational_forecasts.py": {
        "2b21182c0320af376258b3884c0f0ba532aa7f52b1f316936c57fed322fa765c",
        "8cf234cfabfeda54a99ca0f20a391d60df07ef01174967e2a343f8760ac80aff"},
}


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
    if c['profile_recipe'] not in ('local', 'wide30'):
        raise ValueError('profile_recipe must be local or wide30')
    for key in ("floor_study_dir", "output_dir", "package_mo"):
        p = Path(c[key])
        c[key] = str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    if Path(c["floor_study_dir"]) == Path(c["output_dir"]) or Path(c["floor_study_dir"]) in Path(c["output_dir"]).parents:
        raise ValueError("Operational output must be separate from the historical floor study")
    if c['reuse_original_calibration_dir'] is not None:
        p = Path(c['reuse_original_calibration_dir'])
        c['reuse_original_calibration_dir'] = str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
        origin, out = Path(c['reuse_original_calibration_dir']), Path(c['output_dir'])
        if origin == out or origin in out.parents or out in origin.parents:
            raise ValueError('New output must be separate from the retained reuse study')
    return c


def source_hashes():
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in
             ("_mace_operational_profiles.py", "_mace_operational_forecasts.py", "_mace_operational_metrics.py",
              "_mace_floor_study.py", "_mace_profile_bootstrap.py", "_mace_scaler_audit.py", "_mace_scaler_use.py",
              "run_mace_profile_validation.py", "run_mace_scaler_use_audit.py", "run_lstm_floor_sensitivity.py",
              "_mace_original_calibration_cache.py"))]
    paths += [ROOT/"src/rabl/machine_learning"/name for name in ("lstm_pipeline.py", "bagging_ensemble.py")]
    return {str(p.relative_to(ROOT)): fingerprint(p)["sha256"] for p in paths}


def git_state():
    try:
        return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip(),
                    dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


def generated_inputs(c, *, resume_after_validation_fix=False):
    out = Path(c["output_dir"])
    run = json.loads((out/"run_configuration.json").read_text())
    current = source_hashes()
    if run["configuration"] != c:
        raise ValueError("Configuration/executing source changed after generation; use a new output_dir")
    manifest = load_manifest(out)
    expected_recipe = WIDE_RECIPE if c['profile_recipe'] == 'wide30' else RECIPE
    if manifest['recipe'] != expected_recipe:
        raise ValueError('Profile recipe differs from the generated study')
    if manifest["profile_count"] != dict(cal=c["operational_calibration_count"], test=c["operational_test_count"]):
        raise ValueError("Operational split size mismatch")
    if run["source_hashes"] != current:
        if not resume_after_validation_fix:
            raise ValueError("Executing source changed after generation; the known interrupted angle-validation run can use --stage simulate --resume-after-validation-fix")
        migrate_validation_fix(c, run, manifest, current)
    return manifest


def simulation_identity(c, manifest):
    package = Path(c["package_mo"])
    if package.name != "package.mo" or not package.is_file():
        raise FileNotFoundError(f"MicroreactorPK package.mo required: {package}")
    return dict(manifest_identity=manifest["identity"],
                modelica_source_hashes={str(p.relative_to(package.parent)): fingerprint(p)["sha256"]
                                        for p in sorted(package.parent.rglob("*.mo"))},
                runner_source_sha256=fingerprint(ROOT/"src/rabl/interface/pymola.py")["sha256"],
                output_interval=.4, model_name="MicroreactorPK.Experiments.RunOneProfile")


def migrate_validation_fix(c, previous, manifest, current):
    """Explicit, narrow migration; never bypass changed scientific inputs/code."""
    out = Path(c["output_dir"])
    before = previous["source_hashes"]
    if set(before) != set(current):
        raise ValueError("Source inventory changed; cannot migrate this interrupted run")
    changed = [k for k in current if before[k] != current[k]]
    if not changed or any(before[k] not in PRE_FIX_SOURCE_HASHES.get(k.replace("\\", "/"), set()) for k in changed):
        raise ValueError("Only the reviewed be406101 angle-validation source changes can be migrated")
    if (out/"forecasts").exists() or (out/"evaluation").exists():
        raise ValueError("Recovery is limited to interrupted simulations before forecasting/evaluation")
    simpath = out/"simulation_metadata.json"
    sim = json.loads(simpath.read_text())
    if sim.get("completed") or sim.get("config") != c:
        raise ValueError("Requires an incomplete simulation with unchanged configuration")
    if any(sim.get(k) != v for k, v in simulation_identity(c, manifest).items()):
        raise ValueError("Modelica/runner/manifest changed; cannot migrate existing simulations")
    simulation_hashes(out, [p for p in manifest["profiles"] if p["id"] in sim.get("result_hashes", {})], sim)
    for filename, payload in (("run_configuration_before_validation_fix.json", previous),
                              ("simulation_metadata_before_validation_fix.json", sim)):
        if (out/filename).exists():
            raise FileExistsError(f"Recovery snapshot already exists: {out/filename}")
        write_json(out/filename, payload)
    revised = {**previous, "source_hashes": current,
               "validation_fix_migration": dict(origin_commit="be40610122cfd1bdd7186541edc5692d3b207873",
                    changed_sources=changed, previous_source_hashes=before, execution_git=git_state(),
                    manifest_identity=manifest["identity"], regenerated_profiles=False,
                    purpose="Validate native trajectory precision and reuse existing simulations")}
    write_json(out/"run_configuration.json", revised)
    print("Migrated the reviewed angle-validation fix; profile tables/configuration/physics are unchanged", flush=True)


def simulate_stage(c, manifest):
    out, package = Path(c["output_dir"]), Path(c["package_mo"])
    identity = simulation_identity(c, manifest)
    modelica_hashes = identity["modelica_source_hashes"]
    runner_source = ROOT/"src/rabl/interface/pymola.py"
    metapath = out/"simulation_metadata.json"
    if metapath.exists():
        previous = json.loads(metapath.read_text())
        if previous.get("completed"):
            raise FileExistsError("Simulation already completed; proceed to forecast")
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Cannot resume simulation with different inputs/source")
        simulation_hashes(out, [p for p in manifest["profiles"] if p["id"] in previous.get("result_hashes", {})], previous)
    elif any((out/"simulation").rglob("results_*.csv")):
        raise FileExistsError("Existing simulation CSVs have no matching run provenance")
    metadata = dict(**identity, completed=False,
                    result_hashes=previous.get("result_hashes", {}) if metapath.exists() else {},
                    result_validation=previous.get("result_validation", {}) if metapath.exists() else {}, config=c)
    write_json(metapath, metadata)
    names = validate_prior_study(Path(c["floor_study_dir"]))["target_names"]
    for split in ("cal", "test"):
        results = out/"simulation"/split
        results.mkdir(parents=True, exist_ok=True)
        chosen = [p for p in manifest["profiles"] if p["split"] == split]
        cached = [p for p in chosen if (results/(p["id"]+".csv")).exists()]
        # Check existing CSVs before starting more simulations. Reuse a fully
        # completed split without opening Dymola or rerunning any profile.
        for p in cached:
            path = results/(p["id"]+".csv")
            _y, _angle, report = read_simulation(path, names=names, moves=p["moves"])
            metadata["result_hashes"][p["id"]] = fingerprint(path)
            metadata["result_validation"][p["id"]] = report
        write_json(metapath, metadata)
        print(f"Validated/reused {len(cached)}/{len(chosen)} existing {split} simulation CSVs", flush=True)
        if len(cached) == len(chosen):
            continue
        sys.path.insert(0, str(ROOT/"src"))
        from rabl.interface.pymola import BatchConfig, DymolaBatchRunner
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
            _y, _angle, report = read_simulation(path, names=names, moves=p["moves"])
            metadata["result_hashes"][p["id"]] = fingerprint(path)
            metadata["result_validation"][p["id"]] = report
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
        origin_identity = meta.get('original_calibration_manifest_identity', manifest['identity']) if source == 'original' else manifest['identity']
        cals[source], reports = load_forecast(folder/f"{source}_cal.h5", names=names, profile_names=ids,
                                             manifest_identity=origin_identity, stats=stats)
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
                    manifest_identity=manifest["identity"], source_hashes=source_hashes(), git=git_state(),
                    original_calibration_reuse=meta.get('original_calibration_reuse'),
                    profile_recipe=manifest['recipe'])
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


def run(payload, stage, *, resume_after_validation_fix=False):
    if resume_after_validation_fix and stage != "simulate":
        raise ValueError("--resume-after-validation-fix applies only to --stage simulate")
    c = configuration(payload)
    out = Path(c["output_dir"])
    if stage == 'new-only':
        if c['profile_recipe'] != 'wide30' or not c['reuse_original_calibration_dir']:
            raise ValueError('new-only requires wide30 and a completed original-calibration reuse source')
        if out.exists():
            raise FileExistsError('new-only starts a fresh broad study; resume existing studies with individual --stage commands')
        from _mace_original_calibration_cache import validate_cache
        from run_mace_scaler_use_audit import runtime_settings
        print('Preflight: verifying retained original-calibration cache before any new simulations', flush=True)
        prior, _paths, hashes = frozen_inputs(c, ROOT)
        _device, runtime = runtime_settings(c['device'], c['torch_threads'])
        validate_cache(c, prior, hashes, root=ROOT, runtime=runtime)
    stages = ("generate", "simulate", "forecast", "evaluate") if stage in ('all', 'new-only') else (stage,)
    for current in stages:
        print(f"Operational validation stage: {current}", flush=True)
        sources = source_hashes()
        started = False
        try:
            if current == "generate":
                manifest = generate(out, calibration_count=c["operational_calibration_count"],
                                    test_count=c["operational_test_count"], seed=c["profile_seed"],
                                    settling_seconds=c["early_settling_seconds"],
                                    recipe=WIDE_RECIPE if c['profile_recipe'] == 'wide30' else RECIPE)
                write_json(out/"run_configuration.json", dict(configuration=c, source_hashes=sources, git=git_state()))
                if c["plots"]:
                    plot_profiles(out, manifest)
            else:
                manifest = generated_inputs(c, resume_after_validation_fix=resume_after_validation_fix)
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
    p.add_argument("--stage", required=True, choices=("generate", "simulate", "forecast", "evaluate", "all", "new-only"))
    p.add_argument("--resume-after-validation-fix", action="store_true",
                   help="Guarded recovery of the interrupted be406101 angle-validation run; simulate stage only")
    args = p.parse_args()
    run(json.loads(args.config.read_text()), args.stage, resume_after_validation_fix=args.resume_after_validation_fix)


if __name__ == "__main__":
    main()
