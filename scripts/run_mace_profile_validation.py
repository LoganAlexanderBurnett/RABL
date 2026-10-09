"""Bootstrap frozen MACE results, diagnose horizons, and compare joint bands.

Run from the repository root with --config scripts/mace_profile_validation_config.json.
Requires a completed floor study and its original saved forecasts/scaled dataset.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

from _mace_floor_study import calibrate, evaluate, fingerprint, load_forecasts, write_json
from _mace_scaler_audit import decode_names, read_scaler
from _mace_profile_bootstrap import bootstrap_outputs, profile_statistics, resample_statistics
from _mace_horizon_diagnostics import horizon_diagnostics, load_controls, plot_horizons
from run_lstm_floor_sensitivity import config_target_unit

FLOOR = 1e-3


def joint_calibration(profile_target_maxima, *, rank):
    """One max over time AND targets per calibration profile, not max of q_j."""
    scores = np.asarray(profile_target_maxima).max(axis=1)
    if scores.ndim!=1 or not np.all(np.isfinite(scores)) or not 1<=rank<=len(scores):
        raise ValueError("Invalid joint calibration scores/rank")
    return np.partition(scores,rank-1)[rank-1], scores


def validate_prior_study(study):
    metadata = json.loads((study/"study_metadata.json").read_text())
    if (not metadata.get("completed") or not metadata.get("no_training_or_inference")
        or not metadata.get("input_file_sizes_and_mtimes_unchanged")
        or metadata["config"].get("alpha",.05)!=.05
        or metadata["member_count"]!=10):
        raise ValueError("Requires the completed fixed-ensemble floor study at alpha=.05.")
    if not json.loads((study/"scaler_verification.json").read_text())["match"]:
        raise ValueError("Prior scaler verification did not pass")
    profiles=metadata["profile_names"]
    for split,names in profiles.items():
        if len(names)!=len(set(names)) or len(names)!=metadata["config"]["expected_profile_counts"][split]:
            raise ValueError(f"Invalid fixed {split} split")
        for other in profiles:
            if other<split and set(names)&set(profiles[other]):
                raise ValueError("Overlapping fixed profile splits")
    return metadata


def plot_bootstrap(out,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r["scope"]=="aggregate" and r["metric"]=="mean_full_width_relative_reduction"]
    fig,ax=plt.subplots(figsize=(7,3.5),constrained_layout=True)
    for i,r in enumerate(selected):
        if r["ci_lower"] is not None:
            ax.plot([100*r["ci_lower"],100*r["ci_upper"]],[i,i],color="tab:blue")
        if r["estimate"] is not None:
            ax.scatter(100*r["estimate"],i,color="tab:blue")
    ax.set_yticks(range(len(selected)),[r["mode"] for r in selected])
    ax.axvline(0,color="gray",linestyle="--")
    ax.set_xlabel("Mean scaled full-width reduction versus matching absolute CP (%)")
    ax.set_title("Paired whole-profile bootstrap: descriptive 95% percentile intervals")
    ax.grid(axis="x",alpha=.25)
    fig.savefig(out/"bootstrap_width_reductions.png",dpi=180)
    fig.savefig(out/"bootstrap_width_reductions.pdf")
    plt.close(fig)


def run(config,*,config_path=None):
    required={"floor_study_dir","output_dir"}
    allowed=required|{"bootstrap_replicates","bootstrap_seed","bootstrap_batch_size","include_joint",
                      "plots","worst_cells_per_target","control_change_threshold_degrees","input_overrides"}
    if required-config.keys() or config.keys()-allowed:
        raise ValueError("Missing required or unknown configuration keys")
    study,out=Path(config["floor_study_dir"]),Path(config["output_dir"])
    if out.exists():
        raise FileExistsError(f"Choose a new output_dir; already exists: {out}")
    prior=validate_prior_study(study)
    input_keys=("external_scaled_h5_path","calibration_forecasts_h5_path","test_forecasts_h5_path")
    overrides=config.get("input_overrides",{})
    if overrides.keys()-set(input_keys):
        raise ValueError("Only scaled dataset and original audit forecast paths may be overridden")
    inputs={key:Path(overrides.get(key,prior["config"][key])) for key in input_keys}
    report_keys=("study_metadata.json","scaler_verification.json","recalibration.h5","aggregate_metrics.csv")
    inputs.update({"prior/"+name:study/name for name in report_keys})
    if out.resolve()==study.resolve() or study.resolve() in out.resolve().parents:
        raise ValueError("New output_dir must be outside the existing study")
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    replicates=int(config.get("bootstrap_replicates",10000)); seed=int(config.get("bootstrap_seed",20261009))
    batch=int(config.get("bootstrap_batch_size",32)); include_joint=config.get("include_joint",True)
    if not isinstance(include_joint,bool) or replicates<2 or seed<0 or batch<1:
        raise ValueError("Invalid bootstrap configuration")
    for key in ("bootstrap_replicates","bootstrap_seed","bootstrap_batch_size","worst_cells_per_target"):
        if key in config and (isinstance(config[key],bool) or int(config[key])!=config[key]):
            raise ValueError(f"{key} must be an integer")
    if (int(config.get("worst_cells_per_target",3))<1
        or not np.isfinite(float(config.get("control_change_threshold_degrees",1e-4)))
        or float(config.get("control_change_threshold_degrees",1e-4))<0
        or not isinstance(config.get("plots",True),bool)):
        raise ValueError("Invalid diagnostic/plot configuration")
    out.mkdir(parents=True)
    print("Hashing original fixed forecasts/dataset and completed study",flush=True)
    hashes={key:fingerprint(path) for key,path in inputs.items()}
    for key in input_keys:
        if hashes[key]["sha256"]!=prior["input_hashes"][key]["sha256"]:
            raise ValueError(f"Input differs from the completed floor study: {key}")
    names,profiles=prior["target_names"],prior["profile_names"]
    n,steps=len(profiles["test"]),prior["steps"]
    metadata={"completed":False,"config":config,"config_path":config_path,"input_hashes":hashes,
        "floor":FLOOR,"alpha":.05,"profile_names":profiles,"target_names":names,"steps":steps,
        "bootstrap_replicates":replicates,"bootstrap_seed":seed,"bootstrap_batch_size":batch,
        "bootstrap_rng":"NumPy Generator(PCG64), integer indices, common draws across every method/metric",
        "confidence_interval":"95% marginal percentile bootstrap, NOT simultaneous across metrics/scopes",
        "resampling_unit":"complete test profile; all horizons and targets retained together",
        "calibration_fixed_during_bootstrap":True,"no_training_or_inference":True,
        "coverage_convention":prior["coverage_convention"],"metric_arithmetic":prior["metric_arithmetic"],
        "conditional_uncertainty_scope":"Test-profile sampling only, conditional on the trained ensemble and calibration set",
        "independence_assumption":"Test profiles are independent sampling units; related simulation families would require a family-level bootstrap",
        "include_joint":include_joint,"joint_absolute_scale":"unit scale in train-minmax-scaled target coordinates",
        "interval_score_scope":"Average pointwise interval score; not a proper score of a joint trajectory set",
        "source_hashes":[fingerprint(Path(__file__).with_name(name)) for name in
            ("run_mace_profile_validation.py","_mace_profile_bootstrap.py","_mace_horizon_diagnostics.py","_mace_floor_study.py","_mace_scaler_audit.py")]}
    try:
        metadata["code_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
        metadata["code_dirty"]=bool(subprocess.check_output(["git","status","--porcelain"],text=True,stderr=subprocess.DEVNULL).strip())
    except (OSError,subprocess.CalledProcessError):
        metadata["code_commit"]=None
    write_json(out/"validation_metadata.json",metadata)
    stats,smeta=read_scaler(inputs["external_scaled_h5_path"])
    if smeta["target_names"]!=names:
        raise ValueError("Target order differs from verified prior study")
    arrays={}
    for split,path_key in (("cal","calibration_forecasts_h5_path"),("test","test_forecasts_h5_path")):
        arrays[split],checks=load_forecasts(inputs[path_key],profiles[split],scaled_path=inputs["external_scaled_h5_path"],
            split=split,target_names=names,member_count=10,steps=steps,stats=stats)
        # A completed original run had exact aggregation/descaling checks.
        for profile in checks["profiles"]:
            for key in ("member_mean","member_spread","physical_truth_vs_run_descaling"):
                if key in profile and not profile[key]["match"]:
                    raise ValueError(f"Original forecast verification failed: {split}/{profile['profile']}/{key}")
        write_json(out/f"{split}_forecast_verification.json",checks)
    cal,test=arrays["cal"],arrays["test"]
    # Reproduce current quantiles once, before bootstrapping. Bootstrap replicates
    # subsequently hold these frozen; joint q is also calibrated only once.
    quantiles={"absolute":calibrate(cal,alpha=.05,floor=None),"mace":calibrate(cal,alpha=.05,floor=FLOOR)}
    with h5py.File(study/"recalibration.h5","r") as f:
        if decode_names(f.attrs["target_names"])!=names or decode_names(f.attrs["calibration_profile_names"])!=profiles["cal"]:
            raise ValueError("Saved calibration identity differs")
        rank=int(f.attrs["quantile_rank"])
        for method,label in (("absolute","absolute"),("mace","floor_0.001")):
            if rank!=int(quantiles[method]["rank"]):
                raise ValueError("Calibration order statistic rank differs")
            for key in ("horizon","trajectory","trajectory_scores"):
                if not np.array_equal(quantiles[method][key],f[f"{label}/{key}"][()]):
                    raise ValueError(f"Frozen calibration mismatch: {method}/{key}")
    modes=["horizon","trajectory"]+(["joint"] if include_joint else [])
    joint={}
    if include_joint:
        with h5py.File(out/"joint_calibration.h5","w") as f:
            f.attrs["target_names"]=np.asarray(names,dtype="S")
            f.attrs["calibration_profile_names"]=np.asarray(profiles["cal"],dtype="S")
            f.attrs["alpha"]=.05;f.attrs["rank"]=rank;f.attrs["scaled_common_floor"]=FLOOR
            for method in quantiles:
                q,scores=joint_calibration(quantiles[method]["trajectory_scores"],rank=rank)
                joint[method]=q
                g=f.create_group(method)
                g.create_dataset("q_joint",data=q);g.create_dataset("profile_scores",data=scores,compression="gzip")
                g.attrs["score"]="max over time and targets of abs scaled residual"+(" / (spread + 1e-3)" if method=="mace" else "")
    statistics={}
    with h5py.File(out/"profile_statistics.h5","w") as f:
        f.attrs["target_names"]=np.asarray(names,dtype="S");f.attrs["test_profile_names"]=np.asarray(profiles["test"],dtype="S")
        for mode in modes:
            for method in ("absolute","mace"):
                q=joint[method] if mode=="joint" else quantiles[method][mode]
                result=evaluate(test,q,floor=None if method=="absolute" else FLOOR,alpha=.05,span=stats["y"]["span"])
                st=profile_statistics(result)
                statistics[f"{mode}/{method}"]=st
                for key,a in st.items():
                    f.create_dataset(f"{mode}/{method}/{key}",data=a,compression="gzip")
                del result
    samples,digest=resample_statistics(statistics,replicates=replicates,seed=seed,batch_size=batch)
    metadata["resampling_indices_sha256_little_endian_int64"]=digest
    comparisons=bootstrap_outputs(out,statistics,samples,names=names,span=stats["y"]["span"],modes=modes,
        target_units=[config_target_unit(name) for name in names])
    controls=load_controls(inputs["test_forecasts_h5_path"],profiles["test"],steps)
    masks=horizon_diagnostics(out,test=test,cal=cal,controls=controls,names=names,profiles=profiles["test"],
        span=stats["y"]["span"],quantiles={key:value["horizon"] for key,value in quantiles.items()},
        floor=FLOOR,alpha=.05,top_cells=int(config.get("worst_cells_per_target",3)),
        control_threshold=float(config.get("control_change_threshold_degrees",1e-4)))
    if config.get("plots",True):
        plot_bootstrap(out,comparisons);plot_horizons(out,masks,names)
    metadata["input_sizes_and_mtimes_unchanged"]=all(path.stat().st_size==hashes[key]["bytes"] and
        path.stat().st_mtime_ns==hashes[key]["mtime_ns"] for key,path in inputs.items())
    if not metadata["input_sizes_and_mtimes_unchanged"]:
        raise ValueError("Frozen scientific inputs changed during execution")
    metadata["horizon_diagnostic_scope"]="Descriptive/test-exploratory; Wilson intervals unadjusted for multiplicity; pooled regime points are not independent"
    metadata["control_change_threshold_degrees"]=float(config.get("control_change_threshold_degrees",1e-4))
    metadata["completed"]=True
    write_json(out/"validation_metadata.json",metadata)
    print(f"Validation outputs saved to {out}",flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),config_path=args.config)


if __name__=="__main__":
    main()
