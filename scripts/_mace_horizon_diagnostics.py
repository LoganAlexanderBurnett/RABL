"""Descriptive horizon diagnostics on the frozen forecast/calibration run."""
from __future__ import annotations

import h5py
import numpy as np

from _mace_floor_study import evaluate, wilson, write_csv


def load_controls(path, profiles, steps):
    with h5py.File(path, "r") as f:
        controls = np.asarray([f[name]["control" if "control" in f[name] else "u"][()]
                               for name in profiles], dtype=np.float64)
    if controls.shape!=(len(profiles),steps) or not np.all(np.isfinite(controls)):
        raise ValueError("Invalid physical control series")
    return controls


def contiguous_regions(flag):
    edges = np.diff(np.r_[False, flag, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1).tolist(), (np.flatnonzero(edges==-1)-1).tolist()))


def horizon_diagnostics(out, *, test, cal, controls, names, profiles, span,
                        quantiles, floor, alpha, top_cells=3, control_threshold=1e-4):
    if top_cells<1 or not np.isfinite(control_threshold) or control_threshold<0:
        raise ValueError("Invalid horizon diagnostic settings")
    n,h,j = test["truth"].shape
    changes = np.abs(np.diff(controls, axis=1, prepend=controls[:, :1]))
    regimes = {"initial": np.broadcast_to(np.arange(h)==0,(n,h)),
               "control_changing": (changes>control_threshold)&(np.arange(h)>0),
               "control_holding": (changes<=control_threshold)&(np.arange(h)>0)}
    residual = np.abs(test["truth"].astype(np.float64)-test["mean"].astype(np.float64))
    normalized = np.abs(test["truth"]-test["mean"])/(test["spread"]+np.float32(floor))
    cal_result = {}
    base = evaluate(test, quantiles["absolute"], floor=None, alpha=alpha, span=span)
    cells, regions, examples, regime_rows, summaries = [], [], [], [], []
    masks = {}
    for method in ("absolute", "mace"):
        tau = None if method=="absolute" else floor
        result = base if method=="absolute" else evaluate(test,quantiles[method],floor=tau,alpha=alpha,span=span)
        mask = result["mask"]
        masks[method] = mask.mean(axis=0)
        clo,chi = wilson(mask.sum(axis=0),n)
        # Evaluate calibrated endpoints on calibration profiles for diagnostics.
        cal_result[method] = evaluate(cal,quantiles[method],floor=tau,alpha=alpha,span=span)["mask"].mean(axis=0)
        for target,name in enumerate(names):
            cov = mask[:,:,target].mean(axis=0)
            selected = np.argsort(cov,kind="stable")[:min(top_cells,h)]
            for t in range(h):
                cells.append({"method":method,"target":name,"horizon_index":t,
                    "coverage":float(cov[t]),"ci95_lower":float(clo[t,target]),"ci95_upper":float(chi[t,target]),
                    "calibration_endpoint_coverage":float(cal_result[method][t,target]),
                    "coverage_difference_vs_absolute":float(cov[t]-base["mask"][:,t,target].mean()),
                    "mace_only_covered_profiles":None if method=="absolute" else int((mask[:,t,target]&~base["mask"][:,t,target]).sum()),
                    "absolute_only_covered_profiles":None if method=="absolute" else int((~mask[:,t,target]&base["mask"][:,t,target]).sum()),
                    "mean_physical_full_width":float(result["width_physical"][:,t,target].mean()),
                    "mean_physical_abs_error":float(residual[:,t,target].mean()*span[target]),
                    "mean_scaled_spread":float(test["spread"][:,t,target].mean(dtype=np.float64)),
                    "median_floor_share":float(np.median(np.float32(floor)/(test["spread"][:,t,target]+np.float32(floor)))),
                    "normalized_error_p95":float(np.quantile(normalized[:,t,target],.95)),
                    "calibration_quantile":float(quantiles[method][t,target]),
                    "fraction_control_changing":float(regimes["control_changing"][:,t].mean()),
                    "mean_control_change_degrees":float(changes[:,t].mean()),
                    "among_lowest_coverage_cells_for_target":bool(t in selected)})
            flags = {"below_nominal":cov<1-alpha,"unadjusted_wilson_upper_below_nominal":chi[:,target]<1-alpha}
            for criterion,flag in flags.items():
                for start,end in contiguous_regions(flag):
                    regions.append({"method":method,"target":name,"criterion":criterion,
                        "start_horizon_index":start,"end_horizon_index":end,"length":end-start+1,
                        "minimum_coverage":float(cov[start:end+1].min()),"mean_coverage":float(cov[start:end+1].mean())})
            summaries.append({"method":method,"target":name,"mean_coverage":float(cov.mean()),
                "minimum_coverage":float(cov.min()),"worst_horizon_index":int(selected[0]),
                "cells_below_nominal":int(flags["below_nominal"].sum()),
                "cells_wilson_upper_below_nominal":int(flags["unadjusted_wilson_upper_below_nominal"].sum()),
                "longest_below_nominal_region":max([end-start+1 for start,end in contiguous_regions(flags["below_nominal"])],default=0)})
            for t in selected:
                miss = (~mask[:,t,target]).nonzero()[0]
                # Record up to 10 largest physical miss distances at each cell.
                distances = np.maximum(result["lower"][:,t,target]-test["truth"][:,t,target],0).astype(np.float64)+np.maximum(test["truth"][:,t,target]-result["upper"][:,t,target],0).astype(np.float64)
                for i in miss[np.argsort(-distances[miss],kind="stable")[:10]]:
                    examples.append({"method":method,"target":name,"horizon_index":int(t),"profile":profiles[i],
                        "truth_physical":float(test["truth_physical"][i,t,target]),
                        "scaled_truth":float(test["truth"][i,t,target]),"scaled_lower":float(result["lower"][i,t,target]),
                        "scaled_upper":float(result["upper"][i,t,target]),"miss_distance_physical":float(distances[i]*span[target]),
                        "control_degrees":float(controls[i,t]),"control_change_degrees":float(changes[i,t]),
                        "control_regime":"initial" if t==0 else "control_changing" if regimes["control_changing"][i,t] else "control_holding"})
            for regime,selection in regimes.items():
                count = int(selection.sum())
                if not count:
                    continue
                regime_rows.append({"method":method,"target":name,"control_regime":regime,
                    "profile_count":int(selection.any(axis=1).sum()),"point_count":count,
                    "pointwise_coverage":float(mask[:,:,target][selection].mean()),
                    "mean_physical_full_width":float(result["width_physical"][:,:,target][selection].mean()),
                    "mean_physical_interval_score":float(result["score_scaled"][:,:,target][selection].mean()*span[target]),
                    "mean_physical_abs_error":float(residual[:,:,target][selection].mean()*span[target])})
        # The next evaluation does not retain full interval arrays.
        if method=="mace":
            del result
    write_csv(out/"horizon_diagnostics.csv",cells)
    write_csv(out/"horizon_target_summary.csv",summaries)
    # Empty examples/regions can occur in small fixtures or perfectly covered data.
    for filename,rows,columns in (
        ("low_coverage_regions.csv",regions,["method","target","criterion","start_horizon_index","end_horizon_index","length","minimum_coverage","mean_coverage"]),
        ("worst_horizon_miss_examples.csv",examples,["method","target","horizon_index","profile","truth_physical","scaled_truth","scaled_lower","scaled_upper","miss_distance_physical","control_degrees","control_change_degrees","control_regime"]),
    ):
        if rows:
            write_csv(out/filename,rows)
        else:
            (out/filename).write_text(",".join(columns)+"\n",encoding="utf-8")
    write_csv(out/"control_regime_metrics.csv",regime_rows)
    return masks


def plot_horizons(out,masks,names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True,sharex=True)
    for ax,method in zip(axes,("absolute","mace")):
        im=ax.imshow(masks[method].T,aspect="auto",origin="lower",vmin=.90,vmax=1.,cmap="viridis")
        ax.set_yticks(range(len(names)),names)
        ax.set_title(f"{method}: pointwise test coverage at each target and horizon")
        fig.colorbar(im,ax=ax,label="Coverage (color scale clipped at 0.90)")
    axes[-1].set_xlabel("Horizon index")
    fig.savefig(out/"horizon_coverage.png",dpi=180)
    fig.savefig(out/"horizon_coverage.pdf")
    plt.close(fig)
