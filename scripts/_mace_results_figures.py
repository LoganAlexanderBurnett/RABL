"""Read-only manuscript Results figure renderers.

Use archived, validated outputs. No training, simulation, inference, recalibration,
or test-set-dependent threshold/floor tuning. Selected profile identities are saved.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _mace_floor_study import bounds, fingerprint
from _mace_scaler_audit import decode_names, read_scaler
from _mace_operational_profiles import FAMILIES, RECIPE

FLOOR = 1e-3
TIME = np.round(np.linspace(0, 200, 501), 8)
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
METHODS = (("mace", "horizon"), ("mace", "trajectory"),
           ("absolute", "horizon"), ("absolute", "trajectory"))
METHOD_LABELS = ("MACE-Horizon", "MACE-Trajectory",
                 "Absolute CP-Horizon", "Absolute CP-Trajectory")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verified(path, expected=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    info = fingerprint(path)
    if expected is not None and info["sha256"] != expected:
        raise ValueError(f"Saved scientific input does not match completed metadata: {path}")
    return info


def _settings(config, path):
    s = config["results"]
    base = {k: path(s[k]) for k in ("stochastic_forecasts_dir", "stochastic_floor_dir",
                                   "stochastic_validation_dir", "operational_dir", "scaler_file")}
    threshold = float(s["minimum_unsaturated_fraction"])
    if not 0 < threshold <= 1:
        raise ValueError("minimum_unsaturated_fraction must be in (0,1]")
    return s, base, threshold


def _stochastic(config, path, *, require_validation=True):
    s, base, threshold = _settings(config, path)
    forecasts = base["stochastic_forecasts_dir"]
    study = base["stochastic_floor_dir"]
    fm_path = forecasts/"forecast_metadata.json"
    sm_path = study/"study_metadata.json"
    fm, sm = _json(fm_path), _json(sm_path)
    if not fm.get("completed") or not sm.get("completed"):
        raise ValueError("Corrected stochastic forecast or calibration study incomplete")
    if not sm.get("corrected_forecasts", {}).get("forecast_hashes_match"):
        raise ValueError("Not a verified corrected-recurrence stochastic floor study")
    names = fm["target_names"]
    ids = fm["profile_names"]["test"]
    if (len(names) != 15 or len(set(names)) != 15 or len(ids) != 2000
            or ids != sm["profile_names"]["test"] or names != sm["target_names"]
            or sm["config"]["corrected_forecast_metadata_path"] != s["stochastic_forecasts_dir"]+"/forecast_metadata.json"):
        raise ValueError("Stochastic split, methods, or provenance mismatch")
    test_path = forecasts/"test_ensemble_forecasts_audit.h5"
    inputs = [_verified(fm_path), _verified(sm_path), _verified(test_path, fm["output_hashes"][test_path.name]["sha256"])]
    quant_path = study/"recalibration.h5"
    inputs.append(_verified(quant_path))
    if require_validation:
        vm_path = base["stochastic_validation_dir"]/"validation_metadata.json"
        vm = _json(vm_path)
        if (not vm.get("completed") or vm["profile_names"]["test"] != ids
                or vm["target_names"] != names or vm.get("floor") != FLOOR):
            raise ValueError("Completed corrected stochastic profile-validation archive required")
        inputs.append(_verified(vm_path))
    scaler = base["scaler_file"]
    scaler_hash = sm["input_hashes"]["external_scaled_h5_path"]["sha256"]
    inputs.append(_verified(scaler, scaler_hash))
    stats, scaler_meta = read_scaler(scaler)
    if scaler_meta["target_names"] != names:
        raise ValueError("Scaled output coordinate order differs from forecast")
    return {"paths": base, "files": {"test": test_path, "quant": quant_path},
            "names": names, "ids": ids, "stats": stats, "inputs": inputs, "threshold": threshold}


def _operational(config, path):
    s, base, threshold = _settings(config, path)
    root = base["operational_dir"]
    locations = {"manifest": root/"operational_manifest.json",
                 "forecast_meta": root/"forecasts/forecast_metadata.json",
                 "eval_meta": root/"evaluation/evaluation_metadata.json",
                 "test": root/"forecasts/operational_test.h5",
                 "intervals": root/"evaluation/intervals_and_statistics.h5"}
    manifest, fm, em = (_json(locations[k]) for k in ("manifest", "forecast_meta", "eval_meta"))
    if (manifest.get("recipe") != RECIPE or not fm.get("completed") or not em.get("completed")
            or fm.get("feedback") != "native-Y-to-X-affine-float32-v1"
            or fm.get("manifest_identity") != manifest.get("identity")
            or em.get("manifest_identity") != manifest.get("identity")):
        raise ValueError("Operational archive incomplete, incompatible, or non-retained recipe")
    ids = [p["id"] for p in manifest["profiles"] if p["split"] == "test"]
    if ids != em["test_profile_names"] or len(ids) != 2000:
        raise ValueError("Operational test ID ordering/count mismatch")
    names = fm["target_names"]
    if len(names) != 15 or names != em["target_names"]:
        raise ValueError("Operational target identity mismatch")
    inputs = [_verified(locations[k]) for k in ("manifest", "forecast_meta", "eval_meta")]
    inputs.append(_verified(locations["test"], fm["output_hashes"]["operational_test.h5"]["sha256"]))
    inputs.append(_verified(locations["intervals"]))
    stats, scale_meta = read_scaler(base["scaler_file"])
    if scale_meta["target_names"] != names:
        raise ValueError("Operational scaler output order mismatch")
    inputs.append(_verified(base["scaler_file"], fm["input_hashes"]["external_scaled_h5_path"]["sha256"]))
    return {"paths": base, "files": locations, "manifest": manifest, "ids": ids,
            "names": names, "stats": stats, "inputs": inputs, "threshold": threshold}


def unsaturated_fraction(truth, quality_column):
    """Reference truth criterion only; exact x<1 at archived native precision."""
    a = np.asarray(truth)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError("Invalid physical reference trajectory")
    return float(np.mean(a[:, quality_column] < 1.0))


def choose_profile(candidates, threshold):
    """Deterministic middle-ID eligible profile; never rank by errors/coverage."""
    valid = sorted(((name, frac) for name, frac in candidates if frac >= threshold), key=lambda x: x[0])
    if not valid:
        raise ValueError(f"No reference profile qualifies with x_steam,out<1 for >= {threshold:.1%}")
    return valid[len(valid)//2], len(valid)


def _style():
    from _mace_paper_figures import STYLE
    return {**STYLE, "font.size": 8, "legend.fontsize": 7}


def _save(fig, out, stem, options):
    from _mace_paper_figures import save_figure
    return save_figure(fig, out, stem, options)


def _display():
    from _mace_paper_figures import DISPLAY, LABELS
    return DISPLAY, LABELS


def _physical_index(key, names):
    if key == "P_MW":
        return names.index("n"), 6.0
    return names.index(key), 1.0


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _unique(rows, **where):
    matched = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in where.items())]
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one saved metric row {where}; got {len(matched)}")
    return matched[0]


def stochastic_representative(config, out, path, options):
    """Figure 6: 4x4 in exactly the Figure-5 display order (control + 15)."""
    d = _stochastic(config, path)
    names, ids, stats = d["names"], d["ids"], d["stats"]
    quality = names.index("x_steam_out")
    with h5py.File(d["files"]["test"], "r") as f:
        if not f.attrs.get("completed") or decode_names(f.attrs["target_names"]) != names or set(f.keys()) != set(ids):
            raise ValueError("Incomplete or changed stochastic test forecast file")
        candidates = [(pid, unsaturated_fraction(f[pid]["y_true_physical"][()], quality)) for pid in ids]
        (chosen, fraction), eligible = choose_profile(candidates, d["threshold"])
        g = f[chosen]
        mean = g["ensemble_mean_scaled"][()]
        spread = g["ensemble_std_scaled"][()]
        truth = g["y_true_physical"][()]
        control = g["control"][()]
        if (mean.shape != (501, 15) or spread.shape != mean.shape
                or truth.shape != mean.shape or control.shape != (501,)):
            raise ValueError("Stochastic representative has unexpected dimensions")
        with h5py.File(d["files"]["quant"], "r") as qf:
            if decode_names(qf.attrs["target_names"]) != names or qf.attrs["alpha"] != .05:
                raise ValueError("Stochastic calibration coordinate/nominal mismatch")
            bands = {}
            for method, mode in METHODS:
                key = "absolute" if method == "absolute" else "floor_0.001"
                q = qf[f"{key}/{mode}"][()]
                lo, hi = bounds({"mean": mean, "spread": spread}, q,
                                None if method == "absolute" else FLOOR)
                bands[(method, mode)] = (lo.astype(float)*stats["y"]["span"]+stats["y"]["min"],
                                          hi.astype(float)*stats["y"]["span"]+stats["y"]["min"])
    display, labels = _display()
    colors = ("#009E73", "#CC79A7", "#0072B2", "#D55E00")
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(4, 4, figsize=(190/25.4, 168/25.4),
                                 sharex=True, layout="constrained")
        for ax, key, label in zip(axes.flat, display, labels):
            ax.set_title(label, fontsize=8, pad=3)
            if key == "drumAngleDeg":
                ax.plot(TIME, control, color="black", lw=1.0)
            else:
                j, factor = _physical_index(key, names)
                for (method, mode), c in zip(METHODS, colors):
                    lo, hi = bands[(method, mode)]
                    ax.fill_between(TIME, lo[:, j]*factor, hi[:, j]*factor,
                                    color=c, alpha=.055, linewidth=0)
                    ax.plot(TIME, lo[:, j]*factor, color=c, alpha=.75, linewidth=.55)
                    ax.plot(TIME, hi[:, j]*factor, color=c, alpha=.75, linewidth=.55)
                ax.plot(TIME, truth[:, j]*factor, color="black", lw=1.05, zorder=5)
                physical_mean = mean[:, j].astype(float)*stats["y"]["span"][j]+stats["y"]["min"][j]
                ax.plot(TIME, physical_mean*factor, color="#555555", lw=1.0, ls="--", zorder=6)
            ax.set_xlim(0, 200)
            ax.set_xticks([0, 100, 200])
            ax.tick_params(labelsize=6)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(alpha=.15)
        handles = [Line2D([], [], color="black", label="Simulator"),
                   Line2D([], [], color="#555555", ls="--", label="Ensemble mean")]
        handles += [Line2D([], [], color=c, label=label) for c, label in zip(colors, METHOD_LABELS)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.03),
                   ncol=3, frameon=False)
        fig.supxlabel("Time (s)", fontsize=8)
        files = _save(fig, out, "fig06_stochastic_representative", options)
    return {"files": files, "inputs": d["inputs"], "selected_profile_id": chosen,
            "selection_fraction": fraction, "eligible_profiles": eligible,
            "selection_rule": "Middle sorted ID meeting reference x_steam,out<1 fraction threshold",
            "caption": "Corrected autoregressive ensemble forecast for one stochastic evaluation trajectory. "
                       "Simulator truth and ensemble mean are compared with 95% nominal MACE-Horizon, "
                       "MACE-Trajectory, Absolute CP-Horizon and Absolute CP-Trajectory bands, "
                       "all calibrated on the stochastic calibration set. The prescribed drum angle "
                       "occupies the first panel; power equals six times normalized neutron population. "
                       "Bands are not physically clipped. The trajectory was selected by the recorded "
                       "reference steam-quality criterion, not its forecast error."}


def stochastic_uq_performance(config, out, path, options):
    """Figure 7: nominal coverage by calibration objective plus width and score."""
    d = _stochastic(config, path)
    study = d["paths"]["stochastic_floor_dir"]
    metric_files = [study/k for k in ("aggregate_metrics.csv", "target_metrics.csv",
                                     "horizon_target_metrics.csv")]
    inputs = d["inputs"] + [_verified(p) for p in metric_files]
    aggregate, target, horizon = [_read_rows(p) for p in metric_files]
    def selected(rows, method, mode, **other):
        return [r for r in rows if r["method"] == method and r["mode"] == mode
                and ((r["floor"] == "0.001") if method == "mace" else r["floor"] in ("", "None"))
                and all(r[k] == str(v) for k, v in other.items())]
    if any(len(selected(aggregate,m,t)) != 1 for m,t in METHODS):
        raise ValueError("Missing/duplicated corrected floor .001 or absolute aggregates")
    # Mean across all 15 targets at each horizon (not independent observations).
    curve = {}
    for method in ("mace", "absolute"):
        rs = selected(horizon, method, "horizon")
        if len(rs) != 501*15:
            raise ValueError("Incomplete per-horizon coverage coordinates")
        by_h = {}
        for r in rs:
            by_h.setdefault(int(r["horizon_index"]), []).append(float(r["coverage"]))
        curve[method] = [np.mean(by_h[h]) for h in range(501)]
    trajectory = {}
    for method in ("mace", "absolute"):
        rs = selected(target, method, "trajectory")
        if {r["target"] for r in rs} != set(d["names"]):
            raise ValueError("Incomplete per-output trajectory summaries")
        trajectory[method] = [float(_unique(rs, target=n)["objective_coverage"]) for n in d["names"]]
    order = [("absolute", "horizon"), ("mace", "horizon"),
             ("absolute", "trajectory"), ("mace", "trajectory")]
    def agg(method, mode, col):
        return float(_unique(selected(aggregate, method, mode), method=method, mode=mode)[col])
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 2, figsize=(178/25.4, 135/25.4), layout="constrained")
        ax = axes[0,0]
        for meth, c in (("absolute", PALETTE[0]), ("mace", PALETTE[2])):
            ax.plot(TIME, curve[meth], color=c, lw=1.3,
                    label="Absolute CP-Horizon" if meth == "absolute" else "MACE-Horizon")
        ax.axhline(.95, color="gray", ls=":", lw=.9)
        ax.set(title="(a) Horizon coverage", xlabel="Forecast time (s)", ylabel="Mean pointwise coverage", ylim=(0,1.02))
        ax.legend(frameon=False, fontsize=6)
        ax = axes[0,1]
        x = np.arange(15)
        ax.plot(x, trajectory["absolute"], "o", color=PALETTE[0], ms=3,
                label="Absolute CP-Trajectory")
        ax.plot(x, trajectory["mace"], "s", color=PALETTE[2], ms=3,
                label="MACE-Trajectory")
        ax.axhline(.95, color="gray", ls=":", lw=.9)
        ax.set(title="(b) Per-output trajectory coverage", xlabel="Output variable index",
               ylabel="Whole-trajectory coverage", ylim=(0,1.02), xticks=x,
               xticklabels=[str(i) for i in range(1,16)])
        ax.legend(frameon=False, fontsize=6)
        labels = ["Abs-H", "MACE-H", "Abs-T", "MACE-T"]
        for ax, key, title, ylabel in (
                (axes[1,0], "mean_scaled_full_width", "(c) Mean full interval width", "Normalized width"),
                (axes[1,1], "mean_scaled_interval_score", "(d) Mean interval score", "Normalized interval score")):
            ax.bar(np.arange(4), [agg(m,t,key) for m,t in order],
                   color=[PALETTE[0],PALETTE[2],PALETTE[0],PALETTE[2]], alpha=.84)
            ax.set(title=title, ylabel=ylabel, xticks=np.arange(4), xticklabels=labels)
        for ax in axes.flat: ax.grid(alpha=.14, axis="y")
        files = _save(fig, out, "fig07_stochastic_uq_performance", options)
    return {"files": files, "inputs": inputs,
            "caption": "Corrected stochastic evaluation at nominal 95% coverage. "
                       "(a) Mean per-coordinate Horizon coverage over output variables at each time; "
                       "(b) independent per-output full-trajectory coverage across the 15 outputs; "
                       "(c) mean normalized full interval width; and (d) mean normalized interval score. "
                       "Absolute and ensemble-adaptive methods are compared only at matching coverage objectives. "
                       "The horizontal gray line denotes nominal coverage, not a claim of simultaneous coverage."}


def operational_representatives(config, out, path, options):
    """Figure 8: four maneuver groups, trajectory colors, common line semantics."""
    d = _operational(config, path)
    names, ids = d["names"], d["ids"]
    quality = names.index("x_steam_out")
    candidates = {}
    with h5py.File(d["files"]["test"], "r") as f:
        if (not f.attrs.get("completed") or decode_names(f.attrs["target_names"]) != names
                or f.attrs.get("manifest_identity") != d["manifest"]["identity"]):
            raise ValueError("Operational forecasts do not match saved study")
        for p in d["manifest"]["profiles"]:
            if p["split"] != "test": continue
            g = f[p["id"]]
            if g.attrs.get("family") != p["family"] or g.attrs.get("split") != "test":
                raise ValueError("Operational family mismatch")
            candidates.setdefault(p["family"], []).append(
                (p["id"], unsaturated_fraction(g["truth_physical"][()], quality)))
    selected, eligible = {}, {}
    for family in FAMILIES:
        selected[family], eligible[family] = choose_profile(candidates.get(family, []), d["threshold"])
    with h5py.File(d["files"]["test"], "r") as f, h5py.File(d["files"]["intervals"], "r") as iq:
        if decode_names(iq.attrs["test_profile_names"]) != ids:
            raise ValueError("Operational interval profile ordering mismatch")
        panels = []
        for family in FAMILIES:
            pid, frac = selected[family]
            g = f[pid]
            idx = ids.index(pid)
            mean = g["mean"][()]
            truth = g["truth_physical"][()]
            angle = g["control_physical"][()]
            lower = iq["intervals/horizon/operational/mace/lower"][idx]
            upper = iq["intervals/horizon/operational/mace/upper"][idx]
            mask = iq["intervals/horizon/operational/mace/mask"][idx]
            if (mean.shape != (501,15) or truth.shape != mean.shape
                    or angle.shape != (501,) or lower.shape != mean.shape
                    or upper.shape != mean.shape):
                raise ValueError("Operational representative has unexpected dimensions")
            physical = lambda z: z.astype(float)*d["stats"]["y"]["span"]+d["stats"]["y"]["min"]
            if not np.array_equal(mask, (lower<=g["truth"][()]) & (g["truth"][()]<=upper)):
                raise ValueError("Operational saved band/coverage mismatch")
            panels.append((family, angle, truth, physical(mean), physical(lower), physical(upper)))
    with plt.rc_context(_style()):
        fig, axs = plt.subplots(2, 2, figsize=(178/25.4, 118/25.4), sharex=True, layout="constrained")
        specs = [("drum", r"$\theta$ (degree)"),
                 ("Tf", r"$T_f$ (K)"),
                 ("n", r"$P$ (MW)"),
                 ("rho_dollars", r"$\rho_{\$}$ (dollars)")]
        for ax, (key, label) in zip(axs.flat, specs):
            ax.set_title(label)
            for (family, angle, truth, mean, lo, hi), color in zip(panels, PALETTE):
                if key == "drum":
                    ax.plot(TIME, angle, color=color, lw=1.2)
                else:
                    j, scale = _physical_index("P_MW" if key=="n" else key, names)
                    ax.fill_between(TIME, lo[:,j]*scale, hi[:,j]*scale, color=color, alpha=.11, lw=0)
                    ax.plot(TIME, truth[:,j]*scale, color=color, lw=1.25)
                    ax.plot(TIME, mean[:,j]*scale, color=color, lw=1.0, ls="--")
            ax.set_xlim(0,200); ax.set_xticks([0,100,200])
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(alpha=.15)
        legend = [Line2D([], [], color=c, lw=1.8, label=label) for c,label in zip(
            PALETTE, ("Single maneuver + hold", "Successive maneuvers + holds",
                      "Maneuver + return", "Bidirectional + return"))]
        legend += [Line2D([], [], color="gray", label="Simulator"),
                   Line2D([], [], color="gray", ls="--", label="Ensemble mean"),
                   Patch(facecolor="gray", alpha=.15, label="Operational MACE-Horizon")]
        fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(.5,1.12), fontsize=6.5)
        fig.supxlabel("Time (s)")
        files = _save(fig, out, "fig08_operational_representatives", options)
    return {"files": files, "inputs": d["inputs"],
            "selected_profile_ids": {k:v[0] for k,v in selected.items()},
            "selected_unsaturated_fractions": {k:v[1] for k,v in selected.items()},
            "eligible_profile_counts": eligible,
            "selection_rule": "Middle sorted eligible ID within each saved test family; truth x_steam,out<1",
            "caption": "One independently selected simulator trajectory from each of four operation-informed "
                       "maneuver families. Color identifies the maneuver family in every panel; solid curves "
                       "show simulator truth and dashed curves show ensemble mean. Shading denotes only "
                       "MACE-Horizon 95% nominal intervals calibrated on independent operation-informed "
                       "calibration profiles. Power is six times normalized neutron population. "
                       "All selected reference trajectories spend the configured fraction of the record "
                       "with outlet steam quality below unity."}


def spread_error_relationship(config, out, path, options):
    """Figure 9: bin by saved stochastic ensemble spread, never fit on test errors."""
    d = _stochastic(config, path)
    ids = d["ids"]
    spread_path = d["files"]["test"]
    # A deterministic, ID-stratified subsample defines the plotting bins without
    # optimizing bins for observed forecast errors. All 15 million observations
    # contribute to bin-wise means in a second streaming pass.
    sampled = []
    with h5py.File(spread_path, "r") as f:
        for pid in ids[::max(1,len(ids)//64)]:
            sampled.append(f[pid]["ensemble_std_scaled"][()][::20].ravel())
    edges = np.quantile(np.concatenate(sampled), np.linspace(0,1,11))
    edges = np.unique(edges)
    if len(edges)<3:
        raise ValueError("Ensemble spread cannot define informative bins")
    edges[0] = -np.inf; edges[-1] = np.inf
    n = len(edges)-1
    count = np.zeros(n, dtype=np.int64)
    ss = np.zeros(n); error = np.zeros(n)
    width_m = np.zeros(n); width_a = np.zeros(n)
    with h5py.File(d["files"]["quant"], "r") as qf:
        q_m = qf["floor_0.001/horizon"][()]
        q_a = qf["absolute/horizon"][()]
    with h5py.File(spread_path, "r") as f:
        for pid in ids:
            g = f[pid]
            spread = g["ensemble_std_scaled"][()]
            mean = g["ensemble_mean_scaled"][()]
            truth = g["y_true_scaled"][()]
            bins = np.searchsorted(edges, spread.ravel(), side="right")-1
            bins = np.clip(bins, 0, n-1)
            count += np.bincount(bins, minlength=n)
            ss += np.bincount(bins, weights=spread.ravel(), minlength=n)
            error += np.bincount(bins, weights=np.abs(truth-mean).ravel(), minlength=n)
            bm_lo, bm_hi = bounds({"mean":mean,"spread":spread},q_m,FLOOR)
            ba_lo, ba_hi = bounds({"mean":mean,"spread":spread},q_a,None)
            width_m += np.bincount(bins, weights=(bm_hi-bm_lo).ravel(), minlength=n)
            width_a += np.bincount(bins, weights=(ba_hi-ba_lo).ravel(), minlength=n)
    if np.any(count == 0):
        raise ValueError("Empty ensemble-spread bin; lower bin count or inspect input")
    x = ss/count
    observed = error/count
    mace_w = width_m/count
    abs_w = width_a/count
    if not np.all(np.isfinite(np.r_[x,observed,mace_w,abs_w])):
        raise ValueError("Non-finite spread/error bin results")
    with plt.rc_context(_style()):
        fig, axs = plt.subplots(1,2,figsize=(178/25.4,77/25.4),layout="constrained")
        axs[0].plot(x, observed, "o-", color=PALETTE[2], lw=1.25, ms=4)
        axs[0].set(title="(a) Spread versus forecast error",
                   xlabel="Mean ensemble spread (scaled)", ylabel="Mean absolute error (scaled)")
        axs[1].plot(x, mace_w, "s-", label="MACE-Horizon", color=PALETTE[2], lw=1.25, ms=4)
        axs[1].plot(x, abs_w, "o--", label="Absolute CP-Horizon", color=PALETTE[0], lw=1.1, ms=4)
        axs[1].set(title="(b) Allocated interval width",
                   xlabel="Mean ensemble spread (scaled)", ylabel="Mean full interval width (scaled)")
        axs[1].legend(frameon=False)
        for ax in axs: ax.grid(alpha=.2)
        files = _save(fig,out,"fig09_spread_error_relationship",options)
    # Save the numerical points for independent review.
    out_file = out/"spread_error_bins.csv"
    with out_file.open("w",newline="",encoding="utf-8") as fd:
        writer = csv.writer(fd)
        writer.writerow(("bin","count","mean_spread","mean_absolute_error",
                         "mean_mace_horizon_width","mean_absolute_cp_horizon_width"))
        for i in range(n):
            writer.writerow((i,int(count[i]),x[i],observed[i],mace_w[i],abs_w[i]))
    files.append(out_file.name)
    return {"files":files,"inputs":d["inputs"],
            "bin_counts":count.tolist(),"binning":"Ten sampled-spread quantile bins; deterministic test-ID/time subsample, all observations aggregated",
            "caption":"Binned association between spread across ten independently trained autoregressive "
                      "LSTM forecasts and absolute prediction errors on the stochastic test set. "
                      "Bins are defined only from ensemble spread, not from residuals. "
                      "Normalized absolute error and mean full interval widths are pooled over "
                      "profiles, horizons, and output variables. Horizon-calibrated MACE "
                      "and absolute CP widths are compared within common spread bins; "
                      "no conditional-coverage guarantee is implied."}
