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
from matplotlib import patheffects as pe

from _mace_plot_units import plot_index_and_scale, POWER_MW_PER_N

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
    count = s.get("representative_count", 10)
    seed = s.get("representative_seed", 20261016)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("representative_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("representative_seed must be a nonnegative integer")
    return s, base, {"count": count, "seed": seed}


def _stochastic(config, path, *, require_validation=True):
    s, base, selection = _settings(config, path)
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
            "names": names, "ids": ids, "stats": stats, "inputs": inputs,
            "representative_count": selection["count"], "representative_seed": selection["seed"]}


def _operational(config, path):
    s, base, selection = _settings(config, path)
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
            "names": names, "stats": stats, "inputs": inputs,
            "representative_count": selection["count"], "representative_seed": selection["seed"]}


def unsaturated_fraction(truth, quality_column):
    """Reference truth criterion only; exact x<1 at archived native precision."""
    a = np.asarray(truth)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError("Invalid physical reference trajectory")
    return float(np.mean(a[:, quality_column] < 1.0))


def choose_profiles(candidates, count=10, seed=20261016, *, require_unsaturated=True):
    """Uniformly sample qualifying reference profiles without replacement.

    A profile qualifies if the simulator has x_steam,out < 1 at *any* time.
    Selection uses a recorded PCG64 seed and never examines forecast residuals
    or conformal coverage. Returned IDs are sorted for stable output numbering.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("representative_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("representative_seed must be a nonnegative integer")
    ids = [name for name, _ in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate representative profile IDs")
    if any(not np.isfinite(frac) or not 0 <= frac <= 1 for _, frac in candidates):
        raise ValueError("Invalid reference steam-quality fraction")
    if not isinstance(require_unsaturated, bool):
        raise ValueError("require_unsaturated must be boolean")
    eligible = sorted(((name, frac) for name, frac in candidates
                       if frac > 0 or not require_unsaturated),
                      key=lambda item: item[0])
    if len(eligible) < count:
        raise ValueError(
            f"Requested {count} representative profiles "
            + ("with reference x_steam,out < 1 at least once, " if require_unsaturated
               else "from the full saved group, ")
            + f"but only {len(eligible)} qualify. "
            "Choose a smaller --representative-count.")
    selected_indices = np.sort(np.random.default_rng(seed).choice(
        len(eligible), size=count, replace=False))
    return [eligible[int(i)] for i in selected_indices], len(eligible)

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
    return plot_index_and_scale(key, names)


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _unique(rows, **where):
    matched = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in where.items())]
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one saved metric row {where}; got {len(matched)}")
    return matched[0]


def verify_representative_widths(mean, spread, quantiles, stats, names):
    """Numerically audit EVERY plotted conformal band in all 501 x 15 cells.

    Calibrated quantiles are fixed. Compute mathematical endpoints in float64
    to prevent accidental loss of very thin intervals from float32 endpoint
    subtraction; compare rendered widths to the independent analytic 2*q*scale
    in physical units. Also diagnose the native float32 endpoint quantization
    without changing saved scientific evaluation arrays.
    """
    mean32, spread32 = np.asarray(mean), np.asarray(spread)
    if mean32.shape != (501, len(names)) or spread32.shape != mean32.shape:
        raise ValueError("Unexpected stochastic forecast coordinates in interval audit")
    if not (np.isfinite(mean32).all() and np.isfinite(spread32).all()
            and np.all(spread32 >= 0)):
        raise ValueError("Invalid ensemble mean/spread in interval audit")
    yspan = np.asarray(stats["y"]["span"], dtype=np.float64)
    ymin = np.asarray(stats["y"]["min"], dtype=np.float64)
    if (yspan.shape != (len(names),) or ymin.shape != yspan.shape
            or not np.all(np.isfinite(yspan)) or np.any(yspan <= 0)
            or not np.all(np.isfinite(ymin))):
        raise ValueError("Invalid output scaler in interval audit")
    mean64, spread64 = mean32.astype(np.float64), spread32.astype(np.float64)
    center = mean64*yspan + ymin
    bands, rows = {}, []
    for method, mode in METHODS:
        q = np.asarray(quantiles[(method, mode)], dtype=np.float64)
        expected_shape = (501, len(names)) if mode == "horizon" else (len(names),)
        if q.shape != expected_shape or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError(f"Invalid {method}/{mode} calibrated quantiles: {q.shape}")
        scale = spread64 + FLOOR if method == "mace" else np.ones_like(spread64)
        half_scaled = q*scale
        half_physical = half_scaled*yspan
        lower, upper = center-half_physical, center+half_physical
        expected_full = 2*half_physical
        shown_full = upper-lower
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(shown_full < 0)):
            raise ValueError(f"Nonfinite or inverted plotted bounds: {method}/{mode}")
        # Physical-unit cancellation should be much smaller than a meaningful
        # manuscript band. Reject it explicitly rather than rendering a zero.
        allowed = np.maximum(2e-11*np.maximum(1, np.abs(center)), 1e-5*expected_full)
        nonzero = expected_full > 0
        if (np.any(np.abs(shown_full-expected_full) > allowed)
                or np.any(nonzero & (np.abs(shown_full-expected_full)
                                     > .05*expected_full))):
            raise ValueError(f"Plotted endpoints suppress calibrated interval width: {method}/{mode}")
        # The earlier float32 endpoint pathway is audited separately. If that
        # route loses width, the float64 plotted bands remain mathematically
        # faithful to the SAME saved conformal quantiles.
        old_lo, old_hi = bounds({"mean": mean32, "spread": spread32},
                                quantiles[(method, mode)],
                                None if method == "absolute" else FLOOR)
        old_full = (old_hi.astype(np.float64)-old_lo.astype(np.float64))*yspan
        positive = expected_full > 0
        for j, name in enumerate(names):
            mask = positive[:,j]
            reference = expected_full[:,j]
            observed = old_full[:,j]
            # Tiny expected bands at floating-point resolution are explicitly
            # tracked instead of silently being assumed zero.
            frac_zero = float(np.mean((observed[mask] == 0))) if np.any(mask) else 0.
            frac_loss = float(np.mean((observed[mask] < .95*reference[mask]))) if np.any(mask) else 0.
            display_factor = POWER_MW_PER_N if name == "n" else 1.0
            expected_display = reference*display_factor
            actual_display = upper[:,j]*display_factor - lower[:,j]*display_factor
            if (np.any(np.abs(actual_display-expected_display) > allowed[:,j]*display_factor)
                    or np.any((expected_display > 0)
                              & (np.abs(actual_display-expected_display)
                                 > .05*expected_display))):
                raise ValueError(f"Plot-unit conversion suppressed width: {method}/{mode}/{name}")
            rows.append(dict(method=method, calibration=mode, target=name,
                             display_factor=display_factor,
                             expected_width_display_mean=float(np.mean(expected_display)),
                             plotted_width_display_mean=float(np.mean(actual_display)),
                             q_min=float(np.min(q[:,j] if q.ndim==2 else q[j])),
                             q_max=float(np.max(q[:,j] if q.ndim==2 else q[j])),
                             expected_width_physical_mean=float(np.mean(reference)),
                             expected_width_physical_min=float(np.min(reference)),
                             expected_width_physical_max=float(np.max(reference)),
                             plotted_width_physical_mean=float(np.mean(shown_full[:,j])),
                             old_float32_width_physical_mean=float(np.mean(observed)),
                             old_float32_zero_fraction_positive=frac_zero,
                             old_float32_loss_over_5pct_fraction=frac_loss))
        bands[(method, mode)] = (lower, upper)
    return bands, rows


def save_representative_width_audit(path, rows):
    if not rows:
        raise ValueError("No representative conformal-width audit rows")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stochastic_representative(config, out, path, options):
    """Figure 6: one publication-style 4x4 plot per selected stochastic profile.

    Output count is set by --representative-count (default 10). The same saved
    stochastic calibrations are used for every profile; selection uses truth only.
    """
    d = _stochastic(config, path)
    names, ids, stats = d["names"], d["ids"], d["stats"]
    quality = names.index("x_steam_out")
    count, seed = d["representative_count"], d["representative_seed"]
    files, details, audit_rows = [], [], []
    display, labels = _display()
    colors = ("#009E73", "#CC79A7", "#0072B2", "#D55E00")

    with h5py.File(d["files"]["quant"], "r") as qf:
        if decode_names(qf.attrs["target_names"]) != names or float(qf.attrs["alpha"]) != .05:
            raise ValueError("Stochastic calibration coordinate/nominal mismatch")
        quantiles = {}
        for method, mode in METHODS:
            key = "absolute" if method == "absolute" else "floor_0.001"
            quantiles[(method, mode)] = qf[f"{key}/{mode}"][()]

    with h5py.File(d["files"]["test"], "r") as f:
        if (not f.attrs.get("completed") or decode_names(f.attrs["target_names"]) != names
                or set(f.keys()) != set(ids)):
            raise ValueError("Incomplete or changed stochastic test forecast file")
        candidates = [(pid, unsaturated_fraction(f[pid]["y_true_physical"][()], quality))
                      for pid in ids]
        chosen, eligible = choose_profiles(candidates, count=count, seed=seed)
        for figure_number, (pid, fraction) in enumerate(chosen, start=1):
            g = f[pid]
            mean = g["ensemble_mean_scaled"][()]
            spread = g["ensemble_std_scaled"][()]
            truth = g["y_true_physical"][()]
            control = g["control"][()]
            if (mean.shape != (501, 15) or spread.shape != mean.shape
                    or truth.shape != mean.shape or control.shape != (501,)):
                raise ValueError(f"Stochastic representative dimensions differ: {pid}")
            bands, width_rows = verify_representative_widths(
                mean, spread, quantiles, stats, names)
            audit_rows.extend({"profile_id": pid, **row} for row in width_rows)

            with plt.rc_context(_style()):
                fig, axes = plt.subplots(4, 4, figsize=(190/25.4, 168/25.4),
                                         sharex=True)
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
                            ax.plot(TIME, lo[:, j]*factor, color=c, alpha=.75, linewidth=.36)
                            ax.plot(TIME, hi[:, j]*factor, color=c, alpha=.75, linewidth=.55)
                        ax.plot(TIME, truth[:, j]*factor, color="black", lw=1.05, zorder=5)
                        physical_mean = (mean[:, j].astype(float)*stats["y"]["span"][j]
                                         + stats["y"]["min"][j])
                        ax.plot(TIME, physical_mean*factor, color="#555555",
                                lw=.62, ls="--", zorder=6)
                    ax.set_xlim(0, 200)
                    ax.set_xticks([0, 100, 200])
                    ax.tick_params(labelsize=6)
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                    ax.grid(alpha=.15)
                handles = [Line2D([], [], color="black", label="Simulator"),
                           Line2D([], [], color="#555555", ls="--", label="Ensemble mean")]
                handles += [Line2D([], [], color=c, label=label)
                            for c, label in zip(colors, METHOD_LABELS)]
                fig.subplots_adjust(top=.895, bottom=.08, left=.08, right=.99,
                                    wspace=.40, hspace=.55)
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.995),
                           ncol=3, fontsize=7, frameon=False)
                fig.supxlabel("Time (s)", y=.025, fontsize=8)
                stem = f"fig07_stochastic_representative_{figure_number:02d}"
                generated = _save(fig, out, stem, options)
            files.extend(generated)
            caption = (
                f"Corrected autoregressive forecast for stochastic test profile {pid}. "
                f"The simulator reference has outlet steam quality strictly below unity "
                f"for {fraction:.1%} of saved forecast points. Simulator truth (solid black), "
                f"ensemble mean (dashed gray), and nominal 95% MACE-Horizon, "
                f"MACE-Trajectory, Absolute CP-Horizon and Absolute CP-Trajectory "
                f"bounds use a common stochastic calibration set. The prescribed "
                f"drum angle is shown in the first panel; P = 6n MW. "
                f"Uncertainty bounds are not clipped to physical limits.")
            caption_filename = f"{stem}.caption.md"
            (out/caption_filename).write_text(caption+"\n", encoding="utf-8")
            files.append(caption_filename)
            details.append({"number": figure_number, "profile_id": pid,
                            "unsaturated_fraction": fraction, "figure_stem": stem,
                            "caption_file": caption_filename})
            print(f"Stochastic representative {figure_number}/{count}: {pid} "
                  f"(reference x_steam,out<1: {fraction:.2%})", flush=True)
    audit_file = "fig07_representative_interval_width_audit.csv"
    save_representative_width_audit(out/audit_file, audit_rows)
    files.append(audit_file)
    return {
        "files": files, "inputs": d["inputs"], "requested_count": count,
        "generated_count": len(details), "representative_seed": seed,
        "eligible_profiles": eligible, "selected_profiles": details,
        "width_audit_file": audit_file,
        "width_audit_note": ("Independent analytic physical-unit checks for every "
                             "conformal method, selected profile, output and horizon; "
                             "float32 endpoint quantization is reported separately. "
                             "Displayed bands retain unchanged conformal quantiles "
                             "and are constructed in float64 for plotting."),
        "selection_rule": "Uniform seeded no-replacement sample among stochastic test profiles "
                          "with simulator-reference x_steam,out<1 at one or more saved time points; "
                          "no ranking by prediction residuals or coverage",
        "caption": (f"Series of {len(details)} 4-by-4 stochastic-test forecasts using the "
                    "same 95% nominal Horizon and Trajectory MACE and absolute-error CP "
                    "calibrations. All test profiles were selected solely by the presence "
                    "of simulator steam quality below unity. Per-figure captions and "
                    "selected IDs are included in numbered sidecars and metadata.")
    }

def stochastic_uq_performance(config, out, path, options):
    """Figure 8: nominal coverage by calibration objective plus width and score."""
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
        fig, axes = plt.subplots(2, 2, figsize=(178/25.4, 150/25.4), layout="constrained")
        ax = axes[0,0]
        for meth, c in (("absolute", PALETTE[0]), ("mace", PALETTE[2])):
            ax.plot(TIME, curve[meth], color=c, lw=1.3,
                    label="Absolute CP-Horizon" if meth == "absolute" else "MACE-Horizon")
        ax.axhline(.95, color="gray", ls=":", lw=.9)
        ax.set(title="(a) Horizon coverage", xlabel="Forecast time (s)", ylabel="Mean pointwise coverage", ylim=(.9,1.0))
        ax.legend(frameon=False, fontsize=6)
        ax = axes[0,1]
        x = np.arange(15)
        ax.plot(x, trajectory["absolute"], "o", color=PALETTE[0], ms=3,
                label="Absolute CP-Trajectory")
        ax.plot(x, trajectory["mace"], "s", color=PALETTE[2], ms=3,
                label="MACE-Trajectory")
        ax.axhline(.95, color="gray", ls=":", lw=.9)
        symbols = {
            "TN2": r"$T_{N_2}$", "Tm": r"$T_m$", "Thp": r"$T_{hp}$",
            "Tf": r"$T_f$", "Tsg": r"$T_{sg}$",
            "T_steam_out": r"$T_{\mathrm{steam,out}}$",
            "x_steam_out": r"$x_{\mathrm{steam,out}}$",
            "n": r"$P$", "rho_dollars": r"$\rho_{\$}$",
            **{f"c[{i}]": rf"$c_{i}$" for i in range(1,7)},
        }
        if any(n not in symbols for n in d["names"]):
            raise ValueError("Missing output-variable symbol in coverage panel")
        ax.set(title="(b) Per-output trajectory coverage", xlabel="Output variable",
               ylabel="Whole-trajectory coverage", ylim=(.9,1.0), xticks=x)
        ax.set_xticklabels([symbols[n] for n in d["names"]], rotation=55,
                           ha="right", rotation_mode="anchor", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6)
        labels = ["Abs-H", "MACE-H", "Abs-T", "MACE-T"]
        for ax, key, title, ylabel in (
                (axes[1,0], "mean_scaled_full_width", "(c) Mean full interval width", "Normalized width"),
                (axes[1,1], "mean_scaled_interval_score", "(d) Mean interval score", "Normalized interval score")):
            values = [agg(m,t,key) for m,t in order]
            bars = ax.bar(np.arange(4), values,
                          color=[PALETTE[0],PALETTE[2],PALETTE[0],PALETTE[2]], alpha=.84)
            for bar, val in zip(bars, values):
                ax.annotate(f"{val:.3f}", (bar.get_x()+bar.get_width()/2, val),
                            xytext=(0,3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6)
            ax.set_ylim(0, max(values)*1.18)
            ax.set(title=title, ylabel=ylabel, xticks=np.arange(4), xticklabels=labels)
        for ax in axes.flat: ax.grid(alpha=.14, axis="y")
        files = _save(fig, out, "fig08_stochastic_uq_performance", options)
    return {"files": files, "inputs": inputs,
            "caption": "Corrected stochastic evaluation at nominal 95% coverage. "
                       "(a) Mean per-coordinate Horizon coverage over output variables at each time; "
                       "(b) independent per-output full-trajectory coverage across the 15 outputs; "
                       "(c) mean normalized full interval width; and (d) mean normalized interval score. "
                       "Absolute and ensemble-adaptive methods are compared only at matching coverage objectives. "
                       "The horizontal gray line denotes nominal coverage, not a claim of simultaneous coverage."}


def operational_representatives(config, out, path, options):
    """Figure 9: one composite six-panel plot per selected set.

    Each subplot is color-coded by the operational maneuver family (B/C/D/F),
    not by output variable. Saved operational MACE-Horizon intervals only.
    """
    d = _operational(config, path)
    names, ids = d["names"], d["ids"]
    quality = names.index("x_steam_out")
    count, seed = d["representative_count"], d["representative_seed"]
    candidates = {}
    with h5py.File(d["files"]["test"], "r") as f:
        if (not f.attrs.get("completed") or decode_names(f.attrs["target_names"]) != names
                or f.attrs.get("manifest_identity") != d["manifest"]["identity"]):
            raise ValueError("Operational forecasts do not match saved study")
        for p in d["manifest"]["profiles"]:
            if p["split"] != "test":
                continue
            g = f[p["id"]]
            if g.attrs.get("family") != p["family"] or g.attrs.get("split") != "test":
                raise ValueError("Operational family mismatch")
            candidates.setdefault(p["family"], []).append(
                (p["id"], unsaturated_fraction(g["truth_physical"][()], quality)))
    selected, eligible = {}, {}
    for i, family in enumerate(FAMILIES):
        try:
            selected[family], eligible[family] = choose_profiles(
                candidates.get(family, []), count=count, seed=seed+i,
                require_unsaturated=(family != "F"))
        except ValueError as exc:
            raise ValueError(f"Operational family {family}: {exc}") from exc

    files, details = [], []
    with h5py.File(d["files"]["test"], "r") as f, h5py.File(
            d["files"]["intervals"], "r") as iq:
        if decode_names(iq.attrs["test_profile_names"]) != ids:
            raise ValueError("Operational interval profile ordering mismatch")
        profiles_index = {pid: i for i, pid in enumerate(ids)}
        expected = {
            "lower": iq["intervals/horizon/operational/mace/lower"],
            "upper": iq["intervals/horizon/operational/mace/upper"],
            "mask": iq["intervals/horizon/operational/mace/mask"],
        }
        for figure_number in range(1, count+1):
            panels, identities = [], {}
            for family in FAMILIES:
                pid, fraction = selected[family][figure_number-1]
                g = f[pid]
                idx = profiles_index[pid]
                mean = g["mean"][()]
                truth = g["truth_physical"][()]
                angle = g["control_physical"][()]
                lower = expected["lower"][idx]
                upper = expected["upper"][idx]
                mask = expected["mask"][idx]
                if (mean.shape != (501,15) or truth.shape != mean.shape
                        or angle.shape != (501,) or lower.shape != mean.shape
                        or upper.shape != mean.shape):
                    raise ValueError(f"Operational representative dimensions differ: {pid}")
                scaled_truth = g["truth"][()]
                if not np.array_equal(mask, (lower<=scaled_truth) & (scaled_truth<=upper)):
                    raise ValueError(f"Operational saved band/coverage mismatch: {pid}")
                def physical(z):
                    return z.astype(float)*d["stats"]["y"]["span"]+d["stats"]["y"]["min"]
                panels.append((family, angle, truth, physical(mean),
                               physical(lower), physical(upper)))
                identities[family] = {"profile_id": pid, "unsaturated_fraction": fraction}

            with plt.rc_context(_style()):
                fig, axs = plt.subplots(2, 3, figsize=(190/25.4, 135/25.4),
                                        sharex=True)
                # Display P [MW] = 6*n, never plot native dimensionless n
                # as though it were MW. The bounds use the same affine scale.
                specs = [
                    ("drum", r"$\theta$ (degree)"),
                    ("P_MW", r"$P$ (MW)"),
                    ("rho_dollars", r"$\rho_{\$}$ (dollars)"),
                    ("Tf", r"$T_f$ (K)"),
                    ("T_steam_out", r"$T_{\mathrm{steam,out}}$ (K)"),
                    ("x_steam_out", r"$x_{\mathrm{steam,out}}$ (–)"),
                ]
                for ax, (key, label) in zip(axs.flat, specs):
                    ax.set_title(label)
                    for (family, angle, truth, mean, lo, hi), color in zip(panels, PALETTE):
                        if key == "drum":
                            ax.plot(TIME, angle, color=color, lw=1.25)
                        else:
                            j, scale = _physical_index(key, names)
                            ax.fill_between(TIME, lo[:, j]*scale, hi[:, j]*scale,
                                            color=color, alpha=.10, lw=0)
                            ax.plot(TIME, truth[:, j]*scale, color=color, lw=1.3)
                            # Stroke follows each dash, giving a fine black
                            # outline without replacing the group color.
                            mean_line, = ax.plot(
                                TIME, mean[:, j]*scale, color=color,
                                lw=1.05, ls="--")
                            mean_line.set_path_effects([
                                pe.Stroke(linewidth=1.65, foreground="#242424"),
                                pe.Normal()])
                    ax.set_xlim(0, 200)
                    ax.set_xticks([0, 100, 200])
                    ax.tick_params(labelsize=6.5)
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                    ax.grid(alpha=.15)
                legend = [
                    Line2D([], [], color=c, lw=1.8, label=label)
                    for c, label in zip(
                        PALETTE,
                        ("Single maneuver and hold",
                         "Successive maneuvers and holds",
                         "Maneuver and return",
                         "Bidirectional maneuvers"))
                ]
                legend += [
                    Line2D([], [], color="#222222", label="Simulator"),
                    Line2D([], [], color="#777777", ls="--",
                           lw=1.1, label="Ensemble mean",
                           path_effects=[
                               pe.Stroke(linewidth=1.65, foreground="black"),
                               pe.Normal()]),
                    Patch(facecolor="gray", alpha=.15,
                          label="Operational MACE-Horizon"),
                ]
                fig.subplots_adjust(top=.76, bottom=.10, left=.075, right=.985,
                                    wspace=.42, hspace=.46)
                fig.legend(handles=legend, loc="upper center", ncol=2,
                           frameon=False, bbox_to_anchor=(.5,.995), fontsize=6.5)
                fig.supxlabel("Time (s)", y=.022)
                stem = f"fig09_operational_representatives_{figure_number:02d}"
                generated = _save(fig, out, stem, options)
            files.extend(generated)
            caption = (
                "Operational forecast comparison across four maneuver families, "
                + ", ".join(f"{family} ({identities[family]['profile_id']})"
                            for family in FAMILIES)
                + ". Colors identify families across every subplot. Simulator "
                "truth is solid, ensemble mean is dashed with fine black outlines, "
                "and shaded bands are "
                "only MACE-Horizon 95% nominal intervals calibrated using the "
                "independent operation-informed calibration set. P = 6n MW. "
                "Reference profiles in B/C/D have x_steam,out<1 at least once; "
                "F profiles are selected without that requirement.")
            caption_filename = f"{stem}.caption.md"
            (out/caption_filename).write_text(caption+"\n", encoding="utf-8")
            files.append(caption_filename)
            details.append({"number": figure_number, "profiles": identities,
                            "selection_require_unsaturated_by_family":
                                {f: f != "F" for f in FAMILIES},
                            "figure_stem": stem, "caption_file": caption_filename})
            print(f"Operational representative set {figure_number}/{count}: "
                  + ", ".join(f"{k}={identities[k]['profile_id']}" for k in FAMILIES),
                  flush=True)
    return {
        "files": files, "inputs": d["inputs"], "requested_count": count,
        "generated_count": len(details), "representative_seed": seed,
        "eligible_profile_counts": eligible, "selected_sets": details,
        "selection_rule": "Seeded no-replacement samples for B/C/D require simulator "
                          "reference x_steam,out<1 at least once; F samples from all its "
                          "test profiles because its saved population may never reach "
                          "unsaturated outlet conditions. Seeds differ by family index.",
        "selection_require_unsaturated_by_family": {f: (f != "F") for f in FAMILIES},
        "caption": (f"Series of {len(details)} 2-by-2 operational comparisons, each "
                    "containing one distinct representative from each maneuver type. "
                    "Panels show drum angle, P (MW), reactivity in dollars, "
                    "fuel temperature, outlet steam temperature and outlet steam quality. "
                    "Color distinguishes the group rather than the variable, "
                    "and only operationally calibrated MACE-Horizon prediction "
                    "intervals are shown. Families B/C/D require at least one "
                    "simulator reference x_steam,out<1 point; F is sampled from "
                    "all saved F test profiles. Individual IDs and captions "
                    "are saved in metadata and numbered caption files.")
    }

def spread_error_relationship(config, out, path, options):
    """Figure 10: bin by saved stochastic ensemble spread, never fit on test errors."""
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
        axs[0].plot(x, observed, "o-", color="#424242", lw=1.25, ms=4)
        axs[0].set(title="(a) Spread versus forecast error",
                   xlabel="Mean ensemble spread (scaled)", ylabel="Mean absolute error (scaled)")
        axs[1].plot(x, mace_w, "s-", label="MACE-Horizon", color=PALETTE[2], lw=1.25, ms=4)
        axs[1].plot(x, abs_w, "o--", label="Absolute CP-Horizon", color=PALETTE[0], lw=1.1, ms=4)
        axs[1].set(title="(b) Allocated interval width",
                   xlabel="Mean ensemble spread (scaled)", ylabel="Mean full interval width (scaled)")
        axs[1].legend(frameon=False)
        for ax in axs: ax.grid(alpha=.2)
        files = _save(fig,out,"fig10_spread_error_relationship",options)
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
