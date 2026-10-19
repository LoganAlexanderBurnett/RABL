"""Emit manuscript-ready Results Tables 5 and 6 directly from archived metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from _mace_results_figures import _operational, _stochastic, _unique, _read_rows, _verified
from _mace_floor_study import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]


def _path(p):
    import os
    p = Path(p)
    if p.is_absolute():
        return p.resolve()
    output_root = Path(os.environ.get("RABL_OUTPUT_ROOT") or ROOT/"outputs").expanduser().resolve()
    return output_root.joinpath(*p.parts[1:]) if p.parts[0] == "outputs" else (ROOT/p).resolve()


def _fmt(number, *, digits=4):
    number = float(number)
    if not np.isfinite(number):
        raise ValueError("Non-finite manuscript metric")
    return f"{number:.{digits}f}"


def _metric(rows, **where):
    return float(_unique(rows, **where)["estimate"])


def _ci(rows, **where):
    r = _unique(rows, **where)
    return f"{_fmt(r['estimate'])} [{_fmt(r['ci_lower'])}, {_fmt(r['ci_upper'])}]"


def _latex_table(columns, body, caption, label, *, wide=False):
    opener = "table*" if wide else "table"
    lines = [rf"\begin{{{opener}}}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + columns + "}", r"\hline"]
    lines.extend(body)
    lines.extend([r"\hline", r"\end{tabular}", rf"\end{{{opener}}}"])
    return "\n".join(lines) + "\n"



METHODS = (("mace", "horizon", "MACE-H"),
           ("absolute", "horizon", "Abs-H"),
           ("mace", "trajectory", "MACE-T"),
           ("absolute", "trajectory", "Abs-T"))
TARGET_SYMBOLS = {
    "TN2": r"$T_{N_2}$", "Tm": r"$T_m$", "Thp": r"$T_{hp}$",
    "Tf": r"$T_f$", "Tsg": r"$T_{sg}$",
    "T_steam_out": r"$T_{\mathrm{steam,out}}$",
    "x_steam_out": r"$x_{\mathrm{steam,out}}$",
    "n": r"$P$", "rho_dollars": r"$\rho_{\$}$",
    **{f"c[{j}]": rf"$c_{j}$" for j in range(1, 7)},
}
UQ_METRICS = ("objective_coverage", "mean_full_width", "mean_interval_score")
ROW_END = r" \\"

def _stochastic_row(rows, *, method, mode, target=None):
    matches = [r for r in rows if r.get("method") == method and r.get("mode") == mode
               and (target is None or r.get("target") == target)
               and (target is not None or "target" not in r)
               and ((float(r["floor"]) == .001) if method == "mace"
                    else r.get("floor") in ("", "None", None))]
    if len(matches) != 1:
        raise ValueError(f"Expected one stochastic row: {method}/{mode}/{target}; found {len(matches)}")
    return matches[0]

COLUMN_GROUPS = (
    ("thermal", "thermal nodes", ("TN2", "Tm", "Thp", "Tf", "Tsg")),
    ("precursor", "delayed precursors",
     ("c[1]", "c[2]", "c[3]", "c[4]", "c[5]", "c[6]")),
    ("system", "power, reactivity, and steam outputs",
     ("n", "rho_dollars", "T_steam_out", "x_steam_out")),
)
TABLE_METRIC_LABELS = ("Coverage", "Full width", "Interval score")


def _columns(group):
    return group[2] + (("aggregate",) if group[0] == "system" else ())


def _table(title, label, lines, *, fontsize=r"\footnotesize"):
    return "\n".join([
        r"\begin{table}[!htbp]", r"\centering", fontsize,
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\captionsetup{skip=3pt}",
        r"\caption{" + title + "}", r"\label{" + label + "}",
        *lines, r"\end{table}", "",
    ])


def _table_accuracy(group, *, point, aggregate):
    name, description, _ = group
    columns = _columns(group)
    headers = [r"\textbf{Pooled}" if n == "aggregate" else TARGET_SYMBOLS[n]
               for n in columns]
    lines = [
        rf"\begin{{tabular}}{{@{{}}l*{{{len(columns)}}}{{r}}@{{}}}}",
        r"\toprule", "Metric & " + " & ".join(headers) + ROW_END,
        r"\midrule",
    ]
    for label, metric in (("Stoch. MAE", "stochastic_mae"),
                          ("Stoch. RMSE", "stochastic_rmse"),
                          ("Oper. MAE", "operational_mae"),
                          ("Oper. RMSE", "operational_rmse")):
        vals = [aggregate[metric] if n == "aggregate" else point[n][metric]
                for n in columns]
        lines.append(label + " & " + " & ".join(_fmt(v) for v in vals) + ROW_END)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    caption = (f"Normalized forecast accuracy: {description}. Errors divided "
               "by original training ranges.")
    if name == "system":
        caption += " Pooled values span all 15 outputs."
    return _table(caption, "tab:mace_forecast_accuracy:" + name, lines)


def _table_uq(group, *, caption_prefix, reference_prefix, values):
    name, description, _ = group
    columns = _columns(group)
    headers = [r"\textbf{Pooled}" if n == "aggregate" else TARGET_SYMBOLS[n]
               for n in columns]
    lines = [
        rf"\begin{{tabular}}{{@{{}}ll*{{{len(columns)}}}{{r}}@{{}}}}",
        r"\toprule",
        "Method & Metric & " + " & ".join(headers) + ROW_END,
        r"\midrule",
    ]
    for mi, (method, mode, short) in enumerate(METHODS):
        for i, (metric, display) in enumerate(zip(UQ_METRICS, TABLE_METRIC_LABELS)):
            first = r"\multirow{3}{*}{" + short + "}" if i == 0 else ""
            line = first + " & " + display + " & " + " & ".join(
                _fmt(values(method, mode, scope)[metric]) for scope in columns)
            lines.append(line + ROW_END)
        if mi < len(METHODS)-1:
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    caption = (f"{caption_prefix}: {description}. Coverage, normalized full width, "
               "and interval score (95\\% nominal).")
    if name == "system":
        caption += " Pooled values span all 15 outputs."
    return _table(caption, reference_prefix + ":" + name, lines)


def _triplet(v):
    """Coverage/full-width/interval-score triple, all in normalized coordinates."""
    return "/".join(_fmt(v[k], digits=4) for k in UQ_METRICS)

def _select_operational(uq, *, source, kind, mode, scope, metric):
    return _unique(uq, calibration=source, method=kind, mode=mode,
                   phase="all", family="pooled", scope=scope, metric=metric)

def make_tables(config, path=_path):
    """Build detailed manuscript tables only from frozen scientific results."""
    sto = _stochastic(config, path)
    op = _operational(config, path)
    names = sto["names"]
    if names != op["names"] or len(names) != 15 or set(names) != set(TARGET_SYMBOLS):
        raise ValueError("Inconsistent 15-variable stochastic/operational output identity")
    spans = np.asarray(sto["stats"]["y"]["span"], dtype=np.float64)
    if spans.shape != (15,) or not np.all(np.isfinite(spans)) or np.any(spans <= 0):
        raise ValueError("Invalid training-set normalization spans")
    span_by_name = dict(zip(names, spans))
    records = list(sto["inputs"]) + list(op["inputs"])
    ops = op["paths"]["operational_dir"]/"evaluation"
    floor_dir = sto["paths"]["stochastic_floor_dir"]
    validation_dir = sto["paths"]["stochastic_validation_dir"]
    paths = {
        "accuracy": ops/"accuracy_metrics.csv",
        "operational_uq": ops/"uq_method_metrics.csv",
        "operational_comparisons": ops/"paired_comparisons.csv",
        "stochastic_targets": floor_dir/"target_metrics.csv",
        "stochastic_aggregate": floor_dir/"aggregate_metrics.csv",
        "stochastic_joint": validation_dir/"bootstrap_method_metrics.csv",
    }
    records.extend(_verified(p) for p in paths.values())
    accuracy, uq, comparisons, targets, aggregate, joint = (
        _read_rows(paths[key]) for key in paths
    )
    details = []
    n = len(sto["ids"])
    if n != len(op["ids"]):
        raise ValueError("Evaluation populations must have the same expected test count")

    # Per-output stochastic recursive forecast accuracy, with the exact
    # physical-error/training-range normalization used in the original Table 5.
    sa = np.zeros(15, dtype=np.float64)
    ss = np.zeros(15, dtype=np.float64)
    with h5py.File(sto["files"]["test"], "r") as f:
        if set(f.keys()) != set(sto["ids"]):
            raise ValueError("Stochastic test ID inventory differs from manifest")
        for pid in sto["ids"]:
            g = f[pid]
            delta = ((g["ensemble_mean_physical"][()].astype(np.float64)
                     - g["y_true_physical"][()].astype(np.float64)) / spans)
            if delta.shape != (501, 15) or not np.isfinite(delta).all():
                raise ValueError(f"Invalid normalized errors in stochastic trajectory {pid}")
            sa += np.abs(delta).sum(axis=0, dtype=np.float64)
            ss += np.square(delta).sum(axis=0, dtype=np.float64)
    stoch_mae = sa/(n*501)
    stoch_rmse = np.sqrt(ss/(n*501))
    op_mae = np.empty(15)
    op_rmse = np.empty(15)
    for j, name in enumerate(names):
        # The operational per-output archive uses physical units; normalize
        # the point estimates and confidence intervals using the SAME spans.
        for metric, array in (("mae", op_mae), ("rmse", op_rmse)):
            rec = _unique(accuracy, phase="all", family="pooled",
                          scope=name, metric=metric)
            array[j] = float(rec["estimate"])/spans[j]
            for field in ("estimate", "ci_lower", "ci_upper"):
                details.append(dict(dataset="operational", calibration="none",
                                    output=name, method="ensemble", mode="forecast",
                                    metric="normalized_"+metric, units="scaled",
                                    bound=field, value=_fmt(float(rec[field])/spans[j], digits=8)))
    stochastic_mae = float(stoch_mae.mean())
    stochastic_rmse = float(np.sqrt((ss/(n*501)).mean()))
    operational_mae = _metric(accuracy, metric="mae", phase="all",
                              family="pooled", scope="aggregate")
    operational_rmse = _metric(accuracy, metric="rmse", phase="all",
                               family="pooled", scope="aggregate")
    # Confirm operational per-target normalization matches archived aggregate.
    if not np.isclose(op_mae.mean(), operational_mae, rtol=2e-4, atol=1e-8):
        raise ValueError("Operational per-target MAE does not reconcile with pooled aggregate")
    # Accuracy RMSE aggregation averages normalized MSE before taking the sqrt.
    if not np.isclose(np.sqrt(np.mean(op_rmse**2)), operational_rmse, rtol=2e-4, atol=1e-8):
        raise ValueError("Operational per-target RMSE does not reconcile with pooled aggregate")
    for j, name in enumerate(names):
        for metric, val in (("normalized_mae", stoch_mae[j]),
                            ("normalized_rmse", stoch_rmse[j])):
            details.append(dict(dataset="stochastic", calibration="none",
                                output=name, method="ensemble", mode="forecast",
                                metric=metric, units="scaled",
                                bound="estimate", value=_fmt(val, digits=8)))

    def stochastic_values(kind, mode, name):
        row = _stochastic_row(targets, method=kind, mode=mode, target=name)
        span = span_by_name[name]
        return {
            "objective_coverage":float(row["objective_coverage"]),
            "mean_full_width":float(row["mean_physical_full_width"])/span,
            "mean_interval_score":float(row["mean_physical_interval_score"])/span,
        }

    def op_values(source, kind, mode, scope):
        archive_mode = "horizon" if mode == "horizon" else "target_trajectory"
        scale = 1.0 if scope == "aggregate" else span_by_name[scope]
        return {
            metric: float(_select_operational(
                uq, source=source, kind=kind, mode=archive_mode, scope=scope,
                metric=metric)["estimate"])/ (scale if metric.startswith("mean_") else 1.)
            for metric in UQ_METRICS
        }

    # Save exact per-output metrics and available bootstrap intervals in a
    # machine-readable companion, without making the already wide cells illegible.
    for name in names:
        for kind, mode, _ in METHODS:
            values = stochastic_values(kind, mode, name)
            for metric, value in values.items():
                details.append(dict(dataset="stochastic", calibration="stochastic",
                    output=name, method=kind, mode=mode, metric=metric,
                    units="fraction" if metric == "objective_coverage" else "scaled",
                    bound="estimate", value=_fmt(value, digits=8)))
            for source in ("original", "operational"):
                archive_mode = "horizon" if mode == "horizon" else "target_trajectory"
                for metric in UQ_METRICS:
                    rec = _select_operational(uq, source=source, kind=kind,
                        mode=archive_mode, scope=name, metric=metric)
                    scale = (span_by_name[name] if metric.startswith("mean_") else 1.)
                    for bound in ("estimate", "ci_lower", "ci_upper"):
                        details.append(dict(dataset="operational",
                            calibration="stochastic" if source == "original" else "operation-informed",
                            output=name, method=kind, mode=mode, metric=metric,
                            units="fraction" if metric == "objective_coverage" else "scaled",
                            bound=bound, value=_fmt(float(rec[bound])/scale, digits=8)))

    def stochastic_aggregate(kind, mode):
        row = _stochastic_row(aggregate, method=kind, mode=mode)
        return dict(objective_coverage=float(row["objective_coverage"]),
                    mean_full_width=float(row["mean_scaled_full_width"]),
                    mean_interval_score=float(row["mean_scaled_interval_score"]))

    def joint_values(kind):
        output = {}
        for metric in ("objective_coverage", "mean_full_width", "mean_interval_score"):
            rec = _unique(joint, mode="joint", method=kind,
                          scope="aggregate", metric=metric)
            output[metric] = float(rec["estimate"])
            for bound in ("estimate", "ci_lower", "ci_upper"):
                details.append(dict(dataset="stochastic", calibration="stochastic",
                    output="all_15_complete_trajectory", method=kind,
                    mode="joint", metric=metric,
                    units="fraction" if metric == "objective_coverage" else "scaled",
                    bound=bound, value=_fmt(float(rec[bound]), digits=8)))
        return output

    joint_summary = {kind: joint_values(kind) for kind in ("mace", "absolute")}
    point_accuracy = {
        name: {
            "stochastic_mae": float(stoch_mae[j]),
            "stochastic_rmse": float(stoch_rmse[j]),
            "operational_mae": float(op_mae[j]),
            "operational_rmse": float(op_rmse[j]),
        } for j, name in enumerate(names)
    }
    aggregate_accuracy = {
        "stochastic_mae": stochastic_mae,
        "stochastic_rmse": stochastic_rmse,
        "operational_mae": operational_mae,
        "operational_rmse": operational_rmse,
    }
    accuracy_tables = [
        _table_accuracy(group, point=point_accuracy, aggregate=aggregate_accuracy)
        for group in COLUMN_GROUPS
    ]
    stochastic_tables = [
        _table_uq(group, caption_prefix="Stochastic calibration/test",
                  reference_prefix="tab:mace_stochastic_uq",
                  values=lambda kind, mode, scope: (
                      stochastic_aggregate(kind, mode) if scope == "aggregate"
                      else stochastic_values(kind, mode, scope)))
        for group in COLUMN_GROUPS
    ]
    joint_lines = [
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        "Method & Coverage & Mean width & Interval score " + ROW_END,
        r"\midrule",
    ]
    for kind, label in (("mace", "MACE-Joint"), ("absolute", "Abs-Joint")):
        joint_lines.append(label + " & " + " & ".join(
            _fmt(joint_summary[kind][metric]) for metric in UQ_METRICS) + ROW_END)
    joint_lines.extend([r"\bottomrule", r"\end{tabular}"])
    joint_table = _table(
        "Joint complete-multivariate trajectory coverage on the stochastic "
        "test distribution (95\\% nominal).",
        "tab:mace_joint_uq", joint_lines, fontsize=r"\small")
    table5 = "\n".join(accuracy_tables + stochastic_tables + [joint_table])

    transfer_tables = [
        _table_uq(group, caption_prefix="Stochastic calibration, operational test",
                  reference_prefix="tab:mace_operational_transfer",
                  values=lambda kind, mode, scope: op_values(
                      "original", kind, mode, scope))
        for group in COLUMN_GROUPS
    ]
    recalibrated_tables = [
        _table_uq(group, caption_prefix="Operation-informed calibration/test",
                  reference_prefix="tab:mace_operational_recalibration",
                  values=lambda kind, mode, scope: op_values(
                      "operational", kind, mode, scope))
        for group in COLUMN_GROUPS
    ]

    # Paired bootstrap uses the SAME operational evaluation trajectory indices
    # under both calibration sources; no bootstrap is re-run for publication.
    paired_metrics = (
        ("objective_coverage_difference", "coverage"),
        ("mean_full_width_difference", "mean full width"),
        ("mean_interval_score_difference", "mean interval score"))
    paired_summary = {}
    for kind, mode, short_name in METHODS:
        archive_mode = "horizon" if mode == "horizon" else "target_trajectory"
        paired_summary[short_name] = {}
        for scope in ["aggregate"] + names:
            values = []
            for metric, label in paired_metrics:
                row = _unique(comparisons,
                    comparison=kind+":operational_minus_original",
                    mode=archive_mode, phase="all", family="pooled",
                    scope=scope, metric=metric)
                scale = (1.0 if scope == "aggregate" or metric == "objective_coverage_difference"
                         else span_by_name[scope])
                point = float(row["estimate"])/scale
                lo = float(row["ci_lower"])/scale
                hi = float(row["ci_upper"])/scale
                for bound, value in (("estimate",point),("ci_lower",lo),("ci_upper",hi)):
                    details.append(dict(dataset="operational",
                        calibration="operation-informed_minus_stochastic",
                        output=scope, method=kind, mode=mode, metric=metric,
                        units="fraction" if metric == "objective_coverage_difference" else "scaled",
                        bound=bound, value=_fmt(value, digits=8)))
                if scope == "aggregate":
                    paired_summary[short_name][metric] = point
                    values.append(f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            if scope == "aggregate":
                paired_summary[short_name]["formatted"] = values
    paired_lines = [
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"Method & $\Delta$ coverage & $\Delta$ width & $\Delta$ interval score " + ROW_END,
        r"\midrule",
    ]
    for _, _, short in METHODS:
        paired_lines.append(
            short + " & " + " & ".join(paired_summary[short]["formatted"]) + ROW_END)
    paired_lines.extend([r"\bottomrule", r"\end{tabular}"])
    paired_table = _table(
        "Pooled operation-informed minus stochastic calibration: differences "
        "and 95\\% paired whole-trajectory bootstrap intervals.",
        "tab:mace_operational_paired", paired_lines)
    table6 = "\n".join(transfer_tables + recalibrated_tables + [paired_table])

    detail_fields=("dataset","calibration","output","method","mode","metric","units","bound","value")
    # csv.writer is used for proper quoting, including scientific variable labels.
    import io
    stream=io.StringIO()
    writer=csv.DictWriter(stream, fieldnames=detail_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(details)
    detail_csv=stream.getvalue()
    return {
        "table05_forecast_accuracy.tex":table5,
        "table06_operational_uq.tex":table6,
        "table_metrics_detail.csv":detail_csv,
    }, records, {
        "stochastic_mae":stochastic_mae,
        "stochastic_rmse":stochastic_rmse,
        "operational_mae":operational_mae,
        "operational_rmse":operational_rmse,
        "joint_stochastic":joint_summary,
        "uq_stochastic_method_count":len(METHODS),
        "uq_operational_calibration_sources":["original","operational"],
        "per_output_count":len(names),
        "detail_rows":len(details),
        "paired_operational_recalibration":{k:{metric: value for metric,value in v.items() if metric != "formatted"} for k,v in paired_summary.items()},
    }


def run(config_path: Path):
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute(): cfg_path=ROOT/cfg_path
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "results" not in config or "results_tables_output_dir" not in config["results"]:
        raise ValueError("Results table output directory missing from figure config")
    out = _path(config["results"]["results_tables_output_dir"])
    if out.exists():
        raise FileExistsError(f"Results tables already exist; use a versioned output directory: {out}")
    files, inputs, values = make_tables(config)
    if any(out == Path(i["path"]).resolve() or out in Path(i["path"]).resolve().parents for i in inputs):
        raise ValueError("Refusing to overwrite scientific sources")
    for rec in inputs:
        if fingerprint(Path(rec["path"]))["sha256"] != rec["sha256"]:
            raise ValueError(f"Table input changed while aggregating: {rec['path']}")
    out.mkdir(parents=True)
    for name, body in files.items():
        (out/name).write_text(body, encoding="utf-8")
    write_json(out/"table_metadata.json", {"completed":True, "config":config,
               "inputs":inputs, "metrics":values,
               "output_hashes":{key:fingerprint(out/key)["sha256"] for key in files},
               "latex_dependencies":"No extra package required; standard table*, tabular, scriptsize and hline"})
    print("Results LaTeX tables saved:",out)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("scripts/mace_paper_figures_config.json"))
    run(p.parse_args().config)
