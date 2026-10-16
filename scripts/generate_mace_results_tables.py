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


def make_tables(config, path=_path):
    """All metrics are read; no prediction, calibration, or bootstrap is performed."""
    sto = _stochastic(config, path)
    op = _operational(config, path)
    records = list(sto["inputs"]) + list(op["inputs"])
    ops = op["paths"]["operational_dir"]/"evaluation"
    accuracy_path = ops/"accuracy_metrics.csv"
    uq_path = ops/"uq_method_metrics.csv"
    comparison_path = ops/"paired_comparisons.csv"
    records += [_verified(p) for p in (accuracy_path, uq_path, comparison_path)]
    accuracy = _read_rows(accuracy_path)
    uq = _read_rows(uq_path)
    comparisons = _read_rows(comparison_path)
    # Match the saved operational convention: normalise physical error by the
    # original Y training spans, average across all 15 variables and 501 steps.
    span = np.asarray(sto["stats"]["y"]["span"], dtype=np.float64)
    abs_sum = sq_sum = 0.
    count = 0
    with h5py.File(sto["files"]["test"], "r") as f:
        if set(f.keys()) != set(sto["ids"]):
            raise ValueError("Stochastic test profile inventory differs from completed manifest")
        for pid in sto["ids"]:
            g = f[pid]
            errors = ((g["ensemble_mean_physical"][()].astype(np.float64)
                      -g["y_true_physical"][()].astype(np.float64))/span)
            if errors.shape != (501,15) or not np.all(np.isfinite(errors)):
                raise ValueError(f"Invalid scaled forecast errors: {pid}")
            abs_sum += np.abs(errors).sum(dtype=np.float64)
            sq_sum += np.square(errors).sum(dtype=np.float64)
            count += errors.size
    mae_stochastic = abs_sum/count
    rmse_stochastic = np.sqrt(sq_sum/count)
    filt_acc = dict(phase="all", family="pooled", scope="aggregate")
    mae_operational = _metric(accuracy, metric="mae", **filt_acc)
    rmse_operational = _metric(accuracy, metric="rmse", **filt_acc)
    rowend = r" \\"
    table5 = _latex_table("lccc", [
        r"Evaluation distribution & Trajectories & Normalized MAE & Normalized RMSE " + rowend,
        r"\hline",
        f"Stochastic & {len(sto['ids'])} & {_fmt(mae_stochastic)} & {_fmt(rmse_stochastic)} " + rowend,
        f"Operation-informed & {len(op['ids'])} & {_fmt(mae_operational)} & {_fmt(rmse_operational)} " + rowend,
    ], "Aggregate autoregressive ensemble forecast accuracy on independent stochastic and "
       "operation-informed evaluation distributions. Errors are computed in physical units, "
       "normalized by the original training-set output ranges and then equally averaged "
       "over output variables, forecast horizons and trajectories; RMSE is the square root "
       "of the mean normalized squared error.",
       "tab:mace_forecast_accuracy")
    common = dict(mode="horizon", phase="all", family="pooled", scope="aggregate")
    result = {}
    for source in ("original","operational"):
        d = {metric: _unique(uq,calibration=source,method="mace",metric=metric,**common)
             for metric in ("objective_coverage","mean_full_width","mean_interval_score")}
        numerator = float(d["mean_full_width"]["estimate"])
        denom = _metric(uq,calibration=source,method="absolute",metric="mean_full_width",**common)
        if denom <= 0:
            raise ValueError("Absolute-Horizon denominator must be positive")
        result[source] = {
            "coverage": _ci(uq,calibration=source,method="mace",metric="objective_coverage",**common),
            "width": _ci(uq,calibration=source,method="mace",metric="mean_full_width",**common),
            "score": _ci(uq,calibration=source,method="mace",metric="mean_interval_score",**common),
            "width_ratio": _fmt(numerator/denom, digits=3)
        }
    delta = {}
    for key,metric in (("coverage","objective_coverage_difference"),
                       ("width","mean_full_width_difference"),
                       ("score","mean_interval_score_difference")):
        delta[key] = _ci(comparisons,comparison="mace:operational_minus_original",metric=metric,**common)
    # Intervals are bootstrap percentile CI, not CP prediction intervals.
    table6 = _latex_table("lcccc", [
        r"Calibration distribution & Coverage (95\% nominal) & Mean full width & Mean interval score & Width / absolute CP " + rowend,
        r"\hline",
        "Stochastic & "+ " & ".join(result["original"][k] for k in ("coverage","width","score","width_ratio"))+" "+rowend,
        "Operation-informed & "+ " & ".join(result["operational"][k] for k in ("coverage","width","score","width_ratio"))+" "+rowend,
        r"\hline",
        r"Difference (operational $-$ stochastic) & "
        + " & ".join(delta[k] for k in ("coverage","width","score")) + r" & -- " + rowend,
    ], "Operation-informed test-set MACE-Horizon uncertainty metrics with stochastic versus "
       "independent operation-informed conformal calibration. Each estimate is followed "
       "by a descriptive 95\\% paired whole-trajectory bootstrap percentile interval "
       "in brackets, conditional on the frozen ensemble and calibration sets. "
       "Widths and interval scores are equally averaged in scaled output coordinates. "
       "Width ratios compare each MACE-Horizon arm with its matched absolute CP-Horizon "
       "calibration; the difference row compares operational minus stochastic calibration.",
       "tab:mace_operational_recalibration", wide=True)
    return {"table05_forecast_accuracy.tex":table5, "table06_operational_uq.tex":table6}, records, {
        "stochastic_mae":mae_stochastic,"stochastic_rmse":rmse_stochastic,
        "operational_mae":mae_operational,"operational_rmse":rmse_operational,
        "operational_mace_horizon":result, "operational_minus_stochastic":delta}


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
               "latex_dependencies":"No extra package required; uses standard tabular, hline and table/table*"})
    print("Results LaTeX tables saved:",out)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("scripts/mace_paper_figures_config.json"))
    run(p.parse_args().config)
