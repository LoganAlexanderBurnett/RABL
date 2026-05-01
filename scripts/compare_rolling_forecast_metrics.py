"""Compare multiple rolling forecast metrics JSON files in a single subplot figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metrics JSON must be an object: {path}")
    return data


def _label_for(path: Path, idx: int, labels: list[str] | None) -> str:
    if labels is not None:
        return labels[idx]
    return path.stem


def _nan_if_missing(d: dict[str, Any], key: str) -> float:
    val = d.get(key, float("nan"))
    try:
        return float(val)
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple metrics JSON files produced by "
            "compute_and_save_rolling_forecast_metrics()."
        )
    )
    parser.add_argument(
        "metrics_json",
        nargs="+",
        type=Path,
        help="One or more metrics JSON files.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each JSON (same count/order as metrics_json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics_json_comparison.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in (0, len(args.metrics_json)):
        raise SystemExit("--labels must be omitted or have same length as metrics_json inputs.")

    labels = args.labels if args.labels else None
    datasets: list[dict[str, Any]] = [_load_metrics(p) for p in args.metrics_json]
    series_labels = [_label_for(p, i, labels) for i, p in enumerate(args.metrics_json)]

    smape = [_nan_if_missing(d, "smape") for d in datasets]
    nrmse = [_nan_if_missing(d, "nrmse") for d in datasets]

    cov95 = [_nan_if_missing(d, "empirical_coverage_95") for d in datasets]
    cal95 = [_nan_if_missing(d, "calibration_error_95") for d in datasets]
    w95 = [_nan_if_missing(d, "interval_width_95_mean") for d in datasets]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)

    x = np.arange(len(series_labels))

    # Scalar metric bars
    axes[0, 0].bar(x, smape, color="C0")
    axes[0, 0].set_title("sMAPE")
    axes[0, 0].set_xticks(x, series_labels, rotation=20, ha="right")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].bar(x, nrmse, color="C1")
    axes[0, 1].set_title("NRMSE")
    axes[0, 1].set_xticks(x, series_labels, rotation=20, ha="right")
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].bar(x, cal95, color="C3")
    axes[0, 2].set_title("Calibration Error (95%)")
    axes[0, 2].set_xticks(x, series_labels, rotation=20, ha="right")
    axes[0, 2].grid(alpha=0.3)

    # Ensemble-only scalar metrics (nan for single-model)
    axes[1, 0].bar(x, cov95, color="C2")
    axes[1, 0].axhline(0.95, linestyle="--", linewidth=1.0, color="0.4", label="ideal 0.95")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Empirical Coverage (95%)")
    axes[1, 0].set_xticks(x, series_labels, rotation=20, ha="right")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(loc="best", fontsize=8)

    axes[1, 1].bar(x, w95, color="C4")
    axes[1, 1].set_title("Mean 95% Interval Width")
    axes[1, 1].set_xticks(x, series_labels, rotation=20, ha="right")
    axes[1, 1].grid(alpha=0.3)

    # Horizon-wise curves
    ax = axes[1, 2]
    for label, d in zip(series_labels, datasets, strict=True):
        rmse_h = np.asarray(d.get("horizon_mean_rmse", []), dtype=float)
        mae_h = np.asarray(d.get("horizon_mean_mae", []), dtype=float)
        if rmse_h.size:
            ax.plot(np.arange(rmse_h.size), rmse_h, label=f"{label} RMSE", linewidth=1.4)
        if mae_h.size:
            ax.plot(np.arange(mae_h.size), mae_h, linestyle="--", label=f"{label} MAE", linewidth=1.2)
    ax.set_title("Horizon-wise Mean Error")
    ax.set_xlabel("Horizon step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Rolling Forecast Metrics Comparison", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
