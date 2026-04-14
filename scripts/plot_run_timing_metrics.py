"""Plot per-cycle timing metrics from a run_single_experiment metadata file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


TIMING_KEYS = [
    "build_unscaled_dataset_sec",
    "scale_split_dataset_sec",
    "hyperparameter_tuning_sec",
    "ensemble_training_sec",
    "ensemble_metrics_compute_sec",
    "forecast_pdf_render_sec",
    "profile_generation_sec",
    "dymola_simulation_sec",
    "cycle_total_sec",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_metadata",
        type=Path,
        help="Path to run_metadata.json produced by scripts/run_single_experiment.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path for the timing figure. "
            "Defaults to <run_metadata parent>/timing_metrics_vs_cycle.png"
        ),
    )
    return parser


def _load_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("run_metadata.json must contain a JSON object.")
    if "cycles" not in data or not isinstance(data["cycles"], list):
        raise ValueError("run_metadata.json must contain a 'cycles' list.")
    return data


def _extract_cycle_series(cycles: list[dict[str, Any]], timing_key: str) -> list[float]:
    values: list[float] = []
    for idx, cycle_row in enumerate(cycles, start=1):
        if not isinstance(cycle_row, dict):
            raise ValueError(f"Cycle entry #{idx} is not an object.")
        timing = cycle_row.get("timing")
        if not isinstance(timing, dict):
            raise ValueError(f"Cycle entry #{idx} does not contain a 'timing' object.")
        raw_value = timing.get(timing_key)
        if raw_value is None:
            raise ValueError(f"Cycle entry #{idx} is missing timing key '{timing_key}'.")
        values.append(float(raw_value))
    return values


def _plot_timing_metrics(cycles: list[dict[str, Any]], output_path: Path) -> None:
    cycle_indices = [int(cycle_row.get("cycle", i + 1)) for i, cycle_row in enumerate(cycles)]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes_flat = axes.ravel()

    for ax, timing_key in zip(axes_flat, TIMING_KEYS):
        values = _extract_cycle_series(cycles, timing_key)
        ax.plot(cycle_indices, values, marker="o")
        ax.set_title(timing_key)
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.3)

    for ax in axes_flat[-3:]:
        ax.set_xlabel("cycle")

    fig.suptitle("run_single_experiment timing metrics across cycles")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    metadata = _load_metadata(args.run_metadata)
    cycles = metadata["cycles"]
    if not cycles:
        raise SystemExit("No cycles found in run_metadata.json.")

    output_path = args.output
    if output_path is None:
        output_path = args.run_metadata.resolve().parent / "timing_metrics_vs_cycle.png"

    _plot_timing_metrics(cycles=cycles, output_path=output_path)
    print(f"Saved timing metrics plot to: {output_path}")


if __name__ == "__main__":
    main()
