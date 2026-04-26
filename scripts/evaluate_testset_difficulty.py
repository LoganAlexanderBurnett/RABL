from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
import sys
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rabl.machine_learning.build_lstm_dataset import CONTROL_COLUMN, STATE_COLUMNS
from rabl.machine_learning.lstm_pipeline import (
    TARGET_NAMES,
    ProfileDataset,
    build_model,
    rolling_forecast,
    _descale_feature_from_stats,
    _descale_targets_from_stats,
    _load_scaling_stats,
)
from rabl.machine_learning.posthoc_difficulty_eval import (
    aggregate_metric_by_bin,
    bin_series,
    compute_equilibrium_excursions,
)


def _load_steady_state(config_path: Path) -> dict[str, float]:
    spec = importlib.util.spec_from_file_location("rabl_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    steady = getattr(module, "STEADY_STATE", None)
    if not isinstance(steady, dict):
        raise ValueError("STEADY_STATE dictionary not found in config.")
    return steady


def _read_test_profile_names(h5_path: Path) -> list[str]:
    import h5py

    with h5py.File(h5_path, "r") as h5f:
        return sorted(h5f["test"]["files"].keys())


def _infer_checkpoint_io_shapes(model_path: Path) -> tuple[int, int]:
    import torch

    state_dict = torch.load(Path(model_path), map_location="cpu")
    return int(state_dict["lstm.weight_ih_l0"].shape[1]), int(state_dict["output_layer.bias"].shape[0])


def _load_single_model(model_path: Path, *, timesteps: int):
    import torch

    num_features, num_targets = _infer_checkpoint_io_shapes(model_path)
    model = build_model(timesteps=timesteps, num_features=num_features, num_targets=num_targets)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _resolve_model_paths(model_path: Path | None, ensemble_dir: Path | None) -> list[Path]:
    if (model_path is None) == (ensemble_dir is None):
        raise SystemExit("Specify exactly one of --model-path or --ensemble-dir.")
    if model_path is not None:
        if not model_path.exists():
            raise SystemExit(f"Model path not found: {model_path}")
        return [model_path]

    assert ensemble_dir is not None
    if not ensemble_dir.exists():
        raise SystemExit(f"Ensemble dir not found: {ensemble_dir}")
    pt_paths = sorted(ensemble_dir.rglob("*.pt"))
    if not pt_paths:
        raise SystemExit(f"No .pt files found under ensemble dir: {ensemble_dir}")
    return pt_paths


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_histogram(
    rows: list[dict[str, Any]],
    *,
    descriptor: str,
    metric: str,
    output_path: Path,
) -> None:
    labels = [str(r["bin"]) for r in rows]
    means = [float(r["mean"]) for r in rows]
    counts = [int(r["count"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax1.bar(x, means, color="#4C78A8", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel(f"Mean {metric}")
    ax1.set_xlabel(f"{descriptor} bins")
    ax1.set_title(f"{metric} by {descriptor} bin")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, counts, color="#F58518", marker="o", linewidth=1.5)
    ax2.set_ylabel("Bin count")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_boxplot(
    profile_rows: list[dict[str, Any]],
    *,
    descriptor: str,
    metric: str,
    bin_col: str,
    output_path: Path,
) -> None:
    labels = sorted({str(r[bin_col]) for r in profile_rows})
    values = [[float(r[metric]) for r in profile_rows if str(r[bin_col]) == label] for label in labels]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_xlabel(f"{descriptor} bins")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} distribution by {descriptor} bin")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate test-set forecast errors by transient difficulty bins.")
    parser.add_argument("--scaled-h5", type=Path, required=True, help="Scaled/split dataset path containing test split.")
    parser.add_argument("--model-path", type=Path, default=None, help="Single model checkpoint (.pt).")
    parser.add_argument("--ensemble-dir", type=Path, default=None, help="Directory containing ensemble checkpoints (.pt).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for CSV/plots/manifest.")
    parser.add_argument("--n-bins", type=int, default=5, help="Number of bins.")
    parser.add_argument("--binning", type=str, default="quantile", choices=("quantile", "fixed"), help="Binning mode.")
    parser.add_argument("--fixed-edges-theta", type=float, nargs="+", default=None)
    parser.add_argument("--fixed-edges-rho", type=float, nargs="+", default=None)
    parser.add_argument("--fixed-edges-vtheta", type=float, nargs="+", default=None)
    parser.add_argument("--dt", type=float, default=1.0, help="Timestep size for velocity estimate.")
    parser.add_argument("--config-path", type=Path, default=REPO_ROOT / "scripts" / "config.py")
    parser.add_argument("--include-per-target", action="store_true", help="Include per-target MAE/MSE columns.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.scaled_h5.exists():
        raise SystemExit(f"Scaled dataset not found: {args.scaled_h5}")

    model_paths = _resolve_model_paths(args.model_path, args.ensemble_dir)
    test_profile_names = _read_test_profile_names(args.scaled_h5)
    if not test_profile_names:
        raise SystemExit("No test profiles found in scaled dataset.")

    profile_ds = ProfileDataset(args.scaled_h5, test_profile_names, "test")
    first_profile_name, first_x, _first_y = next(iter(profile_ds))
    timesteps = int(first_x.numpy().shape[1])
    print(f"Loaded first test profile: {first_profile_name} (timesteps={timesteps})")

    models = [_load_single_model(path, timesteps=timesteps) for path in model_paths]
    scaling_stats = _load_scaling_stats(args.scaled_h5)
    steady_state = _load_steady_state(args.config_path)

    state_dim = len(STATE_COLUMNS)
    rho_idx = TARGET_NAMES.index("rho_dollars")
    control_idx = state_dim  # first control channel (drumAngleDeg)

    per_profile_rows: list[dict[str, Any]] = []
    for profile_name, x_tensor, y_tensor in ProfileDataset(args.scaled_h5, test_profile_names, "test"):
        x_scaled = x_tensor.numpy()
        y_scaled = y_tensor.numpy()

        pred_stack = []
        for model in models:
            pred_stack.append(rolling_forecast(model, x_scaled, state_dim=state_dim))
        y_pred_scaled = np.mean(np.stack(pred_stack, axis=0), axis=0)

        y_true = _descale_targets_from_stats(scaling_stats, y_scaled)
        y_pred = _descale_targets_from_stats(scaling_stats, y_pred_scaled)

        drum_scaled = x_scaled[:, -1, control_idx]
        drum = _descale_feature_from_stats(scaling_stats, drum_scaled, control_idx)
        rho = y_true[:, rho_idx]

        descriptors = compute_equilibrium_excursions(
            drum_angle_deg=drum,
            rho_dollars=rho,
            drum_equilibrium_deg=float(steady_state[CONTROL_COLUMN]),
            rho_equilibrium_dollars=float(steady_state["rho_dollars"]),
            dt=float(args.dt),
        )

        abs_err = np.abs(y_true - y_pred)
        sq_err = (y_true - y_pred) ** 2
        row: dict[str, Any] = {
            "profile_id": str(profile_name),
            "MAE": float(np.mean(abs_err)),
            "MSE": float(np.mean(sq_err)),
            **descriptors,
        }

        if args.include_per_target:
            for idx, tgt in enumerate(TARGET_NAMES):
                row[f"MAE_{tgt}"] = float(np.mean(abs_err[:, idx]))
                row[f"MSE_{tgt}"] = float(np.mean(sq_err[:, idx]))

        per_profile_rows.append(row)

    per_profile_csv = out_dir / "per_profile_metrics_and_difficulty.csv"
    fieldnames = list(per_profile_rows[0].keys())
    _write_csv(per_profile_csv, fieldnames, per_profile_rows)

    descriptor_specs = [
        ("E_theta_max", args.fixed_edges_theta),
        ("E_rho_max", args.fixed_edges_rho),
        ("V_theta_max", args.fixed_edges_vtheta),
    ]

    generated_paths: list[str] = [str(per_profile_csv)]

    for descriptor, fixed_edges in descriptor_specs:
        values = np.asarray([float(row[descriptor]) for row in per_profile_rows], dtype=float)
        resolved_edges = None if fixed_edges is None else np.asarray(fixed_edges, dtype=float)
        binned = bin_series(values, mode=args.binning, n_bins=args.n_bins, edges=resolved_edges)

        bin_col = f"{descriptor}_bin"
        for row, label in zip(per_profile_rows, binned.labels, strict=True):
            row[bin_col] = str(label)

        agg_rows: list[dict[str, Any]] = []
        for metric in ("MAE", "MSE"):
            stats_rows = aggregate_metric_by_bin(per_profile_rows, metric_col=metric, bin_col=bin_col)
            for r in stats_rows:
                r["metric"] = metric
            agg_rows.extend(stats_rows)

            hist_name = f"{metric.lower()}_hist_by_{descriptor}.png"
            _plot_histogram(
                [r for r in stats_rows],
                descriptor=descriptor,
                metric=metric,
                output_path=out_dir / hist_name,
            )
            generated_paths.append(str(out_dir / hist_name))

            if metric == "MAE":
                box_name = f"mae_box_by_{descriptor}.png"
                _plot_boxplot(
                    per_profile_rows,
                    descriptor=descriptor,
                    metric="MAE",
                    bin_col=bin_col,
                    output_path=out_dir / box_name,
                )
                generated_paths.append(str(out_dir / box_name))

        bins_csv = out_dir / f"bins_{descriptor}_metrics.csv"
        _write_csv(
            bins_csv,
            ["metric", "bin", "count", "mean", "median", "std"],
            sorted(agg_rows, key=lambda r: (r["metric"], r["bin"])),
        )
        generated_paths.append(str(bins_csv))

        edges_json = out_dir / f"bins_{descriptor}_edges.json"
        edges_payload = {
            "descriptor": descriptor,
            "binning_mode": args.binning,
            "n_bins": int(args.n_bins),
            "edges": binned.edges.tolist(),
            "labels": binned.label_names,
        }
        edges_json.write_text(json.dumps(edges_payload, indent=2), encoding="utf-8")
        generated_paths.append(str(edges_json))

    # Persist per-profile table with bin labels included.
    _write_csv(per_profile_csv, list(per_profile_rows[0].keys()), per_profile_rows)

    manifest = {
        "dataset_path": str(args.scaled_h5),
        "model_id": "ensemble" if args.ensemble_dir is not None else Path(model_paths[0]).stem,
        "checkpoint_paths": [str(path) for path in model_paths],
        "steady_state": {
            CONTROL_COLUMN: float(steady_state[CONTROL_COLUMN]),
            "rho_dollars": float(steady_state["rho_dollars"]),
        },
        "binning": {
            "mode": args.binning,
            "n_bins": int(args.n_bins),
            "fixed_edges": {
                "E_theta_max": None if args.fixed_edges_theta is None else list(map(float, args.fixed_edges_theta)),
                "E_rho_max": None if args.fixed_edges_rho is None else list(map(float, args.fixed_edges_rho)),
                "V_theta_max": None if args.fixed_edges_vtheta is None else list(map(float, args.fixed_edges_vtheta)),
            },
        },
        "artifacts": generated_paths,
    }
    manifest_path = out_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved per-profile table: {per_profile_csv}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
