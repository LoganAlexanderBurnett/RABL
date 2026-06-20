from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
)

def _short_bin_label(left: float, right: float, *, is_last: bool) -> str:
    right_bracket = "]" if is_last else ")"
    return f"[{left:.3g}, {right:.3g}{right_bracket}"


def _equal_width_edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = np.nextafter(vmin, np.inf)
    return np.linspace(vmin, vmax, n_bins + 1)


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


def _signed_peak(values: np.ndarray, equilibrium: float) -> float:
    delta = np.asarray(values, dtype=float) - float(equilibrium)
    idx = int(np.argmax(np.abs(delta)))
    return float(np.asarray(values, dtype=float)[idx])


def _read_test_profile_names(h5_path: Path) -> list[str]:
    import h5py

    with h5py.File(h5_path, "r") as h5f:
        return sorted(h5f["test"]["files"].keys())


def _decode_h5_strings(value: Any) -> list[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        return [item.decode("utf-8") if isinstance(item, bytes) else str(item)]
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in arr.tolist()]


def _forecast_columns_for_targets(
    *,
    forecast_h5: Path,
    profile_name: str,
    columns: list[str],
) -> tuple[list[int], list[int], int]:
    missing = ["t", "u(t)"]
    truth_prefix = "x_true(t)_"
    pred_prefix = "x_mean(t)_"
    if not all(f"{truth_prefix}{name}" in columns for name in TARGET_NAMES):
        truth_prefix = "x(t)_"
        pred_prefix = "x^~(t)_"
    missing.extend(f"{truth_prefix}{name}" for name in TARGET_NAMES if f"{truth_prefix}{name}" not in columns)
    missing.extend(f"{pred_prefix}{name}" for name in TARGET_NAMES if f"{pred_prefix}{name}" not in columns)
    missing = [name for name in missing if name not in columns]
    if missing:
        raise ValueError(
            f"Forecast profile {profile_name!r} in {forecast_h5} is missing required columns: {missing}"
        )
    control_idx = columns.index("u(t)")
    true_indices = [columns.index(f"{truth_prefix}{name}") for name in TARGET_NAMES]
    pred_indices = [columns.index(f"{pred_prefix}{name}") for name in TARGET_NAMES]
    return true_indices, pred_indices, control_idx


def _scale_targets_from_stats(stats: dict[str, Any], values: np.ndarray) -> np.ndarray:
    """Scale descaled target values using the dataset's target scaling stats."""
    scaling_type = stats["type"]
    y_stats = stats["y"]
    if scaling_type == "standard":
        return (values - y_stats["mean"]) / y_stats["std"]
    if scaling_type == "minmax":
        return (values - y_stats["min"]) / y_stats["span"]
    raise ValueError(f"Unsupported scaling type: {scaling_type}")


def _infer_checkpoint_arch(model_path: Path) -> dict[str, Any]:
    import torch

    state_dict = torch.load(Path(model_path), map_location="cpu")
    num_features = int(state_dict["lstm.weight_ih_l0"].shape[1])
    num_targets = int(state_dict["output_layer.bias"].shape[0])
    lstm_hidden = int(state_dict["lstm.weight_hh_l0"].shape[1])

    lstm_layer_pattern = re.compile(r"^lstm\.weight_ih_l(\d+)$")
    lstm_layer_ids = sorted(
        int(match.group(1))
        for key in state_dict.keys()
        if (match := lstm_layer_pattern.match(key)) is not None
    )
    n_lstm = (max(lstm_layer_ids) + 1) if lstm_layer_ids else 1

    fc_layer_pattern = re.compile(r"^fc_layers\.(\d+)\.weight$")
    fc_indices = sorted(
        int(match.group(1))
        for key in state_dict.keys()
        if (match := fc_layer_pattern.match(key)) is not None
    )
    if not fc_indices:
        raise RuntimeError(
            f"Checkpoint {model_path} does not contain expected FC layer weights (fc_layers.<idx>.weight)."
        )
    fc_hidden = tuple(
        int(state_dict[f"fc_layers.{idx}.weight"].shape[0])
        for idx in fc_indices
    )
    n_fc = len(fc_hidden)

    return {
        "num_features": num_features,
        "num_targets": num_targets,
        "n_lstm": n_lstm,
        "lstm_hidden": lstm_hidden,
        "n_fc": n_fc,
        "fc_hidden": fc_hidden,
    }


def _load_single_model(model_path: Path, *, timesteps: int):
    import torch

    arch = _infer_checkpoint_arch(model_path)
    model = build_model(
        timesteps=timesteps,
        num_features=int(arch["num_features"]),
        num_targets=int(arch["num_targets"]),
        n_lstm=int(arch["n_lstm"]),
        lstm_hidden=int(arch["lstm_hidden"]),
        n_fc=int(arch["n_fc"]),
        fc_hidden=tuple(int(v) for v in arch["fc_hidden"]),
    )
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
    bars = ax1.bar(x, means, color="#4C78A8", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel(f"Mean {metric}")
    ax1.set_xlabel(f"{descriptor} bins")
    ax1.set_title(f"{metric} by {descriptor} bin")
    ax1.grid(alpha=0.2, axis="y")
    ax1.set_axisbelow(True)
    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)

    for bar, count in zip(bars, counts, strict=True):
        x_pos = bar.get_x() + bar.get_width() / 2.0
        ax1.text(
            x_pos,
            0.05,  # 2% above the bottom of the axis
            f"{count} transients",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=7,
            color="#1F1F1F",
            transform=trans,
        )

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


def _set_log_y_if_positive(ax, values: Any) -> None:
    flat: list[float] = []
    if isinstance(values, np.ndarray):
        flat = [float(v) for v in values.ravel()]
    else:
        for value in values:
            if isinstance(value, (list, tuple, np.ndarray)):
                flat.extend(float(v) for v in np.asarray(value, dtype=float).ravel())
            else:
                flat.append(float(value))
    arr = np.asarray(flat, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if positive.size:
        lower = max(float(np.min(positive)) * 0.5, 1e-12)
        upper = float(np.max(positive)) * 1.25
        if upper <= lower:
            upper = lower * 10.0
        ax.set_yscale("log")
        ax.set_ylim(bottom=lower, top=upper)

def _ordered_target_names(names: Any) -> list[str]:
    unique = {str(name) for name in names}
    ordered = [name for name in TARGET_NAMES if name in unique]
    ordered.extend(sorted(unique.difference(ordered)))
    return ordered

def _target_color_map(target_names: list[str]) -> dict[str, Any]:
    temperature_targets = {"TN2", "Tm", "Thp", "Tf", "Tsg"}
    steam_targets = {"T_steam_out", "x_steam_out"}
    neutronic_targets = {"c[1]", "c[2]", "c[3]", "c[4]", "c[5]", "c[6]", "n", "rho_dollars"}
    groups = [
        (temperature_targets, plt.colormaps.get_cmap("YlOrRd")),
        (steam_targets, plt.colormaps.get_cmap("PuBu")),
        (neutronic_targets, plt.colormaps.get_cmap("YlGn")),
    ]
    colors: dict[str, Any] = {}
    for group_targets, cmap in groups:
        present = [name for name in target_names if name in group_targets]
        for idx, name in enumerate(present):
            frac = 0.35 + 0.55 * (idx / max(1, len(present) - 1))
            colors[name] = cmap(frac)
    fallback = plt.colormaps.get_cmap("tab20")
    for idx, name in enumerate(target_names):
        colors.setdefault(name, fallback(idx % fallback.N))
    return colors

def _plot_summary_grid(
    *,
    per_profile_rows: list[dict[str, Any]],
    descriptor_metrics: dict[str, dict[str, list[dict[str, Any]]]],
    target_overlay: dict[str, dict[str, dict[str, list[float]]]],
    include_per_target: bool,
    output_path: Path,
    log_scale: bool = True,
) -> None:
    descriptors = ["theta_peak", "dtheta_dt_peak", "rho_peak", "drho_dt_peak"]
    columns = ["MAE_hist", "MSE_hist", "MAE_box"]
    descriptor_latex = {
        "theta_peak": r"$\theta_{\mathrm{peak}}$",
        "dtheta_dt_peak": r"$\left(\frac{d\theta}{dt}\right)_{\mathrm{peak}}$",
        "rho_peak": r"$\rho_{\mathrm{peak}}$",
        "drho_dt_peak": r"$\left(\frac{d\rho}{dt}\right)_{\mathrm{peak}}$",
    }
    descriptor_colors = {
        "theta_peak": "#4C78A8",
        "dtheta_dt_peak": "#59A14F",
        "rho_peak": "#F28E2B",
        "drho_dt_peak": "#B07AA1",
    }
    overlay_target_names = _ordered_target_names(
        name
        for descriptor_payload in target_overlay.values()
        for metric_payload in descriptor_payload.values()
        for name in metric_payload
    )
    target_colors = _target_color_map(overlay_target_names)

    fig, axes = plt.subplots(len(descriptors), len(columns), figsize=(18, 18))

    for row_idx, descriptor in enumerate(descriptors):
        bin_col = f"{descriptor}_bin"
        mae_rows = descriptor_metrics[descriptor]["MAE"]
        mse_rows = descriptor_metrics[descriptor]["MSE"]

        for col_idx, panel in enumerate(columns):
            ax = axes[row_idx, col_idx]

            if panel in {"MAE_hist", "MSE_hist"}:
                metric_rows = mae_rows if panel == "MAE_hist" else mse_rows
                metric_name = "MAE" if panel == "MAE_hist" else "MSE"
                labels = [str(r["bin"]) for r in metric_rows]
                display_labels = [str(r.get("bin_display", r["bin"])) for r in metric_rows]
                means = [float(r["mean"]) for r in metric_rows]
                counts = [int(r["count"]) for r in metric_rows]
                x = np.arange(len(labels))

                bars = ax.bar(x, means, color=descriptor_colors[descriptor], alpha=0.9, edgecolor="black", linewidth=0.4)
                ax.set_xticks(x)
                ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=8)
                ax.set_ylabel(f"Mean {metric_name}")
                ax.set_xlabel(f"{descriptor_latex[descriptor]} bins")
                ax.set_title(f"{metric_name} by {descriptor_latex[descriptor]} bin")
                ax.grid(alpha=0.25, axis="y")
                ax.set_axisbelow(True)
                scale_values: list[Any] = [means]

                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

                for bar, count in zip(bars, counts, strict=True):
                    x_pos = bar.get_x() + bar.get_width() / 2.0
                    ax.text(
                        x_pos,
                        0.05,
                        f"{count} transients",
                        rotation=90,
                        va="bottom",
                        ha="center",
                        fontsize=7,
                        color="#1F1F1F",
                        transform=trans,
                    )
                if include_per_target:
                    metric_overlay = target_overlay.get(descriptor, {}).get(metric_name, {})
                    for target_name, target_vals in metric_overlay.items():
                        if len(target_vals) == len(x):
                            scale_values.append(target_vals)
                            ax.plot(
                                x,
                                target_vals,
                                linewidth=0.9,
                                alpha=0.85,
                                marker="o",
                                markersize=2.5,
                                color=target_colors.get(target_name),
                                label=target_name,
                            )
                    if row_idx == 0 and col_idx == 0 and metric_overlay:
                        ax.legend(loc="upper left", fontsize=6, ncol=2, frameon=True)
                if log_scale:
                    _set_log_y_if_positive(ax, scale_values)
            else:
                labels = [str(r["bin"]) for r in mae_rows]
                display_map = {str(r["bin"]): str(r.get("bin_display", r["bin"])) for r in mae_rows}
                values = [
                    [float(r["MAE"]) for r in per_profile_rows if str(r[bin_col]) == label]
                    for label in labels
                ]
                bp = ax.boxplot(values, labels=[display_map[l] for l in labels], showfliers=False, patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor(descriptor_colors[descriptor])
                    patch.set_alpha(0.45)
                ax.set_xlabel(f"{descriptor_latex[descriptor]} bins")
                ax.set_ylabel("MAE")
                ax.set_title(f"MAE distribution by {descriptor_latex[descriptor]} bin")
                if log_scale:
                    _set_log_y_if_positive(ax, values)
                ax.tick_params(axis="x", rotation=25, labelsize=8)
                ax.grid(alpha=0.25, axis="y")
                ax.set_axisbelow(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def evaluate_testset_difficulty(
    *,
    scaled_h5: Path,
    out_dir: Path,
    forecast_h5: Path | None = None,
    model_paths: list[Path] | None = None,
    model_path: Path | None = None,
    ensemble_dir: Path | None = None,
    n_bins: int = 10,
    dt: float = 1.0,
    config_path: Path = REPO_ROOT / "scripts" / "config.py",
    include_per_target: bool = False,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate test-set forecast errors by transient difficulty bins."""
    scaled_h5 = Path(scaled_h5)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scaled_h5.exists():
        raise FileNotFoundError(f"Scaled dataset not found: {scaled_h5}")

    forecast_h5 = Path(forecast_h5) if forecast_h5 is not None else None
    if forecast_h5 is not None:
        if not forecast_h5.exists():
            raise FileNotFoundError(f"Forecast HDF5 not found: {forecast_h5}")
        if model_paths is not None or model_path is not None or ensemble_dir is not None:
            raise ValueError("forecast_h5 is mutually exclusive with model_paths/model_path/ensemble_dir.")
        model_paths = []
    elif model_paths is None:
        model_paths = _resolve_model_paths(model_path, ensemble_dir)
    else:
        if model_path is not None or ensemble_dir is not None:
            raise ValueError("Specify either model_paths or model_path/ensemble_dir, not both.")
        model_paths = [Path(path) for path in model_paths]
        if not model_paths:
            raise ValueError("model_paths must be non-empty.")
        missing_model_paths = [path for path in model_paths if not path.exists()]
        if missing_model_paths:
            raise FileNotFoundError(f"Model path(s) not found: {missing_model_paths[:5]}")

    test_profile_names = _read_test_profile_names(scaled_h5)
    if not test_profile_names:
        raise ValueError("No test profiles found in scaled dataset.")

    scaling_stats = _load_scaling_stats(scaled_h5)
    steady_state = _load_steady_state(Path(config_path))

    state_dim = len(STATE_COLUMNS)
    rho_idx = TARGET_NAMES.index("rho_dollars")
    control_idx = state_dim  # first control channel (drumAngleDeg)

    models = []
    if forecast_h5 is None:
        profile_ds = ProfileDataset(scaled_h5, test_profile_names, "test")
        first_profile_name, first_x, _first_y = next(iter(profile_ds))
        timesteps = int(first_x.numpy().shape[1])
        print(f"Loaded first test profile: {first_profile_name} (timesteps={timesteps})")
        models = [_load_single_model(path, timesteps=timesteps) for path in model_paths]
    else:
        print(f"Reusing existing test forecasts: {forecast_h5}")

    def _evaluate_one(profile_name: str, x_tensor: Any, y_tensor: Any) -> dict[str, Any]:
        x_scaled = x_tensor.numpy()
        y_scaled = y_tensor.numpy()

        pred_stack = []
        for model in models:
            pred_stack.append(rolling_forecast(model, x_scaled, state_dim=state_dim))
        y_pred_scaled = np.mean(np.stack(pred_stack, axis=0), axis=0)

        y_true = _descale_targets_from_stats(scaling_stats, y_scaled)

        drum_scaled = x_scaled[:, -1, control_idx]
        drum = _descale_feature_from_stats(scaling_stats, drum_scaled, control_idx)
        rho = y_true[:, rho_idx]

        v_theta = np.gradient(drum, float(dt))
        drho_dt = np.gradient(rho, float(dt))
        descriptors = {
            "theta_peak": _signed_peak(drum, float(steady_state[CONTROL_COLUMN])),
            "rho_peak": _signed_peak(rho, float(steady_state["rho_dollars"])),
            "dtheta_dt_peak": _signed_peak(v_theta, 0.0),
            "drho_dt_peak": _signed_peak(drho_dt, 0.0),
        }

        abs_err = np.abs(y_scaled - y_pred_scaled)
        sq_err = (y_scaled - y_pred_scaled) ** 2
        row: dict[str, Any] = {
            "profile_id": str(profile_name),
            "MAE": float(np.mean(abs_err)),
            "MSE": float(np.mean(sq_err)),
            **descriptors,
        }

        if include_per_target:
            for idx, tgt in enumerate(TARGET_NAMES):
                row[f"MAE_{tgt}"] = float(np.mean(abs_err[:, idx]))
                row[f"MSE_{tgt}"] = float(np.mean(sq_err[:, idx]))

        return row

    def _evaluate_one_forecast(profile_name: str, table: np.ndarray, columns: list[str]) -> dict[str, Any]:
        true_indices, pred_indices, forecast_control_idx = _forecast_columns_for_targets(
            forecast_h5=forecast_h5 if forecast_h5 is not None else Path("<unknown>"),
            profile_name=profile_name,
            columns=columns,
        )
        y_true = table[:, true_indices]
        y_pred = table[:, pred_indices]
        drum = table[:, forecast_control_idx]
        rho = y_true[:, rho_idx]

        v_theta = np.gradient(drum, float(dt))
        drho_dt = np.gradient(rho, float(dt))
        descriptors = {
            "theta_peak": _signed_peak(drum, float(steady_state[CONTROL_COLUMN])),
            "rho_peak": _signed_peak(rho, float(steady_state["rho_dollars"])),
            "dtheta_dt_peak": _signed_peak(v_theta, 0.0),
            "drho_dt_peak": _signed_peak(drho_dt, 0.0),
        }

        y_true_scaled = _scale_targets_from_stats(scaling_stats, y_true)
        y_pred_scaled = _scale_targets_from_stats(scaling_stats, y_pred)
        abs_err = np.abs(y_true_scaled - y_pred_scaled)
        sq_err = (y_true_scaled - y_pred_scaled) ** 2
        row: dict[str, Any] = {
            "profile_id": str(profile_name),
            "MAE": float(np.mean(abs_err)),
            "MSE": float(np.mean(sq_err)),
            **descriptors,
        }

        if include_per_target:
            for idx, tgt in enumerate(TARGET_NAMES):
                row[f"MAE_{tgt}"] = float(np.mean(abs_err[:, idx]))
                row[f"MSE_{tgt}"] = float(np.mean(sq_err[:, idx]))

        return row

    per_profile_rows: list[dict[str, Any]] = []
    if forecast_h5 is not None:
        import h5py

        with h5py.File(forecast_h5, "r") as h5f:
            missing_profiles = [name for name in test_profile_names if name not in h5f]
            if missing_profiles:
                raise ValueError(
                    f"Forecast HDF5 {forecast_h5} is missing test profiles: {missing_profiles[:10]}"
                )
            for profile_name in test_profile_names:
                group = h5f[profile_name]
                table = group["data"][...].astype(np.float64)
                columns = _decode_h5_strings(group.attrs.get("columns", []))
                per_profile_rows.append(_evaluate_one_forecast(profile_name, table, columns))
    else:
        entries = list(ProfileDataset(scaled_h5, test_profile_names, "test"))
        if int(num_workers) <= 1:
            for profile_name, x_tensor, y_tensor in entries:
                per_profile_rows.append(_evaluate_one(profile_name, x_tensor, y_tensor))
        else:
            with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
                futures = [
                    executor.submit(_evaluate_one, profile_name, x_tensor, y_tensor)
                    for profile_name, x_tensor, y_tensor in entries
                ]
                for fut in as_completed(futures):
                    per_profile_rows.append(fut.result())

    per_profile_csv = out_dir / "per_profile_metrics_and_difficulty.csv"
    fieldnames = list(per_profile_rows[0].keys())
    _write_csv(per_profile_csv, fieldnames, per_profile_rows)

    descriptor_specs = ["theta_peak", "dtheta_dt_peak", "rho_peak", "drho_dt_peak"]

    generated_paths: list[str] = [str(per_profile_csv)]

    descriptor_metric_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    target_overlay: dict[str, dict[str, dict[str, list[float]]]] = {}
    for descriptor in descriptor_specs:
        values = np.asarray([float(row[descriptor]) for row in per_profile_rows], dtype=float)
        resolved_edges = _equal_width_edges(values, n_bins=int(n_bins))
        binned = bin_series(values, mode="fixed", n_bins=int(n_bins), edges=resolved_edges)
        label_to_idx = {label: idx for idx, label in enumerate(binned.label_names)}

        bin_col = f"{descriptor}_bin"
        for row, label in zip(per_profile_rows, binned.labels, strict=True):
            row[bin_col] = str(label)

        agg_rows: list[dict[str, Any]] = []
        descriptor_metric_rows[descriptor] = {}
        target_overlay[descriptor] = {}
        for metric in ("MAE", "MSE"):
            stats_rows = aggregate_metric_by_bin(per_profile_rows, metric_col=metric, bin_col=bin_col)
            for r in stats_rows:
                r["metric"] = metric
                bin_name = str(r["bin"])
                bin_idx = int(label_to_idx[bin_name])
                r["bin_idx"] = bin_idx
                r["bin_display"] = _short_bin_label(
                    float(binned.edges[bin_idx]),
                    float(binned.edges[bin_idx + 1]),
                    is_last=(bin_idx == len(binned.label_names) - 1),
                )
            stats_rows = sorted(stats_rows, key=lambda row: int(row["bin_idx"]))
            agg_rows.extend(stats_rows)
            descriptor_metric_rows[descriptor][metric] = [r for r in stats_rows]
            if include_per_target:
                target_overlay[descriptor][metric] = {}
                for tgt in TARGET_NAMES:
                    per_tgt_metric_col = f"{metric}_{tgt}"
                    target_stats_rows = aggregate_metric_by_bin(
                        per_profile_rows,
                        metric_col=per_tgt_metric_col,
                        bin_col=bin_col,
                    )
                    for tr in target_stats_rows:
                        target_bin = str(tr["bin"])
                        tr["bin_idx"] = int(label_to_idx[target_bin])
                    target_stats_rows = sorted(target_stats_rows, key=lambda row: int(row["bin_idx"]))
                    target_overlay[descriptor][metric][tgt] = [float(tr["mean"]) for tr in target_stats_rows]

        bins_csv = out_dir / f"bins_{descriptor}_metrics.csv"
        _write_csv(
            bins_csv,
            ["metric", "bin", "bin_idx", "bin_display", "count", "mean", "median", "std"],
            sorted(agg_rows, key=lambda r: (str(r["metric"]), int(r["bin_idx"]))),
        )
        generated_paths.append(str(bins_csv))

        edges_json = out_dir / f"bins_{descriptor}_edges.json"
        edges_payload = {
            "descriptor": descriptor,
            "binning_mode": "fixed_equal_width",
            "n_bins": int(n_bins),
            "edges": binned.edges.tolist(),
            "labels": binned.label_names,
        }
        edges_json.write_text(json.dumps(edges_payload, indent=2), encoding="utf-8")
        generated_paths.append(str(edges_json))

    combined_plot = out_dir / "difficulty_4x3_summary.png"
    _plot_summary_grid(
        per_profile_rows=per_profile_rows,
        descriptor_metrics=descriptor_metric_rows,
        target_overlay=target_overlay,
        include_per_target=bool(include_per_target),
        output_path=combined_plot,
        log_scale=True,
    )
    generated_paths.append(str(combined_plot))
    combined_plot_linear = out_dir / "difficulty_4x3_summary_linear.png"
    _plot_summary_grid(
        per_profile_rows=per_profile_rows,
        descriptor_metrics=descriptor_metric_rows,
        target_overlay=target_overlay,
        include_per_target=bool(include_per_target),
        output_path=combined_plot_linear,
        log_scale=False,
    )
    generated_paths.append(str(combined_plot_linear))

    # Persist per-profile table with bin labels included.
    _write_csv(per_profile_csv, list(per_profile_rows[0].keys()), per_profile_rows)

    manifest = {
        "dataset_path": str(scaled_h5),
        "forecast_h5": None if forecast_h5 is None else str(forecast_h5),
        "model_id": (
            "existing_forecast"
            if forecast_h5 is not None
            else ("ensemble" if len(model_paths) > 1 else Path(model_paths[0]).stem)
        ),
        "checkpoint_paths": [str(path) for path in model_paths],
        "binning": {
            "mode": "fixed_equal_width",
            "n_bins": int(n_bins),
        },
        "error_metric_target_space": "scaled",
        "artifacts": generated_paths,
    }
    manifest_path = out_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved per-profile table: {per_profile_csv}")
    print(f"Saved manifest: {manifest_path}")
    return {
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "per_profile_csv": str(per_profile_csv),
        "summary_plot": str(combined_plot),
        "summary_plot_linear": str(combined_plot_linear),
        "artifacts": generated_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate test-set forecast errors by transient difficulty bins.")
    parser.add_argument("--scaled-h5", type=Path, required=True, help="Scaled/split dataset path containing test split.")
    parser.add_argument(
        "--forecast-h5",
        type=Path,
        default=None,
        help="Optional existing rolling_forecasts.h5 to reuse instead of rerunning model inference.",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Single model checkpoint (.pt).")
    parser.add_argument("--ensemble-dir", type=Path, default=None, help="Directory containing ensemble checkpoints (.pt).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for CSV/plots/manifest.")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of equal-width bins per descriptor.")
    parser.add_argument("--dt", type=float, default=1.0, help="Timestep size for velocity estimate.")
    parser.add_argument("--config-path", type=Path, default=REPO_ROOT / "scripts" / "config.py")
    parser.add_argument("--include-per-target", action="store_true", help="Include per-target MAE/MSE columns.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for profile evaluation.")
    args = parser.parse_args()
    evaluate_testset_difficulty(
        scaled_h5=args.scaled_h5,
        forecast_h5=args.forecast_h5,
        model_path=args.model_path,
        ensemble_dir=args.ensemble_dir,
        out_dir=args.out_dir,
        n_bins=args.n_bins,
        dt=args.dt,
        config_path=args.config_path,
        include_per_target=bool(args.include_per_target),
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
