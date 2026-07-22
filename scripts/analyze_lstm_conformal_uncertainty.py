"""Analyze conformal LSTM uncertainty quality from saved conformal forecast outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


TARGET_ORDER: list[str] = [
    "Tf",
    "Tm",
    "Thp",
    "TN2",
    "Tsg",
    "T_steam_out",
    "x_steam_out",
    "c[1]",
    "c[2]",
    "c[3]",
    "c[4]",
    "c[5]",
    "c[6]",
    "n",
    "rho_dollars",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze coverage, horizon behavior, interval efficiency, and error/width relationships "
            "from conformal LSTM forecast outputs."
        ),
    )
    parser.add_argument("--methods-manifest-json", type=Path, default=None, help="Primary input: uq_methods_manifest.json from run_lstm_conformal_prediction.py.")
    parser.add_argument("--metadata-json", type=Path, default=None, help="Legacy path to conformal_calibration_metadata.json.")
    parser.add_argument("--coverage-metrics-json", type=Path, default=None, help="Legacy path to conformal_coverage_metrics.json.")
    parser.add_argument("--forecasts-h5", type=Path, default=None, help="Legacy path to conformal_forecasts.h5.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for analysis JSON/CSVs/plots.")
    parser.add_argument("--difficulty-quantile", type=float, default=0.20, help="Tail fraction for easy/hard bins.")
    parser.add_argument(
        "--show-plot-titles",
        dest="show_plot_titles",
        action="store_true",
        default=True,
        help="Show titles on generated plots (default).",
    )
    parser.add_argument(
        "--no-plot-titles",
        dest="show_plot_titles",
        action="store_false",
        help="Suppress titles on generated plots.",
    )
    parser.add_argument(
        "--max-forecast-plots",
        type=int,
        default=0,
        help=(
            "Number of individual test forecast profiles to plot from conformal_forecasts.h5. "
            "Use 0 to skip these profile-level plots."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def _decode_columns(raw: Any) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in list(raw)]


def _ordered_targets(target_names: list[str]) -> list[str]:
    known = [name for name in TARGET_ORDER if name in target_names]
    remaining = [name for name in target_names if name not in known]
    return known + remaining


def _physical_group_for_target_name(target_name: str) -> str:
    name = target_name.strip().lower()
    if name == "n" or "power" in name:
        return "power"
    if "reactivity" in name or name in {"rho", "rho_dollars"}:
        return "reactivity"
    if "q_to_steam" in name or "steam" in name:
        return "q_to_steam"
    if name.startswith("c[") or name.startswith("c_"):
        return "concentration"
    if name.startswith("t") or "temp" in name:
        return "temperature"
    return "other"


def _vary_group_color(base_color: str, *, shade_idx: int, n_shades: int) -> str:
    if n_shades <= 1:
        return base_color
    rgb = np.asarray(mcolors.to_rgb(base_color), dtype=np.float64)
    center = 0.5 * (n_shades - 1)
    offset = (shade_idx - center) / max(center, 1.0)
    if offset >= 0:
        mix = 0.22 * offset
        out = rgb * (1.0 - mix) + np.ones(3, dtype=np.float64) * mix
    else:
        mix = 0.18 * (-offset)
        out = rgb * (1.0 - mix)
    return mcolors.to_hex(np.clip(out, 0.0, 1.0))


def _target_color_map(target_names: list[str]) -> dict[str, str]:
    group_base_colors = {
        "temperature": "red",
        "concentration": "green",
        "power": "deeppink",
        "reactivity": "black",
        "q_to_steam": "gray",
        "other": "C4",
    }
    groups = [_physical_group_for_target_name(name) for name in target_names]
    group_counts: dict[str, int] = {}
    group_indices: dict[str, int] = {}
    for group in groups:
        group_counts[group] = group_counts.get(group, 0) + 1
    colors: dict[str, str] = {}
    for name, group in zip(target_names, groups, strict=True):
        idx_in_group = group_indices.get(group, 0)
        group_indices[group] = idx_in_group + 1
        shade_idx = 0 if group_counts[group] <= 1 else idx_in_group
        colors[name] = _vary_group_color(group_base_colors[group], shade_idx=shade_idx, n_shades=group_counts[group])
    return colors



def _pretty_target_label(target_name: str) -> str:
    mapping = {
        "TN2": r"$T_{N2}$",
        "Tm": r"$T_m$",
        "Thp": r"$T_{hp}$",
        "Tf": r"$T_f$",
        "Tsg": r"$T_{sg}$",
        "n": r"$n$",
        "rho_dollars": r"$\rho_{\$}$",
        "T_steam_out": r"$T_{\mathrm{steam,out}}$",
        "x_steam_out": r"$x_{\mathrm{steam,out}}$",
    }
    if target_name in mapping:
        return mapping[target_name]
    if target_name.startswith("c[") and target_name.endswith("]"):
        return rf"$c_{{{target_name[2:-1]}}}$"
    return target_name.replace("_", r"\_")


def _extract_profile_arrays(
    table: np.ndarray,
    columns: list[str],
    target_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = table[:, columns.index("t")]
    u = table[:, columns.index("u(t)")]
    y_true = np.column_stack([table[:, columns.index(f"x_true(t)_{name}")] for name in target_names])
    y_pred = np.column_stack([table[:, columns.index(f"x_pred(t)_{name}")] for name in target_names])
    lower = np.column_stack([table[:, columns.index(f"x_lower_conformal(t)_{name}")] for name in target_names])
    upper = np.column_stack([table[:, columns.index(f"x_upper_conformal(t)_{name}")] for name in target_names])
    width = np.column_stack([table[:, columns.index(f"x_width_conformal(t)_{name}")] for name in target_names])
    return t, u, y_true, y_pred, lower, upper, width


def load_forecasts(
    forecasts_h5: Path,
    target_names: list[str],
) -> dict[str, Any]:
    profile_records: list[dict[str, Any]] = []
    with h5py.File(forecasts_h5, "r") as h5f:
        is_joint = str(h5f.attrs.get("conformal_method", "")) == "joint_ensemble_normalized"
        profile_names = sorted(h5f.keys())
        if not profile_names:
            raise ValueError(f"No profile groups found in {forecasts_h5}.")
        for profile_name in profile_names:
            group = h5f[profile_name]
            if is_joint:
                required = {"t", "u", "y_true", "y_pred", "lower", "upper", "interval_width", "spread_scaled"}
                missing = required.difference(group.keys())
                if missing:
                    raise KeyError(f"Joint profile {profile_name!r} is missing datasets: {sorted(missing)}")
                profile_records.append({
                    "profile": str(profile_name), "t": group["t"][...], "u": group["u"][...],
                    "y_true": group["y_true"][...], "y_pred": group["y_pred"][...],
                    "lower": group["lower"][...], "upper": group["upper"][...],
                    "width": group["interval_width"][...], "spread_scaled": group["spread_scaled"][...],
                })
            else:
                if "data" not in group:
                    raise KeyError(f"Profile {profile_name!r} is missing dataset 'data'.")
                columns = _decode_columns(group.attrs.get("columns", h5f.attrs.get("columns", [])))
                table = group["data"][...].astype(np.float64)
                t, u, y_true, y_pred, lower, upper, width = _extract_profile_arrays(table, columns, target_names)
                profile_records.append({"profile": str(profile_name), "t": t, "u": u, "y_true": y_true,
                    "y_pred": y_pred, "lower": lower, "upper": upper, "width": width})

    max_steps = max(record["y_true"].shape[0] for record in profile_records)
    n_profiles = len(profile_records)
    n_targets = len(target_names)
    y_true_all = np.full((n_profiles, max_steps, n_targets), np.nan, dtype=np.float64)
    y_pred_all = np.full_like(y_true_all, np.nan)
    lower_all = np.full_like(y_true_all, np.nan)
    upper_all = np.full_like(y_true_all, np.nan)
    width_all = np.full_like(y_true_all, np.nan)
    t_all = np.full((n_profiles, max_steps), np.nan, dtype=np.float64)

    for profile_idx, record in enumerate(profile_records):
        steps = record["y_true"].shape[0]
        y_true_all[profile_idx, :steps, :] = record["y_true"]
        y_pred_all[profile_idx, :steps, :] = record["y_pred"]
        lower_all[profile_idx, :steps, :] = record["lower"]
        upper_all[profile_idx, :steps, :] = record["upper"]
        width_all[profile_idx, :steps, :] = record["width"]
        t_all[profile_idx, :steps] = record["t"]

    return {
        "profiles": profile_records,
        "profile_names": [record["profile"] for record in profile_records],
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "lower": lower_all,
        "upper": upper_all,
        "width": width_all,
        "t": t_all,
        "conformal_method": "joint_ensemble_normalized" if is_joint else "absolute",
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    if x.size < 3:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr)


def _nanquantile(values: np.ndarray, q: float, axis: int | tuple[int, ...]) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanquantile(values, q, axis=axis)


def compute_analysis(
    forecast_data: dict[str, Any],
    metadata: dict[str, Any],
    coverage_metrics: dict[str, Any],
    target_names: list[str],
    difficulty_quantile: float,
) -> dict[str, Any]:
    y_true = forecast_data["y_true"]
    y_pred = forecast_data["y_pred"]
    lower = forecast_data["lower"]
    upper = forecast_data["upper"]
    width = forecast_data["width"]

    abs_error = np.abs(y_true - y_pred)
    inside = (lower <= y_true) & (y_true <= upper)
    inside = np.where(np.isfinite(abs_error), inside.astype(np.float64), np.nan)
    half_width = width / 2.0
    joint_trajectory_coverage = float(np.mean([np.all((record["lower"] <= record["y_true"]) & (record["y_true"] <= record["upper"])) for record in forecast_data["profiles"]]))
    targetwise_trajectory_coverage = {
        name: float(np.mean([np.all((record["lower"][:, idx] <= record["y_true"][:, idx]) & (record["y_true"][:, idx] <= record["upper"][:, idx])) for record in forecast_data["profiles"]]))
        for idx, name in enumerate(target_names)
    }

    coverage_by_horizon_target = np.nanmean(inside, axis=0)
    mean_abs_error_by_horizon_target = np.nanmean(abs_error, axis=0)
    mean_width_by_horizon_target = np.nanmean(width, axis=0)
    mean_half_width_by_horizon_target = mean_width_by_horizon_target / 2.0
    test_q95_abs_error_by_horizon_target = _nanquantile(abs_error, 0.95, axis=0)
    test_q95_to_half_width_ratio = test_q95_abs_error_by_horizon_target / np.where(
        mean_half_width_by_horizon_target > 0.0,
        mean_half_width_by_horizon_target,
        np.nan,
    )

    spearman_by_target = {
        name: _spearman(mean_abs_error_by_horizon_target[:, idx], mean_half_width_by_horizon_target[:, idx])
        for idx, name in enumerate(target_names)
    }

    profile_mae = np.nanmean(abs_error, axis=(1, 2))
    profile_coverage = np.nanmean(inside, axis=(1, 2))
    q = float(difficulty_quantile)
    if not (0.0 < q < 0.5):
        raise ValueError("difficulty_quantile must be in (0.0, 0.5).")
    low_threshold = float(np.nanquantile(profile_mae, q))
    high_threshold = float(np.nanquantile(profile_mae, 1.0 - q))
    difficulty_masks = {
        "easy": profile_mae <= low_threshold,
        "medium": (profile_mae > low_threshold) & (profile_mae < high_threshold),
        "hard": profile_mae >= high_threshold,
    }
    difficulty_summary: dict[str, dict[str, float | int]] = {}
    for bin_name, mask in difficulty_masks.items():
        difficulty_summary[bin_name] = {
            "n_profiles": int(np.sum(mask)),
            "mean_profile_mae": float(np.nanmean(profile_mae[mask])) if np.any(mask) else float("nan"),
            "mean_coverage": float(np.nanmean(profile_coverage[mask])) if np.any(mask) else float("nan"),
        }

    average_width_by_target = np.nanmean(width, axis=(0, 1))
    median_width_by_target = np.nanmedian(width, axis=(0, 1))
    p90_width_by_target = _nanquantile(width, 0.90, axis=(0, 1))
    target_range = np.nanmax(y_true, axis=(0, 1)) - np.nanmin(y_true, axis=(0, 1))
    target_std = np.nanstd(y_true, axis=(0, 1))
    normalized_width_by_range = average_width_by_target / np.where(target_range > 0.0, target_range, np.nan)
    normalized_width_by_std = average_width_by_target / np.where(target_std > 0.0, target_std, np.nan)

    interval_efficiency = {
        name: {
            "average_width": float(average_width_by_target[idx]),
            "median_width": float(median_width_by_target[idx]),
            "p90_width": float(p90_width_by_target[idx]),
            "target_range": float(target_range[idx]),
            "target_std": float(target_std[idx]),
            "average_width_over_range": float(normalized_width_by_range[idx]),
            "average_width_over_std": float(normalized_width_by_std[idx]),
        }
        for idx, name in enumerate(target_names)
    }

    mean_coverage_by_horizon = np.nanmean(coverage_by_horizon_target, axis=1)
    mean_abs_error_by_horizon = np.nanmean(mean_abs_error_by_horizon_target, axis=1)
    mean_half_width_by_horizon = np.nanmean(mean_half_width_by_horizon_target, axis=1)
    mean_q95_ratio_by_horizon = np.nanmean(test_q95_to_half_width_ratio, axis=1)

    return {
        "alpha": float(metadata.get("alpha", float("nan"))),
        "nominal_coverage": float(1.0 - float(metadata.get("alpha", 0.05))),
        "horizon_mode": str(metadata.get("horizon_mode", "")),
        "conformal_method": str(forecast_data.get("conformal_method", metadata.get("conformal_method", "absolute"))),
        "primary_joint_trajectory_coverage": joint_trajectory_coverage,
        "targetwise_trajectory_coverage": targetwise_trajectory_coverage,
        "n_profiles": int(y_true.shape[0]),
        "n_horizons": int(y_true.shape[1]),
        "n_targets": int(y_true.shape[2]),
        "target_names": target_names,
        "coverage_metrics_json": coverage_metrics,
        "mean_coverage_by_horizon": mean_coverage_by_horizon.tolist(),
        "mean_abs_error_by_horizon": mean_abs_error_by_horizon.tolist(),
        "mean_half_width_by_horizon": mean_half_width_by_horizon.tolist(),
        "mean_test_q95_to_half_width_ratio_by_horizon": mean_q95_ratio_by_horizon.tolist(),
        "spearman_error_half_width_by_target": spearman_by_target,
        "profile_difficulty": {
            "difficulty_quantile": q,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "bins": difficulty_summary,
        },
        "interval_efficiency_by_target": interval_efficiency,
        "calibration_vs_test_residual_note": (
            "The required inputs do not contain full calibration residual arrays. "
            "This script therefore compares the test absolute-error 95th percentile to the saved conformal "
            "half-width, which is a calibration quantile proxy after descaling, rather than comparing full "
            "calibration residual distributions."
        ),
        "arrays": {
            "coverage_by_horizon_target": coverage_by_horizon_target.tolist(),
            "mean_abs_error_by_horizon_target": mean_abs_error_by_horizon_target.tolist(),
            "mean_width_by_horizon_target": mean_width_by_horizon_target.tolist(),
            "mean_half_width_by_horizon_target": mean_half_width_by_horizon_target.tolist(),
            "test_q95_abs_error_by_horizon_target": test_q95_abs_error_by_horizon_target.tolist(),
            "test_q95_to_half_width_ratio": test_q95_to_half_width_ratio.tolist(),
        },
    }


def _save_horizon_csv(path: Path, analysis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["horizon", "mean_coverage", "mean_abs_error", "mean_half_width", "mean_test_q95_to_half_width_ratio"])
        rows = zip(
            analysis["mean_coverage_by_horizon"],
            analysis["mean_abs_error_by_horizon"],
            analysis["mean_half_width_by_horizon"],
            analysis["mean_test_q95_to_half_width_ratio_by_horizon"],
            strict=True,
        )
        for horizon, values in enumerate(rows):
            writer.writerow([horizon, *values])


def _save_target_csv(path: Path, analysis: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "target",
                "spearman_error_half_width",
                "average_width",
                "median_width",
                "p90_width",
                "target_range",
                "target_std",
                "average_width_over_range",
                "average_width_over_std",
            ]
        )
        for target in analysis["target_names"]:
            efficiency = analysis["interval_efficiency_by_target"][target]
            writer.writerow(
                [
                    target,
                    analysis["spearman_error_half_width_by_target"][target],
                    efficiency["average_width"],
                    efficiency["median_width"],
                    efficiency["p90_width"],
                    efficiency["target_range"],
                    efficiency["target_std"],
                    efficiency["average_width_over_range"],
                    efficiency["average_width_over_std"],
                ]
            )


def _plot_mean_coverage_by_horizon(analysis: dict[str, Any], out_dir: Path, *, show_titles: bool) -> None:
    plt.rcParams.update({"font.size": 18})
    nominal = analysis["nominal_coverage"]
    y = np.asarray(analysis["mean_coverage_by_horizon"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y, label="mean coverage across targets")
    ax.axhline(nominal, color="black", linestyle="--", label=f"nominal={nominal:.3f}")
    ax.set_xlabel(r"Forecast horizon $t$")
    ax.set_ylabel(r"Empirical coverage $\hat{P}(y_t \in C_t)$")
    if show_titles:
        ax.set_title("Mean conformal coverage by forecast horizon")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_by_horizon_mean.png", dpi=150)
    plt.close(fig)


def _plot_target_grid(
    data: np.ndarray,
    target_names: list[str],
    *,
    out_path: Path,
    ylabel: str,
    title: str,
    nominal: float | None = None,
    show_titles: bool = True,
) -> None:
    plt.rcParams.update({"font.size": 18})
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
    axes = np.atleast_1d(axes).ravel()
    order = _ordered_targets(target_names)
    name_to_idx = {name: idx for idx, name in enumerate(target_names)}
    colors = _target_color_map(order)
    for plot_idx, name in enumerate(order):
        ax = axes[plot_idx]
        ax.plot(data[:, name_to_idx[name]], color=colors[name])
        if nominal is not None:
            ax.axhline(nominal, color="black", linestyle="--", linewidth=1.0)
        if show_titles:
            ax.set_title(_pretty_target_label(name))
        ax.set_xlabel(r"Forecast horizon $t$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
    for ax in axes[len(order):]:
        ax.axis("off")
    if show_titles:
        fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_error_vs_width_grid(analysis: dict[str, Any], out_dir: Path, *, show_titles: bool) -> None:
    plt.rcParams.update({"font.size": 18})
    target_names = list(analysis["target_names"])
    error = np.asarray(analysis["arrays"]["mean_abs_error_by_horizon_target"], dtype=float)
    half_width = np.asarray(analysis["arrays"]["mean_half_width_by_horizon_target"], dtype=float)
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
    axes = np.atleast_1d(axes).ravel()
    order = _ordered_targets(target_names)
    name_to_idx = {name: idx for idx, name in enumerate(target_names)}
    colors = _target_color_map(order)
    for plot_idx, name in enumerate(order):
        idx = name_to_idx[name]
        ax = axes[plot_idx]
        ax.plot(error[:, idx], label=r"Mean absolute error", color=colors[name])
        ax.plot(half_width[:, idx], label=r"Conformal half-width", color=colors[name], linestyle="--")
        if show_titles:
            ax.set_title(_pretty_target_label(name))
        ax.set_xlabel(r"Forecast horizon $t$")
        ax.set_ylabel(r"Physical units")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=18, loc="best")
    for ax in axes[len(order):]:
        ax.axis("off")
    if show_titles:
        fig.suptitle("Mean absolute error vs conformal half-width by forecast horizon", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "mean_abs_error_vs_half_width_by_target.png", dpi=150)
    plt.close(fig)


def _plot_spearman(analysis: dict[str, Any], out_dir: Path, *, show_titles: bool) -> None:
    plt.rcParams.update({"font.size": 18})
    target_names = _ordered_targets(list(analysis["target_names"]))
    values = [analysis["spearman_error_half_width_by_target"][name] for name in target_names]
    colors = _target_color_map(target_names)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(target_names, values, color=[colors[name] for name in target_names])
    ax.set_ylabel(r"Spearman rank correlation $\rho_s$")
    if show_titles:
        ax.set_title("Horizon-level correlation: mean absolute error vs conformal half-width")
    ax.set_xticks(np.arange(len(target_names)), [_pretty_target_label(name) for name in target_names])
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "spearman_error_half_width_by_target.png", dpi=150)
    plt.close(fig)


def _plot_difficulty_bins(analysis: dict[str, Any], out_dir: Path, *, show_titles: bool) -> None:
    plt.rcParams.update({"font.size": 18})
    bins = analysis["profile_difficulty"]["bins"]
    labels = ["easy", "medium", "hard"]
    coverage = [bins[label]["mean_coverage"] for label in labels]
    mae = [bins[label]["mean_profile_mae"] for label in labels]
    nominal = analysis["nominal_coverage"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    difficulty_colors = {"easy": "green", "medium": "gold", "hard": "red"}
    bar_colors = [difficulty_colors[label] for label in labels]
    axes[0].bar(labels, coverage, color=bar_colors)
    axes[0].axhline(nominal, color="black", linestyle="--", label=f"nominal={nominal:.3f}")
    axes[0].set_ylabel(r"Empirical coverage")
    if show_titles:
        axes[0].set_title("Coverage by profile-difficulty bin")
    axes[0].legend(loc="best")
    axes[1].bar(labels, mae, color=bar_colors)
    axes[1].set_ylabel(r"Mean profile MAE")
    if show_titles:
        axes[1].set_title("Mean absolute error by difficulty bin")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_by_profile_difficulty.png", dpi=150)
    plt.close(fig)


def _plot_interval_efficiency(analysis: dict[str, Any], out_dir: Path, *, show_titles: bool) -> None:
    plt.rcParams.update({"font.size": 18})
    target_names = _ordered_targets(list(analysis["target_names"]))
    by_target = analysis["interval_efficiency_by_target"]
    colors = _target_color_map(target_names)
    bar_colors = [colors[name] for name in target_names]
    over_std = [by_target[name]["average_width_over_std"] for name in target_names]
    over_range = [by_target[name]["average_width_over_range"] for name in target_names]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].bar(target_names, over_std, color=bar_colors)
    axes[0].set_ylabel(r"$\overline{W} / \sigma(y_{\mathrm{true}})$")
    if show_titles:
        axes[0].set_title("Interval width normalized by target standard deviation")
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[1].bar(target_names, over_range, color=bar_colors)
    axes[1].set_ylabel(r"$\overline{W} / \mathrm{range}(y_{\mathrm{true}})$")
    if show_titles:
        axes[1].set_title("Interval width normalized by target range")
    axes[1].set_xticks(np.arange(len(target_names)), [_pretty_target_label(name) for name in target_names])
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "interval_efficiency_by_target.png", dpi=150)
    plt.close(fig)


def _plot_forecast_profiles(
    forecast_data: dict[str, Any],
    target_names: list[str],
    out_dir: Path,
    *,
    max_plots: int,
    show_titles: bool,
) -> None:
    if max_plots <= 0:
        return
    plt.rcParams.update({"font.size": 18})
    plot_dir = out_dir / "forecast_profiles"
    plot_dir.mkdir(parents=True, exist_ok=True)
    order = _ordered_targets(target_names)
    name_to_idx = {name: idx for idx, name in enumerate(target_names)}
    colors = _target_color_map(order)
    rows, cols = 4, 4
    for record in forecast_data["profiles"][:max_plots]:
        fig, axes = plt.subplots(rows, cols, figsize=(24, 16), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        t = np.asarray(record["t"], dtype=float)
        axes[0].plot(t, record["u"], color="black", label=r"$u(t)$")
        if show_titles:
            axes[0].set_title(r"Control $u(t)$")
        axes[0].set_xlabel(r"Time $t$")
        axes[0].set_ylabel(r"$u(t)$")
        axes[0].grid(True, alpha=0.2)
        legend_handles = []
        legend_labels = []

        for plot_idx, name in enumerate(order, start=1):
            ax = axes[plot_idx]
            idx = name_to_idx[name]
            color = colors[name]
            truth_line = ax.plot(t, record["y_true"][:, idx], label="Truth", color="black")[0]
            pred_line = ax.plot(t, record["y_pred"][:, idx], label="Prediction", color=color)[0]
            lower_line = ax.plot(t, record["lower"][:, idx], label="Lower", color="tab:orange", linewidth=0.8)[0]
            upper_line = ax.plot(t, record["upper"][:, idx], label="Upper", color="tab:orange", linewidth=0.8)[0]
            interval = ax.fill_between(
                t,
                record["lower"][:, idx],
                record["upper"][:, idx],
                color="tab:orange",
                alpha=0.2,
                label="Conformal interval",
            )
            if not legend_handles:
                legend_handles = [truth_line, pred_line, lower_line, upper_line, interval]
                legend_labels = [handle.get_label() for handle in legend_handles]
            pretty_label = _pretty_target_label(name)
            if show_titles:
                ax.set_title(pretty_label)
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(pretty_label)
            ax.grid(True, alpha=0.2)

        if legend_handles:
            axes[3].legend(legend_handles, legend_labels, fontsize=16, loc="upper right")

        for ax in axes[len(order) + 1:]:
            ax.axis("off")
        if show_titles:
            fig.suptitle(f"Conformal forecast - {record['profile']}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe_name = str(record["profile"]).replace("/", "_")
        fig.savefig(plot_dir / f"conformal_forecast_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_plots(analysis: dict[str, Any], out_dir: Path, *, show_titles: bool) -> None:
    target_names = list(analysis["target_names"])
    coverage = np.asarray(analysis["arrays"]["coverage_by_horizon_target"], dtype=float)
    ratio = np.asarray(analysis["arrays"]["test_q95_to_half_width_ratio"], dtype=float)
    _plot_mean_coverage_by_horizon(analysis, out_dir, show_titles=show_titles)
    _plot_target_grid(
        coverage,
        target_names,
        out_path=out_dir / "coverage_by_horizon_target_grid.png",
        ylabel=r"Empirical coverage $\hat{P}(y_t \in C_t)$",
        title="Coverage by forecast horizon and target",
        nominal=analysis["nominal_coverage"],
        show_titles=show_titles,
    )
    _plot_error_vs_width_grid(analysis, out_dir, show_titles=show_titles)
    _plot_target_grid(
        ratio,
        target_names,
        out_path=out_dir / "test_q95_to_conformal_half_width_ratio_grid.png",
        ylabel=r"$Q_{0.95}(|e_t|) / h_t$",
        title="Test 95th-percentile error relative to conformal half-width",
        nominal=1.0,
        show_titles=show_titles,
    )
    _plot_spearman(analysis, out_dir, show_titles=show_titles)
    _plot_difficulty_bins(analysis, out_dir, show_titles=show_titles)
    _plot_interval_efficiency(analysis, out_dir, show_titles=show_titles)



METHOD_COLORS = {
    "ensemble_conformal_target_trajectory": "#1f77b4",
    "ensemble_conformal_target_horizon": "#ff7f0e",
    "absolute_conformal_target_horizon": "#2ca02c",
    "absolute_conformal_target_trajectory": "#d62728",
    "raw_ensemble_2sigma": "#9467bd",
}


def _load_standardized_method_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5f:
        target_names = _decode_columns(h5f.attrs["target_names"])
        alpha = float(h5f.attrs.get("alpha", np.nan))
        nominal = float(h5f.attrs.get("nominal_coverage", np.nan))
        profiles: dict[str, dict[str, np.ndarray]] = {}
        for profile_name in sorted(h5f.keys()):
            group = h5f[profile_name]
            profiles[profile_name] = {key: group[key][...] for key in group.keys()}
    return {"target_names": target_names, "alpha": alpha, "nominal_coverage": nominal, "profiles": profiles}


def _validate_method_comparison(method_data: dict[str, dict[str, Any]]) -> None:
    first_id = next(iter(method_data))
    first = method_data[first_id]
    profile_names = sorted(first["profiles"])
    target_names = first["target_names"]
    for method_id, data in method_data.items():
        if data["target_names"] != target_names:
            raise ValueError(f"Target order mismatch for {method_id}.")
        if sorted(data["profiles"]) != profile_names:
            raise ValueError(f"Profile set mismatch for {method_id}.")
        if not np.isclose(data["alpha"], first["alpha"], equal_nan=True):
            raise ValueError(f"Alpha mismatch for {method_id}.")
        for profile in profile_names:
            base = first["profiles"][profile]
            cur = data["profiles"][profile]
            for key in ("t", "u", "y_true", "y_true_scaled", "y_pred", "y_pred_scaled"):
                if not np.allclose(base[key], cur[key], rtol=0.0, atol=1e-7):
                    raise ValueError(f"Controlled comparison field {key!r} differs for method {method_id}, profile {profile}.")


def _method_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    profiles = data["profiles"]
    y_true = np.stack([profiles[name]["y_true"] for name in sorted(profiles)], axis=0)
    y_pred = np.stack([profiles[name]["y_pred"] for name in sorted(profiles)], axis=0)
    lower = np.stack([profiles[name]["lower"] for name in sorted(profiles)], axis=0)
    upper = np.stack([profiles[name]["upper"] for name in sorted(profiles)], axis=0)
    width = np.stack([profiles[name]["interval_width"] for name in sorted(profiles)], axis=0)
    return y_true, y_pred, lower, upper, width


def _interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    width = upper - lower
    return width + (2.0 / float(alpha)) * (np.maximum(lower - y_true, 0.0) + np.maximum(y_true - upper, 0.0))


def analyze_methods_manifest(manifest_json: Path, out_dir: Path, *, max_forecast_plots: int = 0) -> None:
    manifest = _load_json(manifest_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = manifest.get("uq_methods", [])
    if not methods:
        raise ValueError("uq_methods_manifest.json contains no uq_methods entries.")
    method_data = {entry["method_id"]: _load_standardized_method_h5(Path(entry["test_forecasts_path"])) for entry in methods}
    _validate_method_comparison(method_data)
    target_names = next(iter(method_data.values()))["target_names"]
    alpha = float(manifest["alpha"])
    nominal = 1.0 - alpha
    overall_rows = []
    target_rows = []
    horizon_rows = []
    tradeoff_rows = []
    all_json: dict[str, Any] = {"methods": {}}
    for entry in methods:
        method_id = entry["method_id"]
        label = entry["method_label"]
        metrics = _load_json(Path(entry["metrics_path"]))
        y_true, y_pred, lower, upper, width = _method_arrays(method_data[method_id])
        covered = (lower <= y_true) & (y_true <= upper)
        score = _interval_score(y_true, lower, upper, alpha)
        target_point = np.mean(covered, axis=(0, 1))
        target_traj = np.mean(np.all(covered, axis=1), axis=0)
        target_std = np.std(y_true.reshape(-1, y_true.shape[-1]), axis=0)
        mean_width_target = np.mean(width, axis=(0, 1))
        primary_type = metrics.get("primary_coverage_type", entry.get("primary_coverage_type", ""))
        overall_rows.append({
            "method_id": method_id,
            "method_label": label,
            "primary_coverage_type": primary_type,
            "primary_empirical_coverage": metrics.get("primary_empirical_coverage", np.nan),
            "overall_pointwise_coverage": float(np.mean(covered)),
            "mean_targetwise_trajectory_coverage": float(np.mean(target_traj)),
            "complete_multivariate_trajectory_coverage": float(np.mean(np.all(covered, axis=(1, 2)))),
            "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width)),
            "mean_normalized_width": float(np.nanmean(mean_width_target / np.where(target_std > 0, target_std, np.nan))),
            "mean_interval_score": float(np.mean(score)),
            "worst_target_coverage": float(np.min(target_point)),
            "max_abs_target_coverage_deviation_from_nominal": float(np.max(np.abs(target_point - nominal))),
            "mean_abs_target_coverage_deviation_from_nominal": float(np.mean(np.abs(target_point - nominal))),
        })
        tradeoff_rows.append({"method_id": method_id, "method_label": label, "coverage": float(np.mean(covered)), "mean_width": float(np.mean(width))})
        for j, target in enumerate(target_names):
            target_rows.append({
                "method_id": method_id, "method_label": label, "target": target,
                "targetwise_trajectory_coverage": float(target_traj[j]),
                "pointwise_coverage": float(target_point[j]),
                "mean_width": float(np.mean(width[:, :, j])),
                "median_width": float(np.median(width[:, :, j])),
                "p90_width": float(np.quantile(width[:, :, j], 0.9)),
                "normalized_width_by_std": float(np.mean(width[:, :, j]) / target_std[j]) if target_std[j] > 0 else float("nan"),
                "interval_score": float(np.mean(score[:, :, j])),
                "coverage_gap_from_nominal": float(target_point[j] - nominal),
            })
        for t in range(y_true.shape[1]):
            horizon_rows.append({
                "method_id": method_id, "method_label": label, "horizon": t,
                "pointwise_coverage": float(np.mean(covered[:, t, :])),
                "mean_width": float(np.mean(width[:, t, :])),
                "mean_absolute_error": float(np.mean(np.abs(y_true[:, t, :] - y_pred[:, t, :]))),
                "interval_score": float(np.mean(score[:, t, :])),
                "n_valid_profiles": int(y_true.shape[0]),
            })
        all_json["methods"][method_id] = {"overall": overall_rows[-1]}
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    write_csv(out_dir / "method_overall_summary.csv", overall_rows)
    write_csv(out_dir / "method_target_summary.csv", target_rows)
    write_csv(out_dir / "method_horizon_summary.csv", horizon_rows)
    write_csv(out_dir / "coverage_width_tradeoff.csv", tradeoff_rows)
    (out_dir / "uq_comparison_analysis.json").write_text(json.dumps(all_json, indent=2), encoding="utf-8")
    _plot_multi_method_summaries(out_dir, overall_rows, target_rows, horizon_rows, target_names, nominal)
    print(f"Saved multi-method UQ comparison analysis to: {out_dir}")


def _plot_multi_method_summaries(out_dir: Path, overall_rows, target_rows, horizon_rows, target_names, nominal: float) -> None:
    plot_dir = out_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    methods = [row["method_id"] for row in overall_rows]
    labels = {row["method_id"]: row["method_label"] for row in overall_rows}
    fig, ax = plt.subplots(figsize=(10, 6))
    for method_id in methods:
        rows = [r for r in horizon_rows if r["method_id"] == method_id]
        ax.plot([r["horizon"] for r in rows], [r["pointwise_coverage"] for r in rows], label=labels[method_id], color=METHOD_COLORS.get(method_id))
    ax.axhline(nominal, color="black", linestyle="--", linewidth=1.0, label="Nominal")
    ax.set_xlabel("Forecast horizon"); ax.set_ylabel("Pointwise coverage"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(plot_dir / "pointwise_coverage_by_horizon.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for method_id in methods:
        rows = [r for r in horizon_rows if r["method_id"] == method_id]
        ax.plot([r["horizon"] for r in rows], [r["mean_width"] for r in rows], label=labels[method_id], color=METHOD_COLORS.get(method_id))
    ax.set_xlabel("Forecast horizon"); ax.set_ylabel("Mean interval width"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(plot_dir / "mean_interval_width_by_horizon.png", dpi=150); plt.close(fig)
    x = np.arange(len(target_names)); width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, method_id in enumerate(methods):
        rows = [r for r in target_rows if r["method_id"] == method_id]
        ax.bar(x + (idx - (len(methods)-1)/2)*width, [r["targetwise_trajectory_coverage"] for r in rows], width=width, label=labels[method_id], color=METHOD_COLORS.get(method_id))
    ax.axhline(nominal, color="black", linestyle="--", linewidth=1.0); ax.set_xticks(x); ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_ylabel("Targetwise trajectory coverage"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plot_dir / "targetwise_trajectory_coverage_by_target.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in overall_rows:
        ax.scatter(row["mean_width"], row["overall_pointwise_coverage"], label=row["method_label"], color=METHOD_COLORS.get(row["method_id"]), s=60)
    ax.axhline(nominal, color="black", linestyle="--", linewidth=1.0); ax.set_xlabel("Mean interval width"); ax.set_ylabel("Overall pointwise coverage")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2); fig.tight_layout(); fig.savefig(plot_dir / "coverage_width_tradeoff.png", dpi=150); plt.close(fig)

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.methods_manifest_json is not None:
        analyze_methods_manifest(args.methods_manifest_json, args.out_dir, max_forecast_plots=args.max_forecast_plots)
        return
    if args.metadata_json is None or args.coverage_metrics_json is None or args.forecasts_h5 is None:
        raise ValueError("Provide --methods-manifest-json, or provide all legacy inputs: --metadata-json, --coverage-metrics-json, and --forecasts-h5.")
    metadata = _load_json(args.metadata_json)
    coverage_metrics = _load_json(args.coverage_metrics_json)
    target_names = [str(name) for name in metadata.get("target_names", [])]
    if not target_names:
        raise ValueError("metadata_json must contain a non-empty target_names list.")

    forecast_data = load_forecasts(args.forecasts_h5, target_names)
    analysis = compute_analysis(
        forecast_data,
        metadata,
        coverage_metrics,
        target_names,
        difficulty_quantile=args.difficulty_quantile,
    )

    (args.out_dir / "conformal_uncertainty_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    _save_horizon_csv(args.out_dir / "horizon_summary.csv", analysis)
    _save_target_csv(args.out_dir / "target_summary.csv", analysis)
    save_plots(analysis, args.out_dir, show_titles=args.show_plot_titles)
    _plot_forecast_profiles(
        forecast_data,
        target_names,
        args.out_dir,
        max_plots=args.max_forecast_plots,
        show_titles=args.show_plot_titles,
    )

    print(f"Saved conformal uncertainty analysis to: {args.out_dir}")
    print(f"Mean test coverage: {np.nanmean(analysis['mean_coverage_by_horizon']):.6f}")
    print("Profile difficulty coverage:")
    for bin_name, summary in analysis["profile_difficulty"]["bins"].items():
        print(
            f"  {bin_name}: n={summary['n_profiles']}, "
            f"coverage={summary['mean_coverage']:.6f}, mean_mae={summary['mean_profile_mae']:.6g}"
        )


if __name__ == "__main__":
    main()
