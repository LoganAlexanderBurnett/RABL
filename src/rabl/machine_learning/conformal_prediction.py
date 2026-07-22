"""Conformal prediction utilities for autoregressive LSTM forecasts.

This module is intentionally post-hoc: it calibrates already-trained LSTM models
on held-out calibration profiles and does not interact with RABL active-learning
or branch-selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .lstm_pipeline import (
    FORECAST_PLOT_TARGET_ORDER,
    STATE_DIM,
    TARGET_NAMES,
    _descale_feature_from_stats,
    _descale_targets_from_stats,
    _disable_y_offset_if_requested,
    _extract_control_series,
    _reorder_forecast_plot_targets,
    rolling_forecast,
)


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


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = 0.05
    horizon_mode: str = "per_horizon"
    residual_space: str = "scaled"
    state_dim: int = STATE_DIM
    control_channel: int = 0

    def validate(self) -> None:
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}.")
        if self.horizon_mode not in {"per_horizon", "global"}:
            raise ValueError("horizon_mode must be 'per_horizon' or 'global'.")
        if self.residual_space != "scaled":
            raise ValueError("Only residual_space='scaled' is currently supported.")
        if int(self.state_dim) < 1:
            raise ValueError("state_dim must be >= 1.")
        if int(self.control_channel) < 0:
            raise ValueError("control_channel must be >= 0.")


def _conformal_quantile(values: np.ndarray, *, alpha: float) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 1:
        raise ValueError("Cannot compute conformal quantile from an empty residual set.")
    sorted_values = np.sort(clean)
    k = int(ceil((clean.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), clean.size)
    return float(sorted_values[k - 1])


def calibrate_autoregressive_conformal(
    model: nn.Module,
    cal_profile_ds: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    alpha: float = 0.05,
    horizon_mode: str = "per_horizon",
    state_dim: int = STATE_DIM,
) -> dict[str, np.ndarray | float | int | str]:
    """Calibrate conformal residual quantiles from autoregressive calibration forecasts."""
    config = ConformalConfig(alpha=alpha, horizon_mode=horizon_mode, state_dim=state_dim)
    config.validate()

    residuals: list[np.ndarray] = []
    for _profile_name, x_profile, y_profile in cal_profile_ds:
        x_np = x_profile.detach().cpu().numpy() if isinstance(x_profile, torch.Tensor) else np.asarray(x_profile)
        y_true = y_profile.detach().cpu().numpy() if isinstance(y_profile, torch.Tensor) else np.asarray(y_profile)
        y_true = y_true.astype(np.float32, copy=False)
        y_pred = rolling_forecast(model, x_np.astype(np.float32, copy=False), state_dim=state_dim)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Calibration y_true/y_pred shapes differ: {y_true.shape} vs {y_pred.shape}.")
        residuals.append(np.abs(y_true - y_pred).astype(np.float32, copy=False))

    if not residuals:
        raise ValueError("No calibration profiles were provided; conformal calibration requires a cal split.")

    n_horizons = max(arr.shape[0] for arr in residuals)
    n_targets = residuals[0].shape[1]
    if any(arr.ndim != 2 or arr.shape[1] != n_targets for arr in residuals):
        raise ValueError("All calibration residual arrays must be 2D with a consistent target dimension.")

    if horizon_mode == "per_horizon":
        q_hat = np.empty((n_horizons, n_targets), dtype=np.float32)
        for t in range(n_horizons):
            at_horizon = [arr[t] for arr in residuals if arr.shape[0] > t]
            if not at_horizon:
                raise ValueError(f"No calibration residuals available at horizon {t}.")
            horizon_residuals = np.vstack(at_horizon)
            for j in range(n_targets):
                q_hat[t, j] = _conformal_quantile(horizon_residuals[:, j], alpha=alpha)
    else:
        pooled = np.vstack(residuals)
        q_hat = np.asarray([_conformal_quantile(pooled[:, j], alpha=alpha) for j in range(n_targets)], dtype=np.float32)

    return {
        "q_hat": q_hat,
        "alpha": float(alpha),
        "horizon_mode": horizon_mode,
        "n_cal_profiles": int(len(residuals)),
        "n_targets": int(n_targets),
        "n_horizons": int(n_horizons),
    }


def apply_conformal_intervals(
    y_pred_scaled: np.ndarray,
    q_hat: np.ndarray,
    *,
    horizon_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_pred_scaled = np.asarray(y_pred_scaled, dtype=np.float32)
    q_hat = np.asarray(q_hat, dtype=np.float32)
    if y_pred_scaled.ndim != 2:
        raise ValueError(f"y_pred_scaled must be 2D (steps, targets); got {y_pred_scaled.shape}.")
    if horizon_mode == "per_horizon":
        if q_hat.shape[1:] != y_pred_scaled.shape[1:] or q_hat.shape[0] < y_pred_scaled.shape[0]:
            raise ValueError(f"per_horizon q_hat must have shape at least {y_pred_scaled.shape}; got {q_hat.shape}.")
        q = q_hat[: y_pred_scaled.shape[0], :]
    elif horizon_mode == "global":
        if q_hat.shape != (y_pred_scaled.shape[1],):
            raise ValueError(f"global q_hat must have shape ({y_pred_scaled.shape[1]},); got {q_hat.shape}.")
        q = q_hat[None, :]
    else:
        raise ValueError("horizon_mode must be 'per_horizon' or 'global'.")
    lower = y_pred_scaled - q
    upper = y_pred_scaled + q
    return lower.astype(np.float32), upper.astype(np.float32)


def _assemble_conformal_table(t_series, u_series, y_true, y_pred, lower, upper) -> np.ndarray:
    if not (y_true.shape == y_pred.shape == lower.shape == upper.shape):
        raise ValueError("y_true, y_pred, lower, and upper must have identical shapes.")
    width = upper - lower
    return np.column_stack([t_series, u_series, y_true, y_pred, lower, upper, width]).astype(np.float32)


def conformal_rolling_forecast_profile(
    model: nn.Module,
    profile_name: str,
    x_profile: np.ndarray,
    y_profile: np.ndarray,
    *,
    conformal_result: dict[str, Any],
    scaling_stats: dict[str, Any],
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
) -> dict[str, Any]:
    x_profile = np.asarray(x_profile, dtype=np.float32)
    y_scaled = np.asarray(y_profile, dtype=np.float32)
    y_pred_scaled = rolling_forecast(model, x_profile, state_dim=state_dim)
    lower_scaled, upper_scaled = apply_conformal_intervals(
        y_pred_scaled,
        np.asarray(conformal_result["q_hat"]),
        horizon_mode=str(conformal_result["horizon_mode"]),
    )
    if not np.all(lower_scaled <= upper_scaled):
        raise ValueError(f"Conformal lower bound exceeds upper bound for profile {profile_name}.")

    y_true = _descale_targets_from_stats(scaling_stats, y_scaled)
    y_pred = _descale_targets_from_stats(scaling_stats, y_pred_scaled)
    lower = _descale_targets_from_stats(scaling_stats, lower_scaled)
    upper = _descale_targets_from_stats(scaling_stats, upper_scaled)
    if y_true.shape != y_pred.shape or lower.shape != upper.shape or y_true.shape != lower.shape:
        raise ValueError(f"Descaled conformal arrays have inconsistent shapes for profile {profile_name}.")

    t_series = np.arange(y_pred.shape[0], dtype=np.float32)
    u_series = _extract_control_series(x_profile, state_dim=state_dim, control_channel=control_channel)
    u_series = _descale_feature_from_stats(scaling_stats, u_series, state_dim + control_channel)
    table = _assemble_conformal_table(t_series, u_series, y_true, y_pred, lower, upper)
    return {
        "profile": str(profile_name),
        "table": table,
        "y_true": y_true,
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper,
        "scaled": {"y_true": y_scaled, "y_pred": y_pred_scaled, "lower": lower_scaled, "upper": upper_scaled},
    }


def save_conformal_forecasts_hdf5(
    forecasts: list[dict[str, Any]],
    *,
    output_path: Path,
    target_names: list[str] | None = None,
) -> None:
    target_names = list(TARGET_NAMES if target_names is None else target_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (["t", "u(t)"]
        + [f"x_true(t)_{name}" for name in target_names]
        + [f"x_pred(t)_{name}" for name in target_names]
        + [f"x_lower_conformal(t)_{name}" for name in target_names]
        + [f"x_upper_conformal(t)_{name}" for name in target_names]
        + [f"x_width_conformal(t)_{name}" for name in target_names])
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["columns"] = np.asarray(columns, dtype="S")
        for entry in forecasts:
            group = h5f.create_group(str(entry["profile"]))
            group.create_dataset("data", data=entry["table"])
            group.attrs["columns"] = np.asarray(columns, dtype="S")


def plot_conformal_forecast_profile_grid(
    *,
    x_profile: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_names: list[str],
    title: str,
    save_path: Path | None = None,
    control_name: str = "drumAngleDeg",
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
    close_figure: bool = True,
) -> plt.Figure:
    if not (y_true.shape == y_pred.shape == lower.shape == upper.shape):
        raise ValueError("y_true, y_pred, lower, and upper must have identical shapes.")
    target_names, reordered = _reorder_forecast_plot_targets(target_names, y_true, y_pred, lower, upper)
    y_true, y_pred, lower, upper = reordered
    control_series = x_profile[:, -1, state_dim + control_channel]
    nplots = len(target_names) + 1
    rows, cols = 4, 4
    if nplots > rows * cols:
        raise ValueError(f"Plot requires {nplots} panels but 4x4 supports only 16.")
    plt.rcParams.update({"font.size": 18})
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
    axes = np.atleast_1d(axes).ravel()
    axes[0].plot(control_series, label=r"$u(t)$")
    axes[0].set_title(r"Control $u(t)$")
    axes[0].set_xlabel(r"Forecast horizon $t$")
    axes[0].set_ylabel(r"$u(t)$")
    axes[0].grid(True, alpha=0.2)
    legend_handles = []
    legend_labels = []
    steps = np.arange(y_true.shape[0])
    for i, name in enumerate(target_names):
        ax = axes[i + 1]
        pretty_label = _pretty_target_label(name)
        truth_line = ax.plot(steps, y_true[:, i], label="Truth", color="black")[0]
        pred_line = ax.plot(steps, y_pred[:, i], label="Prediction", color="blue")[0]
        lower_line = ax.plot(steps, lower[:, i], label="Lower", color="tab:orange", linewidth=0.8)[0]
        upper_line = ax.plot(steps, upper[:, i], label="Upper", color="tab:orange", linewidth=0.8)[0]
        interval = ax.fill_between(steps, lower[:, i], upper[:, i], color="tab:orange", alpha=0.2, label="Conformal interval")
        if not legend_handles:
            legend_handles = [truth_line, pred_line, lower_line, upper_line, interval]
            legend_labels = [handle.get_label() for handle in legend_handles]
        ax.set_title(pretty_label)
        ax.set_xlabel(r"Forecast horizon $t$")
        ax.set_ylabel(pretty_label)
        _disable_y_offset_if_requested(ax, name)
        ax.grid(True, alpha=0.2)
    if legend_handles:
        axes[3].legend(legend_handles, legend_labels, fontsize=16, loc="upper right")
    for ax in axes[nplots:]:
        ax.axis("off")
    fig.suptitle(title, y=0.98, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved conformal forecast plot to: {save_path}")
    if close_figure:
        plt.close(fig)
    return fig


def _sigma_floor_vector(sigma_floor: float | np.ndarray, n_targets: int) -> np.ndarray:
    """Validate and explicitly normalize a scaled-space sigma floor to (targets,)."""
    floor = np.asarray(sigma_floor, dtype=np.float64)
    if floor.ndim == 0:
        floor = np.full(n_targets, float(floor), dtype=np.float64)
    elif floor.shape != (n_targets,):
        raise ValueError(f"sigma_floor must be a scalar or shape ({n_targets},); got {floor.shape}.")
    if not np.all(np.isfinite(floor)) or np.any(floor < 0.0):
        raise ValueError("sigma_floor values must be finite and nonnegative.")
    return floor.astype(np.float32)


def _ensemble_scaled_forecast(
    models: list[nn.Module], x_profile: np.ndarray, *, state_dim: int, ddof: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the bagging ensemble's autoregressive scaled-space forecast primitive."""
    if len(models) < 2:
        raise ValueError("Joint ensemble-normalized conformal prediction requires at least two models.")
    if ddof < 0 or ddof >= len(models):
        raise ValueError(f"ddof must be in [0, {len(models) - 1}]; got {ddof}.")
    # Import locally to keep ordinary conformal prediction independent of bagging.
    from .bagging_ensemble import ensemble_member_predictions_scaled

    member_predictions, mean, _default_spread = ensemble_member_predictions_scaled(models, x_profile, state_dim=state_dim)
    spread = np.std(member_predictions, axis=0, ddof=ddof).astype(np.float32)
    if not (member_predictions.ndim == 3 and mean.shape == spread.shape == member_predictions.shape[1:]):
        raise ValueError("Invalid ensemble forecast shapes.")
    return member_predictions, mean.astype(np.float32), spread


def calibrate_joint_ensemble_normalized_conformal(
    models: list[nn.Module],
    cal_profile_ds: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    alpha: float = 0.05,
    sigma_floor: float | np.ndarray = 1e-6,
    state_dim: int = STATE_DIM,
    ddof: int = 0,
) -> dict[str, Any]:
    """Calibrate one max-over-time-and-target normalized score per profile.

    Calibration remains entirely in scaled target space.  Each profile is one
    exchangeable calibration observation; individual timesteps are never pooled.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    scores: list[float] = []
    profile_names: list[str] = []
    raw_below_floor: list[np.ndarray] = []
    n_targets: int | None = None
    horizons: list[int] = []
    for profile_name, x_profile, y_profile in cal_profile_ds:
        x_np = x_profile.detach().cpu().numpy() if isinstance(x_profile, torch.Tensor) else np.asarray(x_profile)
        y_true = y_profile.detach().cpu().numpy() if isinstance(y_profile, torch.Tensor) else np.asarray(y_profile)
        members, mean, spread = _ensemble_scaled_forecast(models, x_np, state_dim=state_dim, ddof=ddof)
        if y_true.shape != mean.shape:
            raise ValueError(f"Calibration truth/mean shape mismatch for {profile_name}: {y_true.shape} vs {mean.shape}.")
        if n_targets is None:
            n_targets = int(mean.shape[1])
            floor = _sigma_floor_vector(sigma_floor, n_targets)
        elif mean.shape[1] != n_targets:
            raise ValueError("Calibration profiles have inconsistent target dimensions.")
        effective_scale = spread + floor[None, :]
        normalized_error = np.abs(y_true - mean) / effective_scale
        score = float(np.max(normalized_error))
        if not np.isfinite(score):
            raise ValueError(f"Non-finite calibration score for profile {profile_name}.")
        scores.append(score)
        profile_names.append(str(profile_name))
        horizons.append(int(mean.shape[0]))
        raw_below_floor.append(spread < floor[None, :])
    if not scores or n_targets is None:
        raise ValueError("No calibration profiles were provided; a non-empty cal split is required.")
    scores_array = np.asarray(scores, dtype=np.float32)
    q_joint = _conformal_quantile(scores_array, alpha=alpha)
    below = np.concatenate(raw_below_floor, axis=0)
    return {
        "conformal_method": "joint_ensemble_normalized",
        "alpha": float(alpha),
        "q_joint": float(q_joint),
        "calibration_scores": scores_array,
        "calibration_profile_names": profile_names,
        "n_cal_profiles": len(scores),
        "n_targets": n_targets,
        "n_horizons_min": int(min(horizons)),
        "n_horizons_max": int(max(horizons)),
        "ensemble_ddof": int(ddof),
        "sigma_floor": floor,
        "raw_spread_below_floor_fraction": float(np.mean(below)),
        "raw_spread_below_floor_fraction_by_target": np.mean(below, axis=0).astype(np.float32),
        "residual_space": "scaled",
    }


def joint_ensemble_conformal_forecast_profile(
    models: list[nn.Module], profile_name: str, x_profile: np.ndarray, y_profile: np.ndarray, *,
    conformal_result: dict[str, Any], scaling_stats: dict[str, Any], state_dim: int = STATE_DIM,
    control_channel: int = 0,
) -> dict[str, Any]:
    """Forecast one profile with profile/time/target adaptive joint intervals."""
    x_np = np.asarray(x_profile, dtype=np.float32)
    y_scaled = np.asarray(y_profile, dtype=np.float32)
    members, mean_scaled, spread_scaled = _ensemble_scaled_forecast(
        models, x_np, state_dim=state_dim, ddof=int(conformal_result["ensemble_ddof"])
    )
    if y_scaled.shape != mean_scaled.shape:
        raise ValueError(f"Truth/ensemble mean shape mismatch for {profile_name}.")
    floor = _sigma_floor_vector(conformal_result["sigma_floor"], mean_scaled.shape[1])
    effective_scale = spread_scaled + floor[None, :]
    q_joint = float(conformal_result["q_joint"])
    if not np.isfinite(q_joint):
        raise ValueError("q_joint must be finite.")
    lower_scaled = mean_scaled - q_joint * effective_scale
    upper_scaled = mean_scaled + q_joint * effective_scale
    if not np.all(np.isfinite(lower_scaled)) or not np.all(lower_scaled <= upper_scaled):
        raise ValueError(f"Invalid joint conformal intervals for {profile_name}.")
    normalized_abs_error = np.abs(y_scaled - mean_scaled) / effective_scale
    y_true = _descale_targets_from_stats(scaling_stats, y_scaled)
    y_mean = _descale_targets_from_stats(scaling_stats, mean_scaled)
    lower = _descale_targets_from_stats(scaling_stats, lower_scaled)
    upper = _descale_targets_from_stats(scaling_stats, upper_scaled)
    t = np.arange(mean_scaled.shape[0], dtype=np.float32)
    u = _extract_control_series(x_np, state_dim=state_dim, control_channel=control_channel)
    u = _descale_feature_from_stats(scaling_stats, u, state_dim + control_channel)
    return {
        "profile": str(profile_name), "t": t, "u": u, "y_true": y_true, "y_pred": y_mean,
        "lower": lower, "upper": upper,
        "member_predictions_scaled": members, "mean_scaled": mean_scaled, "spread_scaled": spread_scaled,
        "effective_scale_scaled": effective_scale, "normalized_abs_error": normalized_abs_error,
        "scaled": {"y_true": y_scaled, "lower": lower_scaled, "upper": upper_scaled},
    }


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    x_rank = np.argsort(np.argsort(np.asarray(x)[mask]))
    y_rank = np.argsort(np.argsort(np.asarray(y)[mask]))
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def joint_ensemble_coverage_metrics(
    forecasts: list[dict[str, Any]], target_names: list[str], *, sigma_floor: float | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute complete-trajectory coverage separately from marginal coverage."""
    if not forecasts:
        raise ValueError("Cannot compute coverage for no forecasts.")
    covered_profiles: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for entry in forecasts:
        inside = (entry["lower"] <= entry["y_true"]) & (entry["y_true"] <= entry["upper"])
        covered_profiles.append(inside)
        widths.append(entry["upper"] - entry["lower"])
        spreads.append(entry["spread_scaled"])
        errors.append(np.abs(entry["mean_scaled"] - entry["scaled"]["y_true"]))
    trajectory = np.asarray([bool(np.all(mask)) for mask in covered_profiles])
    targetwise = np.asarray([np.all(mask, axis=0) for mask in covered_profiles])
    points = np.concatenate(covered_profiles, axis=0)
    width = np.concatenate(widths, axis=0)
    spread = np.concatenate(spreads, axis=0)
    error = np.concatenate(errors, axis=0)
    metrics: dict[str, Any] = {
        "primary_joint_trajectory_coverage": float(np.mean(trajectory)),
        "trajectory_covered_by_profile": trajectory.tolist(),
        "targetwise_trajectory_coverage": {name: float(value) for name, value in zip(target_names, np.mean(targetwise, axis=0))},
        "marginal_pointwise_coverage_overall": float(np.mean(points)),
        "marginal_pointwise_coverage_by_target": {name: float(value) for name, value in zip(target_names, np.mean(points, axis=0))},
        "mean_interval_width_overall": float(np.mean(width)), "median_interval_width_overall": float(np.median(width)),
        "mean_interval_width_by_target": {name: float(value) for name, value in zip(target_names, np.mean(width, axis=0))},
        "mean_ensemble_spread": float(np.mean(spread)), "median_ensemble_spread": float(np.median(spread)),
        "spearman_spread_absolute_error_overall": _spearman_rank_correlation(spread.ravel(), error.ravel()),
        "spearman_spread_absolute_error_by_target": {
            name: _spearman_rank_correlation(spread[:, idx], error[:, idx])
            for idx, name in enumerate(target_names)
        },
    }
    if sigma_floor is not None:
        floor = _sigma_floor_vector(sigma_floor, spread.shape[1])
        below = spread < floor[None, :]
        metrics["raw_spread_below_floor_fraction"] = float(np.mean(below))
        metrics["raw_spread_below_floor_fraction_by_target"] = {
            name: float(value) for name, value in zip(target_names, np.mean(below, axis=0))
        }
    # Variable horizons are represented without zero padding.
    max_horizon = max(entry["y_true"].shape[0] for entry in forecasts)
    metrics["marginal_pointwise_coverage_by_horizon"] = [
        float(np.mean(np.concatenate([mask[t:t + 1] for mask in covered_profiles if mask.shape[0] > t], axis=0)))
        for t in range(max_horizon)
    ]
    metrics["mean_interval_width_by_horizon"] = [
        float(np.mean(np.concatenate([arr[t:t + 1] for arr in widths if arr.shape[0] > t], axis=0)))
        for t in range(max_horizon)
    ]
    return metrics


def save_joint_ensemble_conformal_forecasts_hdf5(
    forecasts: list[dict[str, Any]], *, output_path: Path, metadata: dict[str, Any],
    target_names: list[str] | None = None, save_member_forecasts: bool = True,
) -> None:
    """Save auditable joint conformal arrays without overwriting plain outputs."""
    names = list(TARGET_NAMES if target_names is None else target_names)
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["conformal_method"] = "joint_ensemble_normalized"
        h5f.attrs["target_names"] = np.asarray(names, dtype="S")
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, np.integer, np.floating)):
                h5f.attrs[key] = value
            elif isinstance(value, np.ndarray):
                h5f.attrs[key] = value
        for entry in forecasts:
            group = h5f.create_group(entry["profile"])
            group.create_dataset("t", data=entry["t"])
            group.create_dataset("u", data=entry["u"])
            for key in ("y_true", "y_pred", "lower", "upper", "mean_scaled", "spread_scaled", "effective_scale_scaled", "normalized_abs_error"):
                group.create_dataset(key, data=entry[key], compression="gzip")
            group.create_dataset("interval_width", data=entry["upper"] - entry["lower"], compression="gzip")
            if save_member_forecasts:
                group.create_dataset("member_predictions_scaled", data=entry["member_predictions_scaled"], compression="gzip")


UQ_METHODS: dict[str, dict[str, Any]] = {
    "ensemble_conformal_target_trajectory": {
        "label": "Ensemble conformal — target trajectory",
        "temporal_mode": "trajectory",
        "residual_type": "ensemble_normalized",
        "primary_coverage_type": "targetwise_trajectory_coverage",
        "uses_conformal_quantile": True,
        "uses_ensemble_normalization": True,
    },
    "ensemble_conformal_target_horizon": {
        "label": "Ensemble conformal — target/horizon",
        "temporal_mode": "per_horizon",
        "residual_type": "ensemble_normalized",
        "primary_coverage_type": "marginal_pointwise_coverage_by_horizon_target",
        "uses_conformal_quantile": True,
        "uses_ensemble_normalization": True,
    },
    "absolute_conformal_target_horizon": {
        "label": "Absolute conformal — target/horizon",
        "temporal_mode": "per_horizon",
        "residual_type": "absolute",
        "primary_coverage_type": "marginal_pointwise_coverage_by_horizon_target",
        "uses_conformal_quantile": True,
        "uses_ensemble_normalization": False,
    },
    "absolute_conformal_target_trajectory": {
        "label": "Absolute conformal — target trajectory",
        "temporal_mode": "trajectory",
        "residual_type": "absolute",
        "primary_coverage_type": "targetwise_trajectory_coverage",
        "uses_conformal_quantile": True,
        "uses_ensemble_normalization": False,
    },
    "raw_ensemble_2sigma": {
        "label": "Raw ensemble ±2σ",
        "temporal_mode": "pointwise_uncalibrated",
        "residual_type": "raw_ensemble_spread",
        "primary_coverage_type": "empirical_only_no_conformal_guarantee",
        "uses_conformal_quantile": False,
        "uses_ensemble_normalization": False,
    },
}
DEFAULT_UQ_METHODS = tuple(UQ_METHODS.keys())


def _quantile_and_index(values: np.ndarray, *, alpha: float) -> tuple[float, int, int]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 1:
        raise ValueError("Cannot compute conformal quantile from an empty score set.")
    sorted_values = np.sort(clean)
    k = int(ceil((clean.size + 1) * (1.0 - float(alpha))))
    k = min(max(k, 1), clean.size)
    return float(sorted_values[k - 1]), int(k), int(clean.size)


def compute_ensemble_profile_forecast(
    models: list[nn.Module],
    profile_name: str,
    x_profile: np.ndarray,
    y_profile: np.ndarray,
    *,
    scaling_stats: dict[str, Any],
    state_dim: int = STATE_DIM,
    control_channel: int = 0,
    ddof: int = 0,
) -> dict[str, Any]:
    """Compute one reusable scaled-space ensemble forecast for one profile."""
    x_np = np.asarray(x_profile, dtype=np.float32)
    y_scaled = np.asarray(y_profile, dtype=np.float32)
    members, mean_scaled, spread_scaled = _ensemble_scaled_forecast(models, x_np, state_dim=state_dim, ddof=ddof)
    if y_scaled.shape != mean_scaled.shape:
        raise ValueError(f"Truth/ensemble mean shape mismatch for {profile_name}: {y_scaled.shape} vs {mean_scaled.shape}.")
    t = np.arange(mean_scaled.shape[0], dtype=np.float32)
    u_scaled = _extract_control_series(x_np, state_dim=state_dim, control_channel=control_channel)
    u = _descale_feature_from_stats(scaling_stats, u_scaled, state_dim + control_channel)
    return {
        "profile": str(profile_name),
        "t": t,
        "u": u.astype(np.float32),
        "y_true_scaled": y_scaled.astype(np.float32),
        "member_predictions_scaled": members.astype(np.float32),
        "mean_scaled": mean_scaled.astype(np.float32),
        "spread_scaled": spread_scaled.astype(np.float32),
        "y_true": _descale_targets_from_stats(scaling_stats, y_scaled).astype(np.float32),
        "y_pred": _descale_targets_from_stats(scaling_stats, mean_scaled).astype(np.float32),
    }


def calibrate_ensemble_normalized_conformal(
    calibration_forecasts: list[dict[str, Any]],
    *,
    alpha: float,
    sigma_floor: float | np.ndarray = 1e-6,
    temporal_mode: str,
) -> dict[str, Any]:
    """Calibrate ensemble-normalized conformal quantiles per target."""
    if temporal_mode not in {"trajectory", "per_horizon"}:
        raise ValueError("temporal_mode must be 'trajectory' or 'per_horizon'.")
    if not calibration_forecasts:
        raise ValueError("At least one calibration forecast is required.")
    n_targets = int(calibration_forecasts[0]["mean_scaled"].shape[1])
    floor = _sigma_floor_vector(sigma_floor, n_targets)
    normalized: list[np.ndarray] = []
    profile_names: list[str] = []
    for entry in calibration_forecasts:
        y_true = np.asarray(entry["y_true_scaled"], dtype=np.float32)
        mean = np.asarray(entry["mean_scaled"], dtype=np.float32)
        spread = np.asarray(entry["spread_scaled"], dtype=np.float32)
        if y_true.shape != mean.shape or mean.shape != spread.shape or mean.shape[1] != n_targets:
            raise ValueError(f"Invalid calibration forecast shapes for profile {entry['profile']}.")
        normalized.append(np.abs(y_true - mean) / (spread + floor[None, :]))
        profile_names.append(str(entry["profile"]))
    max_horizon = max(arr.shape[0] for arr in normalized)
    if temporal_mode == "trajectory":
        scores = np.vstack([np.max(arr, axis=0) for arr in normalized]).astype(np.float32)
        q = np.empty(n_targets, dtype=np.float32)
        k = np.empty(n_targets, dtype=np.int32)
        n = np.empty(n_targets, dtype=np.int32)
        for j in range(n_targets):
            q[j], k[j], n[j] = _quantile_and_index(scores[:, j], alpha=alpha)
        return {
            "method_id": "ensemble_conformal_target_trajectory",
            "alpha": float(alpha),
            "temporal_mode": temporal_mode,
            "residual_type": "ensemble_normalized",
            "q_by_target": q,
            "calibration_scores_by_profile_target": scores,
            "quantile_index_by_target": k,
            "score_count_by_target": n,
            "calibration_profile_names": profile_names,
            "sigma_floor": floor,
        }
    q_ht = np.empty((max_horizon, n_targets), dtype=np.float32)
    k_ht = np.empty((max_horizon, n_targets), dtype=np.int32)
    n_h = np.empty(max_horizon, dtype=np.int32)
    for t in range(max_horizon):
        rows = [arr[t] for arr in normalized if arr.shape[0] > t]
        n_h[t] = len(rows)
        stacked = np.vstack(rows)
        for j in range(n_targets):
            q_ht[t, j], k_ht[t, j], _ = _quantile_and_index(stacked[:, j], alpha=alpha)
    return {
        "method_id": "ensemble_conformal_target_horizon",
        "alpha": float(alpha),
        "temporal_mode": temporal_mode,
        "residual_type": "ensemble_normalized",
        "q_by_horizon_target": q_ht,
        "calibration_count_by_horizon": n_h,
        "quantile_index_by_horizon_target": k_ht,
        "calibration_profile_names": profile_names,
        "sigma_floor": floor,
    }


def calibrate_absolute_conformal(
    calibration_forecasts: list[dict[str, Any]],
    *,
    alpha: float,
    temporal_mode: str,
) -> dict[str, Any]:
    """Calibrate absolute-residual conformal quantiles around ensemble means."""
    if temporal_mode not in {"trajectory", "per_horizon"}:
        raise ValueError("temporal_mode must be 'trajectory' or 'per_horizon'.")
    if not calibration_forecasts:
        raise ValueError("At least one calibration forecast is required.")
    residuals: list[np.ndarray] = []
    profile_names: list[str] = []
    n_targets = int(calibration_forecasts[0]["mean_scaled"].shape[1])
    for entry in calibration_forecasts:
        y_true = np.asarray(entry["y_true_scaled"], dtype=np.float32)
        mean = np.asarray(entry["mean_scaled"], dtype=np.float32)
        if y_true.shape != mean.shape or mean.shape[1] != n_targets:
            raise ValueError(f"Invalid absolute calibration shapes for profile {entry['profile']}.")
        residuals.append(np.abs(y_true - mean).astype(np.float32))
        profile_names.append(str(entry["profile"]))
    max_horizon = max(arr.shape[0] for arr in residuals)
    if temporal_mode == "trajectory":
        scores = np.vstack([np.max(arr, axis=0) for arr in residuals]).astype(np.float32)
        q = np.empty(n_targets, dtype=np.float32)
        k = np.empty(n_targets, dtype=np.int32)
        n = np.empty(n_targets, dtype=np.int32)
        for j in range(n_targets):
            q[j], k[j], n[j] = _quantile_and_index(scores[:, j], alpha=alpha)
        return {
            "method_id": "absolute_conformal_target_trajectory",
            "alpha": float(alpha),
            "temporal_mode": temporal_mode,
            "residual_type": "absolute",
            "q_by_target": q,
            "calibration_scores_by_profile_target": scores,
            "quantile_index_by_target": k,
            "score_count_by_target": n,
            "calibration_profile_names": profile_names,
        }
    q_ht = np.empty((max_horizon, n_targets), dtype=np.float32)
    k_ht = np.empty((max_horizon, n_targets), dtype=np.int32)
    n_h = np.empty(max_horizon, dtype=np.int32)
    for t in range(max_horizon):
        rows = [arr[t] for arr in residuals if arr.shape[0] > t]
        n_h[t] = len(rows)
        stacked = np.vstack(rows)
        for j in range(n_targets):
            q_ht[t, j], k_ht[t, j], _ = _quantile_and_index(stacked[:, j], alpha=alpha)
    return {
        "method_id": "absolute_conformal_target_horizon",
        "alpha": float(alpha),
        "temporal_mode": temporal_mode,
        "residual_type": "absolute",
        "q_by_horizon_target": q_ht,
        "calibration_count_by_horizon": n_h,
        "quantile_index_by_horizon_target": k_ht,
        "calibration_profile_names": profile_names,
    }


def apply_uq_method(
    forecast: dict[str, Any],
    *,
    method_id: str,
    calibration_result: dict[str, Any] | None,
    scaling_stats: dict[str, Any],
    alpha: float,
    sigma_floor: float | np.ndarray = 1e-6,
    include_member_predictions: bool = False,
) -> dict[str, Any]:
    """Apply one of the five UQ methods to a cached ensemble forecast."""
    if method_id not in UQ_METHODS:
        raise ValueError(f"Unsupported UQ method: {method_id}")
    mean = np.asarray(forecast["mean_scaled"], dtype=np.float32)
    spread = np.asarray(forecast["spread_scaled"], dtype=np.float32)
    n_steps, n_targets = mean.shape
    effective_scale = None
    normalized_abs_error = None
    if method_id == "raw_ensemble_2sigma":
        lower_scaled = mean - 2.0 * spread
        upper_scaled = mean + 2.0 * spread
    else:
        if calibration_result is None:
            raise ValueError(f"calibration_result is required for method {method_id}.")
        if UQ_METHODS[method_id]["uses_ensemble_normalization"]:
            floor = _sigma_floor_vector(calibration_result.get("sigma_floor", sigma_floor), n_targets)
            effective_scale = spread + floor[None, :]
            normalized_abs_error = np.abs(np.asarray(forecast["y_true_scaled"]) - mean) / effective_scale
            scale = effective_scale
        else:
            scale = np.ones_like(mean, dtype=np.float32)
        if "q_by_target" in calibration_result:
            q = np.asarray(calibration_result["q_by_target"], dtype=np.float32)[None, :]
        elif "q_by_horizon_target" in calibration_result:
            q_all = np.asarray(calibration_result["q_by_horizon_target"], dtype=np.float32)
            if q_all.shape[0] < n_steps or q_all.shape[1] != n_targets:
                raise ValueError(f"Quantile shape {q_all.shape} cannot cover forecast shape {mean.shape}.")
            q = q_all[:n_steps, :]
        else:
            raise ValueError(f"Calibration result for {method_id} has no recognized quantile array.")
        lower_scaled = mean - q * scale
        upper_scaled = mean + q * scale
    if not np.all(np.isfinite(lower_scaled)) or not np.all(lower_scaled <= upper_scaled):
        raise ValueError(f"Invalid intervals for method {method_id}, profile {forecast['profile']}.")
    y_true = np.asarray(forecast["y_true"], dtype=np.float32)
    y_pred = np.asarray(forecast["y_pred"], dtype=np.float32)
    lower = _descale_targets_from_stats(scaling_stats, lower_scaled).astype(np.float32)
    upper = _descale_targets_from_stats(scaling_stats, upper_scaled).astype(np.float32)
    entry = {
        "profile": str(forecast["profile"]),
        "t": np.asarray(forecast["t"], dtype=np.float32),
        "u": np.asarray(forecast["u"], dtype=np.float32),
        "y_true": y_true,
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper,
        "interval_width": upper - lower,
        "y_true_scaled": np.asarray(forecast["y_true_scaled"], dtype=np.float32),
        "y_pred_scaled": mean,
        "lower_scaled": lower_scaled.astype(np.float32),
        "upper_scaled": upper_scaled.astype(np.float32),
        "spread_scaled": spread,
    }
    if effective_scale is not None:
        entry["effective_scale_scaled"] = effective_scale.astype(np.float32)
        entry["normalized_abs_error"] = normalized_abs_error.astype(np.float32)
    if include_member_predictions:
        entry["member_predictions_scaled"] = np.asarray(forecast["member_predictions_scaled"], dtype=np.float32)
    return entry


def compute_uq_coverage_metrics(
    forecasts: list[dict[str, Any]],
    target_names: list[str],
    *,
    alpha: float,
    primary_coverage_type: str,
    no_conformal_guarantee: bool = False,
) -> dict[str, Any]:
    """Compute standardized coverage/efficiency metrics for any UQ method."""
    if not forecasts:
        raise ValueError("Cannot compute UQ coverage metrics for no forecasts.")
    nominal = 1.0 - float(alpha)
    covered = [(e["lower"] <= e["y_true"]) & (e["y_true"] <= e["upper"]) for e in forecasts]
    widths = [np.asarray(e["interval_width"], dtype=np.float64) for e in forecasts]
    errors = [np.abs(np.asarray(e["y_true"], dtype=np.float64) - np.asarray(e["y_pred"], dtype=np.float64)) for e in forecasts]
    y_true_all = np.concatenate([np.asarray(e["y_true"], dtype=np.float64) for e in forecasts], axis=0)
    points = np.concatenate(covered, axis=0)
    width_all = np.concatenate(widths, axis=0)
    err_all = np.concatenate(errors, axis=0)
    target_std = np.std(y_true_all, axis=0)
    target_range = np.ptp(y_true_all, axis=0)
    safe_std = np.where(target_std > 0, target_std, np.nan)
    safe_range = np.where(target_range > 0, target_range, np.nan)
    lower_miss = [np.maximum(np.asarray(e["lower"]) - np.asarray(e["y_true"]), 0.0) for e in forecasts]
    upper_miss = [np.maximum(np.asarray(e["y_true"]) - np.asarray(e["upper"]), 0.0) for e in forecasts]
    interval_scores = [w + (2.0 / float(alpha)) * (lo + hi) for w, lo, hi in zip(widths, lower_miss, upper_miss)]
    score_all = np.concatenate(interval_scores, axis=0)
    targetwise_trajectory = np.asarray([np.all(mask, axis=0) for mask in covered], dtype=bool)
    complete_trajectory = np.asarray([np.all(mask) for mask in covered], dtype=bool)
    max_horizon = max(mask.shape[0] for mask in covered)
    by_horizon = []
    by_horizon_target = np.full((max_horizon, len(target_names)), np.nan, dtype=np.float64)
    width_by_horizon = []
    mae_by_horizon = []
    score_by_horizon = []
    n_valid = []
    for t in range(max_horizon):
        masks_t = [mask[t] for mask in covered if mask.shape[0] > t]
        widths_t = [arr[t] for arr in widths if arr.shape[0] > t]
        errors_t = [arr[t] for arr in errors if arr.shape[0] > t]
        scores_t = [arr[t] for arr in interval_scores if arr.shape[0] > t]
        stacked_mask = np.vstack(masks_t)
        by_horizon.append(float(np.mean(stacked_mask)))
        by_horizon_target[t, :] = np.mean(stacked_mask, axis=0)
        width_by_horizon.append(float(np.mean(np.vstack(widths_t))))
        mae_by_horizon.append(float(np.mean(np.vstack(errors_t))))
        score_by_horizon.append(float(np.mean(np.vstack(scores_t))))
        n_valid.append(len(masks_t))
    point_by_target = np.mean(points, axis=0)
    target_traj = np.mean(targetwise_trajectory, axis=0)
    primary = float(np.nanmean(target_traj)) if primary_coverage_type == "targetwise_trajectory_coverage" else float(np.nanmean(by_horizon_target))
    target_gap = point_by_target - nominal
    return {
        "alpha": float(alpha),
        "nominal_coverage": nominal,
        "primary_coverage_type": primary_coverage_type,
        "primary_empirical_coverage": primary,
        "no_conformal_guarantee": bool(no_conformal_guarantee),
        "marginal_pointwise_coverage_overall": float(np.mean(points)),
        "marginal_pointwise_coverage_by_target": {n: float(v) for n, v in zip(target_names, point_by_target)},
        "marginal_pointwise_coverage_by_horizon": by_horizon,
        "marginal_pointwise_coverage_by_horizon_target": by_horizon_target.tolist(),
        "targetwise_trajectory_coverage": {n: float(v) for n, v in zip(target_names, target_traj)},
        "complete_multivariate_trajectory_coverage": float(np.mean(complete_trajectory)),
        "mean_interval_width_overall": float(np.mean(width_all)),
        "median_interval_width_overall": float(np.median(width_all)),
        "p90_interval_width_overall": float(np.quantile(width_all, 0.9)),
        "mean_interval_width_by_target": {n: float(v) for n, v in zip(target_names, np.mean(width_all, axis=0))},
        "median_interval_width_by_target": {n: float(v) for n, v in zip(target_names, np.median(width_all, axis=0))},
        "p90_interval_width_by_target": {n: float(v) for n, v in zip(target_names, np.quantile(width_all, 0.9, axis=0))},
        "mean_interval_width_by_horizon": width_by_horizon,
        "mean_absolute_error_by_horizon": mae_by_horizon,
        "mean_width_normalized_by_target_std": {n: float(v) for n, v in zip(target_names, np.nanmean(width_all, axis=0) / safe_std)},
        "mean_width_normalized_by_target_range": {n: float(v) for n, v in zip(target_names, np.nanmean(width_all, axis=0) / safe_range)},
        "mean_interval_score_overall": float(np.mean(score_all)),
        "mean_interval_score_by_target": {n: float(v) for n, v in zip(target_names, np.mean(score_all, axis=0))},
        "mean_interval_score_by_horizon": score_by_horizon,
        "worst_target_coverage": float(np.min(point_by_target)),
        "maximum_absolute_targetwise_coverage_deviation_from_nominal": float(np.max(np.abs(target_gap))),
        "mean_absolute_targetwise_coverage_deviation_from_nominal": float(np.mean(np.abs(target_gap))),
        "valid_profile_count_by_horizon": n_valid,
    }


def save_shared_ensemble_predictions_hdf5(
    forecasts: list[dict[str, Any]], *, output_path: Path, target_names: list[str], ensemble_ddof: int
) -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["target_names"] = np.asarray(target_names, dtype="S")
        h5f.attrs["ensemble_member_count"] = int(forecasts[0]["member_predictions_scaled"].shape[0]) if forecasts else 0
        h5f.attrs["ensemble_ddof"] = int(ensemble_ddof)
        for entry in forecasts:
            group = h5f.create_group(str(entry["profile"]))
            for key in ("t", "u", "y_true_scaled", "member_predictions_scaled", "mean_scaled", "spread_scaled"):
                group.create_dataset(key, data=entry[key], compression="gzip", chunks=True)


def save_uq_forecasts_hdf5(
    forecasts: list[dict[str, Any]], *, output_path: Path, metadata: dict[str, Any], target_names: list[str]
) -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                h5f.attrs[key] = value
        h5f.attrs["target_names"] = np.asarray(target_names, dtype="S")
        for entry in forecasts:
            group = h5f.create_group(str(entry["profile"]))
            for key in (
                "t", "u", "y_true", "y_pred", "lower", "upper", "interval_width",
                "y_true_scaled", "y_pred_scaled", "lower_scaled", "upper_scaled", "spread_scaled",
            ):
                group.create_dataset(key, data=entry[key], compression="gzip", chunks=True)
            for key in ("effective_scale_scaled", "normalized_abs_error", "member_predictions_scaled"):
                if key in entry:
                    group.create_dataset(key, data=entry[key], compression="gzip", chunks=True)
