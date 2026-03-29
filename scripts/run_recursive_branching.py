"""Standalone + programmatic recursive branching runner with visualization.

Usage (standalone):
    python scripts/run_recursive_branching.py \
        --model-path outputs/ml_results/ensemble/model_0/model.pt \
                     outputs/ml_results/ensemble/model_1/model.pt \
        --lstm-hidden 64 \
        --fc-hidden 64 \
        --output-dir outputs/recursive_branching
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rabl.machine_learning.bagging_ensemble import (
    _save_ensemble_rolling_forecasts_hdf5,
)
from rabl.machine_learning.branchpoint_finder import finite_difference
from rabl.machine_learning.recursive_branching import (
    LSTMEnsembleForecaster,
    RecursiveBranchingResult,
    generate_root_profile,
    load_trained_ensemble,
    run_recursive_branching,
    save_recursive_branching_output,
)
from rabl.machine_learning.lstm_pipeline import save_forecast_profiles_pdf
from rabl.machine_learning.lstm_pipeline import _descale_targets_from_stats, _load_scaling_stats
from rabl.variography.DrumVariography import DrumProfile, DrumProfileGenerator
from rabl.machine_learning.build_lstm_dataset import CONTROL_COLUMN, STATE_COLUMNS

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "config.py"


@dataclass(frozen=True)
class RecursiveBranchingRunConfig:
    model_paths: tuple[Path, ...]
    bagged_h5_path: Path
    output_dir: Path
    weights_npy_path: Path | None = None

    T: float = 200.0
    dt: float = 0.4
    Nk: int = 3
    Nb: int = 2

    baseline_angle_deg: float = 45.0
    seed: int = 123
    Nr: int = 1
    visualize: bool = True

    lookback: int = 12
    n_features: int = 13
    state_dim: int | None = None
    num_targets: int | None = None
    control_channel: int = 0

    n_lstm: int = 1
    lstm_hidden: int = 64
    n_fc: int = 1
    fc_hidden: tuple[int, ...] = (64,)

    finite_difference_order: int = 4
    kernel: str = "matern52"
    device: str = "cpu"
    config_path: Path = DEFAULT_CONFIG_PATH




def _infer_checkpoint_io_shapes(model_path: Path) -> tuple[int, int]:
    """Infer (num_features, num_targets) from a saved LSTMRegressor checkpoint."""
    state_dict = torch.load(Path(model_path), map_location="cpu")
    try:
        input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
        output_size = int(state_dict["output_layer.bias"].shape[0])
    except KeyError as exc:
        raise KeyError(
            f"Checkpoint '{model_path}' is missing expected keys for shape inference. "
            "Expected at least 'lstm.weight_ih_l0' and 'output_layer.bias'."
        ) from exc
    return input_size, output_size


def _resolve_model_io_shapes(config: RecursiveBranchingRunConfig) -> tuple[int, int]:
    """Resolve model input/output sizes from checkpoints and validate consistency."""
    if not config.model_paths:
        raise ValueError("At least one model path is required.")

    inferred = [_infer_checkpoint_io_shapes(path) for path in config.model_paths]
    first_in, first_out = inferred[0]
    mismatched = [
        (str(path), in_size, out_size)
        for path, (in_size, out_size) in zip(config.model_paths, inferred, strict=True)
        if in_size != first_in or out_size != first_out
    ]
    if mismatched:
        details = "; ".join(
            f"{path} -> (in={in_size}, out={out_size})" for path, in_size, out_size in mismatched
        )
        raise ValueError(
            "Ensemble checkpoints do not share the same input/output dimensions. "
            f"Expected all to match first checkpoint (in={first_in}, out={first_out}). "
            f"Mismatches: {details}"
        )

    if config.n_features != first_in:
        print(
            "[shape-infer] Overriding n_features from config "
            f"({config.n_features}) to checkpoint value ({first_in})."
        )
    if config.num_targets is not None and config.num_targets != first_out:
        print(
            "[shape-infer] Overriding num_targets from config "
            f"({config.num_targets}) to checkpoint value ({first_out})."
        )

    return first_in, first_out

def _load_config_module(config_path: Path) -> ModuleType:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("rabl_recursive_branching_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _steady_state_rows(steady_state: dict, *, state_dim: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    if lookback < 1:
        raise ValueError(f"lookback must be >=1, got {lookback}.")

    control_value = steady_state.get(CONTROL_COLUMN)
    if control_value is None:
        raise ValueError(f"STEADY_STATE must contain key '{CONTROL_COLUMN}'.")

    missing = [key for key in STATE_COLUMNS if key not in steady_state]
    if missing:
        raise ValueError(f"STEADY_STATE is missing required state keys: {missing}")
    state_values = [steady_state[key] for key in STATE_COLUMNS]
    if state_dim > len(state_values):
        raise ValueError(
            "STEADY_STATE does not contain enough non-control entries for requested state_dim. "
            f"Found {len(state_values)} non-control entries, state_dim={state_dim}."
        )

    state_row = np.asarray(state_values[:state_dim], dtype=np.float32)
    control_row = np.asarray([float(control_value)], dtype=np.float32)
    state_pad = np.repeat(state_row[None, :], lookback, axis=0)
    control_pad = np.repeat(control_row[None, :], lookback, axis=0)
    return state_pad, control_pad


def _make_control_window(u_series: np.ndarray, step: int, lookback: int, *, control_steady: float) -> np.ndarray:
    start = max(0, step - lookback + 1)
    history = u_series[start : step + 1]
    if history.size == 0:
        raise RuntimeError("Control history unexpectedly empty while building windows.")
    out = np.empty(lookback, dtype=np.float32)
    out[: lookback - history.size] = np.float32(control_steady)
    out[lookback - history.size :] = history
    return out


def _descale_uncertainty_widths_from_stats(stats: dict, sigma_values: np.ndarray) -> np.ndarray:
    """Descale uncertainty widths (e.g., 2σ) without applying offsets.

    Mean predictions use a full inverse transform; uncertainty widths should only
    be multiplied by the target scale (std/span).
    """
    scaling_type = stats["type"]
    y_stats = stats["y"]
    if scaling_type == "standard":
        return sigma_values * y_stats["std"]
    if scaling_type == "minmax":
        return sigma_values * y_stats["span"]
    raise ValueError(f"Unsupported scaling type: {scaling_type}")


def build_profile_to_x_adapter(
    *,
    n_steps: int,
    lookback: int,
    n_features: int,
    steady_state: dict,
    state_dim: int,
    control_channel: int,
    scaling_stats: dict,
) -> Callable[[DrumProfile], np.ndarray]:
    if n_steps < 1:
        raise ValueError("n_steps must be >=1.")
    if lookback < 1:
        raise ValueError("lookback must be >=1.")
    if n_features <= state_dim:
        raise ValueError("n_features must be > state_dim (at least one control feature).")

    control_dim = n_features - state_dim
    if not (0 <= control_channel < control_dim):
        raise ValueError(f"control_channel={control_channel} out of range [0, {control_dim - 1}]")

    control_feature_idx = state_dim + control_channel
    state_pad, control_pad = _steady_state_rows(steady_state, state_dim=state_dim, lookback=lookback)
    control_steady = float(control_pad[0, 0])

    def _adapter(profile: DrumProfile) -> np.ndarray:
        u_series = np.asarray(profile.theta_deg, dtype=np.float32)
        if u_series.ndim != 1:
            raise ValueError(f"Expected 1D theta_deg control series, got shape {u_series.shape}")
        if u_series.size != n_steps:
            raise ValueError(f"Profile horizon mismatch: expected {n_steps} steps, got {u_series.size}.")

        x_profile = np.zeros((n_steps, lookback, n_features), dtype=np.float32)
        x_profile[:, :, :state_dim] = state_pad[None, :, :]
        for step in range(n_steps):
            x_profile[step, :, control_feature_idx] = _make_control_window(
                u_series,
                step,
                lookback,
                control_steady=control_steady,
            )
        if scaling_stats["type"] == "standard":
            mean = scaling_stats["x"]["mean"].astype(np.float32)
            std = scaling_stats["x"]["std"].astype(np.float32)
            x_profile = (x_profile - mean[None, None, :]) / std[None, None, :]
        elif scaling_stats["type"] == "minmax":
            x_min = scaling_stats["x"]["min"].astype(np.float32)
            x_span = scaling_stats["x"]["span"].astype(np.float32)
            x_profile = (x_profile - x_min[None, None, :]) / x_span[None, None, :]
        else:
            raise ValueError(f"Unsupported scaling type: {scaling_stats['type']}")
        return x_profile

    return _adapter


def _build_time_grid(T: float, dt: float) -> np.ndarray:
    if T <= 0.0:
        raise ValueError("T must be > 0.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")
    t_grid = np.arange(0.0, T + dt, dt, dtype=float)
    if not np.isclose(t_grid[-1], T):
        t_grid = np.append(t_grid, T)
    return t_grid


def _plot_root_profile(root_profile: DrumProfile, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(root_profile.t, root_profile.theta_deg, color="black", linewidth=2.0)
    ax.set_title("Generated Root Drum Angle Profile")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drum Angle (deg)")
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_branched_profiles(result: RecursiveBranchingResult, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    root_id = "profile_000000"
    interval_colors = _interval_colors(len(result.intervals))

    for profile_id, node in result.final_profiles.items():
        t = np.asarray(node.profile.t, dtype=float)
        u = np.asarray(node.profile.theta_deg, dtype=float)
        if profile_id == root_id:
            ax.plot(t, u, color="black", linewidth=2.0, zorder=3, label="root")
            continue

        k = 0 if node.created_in_interval is None else node.created_in_interval
        color = interval_colors[k % len(interval_colors)] if interval_colors else "darkcyan"
        if node.branch_time is None:
            ax.plot(t, u, color=color, linewidth=1.2, alpha=0.85)
        else:
            mask = t >= node.branch_time
            ax.plot(t[mask], u[mask], color=color, linewidth=1.2, alpha=0.9)
            branch_idx = int(np.argmin(np.abs(t - node.branch_time)))
            ax.plot(t[branch_idx], u[branch_idx], "o", color="black", markersize=3.5, zorder=4)

    for interval in result.intervals:
        ax.axvline(interval.start, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    if result.intervals:
        ax.axvline(result.intervals[-1].end, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    if result.intervals:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=interval_colors[interval.index % len(interval_colors)],
                linewidth=2.0,
                label=f"Spawned in Interval {interval.index + 1}",
            )
            for interval in result.intervals
        ]
        ax.legend(handles=legend_handles, loc="best", frameon=True)

    ax.set_title("Branched Drum Profiles Across Intervals")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drum Angle (deg)")
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _interval_colors(n_intervals: int) -> list[str]:
    """Match the interval color scheme used in generate_branched_control_profiles.py."""
    palette = ["darkcyan", "aquamarine", "mediumturquoise"]
    return [palette[i % len(palette)] for i in range(max(0, n_intervals))]


def _print_run_summary(
    *,
    config: RecursiveBranchingRunConfig,
    n_steps: int,
    n_features: int,
    state_dim: int,
    num_targets: int,
    ell: float,
    nugget_v: float,
    sill_v: float,
    models: list,
    result: RecursiveBranchingResult,
    scaling_type: str,
) -> None:
    print("\n=== Recursive Branching Run Configuration ===")
    print(f"Variogram: kernel={config.kernel}, ell={ell}, nugget={nugget_v}, sill={sill_v:.6f}")
    print(f"Time/Grid: n_steps={n_steps}, dt={config.dt}, T={config.T}, Nk={config.Nk}, Nb={config.Nb}, Nr={config.Nr}")
    print(f"Forecast shape: state_dim={state_dim}, num_targets={num_targets}, lookback={config.lookback}, n_features={n_features}")
    print(f"Finite difference order: {config.finite_difference_order}")
    print(f"Scaling: type={scaling_type}, source={config.bagged_h5_path}")
    print(f"Ensemble members: {len(models)}")
    for idx, path in enumerate(config.model_paths):
        print(
            f"  - model[{idx}] file={path} "
            f"arch=(n_lstm={config.n_lstm}, lstm_hidden={config.lstm_hidden}, n_fc={config.n_fc}, fc_hidden={config.fc_hidden})"
        )

    per_interval_counts = {interval.index: 0 for interval in result.intervals}
    for event in result.branch_events:
        per_interval_counts[event.interval_index] = per_interval_counts.get(event.interval_index, 0) + 1

    branched_profiles = len(result.final_profiles) - 1
    print(f"\nTotal profiles: {len(result.final_profiles)} (branched/new={branched_profiles})")
    print(f"Profiles branched per interval: {per_interval_counts}\n")


def _save_branching_ensemble_forecasts(
    *,
    result: RecursiveBranchingResult,
    forecaster: LSTMEnsembleForecaster,
    output_dir: Path,
    target_names: list[str],
    derivative_order: int,
    dt: float,
    scaling_stats: dict,
) -> Path:
    forecasts: list[dict[str, np.ndarray | str]] = []
    for profile_id, node in sorted(result.final_profiles.items()):
        pred_stack_scaled = forecaster.forecast(node.profile)
        y_mean_scaled = np.mean(pred_stack_scaled, axis=0)
        y_two_sigma_scaled = 2.0 * np.std(pred_stack_scaled, axis=0, ddof=0)
        y_dsigma_dt_scaled = finite_difference(y_two_sigma_scaled, order=derivative_order, dt=dt)

        y_mean = _descale_targets_from_stats(scaling_stats, y_mean_scaled)
        y_two_sigma = _descale_uncertainty_widths_from_stats(scaling_stats, y_two_sigma_scaled)

        t_series = np.asarray(node.profile.t, dtype=np.float32)
        u_series = np.asarray(node.profile.theta_deg, dtype=np.float32)

        # Branching workflow has no measured truth series; duplicate mean for x_true fields
        # to keep schema compatible with existing ensemble plotting utilities.
        # NOTE: y_mean / y_2sigma are descaled physical values; dx_sigma_dt is kept scaled.
        table = np.column_stack([t_series, u_series, y_mean, y_mean, y_two_sigma]).astype(np.float32)
        forecasts.append({
            "profile": profile_id,
            "table": table,
            # Keep uncertainty derivative in scaled units.
            "dx_sigma_dt": y_dsigma_dt_scaled.astype(np.float32),
        })

    forecast_h5 = output_dir / "ensemble_forecasts.h5"
    _save_ensemble_rolling_forecasts_hdf5(
        forecasts,
        output_path=forecast_h5,
        target_names=target_names,
    )
    return forecast_h5


def _run_single_recursive_branching_workflow(
    config: RecursiveBranchingRunConfig,
    *,
    root_index: int,
    run_output_dir: Path,
    profiles_h5_path: Path,
) -> RecursiveBranchingResult:
    run_output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_module(config.config_path)
    ell = float(cfg.ELL)
    nugget_v = float(cfg.NUGGET_V_DEG2_S2)
    sigma_theta_target = float(cfg.SIGMA_THETA_TARGET)
    steady_state = dict(cfg.STEADY_STATE)
    scaling_stats = _load_scaling_stats(config.bagged_h5_path)

    t_grid = _build_time_grid(config.T, config.dt)
    generator = DrumProfileGenerator(
        kernel=config.kernel,
        ell=ell,
        sill_v_deg2_s2=0.0,
        nugget_v_deg2_s2=nugget_v,
    )
    _ell, sill_v, _nugget = generator.solve_params_for_sigma_theta(
        t_grid=t_grid,
        sigma_theta_target=sigma_theta_target,
        ell=ell,
        nugget=nugget_v,
        update_instance=True,
    )

    process_start = perf_counter()

    root_profile = generate_root_profile(
        generator,
        t_grid=t_grid,
        baseline_angle_deg=config.baseline_angle_deg,
        seed=config.seed + root_index,
    )
    if config.visualize:
        _plot_root_profile(root_profile, run_output_dir / "root_profile.png")

    n_steps = int(root_profile.t.size)
    n_features, num_targets = _resolve_model_io_shapes(config)
    state_dim = (n_features - 1) if config.state_dim is None else int(config.state_dim)
    if num_targets != state_dim:
        raise ValueError(
            "Resolved checkpoint output size does not match state_dim. "
            f"num_targets={num_targets}, state_dim={state_dim}. "
            "Pass --state-dim explicitly if your model output/state split differs."
        )

    profile_to_x = build_profile_to_x_adapter(
        n_steps=n_steps,
        lookback=config.lookback,
        n_features=n_features,
        steady_state=steady_state,
        state_dim=state_dim,
        control_channel=config.control_channel,
        scaling_stats=scaling_stats,
    )

    models = load_trained_ensemble(
        [Path(p) for p in config.model_paths],
        timesteps=config.lookback,
        num_features=n_features,
        num_targets=num_targets,
        n_lstm=config.n_lstm,
        lstm_hidden=config.lstm_hidden,
        lstm_dropout=0.0,
        n_fc=config.n_fc,
        fc_hidden=tuple(config.fc_hidden),
        device=config.device,
    )

    forecaster = LSTMEnsembleForecaster(models, profile_to_x=profile_to_x, state_dim=state_dim)

    if config.weights_npy_path is None:
        weights = np.ones((num_targets,), dtype=np.float64)
    else:
        weights = np.load(config.weights_npy_path).astype(np.float64)
    if weights.ndim != 1 or weights.shape[0] != num_targets:
        raise ValueError(
            "weights vector must have shape (num_targets,). "
            f"Got {weights.shape}, num_targets={num_targets}."
        )

    result = run_recursive_branching(
        forecaster=forecaster,
        generator=generator,
        root_profile=root_profile,
        n_intervals=config.Nk,
        n_branches=config.Nb,
        weights=weights,
        finite_difference_order=config.finite_difference_order,
        seed=config.seed + root_index,
        verbose=True,
    )

    root_group_name = f"root_{root_index + 1:03d}"
    save_recursive_branching_output(
        result,
        profiles_h5_path=profiles_h5_path,
        root_group_name=root_group_name,
    )
    metadata = {
        "root_group_name": root_group_name,
        "intervals": [
            {"index": interval.index, "start": interval.start, "end": interval.end}
            for interval in result.intervals
        ],
        "n_profiles": len(result.final_profiles),
        "branch_events": [
            {
                "child_profile_id": event.child_profile_id,
                "parent_profile_id": event.parent_profile_id,
                "interval_index": event.interval_index,
                "branch_time": event.branch_time,
                "branch_label": event.branch_label,
            }
            for event in result.branch_events
        ],
    }
    (run_output_dir / "branch_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if config.visualize:
        _plot_branched_profiles(result, run_output_dir / "branched_profiles.png")

    target_names = list(STATE_COLUMNS[:num_targets])
    forecast_h5 = _save_branching_ensemble_forecasts(
        result=result,
        forecaster=forecaster,
        output_dir=run_output_dir,
        target_names=target_names,
        derivative_order=config.finite_difference_order,
        dt=config.dt,
        scaling_stats=scaling_stats,
    )
    process_elapsed_seconds = perf_counter() - process_start

    if config.visualize:
        save_forecast_profiles_pdf(
            forecast_h5_path=forecast_h5,
            output_pdf_path=run_output_dir / "ensemble_forecasts_with_derivative.pdf",
            target_names=target_names,
            state_dim=state_dim,
            control_channel=config.control_channel,
            mode="ensemble",
            include_uncertainty_derivative=True,
            derivative_order=config.finite_difference_order,
            derivative_dt=config.dt,
        )

    _print_run_summary(
        config=config,
        n_steps=n_steps,
        n_features=n_features,
        state_dim=state_dim,
        num_targets=num_targets,
        ell=ell,
        nugget_v=nugget_v,
        sill_v=sill_v,
        models=models,
        result=result,
        scaling_type=scaling_stats["type"],
    )

    expected_profiles = (config.Nb + 1) ** config.Nk
    print(
        "\nRecursive branching complete:\n"
        f"  final_profiles={len(result.final_profiles)}\n"
        f"  expected_profiles={expected_profiles}\n"
        f"  branch_events={len(result.branch_events)}\n"
        f"  process_seconds_excluding_visualization={process_elapsed_seconds:.3f}\n"
        f"  output_dir={run_output_dir}\n"
    )

    return result


def run_recursive_branching_workflow(config: RecursiveBranchingRunConfig) -> RecursiveBranchingResult | list[RecursiveBranchingResult]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.Nr < 1:
        raise ValueError(f"Nr must be >= 1, got {config.Nr}.")

    results: list[RecursiveBranchingResult] = []
    multiple_roots = config.Nr > 1
    profiles_h5_path = config.output_dir / "profiles.h5"
    for root_index in range(config.Nr):
        run_output_dir = config.output_dir / f"root_{root_index:03d}" if multiple_roots else config.output_dir
        print(f"\n=== Root Profile {root_index + 1}/{config.Nr} ===\n")
        result = _run_single_recursive_branching_workflow(
            config,
            root_index=root_index,
            run_output_dir=run_output_dir,
            profiles_h5_path=profiles_h5_path,
        )
        results.append(result)

    return results[0] if config.Nr == 1 else results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recursive branching as a standalone script.")
    parser.add_argument("--model-path", type=Path, nargs="+", required=True, help="One or more .pt ensemble checkpoints.")
    parser.add_argument("--bagged-h5-path", type=Path, required=True, help="Path to bagged/scaled HDF5 containing 'scaling' group.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for branching outputs and plots.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help=f"Path to config.py (default: {DEFAULT_CONFIG_PATH}).")

    parser.add_argument("--T", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--Nk", type=int, default=3, help="Number of intervals across the horizon.")
    parser.add_argument("--Nb", type=int, default=2, help="Number of children generated at each branch point.")
    parser.add_argument("--Nr", type=int, default=1, help="Number of root profiles to run through the branching workflow.")

    parser.add_argument("--baseline-angle-deg", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--n-features", type=int, default=13)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--num-targets", type=int, default=None)
    parser.add_argument("--control-channel", type=int, default=0)

    parser.add_argument("--n-lstm", type=int, default=1)
    parser.add_argument("--lstm-hidden", type=int, required=True)
    parser.add_argument("--n-fc", type=int, default=1)
    parser.add_argument("--fc-hidden", type=int, nargs="+", required=True)

    parser.add_argument("--finite-difference-order", type=int, default=4, choices=[2, 4])
    parser.add_argument("--kernel", type=str, default="matern52", choices=["matern32", "matern52"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--visualize",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to generate PNG/PDF visualizations (1=yes, 0=no).",
    )
    parser.add_argument(
        "--weights-npy",
        type=Path,
        default=None,
        help="Optional .npy vector of per-target non-negative weights (shape: [num_targets]).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = RecursiveBranchingRunConfig(
        model_paths=tuple(args.model_path),
        bagged_h5_path=args.bagged_h5_path,
        output_dir=args.output_dir,
        T=args.T,
        dt=args.dt,
        Nk=args.Nk,
        Nb=args.Nb,
        Nr=args.Nr,
        baseline_angle_deg=args.baseline_angle_deg,
        seed=args.seed,
        visualize=bool(args.visualize),
        lookback=args.lookback,
        n_features=args.n_features,
        state_dim=args.state_dim,
        num_targets=args.num_targets,
        control_channel=args.control_channel,
        n_lstm=args.n_lstm,
        lstm_hidden=args.lstm_hidden,
        n_fc=args.n_fc,
        fc_hidden=tuple(args.fc_hidden),
        finite_difference_order=args.finite_difference_order,
        kernel=args.kernel,
        device=args.device,
        config_path=args.config,
        weights_npy_path=args.weights_npy,
    )
    run_recursive_branching_workflow(run_config)


if __name__ == "__main__":
    main()
