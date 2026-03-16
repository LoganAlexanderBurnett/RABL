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
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rabl.machine_learning.bagging_ensemble import (
    _save_ensemble_rolling_forecasts_hdf5,
    plot_ensemble_forecast_profile_grid,
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
from rabl.variography.DrumVariography import DrumProfile, DrumProfileGenerator

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "config.py"


@dataclass(frozen=True)
class RecursiveBranchingRunConfig:
    model_paths: tuple[Path, ...]
    output_dir: Path

    T: float = 200.0
    dt: float = 0.4
    n_intervals: int = 3
    n_branches: int = 2

    baseline_angle_deg: float = 45.0
    seed: int = 123

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

    control_value = steady_state.get("drumAngleDeg")
    if control_value is None:
        raise ValueError("STEADY_STATE must contain key 'drumAngleDeg'.")

    state_values = [value for key, value in steady_state.items() if key != "drumAngleDeg"]
    if len(state_values) < state_dim:
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


def build_profile_to_x_adapter(
    *,
    n_steps: int,
    lookback: int,
    n_features: int,
    steady_state: dict,
    state_dim: int,
    control_channel: int,
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
    cmap = plt.get_cmap("viridis")
    max_interval = max((i.index for i in result.intervals), default=0)

    for profile_id, node in result.final_profiles.items():
        t = np.asarray(node.profile.t, dtype=float)
        u = np.asarray(node.profile.theta_deg, dtype=float)
        if profile_id == root_id:
            ax.plot(t, u, color="black", linewidth=2.0, zorder=3, label="root")
            continue

        k = 0 if node.created_in_interval is None else node.created_in_interval
        color = cmap(k / max(1, max_interval))
        if node.branch_time is None:
            ax.plot(t, u, color=color, linewidth=1.2, alpha=0.85)
        else:
            mask = t >= node.branch_time
            ax.plot(t[mask], u[mask], color=color, linewidth=1.2, alpha=0.9)

    for interval in result.intervals:
        ax.axvline(interval.start, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    if result.intervals:
        ax.axvline(result.intervals[-1].end, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title("Branched Drum Profiles Across Intervals")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drum Angle (deg)")
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
) -> None:
    print("=== Recursive Branching Run Configuration ===")
    print(f"Variogram: kernel={config.kernel}, ell={ell}, nugget={nugget_v}, sill={sill_v:.6f}")
    print(f"Time/Grid: n_steps={n_steps}, dt={config.dt}, T={config.T}")
    print(f"Forecast shape: state_dim={state_dim}, num_targets={num_targets}, lookback={config.lookback}, n_features={n_features}")
    print(f"Finite difference order: {config.finite_difference_order}")
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
    print(f"Total profiles: {len(result.final_profiles)} (branched/new={branched_profiles})")
    print(f"Profiles branched per interval: {per_interval_counts}")


def _save_branching_ensemble_forecasts(
    *,
    result: RecursiveBranchingResult,
    forecaster: LSTMEnsembleForecaster,
    output_dir: Path,
    target_names: list[str],
    derivative_order: int,
    dt: float,
) -> Path:
    forecasts: list[dict[str, np.ndarray | str]] = []
    for profile_id, node in sorted(result.final_profiles.items()):
        pred_stack = forecaster.forecast(node.profile)
        y_mean = np.mean(pred_stack, axis=0)
        y_two_sigma = 2.0 * np.std(pred_stack, axis=0, ddof=0)
        y_dsigma_dt = finite_difference(y_two_sigma, order=derivative_order, dt=dt)

        t_series = np.asarray(node.profile.t, dtype=np.float32)
        u_series = np.asarray(node.profile.theta_deg, dtype=np.float32)

        # Branching workflow has no measured truth series; duplicate mean for x_true fields
        # to keep schema compatible with existing ensemble plotting utilities.
        table = np.column_stack([t_series, u_series, y_mean, y_mean, y_two_sigma]).astype(np.float32)
        forecasts.append({
            "profile": profile_id,
            "table": table,
            "dx_sigma_dt": y_dsigma_dt.astype(np.float32),
        })

    forecast_h5 = output_dir / "ensemble_forecasts.h5"
    _save_ensemble_rolling_forecasts_hdf5(
        forecasts,
        output_path=forecast_h5,
        target_names=target_names,
    )
    return forecast_h5


def run_recursive_branching_workflow(config: RecursiveBranchingRunConfig) -> RecursiveBranchingResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_module(config.config_path)
    ell = float(cfg.ELL)
    nugget_v = float(cfg.NUGGET_V_DEG2_S2)
    sigma_theta_target = float(cfg.SIGMA_THETA_TARGET)
    steady_state = dict(cfg.STEADY_STATE)

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

    root_profile = generate_root_profile(
        generator,
        t_grid=t_grid,
        baseline_angle_deg=config.baseline_angle_deg,
        seed=config.seed,
    )
    _plot_root_profile(root_profile, config.output_dir / "root_profile.png")

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

    result = run_recursive_branching(
        forecaster=forecaster,
        generator=generator,
        root_profile=root_profile,
        n_intervals=config.n_intervals,
        n_branches=config.n_branches,
        finite_difference_order=config.finite_difference_order,
        seed=config.seed,
    )

    save_recursive_branching_output(result, config.output_dir)
    _plot_branched_profiles(result, config.output_dir / "branched_profiles.png")

    target_names = [f"state_{i}" for i in range(num_targets)]
    forecast_h5 = _save_branching_ensemble_forecasts(
        result=result,
        forecaster=forecaster,
        output_dir=config.output_dir,
        target_names=target_names,
        derivative_order=config.finite_difference_order,
        dt=config.dt,
    )

    forecast_plot_dir = config.output_dir / "ensemble_forecast_plots"
    forecast_plot_dir.mkdir(parents=True, exist_ok=True)
    for profile_id in sorted(result.final_profiles.keys()):
        plot_ensemble_forecast_profile_grid(
            forecast_h5,
            profile_name=profile_id,
            save_path=forecast_plot_dir / f"{profile_id}.png",
            target_names=target_names,
            plot_uncertainty_derivative=True,
            close_figure=True,
        )

    save_forecast_profiles_pdf(
        forecast_h5_path=forecast_h5,
        output_pdf_path=config.output_dir / "ensemble_forecasts_with_derivative.pdf",
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
    )

    expected_profiles = (config.n_branches + 1) ** config.n_intervals
    print(
        "Recursive branching complete: "
        f"final_profiles={len(result.final_profiles)} expected_profiles={expected_profiles} "
        f"branch_events={len(result.branch_events)} output_dir={config.output_dir}"
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recursive branching as a standalone script.")
    parser.add_argument("--model-path", type=Path, nargs="+", required=True, help="One or more .pt ensemble checkpoints.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for branching outputs and plots.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help=f"Path to config.py (default: {DEFAULT_CONFIG_PATH}).")

    parser.add_argument("--T", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--n-intervals", type=int, default=3)
    parser.add_argument("--n-branches", type=int, default=2)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = RecursiveBranchingRunConfig(
        model_paths=tuple(args.model_path),
        output_dir=args.output_dir,
        T=args.T,
        dt=args.dt,
        n_intervals=args.n_intervals,
        n_branches=args.n_branches,
        baseline_angle_deg=args.baseline_angle_deg,
        seed=args.seed,
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
    )
    run_recursive_branching_workflow(run_config)


if __name__ == "__main__":
    main()
