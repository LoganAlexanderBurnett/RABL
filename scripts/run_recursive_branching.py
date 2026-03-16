"""Standalone + programmatic recursive branching runner.

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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rabl.machine_learning.recursive_branching import (
    LSTMEnsembleForecaster,
    RecursiveBranchingResult,
    generate_root_profile,
    load_trained_ensemble,
    run_recursive_branching,
    save_recursive_branching_output,
)
from rabl.variography.DrumVariography import DrumProfile, DrumProfileGenerator

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "config.py"


@dataclass(frozen=True)
class RecursiveBranchingRunConfig:
    """Settings for one recursive branching run."""

    model_paths: tuple[Path, ...]
    output_dir: Path

    T: float = 200.0
    dt: float = 0.4
    n_intervals: int = 3
    n_branches: int = 2

    baseline_angle_deg: float = 45.0
    seed: int = 123

    # Feature layout (one control feature + state features)
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
    """Build adapter mapping `DrumProfile -> x_profile` for rolling forecast.

    The first state window is initialized from `STEADY_STATE`, mirroring the
    steady-state padding concept used by dataset construction.
    """
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
            raise ValueError(
                f"Profile horizon mismatch: expected {n_steps} steps, got {u_series.size}."
            )

        x_profile = np.zeros((n_steps, lookback, n_features), dtype=np.float32)
        # Initialize all state channels from steady state; rolling_forecast uses x_profile[0] state window.
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


def run_recursive_branching_workflow(config: RecursiveBranchingRunConfig) -> RecursiveBranchingResult:
    """Run the recursive branching workflow and save outputs."""
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

    n_steps = int(root_profile.t.size)
    state_dim = (config.n_features - 1) if config.state_dim is None else int(config.state_dim)
    num_targets = state_dim if config.num_targets is None else int(config.num_targets)

    profile_to_x = build_profile_to_x_adapter(
        n_steps=n_steps,
        lookback=config.lookback,
        n_features=config.n_features,
        steady_state=steady_state,
        state_dim=state_dim,
        control_channel=config.control_channel,
    )

    models = load_trained_ensemble(
        [Path(p) for p in config.model_paths],
        timesteps=config.lookback,
        num_features=config.n_features,
        num_targets=num_targets,
        n_lstm=config.n_lstm,
        lstm_hidden=config.lstm_hidden,
        lstm_dropout=0.0,
        n_fc=config.n_fc,
        fc_hidden=tuple(config.fc_hidden),
        device=config.device,
    )

    forecaster = LSTMEnsembleForecaster(
        models,
        profile_to_x=profile_to_x,
        state_dim=state_dim,
    )

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

    expected_profiles = (config.n_branches + 1) ** config.n_intervals
    print(
        "Recursive branching complete: "
        f"final_profiles={len(result.final_profiles)} "
        f"expected_profiles={expected_profiles} "
        f"branch_events={len(result.branch_events)} "
        f"n_steps={n_steps} lookback={config.lookback} n_features={config.n_features} "
        f"state_dim={state_dim} num_targets={num_targets} "
        f"ell={ell} nugget_v={nugget_v} sill_v={sill_v:.6f} sigma_theta_target={sigma_theta_target} "
        f"output_dir={config.output_dir}"
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recursive branching as a standalone script.")
    parser.add_argument("--model-path", type=Path, nargs="+", required=True, help="One or more .pt ensemble checkpoints.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for profiles.h5 and branch_metadata.json.")
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
