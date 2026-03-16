"""Run persistent recursive branching using trained ensemble forecasters.

This script wires together the full workflow from
`rabl.machine_learning.recursive_branching`:

1. Load trained ensemble members.
2. Generate the initial root profile with DrumVariography.
3. Partition horizon [0, T] into N_k intervals.
4. Initialize U_0 = {u_root}.
5. For each interval, forecast every profile, pick branch time from max
   uncertainty-derivative metric, and branch N_b children while keeping parents.
6. Repeat through all intervals.
7. Save final profiles and branch metadata.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
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
    generate_root_profile,
    load_trained_ensemble,
    run_recursive_branching,
    save_recursive_branching_output,
)
from rabl.variography.DrumVariography import DrumProfile, DrumProfileGenerator


DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "config.py"


def _load_config_module(config_path: Path) -> ModuleType:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("rabl_recursive_branching_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_control_window(u_series: np.ndarray, step: int, lookback: int) -> np.ndarray:
    """Create a right-aligned lookback window for control at a forecast step."""
    start = max(0, step - lookback + 1)
    history = u_series[start : step + 1]
    if history.size == 0:
        raise RuntimeError("Control history unexpectedly empty while building windows.")

    out = np.empty(lookback, dtype=np.float32)
    out[: lookback - history.size] = history[0]
    out[lookback - history.size :] = history
    return out


def build_profile_to_x_adapter(
    template_x: np.ndarray,
    *,
    state_dim: int,
    control_channel: int,
) -> Callable[[DrumProfile], np.ndarray]:
    """Build adapter mapping `DrumProfile -> x_profile` for rolling forecast.

    The adapter starts from a caller-provided template tensor
    `(n_steps, lookback, n_features)` and injects each candidate profile's
    control signal (`theta_deg`) into one chosen control feature channel while
    leaving all other channels unchanged.
    """
    x_template = np.asarray(template_x, dtype=np.float32)
    if x_template.ndim != 3:
        raise ValueError(f"template_x must be 3D (steps, lookback, features), got {x_template.shape}")

    n_steps, lookback, n_features = x_template.shape
    control_dim = n_features - state_dim
    if control_dim <= 0:
        raise ValueError(
            f"Invalid feature layout: n_features={n_features}, state_dim={state_dim}. "
            "Need at least one control channel."
        )
    if not (0 <= control_channel < control_dim):
        raise ValueError(f"control_channel={control_channel} out of range [0, {control_dim - 1}]")

    control_feature_idx = state_dim + control_channel

    def _adapter(profile: DrumProfile) -> np.ndarray:
        u_series = np.asarray(profile.theta_deg, dtype=np.float32)
        if u_series.ndim != 1:
            raise ValueError(f"Expected 1D theta_deg control series, got shape {u_series.shape}")
        if u_series.size != n_steps:
            raise ValueError(
                "Template x_profile step count must match profile horizon. "
                f"Got template steps={n_steps}, profile steps={u_series.size}."
            )

        x_profile = x_template.copy()
        for step in range(n_steps):
            x_profile[step, :, control_feature_idx] = _make_control_window(u_series, step, lookback)
        return x_profile

    return _adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recursive branching with trained LSTM ensemble members.")

    parser.add_argument("--model-path", type=Path, nargs="+", required=True, help="Paths to ensemble model .pt files.")
    parser.add_argument(
        "--x-profile-template",
        type=Path,
        required=True,
        help="Path to .npy template with shape (n_steps, lookback, n_features).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for branching outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help=f"Path to config.py (default: {DEFAULT_CONFIG_PATH}).")

    parser.add_argument("--T", type=float, default=200.0, help="Final horizon time.")
    parser.add_argument("--dt", type=float, default=0.4, help="Time step for grid generation.")
    parser.add_argument("--n-intervals", type=int, default=3, help="Number of horizon intervals N_k.")
    parser.add_argument("--n-branches", type=int, default=2, help="Number of children per profile per interval N_b.")

    parser.add_argument("--baseline-angle-deg", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--state-dim", type=int, default=13)
    parser.add_argument("--num-targets", type=int, default=13)
    parser.add_argument("--control-channel", type=int, default=0)

    parser.add_argument("--n-lstm", type=int, default=1)
    parser.add_argument("--lstm-hidden", type=int, required=True)
    parser.add_argument("--n-fc", type=int, default=1)
    parser.add_argument("--fc-hidden", type=int, nargs="+", required=True)

    parser.add_argument("--finite-difference-order", type=int, default=4, choices=[2, 4])

    parser.add_argument("--kernel", type=str, default="matern52", choices=["matern32", "matern52"])

    parser.add_argument("--device", type=str, default="cpu", help="Torch device for model loading/inference.")

    return parser.parse_args()


def _build_time_grid(T: float, dt: float) -> np.ndarray:
    if T <= 0.0:
        raise ValueError("T must be > 0.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")

    t_grid = np.arange(0.0, T + dt, dt, dtype=float)
    if not np.isclose(t_grid[-1], T):
        t_grid = np.append(t_grid, T)
    return t_grid


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_module(args.config)
    ell = float(cfg.ELL)
    nugget_v = float(cfg.NUGGET_V_DEG2_S2)
    sigma_theta_target = float(cfg.SIGMA_THETA_TARGET)

    x_template = np.load(args.x_profile_template)
    if x_template.ndim != 3:
        raise ValueError(f"x-profile-template must load to 3D array; got {x_template.shape}.")

    profile_to_x = build_profile_to_x_adapter(
        x_template,
        state_dim=args.state_dim,
        control_channel=args.control_channel,
    )

    t_grid = _build_time_grid(args.T, args.dt)

    # Compute sill from config sigma target using DrumVariography helper.
    generator = DrumProfileGenerator(
        kernel=args.kernel,
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
        baseline_angle_deg=args.baseline_angle_deg,
        seed=args.seed,
    )

    n_steps, lookback, num_features = x_template.shape
    if n_steps != root_profile.t.size:
        raise ValueError(
            "Time-grid/profile length mismatch with x-profile template. "
            f"template steps={n_steps}, generated steps={root_profile.t.size}."
        )

    models = load_trained_ensemble(
        [Path(p) for p in args.model_path],
        timesteps=lookback,
        num_features=num_features,
        num_targets=args.num_targets,
        n_lstm=args.n_lstm,
        lstm_hidden=args.lstm_hidden,
        # Dropout is not needed for inference; eval mode disables it.
        lstm_dropout=0.0,
        n_fc=args.n_fc,
        fc_hidden=tuple(args.fc_hidden),
        device=args.device,
    )

    forecaster = LSTMEnsembleForecaster(
        models,
        profile_to_x=profile_to_x,
        state_dim=args.state_dim,
    )

    result = run_recursive_branching(
        forecaster=forecaster,
        generator=generator,
        root_profile=root_profile,
        n_intervals=args.n_intervals,
        n_branches=args.n_branches,
        finite_difference_order=args.finite_difference_order,
        seed=args.seed,
    )

    save_recursive_branching_output(result, args.output_dir)

    expected_profiles = (args.n_branches + 1) ** args.n_intervals
    print(
        "Recursive branching complete: "
        f"final_profiles={len(result.final_profiles)} "
        f"expected_profiles={expected_profiles} "
        f"branch_events={len(result.branch_events)} "
        f"ell={ell} nugget_v={nugget_v} sill_v={sill_v:.6f} sigma_theta_target={sigma_theta_target} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
