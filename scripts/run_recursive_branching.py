"""Programmatic recursive branching runner.

This module is intended for direct Python use (not command-line parsing).
Import `run_recursive_branching_workflow(...)` from notebooks, services, or
other scripts to execute the end-to-end recursive branching workflow.
"""

from __future__ import annotations

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
    """Programmatic settings for one recursive branching run."""

    model_paths: tuple[Path, ...]
    x_profile_template_path: Path
    output_dir: Path

    # Horizon/branching defaults
    T: float = 200.0
    dt: float = 0.4
    n_intervals: int = 3
    n_branches: int = 2

    baseline_angle_deg: float = 45.0
    seed: int = 123

    # LSTM inference shape/options
    state_dim: int = 13
    num_targets: int = 13
    control_channel: int = 0
    n_lstm: int = 1
    lstm_hidden: int = 64
    n_fc: int = 1
    fc_hidden: tuple[int, ...] = (64,)

    finite_difference_order: int = 4
    kernel: str = "matern52"
    device: str = "cpu"

    # Variography config source
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


def _make_control_window(u_series: np.ndarray, step: int, lookback: int) -> np.ndarray:
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
    """Build adapter mapping `DrumProfile -> x_profile` for rolling forecast."""
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
    """Run the recursive branching workflow programmatically and save outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_module(config.config_path)
    ell = float(cfg.ELL)
    nugget_v = float(cfg.NUGGET_V_DEG2_S2)
    sigma_theta_target = float(cfg.SIGMA_THETA_TARGET)

    x_template = np.load(config.x_profile_template_path)
    if x_template.ndim != 3:
        raise ValueError(f"x-profile-template must load to 3D array; got {x_template.shape}.")

    profile_to_x = build_profile_to_x_adapter(
        x_template,
        state_dim=config.state_dim,
        control_channel=config.control_channel,
    )

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

    n_steps, lookback, num_features = x_template.shape
    if n_steps != root_profile.t.size:
        raise ValueError(
            "Time-grid/profile length mismatch with x-profile template. "
            f"template steps={n_steps}, generated steps={root_profile.t.size}."
        )

    models = load_trained_ensemble(
        [Path(p) for p in config.model_paths],
        timesteps=lookback,
        num_features=num_features,
        num_targets=config.num_targets,
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
        state_dim=config.state_dim,
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
        f"ell={ell} nugget_v={nugget_v} sill_v={sill_v:.6f} sigma_theta_target={sigma_theta_target} "
        f"output_dir={config.output_dir}"
    )

    return result


if __name__ == "__main__":
    raise SystemExit(
        "This module is now intended for programmatic use. "
        "Import run_recursive_branching_workflow(...) and call it from Python code."
    )
