"""Recursive branching utilities for uncertainty-driven drum-profile exploration.

This module orchestrates a persistent branching workflow:

1. Load trained ensemble members.
2. Generate an initial root drum profile on ``[0, T]``.
3. Partition the horizon into ``N_k`` intervals.
4. Repeatedly forecast each profile, find branch times from uncertainty derivatives,
   branch ``N_b`` children, and keep parents.
5. Save the final profile collection and branch metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import h5py
import numpy as np
import torch

from rabl.machine_learning.branchpoint_finder import finite_difference
from rabl.machine_learning.lstm_pipeline import build_model, rolling_forecast
from rabl.variography.DrumVariography import DrumProfile, DrumProfileGenerator


class EnsembleForecaster(Protocol):
    """Protocol for model-ensemble forecast adapters used by branching."""

    def forecast(self, profile: DrumProfile) -> np.ndarray:
        """Return per-estimator forecasts with shape ``(n_models, n_steps, n_targets)``."""


@dataclass(frozen=True)
class Interval:
    """Closed-open time interval ``[start, end)`` except for final interval end."""

    start: float
    end: float
    index: int


@dataclass(frozen=True)
class ProfileNode:
    """One profile tracked by the recursive branching process."""

    profile_id: str
    profile: DrumProfile
    parent_profile_id: str | None = None
    created_in_interval: int | None = None
    branch_time: float | None = None
    branch_label: int | None = None


@dataclass(frozen=True)
class BranchEvent:
    """Metadata for a single child produced by branching."""

    child_profile_id: str
    parent_profile_id: str
    interval_index: int
    branch_time: float
    branch_label: int


@dataclass(frozen=True)
class RecursiveBranchingResult:
    """Final outputs of the recursive branching workflow."""

    intervals: list[Interval]
    final_profiles: dict[str, ProfileNode]
    branch_events: list[BranchEvent]


class LSTMEnsembleForecaster:
    """Forecast adapter for saved LSTM ensemble members.

    Parameters
    ----------
    models:
        Loaded ensemble members.
    profile_to_x:
        Callable that converts a :class:`DrumProfile` into an LSTM feature tensor
        with shape ``(n_steps, lookback, n_features)`` expected by
        :func:`rolling_forecast`.
    state_dim:
        Number of state channels used by the LSTM pipeline.
    """

    def __init__(
        self,
        models: list[torch.nn.Module],
        *,
        profile_to_x: Callable[[DrumProfile], np.ndarray],
        state_dim: int,
    ) -> None:
        if not models:
            raise ValueError("models must be non-empty.")
        self._models = models
        self._profile_to_x = profile_to_x
        self._state_dim = state_dim

    def forecast(self, profile: DrumProfile) -> np.ndarray:
        x_profile = self._profile_to_x(profile)
        per_model: list[np.ndarray] = []
        for model in self._models:
            pred = rolling_forecast(model, x_profile, state_dim=self._state_dim)
            per_model.append(pred)
        return np.stack(per_model, axis=0)


def load_trained_ensemble(
    model_paths: list[Path],
    *,
    timesteps: int,
    num_features: int,
    num_targets: int,
    n_lstm: int = 1,
    lstm_hidden: int = 64,
    lstm_dropout: float = 0.0,
    n_fc: int = 1,
    fc_hidden: tuple[int, ...] = (64,),
    device: str | torch.device = "cpu",
) -> list[torch.nn.Module]:
    """Load all trained ensemble members and set eval mode."""
    resolved_device = torch.device(device)
    models: list[torch.nn.Module] = []
    for model_path in model_paths:
        model = build_model(
            timesteps=timesteps,
            num_features=num_features,
            num_targets=num_targets,
            n_lstm=n_lstm,
            lstm_hidden=lstm_hidden,
            lstm_dropout=lstm_dropout,
            n_fc=n_fc,
            fc_hidden=fc_hidden,
        )
        state_dict = torch.load(Path(model_path), map_location=resolved_device)
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        model.eval()
        models.append(model)
    return models


def generate_root_profile(
    generator: DrumProfileGenerator,
    t_grid: np.ndarray,
    *,
    baseline_angle_deg: float = 45.0,
    seed: int | None = None,
) -> DrumProfile:
    """Generate the initial root trajectory ``u_root(t)`` across ``[0, T]``."""
    return generator.generate(
        t_grid=np.asarray(t_grid, dtype=float),
        n_realizations=1,
        baseline_angle_deg=baseline_angle_deg,
        seed=seed,
    )[0]


def partition_horizon(t_grid: np.ndarray, n_intervals: int) -> list[Interval]:
    """Partition the global horizon into contiguous time intervals."""
    t = np.asarray(t_grid, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t_grid must be a 1D array with at least two points.")
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1.")

    boundaries = np.linspace(t[0], t[-1], n_intervals + 1)
    intervals = [
        Interval(start=float(boundaries[idx]), end=float(boundaries[idx + 1]), index=idx)
        for idx in range(n_intervals)
    ]
    return intervals


def _interval_mask(t: np.ndarray, interval: Interval, *, is_last: bool) -> np.ndarray:
    if is_last:
        return (t >= interval.start) & (t <= interval.end)
    return (t >= interval.start) & (t < interval.end)


def _select_branch_time(
    t: np.ndarray,
    d_unc_dt: np.ndarray,
    weights: np.ndarray,
    *,
    interval: Interval,
    is_last_interval: bool,
    positive_only: bool = True,
) -> tuple[float, np.ndarray, bool]:
    if d_unc_dt.ndim != 2:
        raise ValueError(f"Expected d_unc_dt shape (n_steps,n_targets), got {d_unc_dt.shape}.")
    if d_unc_dt.shape[0] != t.shape[0]:
        raise ValueError(
            "Derivative timeline length does not match t-grid length. "
            f"Got {d_unc_dt.shape[0]} vs {t.shape[0]}."
        )
    if weights.ndim != 1 or weights.shape[0] != d_unc_dt.shape[1]:
        raise ValueError(
            "weights must be 1D with length equal to target dimension. "
            f"Got weights={weights.shape}, targets={d_unc_dt.shape[1]}."
        )

    metric_components = np.maximum(d_unc_dt, 0.0) if positive_only else d_unc_dt
    weighted_sq = (metric_components ** 2) * weights[None, :]
    score = np.sqrt(np.sum(weighted_sq, axis=1))

    mask = _interval_mask(t, interval, is_last=is_last_interval)
    if not np.any(mask):
        raise RuntimeError(f"No time points found in interval {interval.index}: [{interval.start}, {interval.end}].")

    masked_idx = np.where(mask)[0]
    score_interval = score[masked_idx]
    all_non_positive = bool(np.all(score_interval <= 0.0))
    local_argmax = int(np.argmax(score_interval))
    return float(t[masked_idx[local_argmax]]), score, all_non_positive


def run_recursive_branching(
    *,
    forecaster: EnsembleForecaster,
    generator: DrumProfileGenerator,
    root_profile: DrumProfile,
    n_intervals: int,
    n_branches: int,
    weights: np.ndarray,
    finite_difference_order: int = 4,
    seed: int | None = None,
    verbose: bool = True,
) -> RecursiveBranchingResult:
    """Execute persistent recursive branching across all intervals.

    The parent profile is retained after branching, so profile counts evolve as
    ``|U_{k+1}| = (n_branches + 1) * |U_k|`` and ``|U_{N_k}| = (n_branches+1)^{N_k}``
    when starting from one root profile.
    """
    if n_branches < 1:
        raise ValueError("n_branches must be >= 1.")
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights.shape}.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")

    t_grid = np.asarray(root_profile.t, dtype=float)
    intervals = partition_horizon(t_grid, n_intervals=n_intervals)

    rng = np.random.default_rng(seed)
    profiles: dict[str, ProfileNode] = {"profile_000000": ProfileNode(profile_id="profile_000000", profile=root_profile)}
    branch_events: list[BranchEvent] = []
    next_profile_idx = 1

    for interval in intervals:
        current_nodes = list(profiles.values())
        for node in current_nodes:
            forecasts = forecaster.forecast(node.profile)  # (M, S, D)
            if forecasts.ndim != 3:
                raise ValueError(f"Expected forecasts shape (n_models, n_steps, n_targets), got {forecasts.shape}.")

            uncertainty_2sigma = 2.0 * np.std(forecasts, axis=0, ddof=0)  # (S, D)
            if uncertainty_2sigma.shape[0] != t_grid.shape[0]:
                raise ValueError(
                    "Uncertainty timeline length does not match profile timeline. "
                    f"Got {uncertainty_2sigma.shape[0]} vs {t_grid.shape[0]}."
                )
            if uncertainty_2sigma.shape[1] != weights.shape[0]:
                raise ValueError(
                    "Target dimension from forecasts does not match weight length. "
                    f"Got targets={uncertainty_2sigma.shape[1]}, weights={weights.shape[0]}"
                )

            dt = float(np.mean(np.diff(t_grid)))
            d_unc_dt = finite_difference(uncertainty_2sigma, order=finite_difference_order, dt=dt)

            t_branch, _score, all_non_positive = _select_branch_time(
                t_grid,
                d_unc_dt,
                weights,
                interval=interval,
                is_last_interval=(interval.index == len(intervals) - 1),
            )

            if all_non_positive:
                if verbose:
                    print(
                        f"\n[branch-skip] profile={node.profile_id} interval={interval.index} "
                        "all uncertainty-derivative components were non-positive; skipping branching.\n"
                    )
                continue

            child_profiles = generator.branch_N_times(
                node.profile,
                t_branch=t_branch,
                n_branches=n_branches,
                seed=int(rng.integers(low=0, high=np.iinfo(np.int32).max)),
            )
            if verbose:
                print(
                    f"\n[branch] interval={interval.index} parent={node.profile_id} "
                    f"t_branch={t_branch:.3f} generated={len(child_profiles)}\n"
                )

            for branch_label, child_profile in enumerate(child_profiles):
                child_id = f"profile_{next_profile_idx:06d}"
                next_profile_idx += 1

                profiles[child_id] = ProfileNode(
                    profile_id=child_id,
                    profile=child_profile,
                    parent_profile_id=node.profile_id,
                    created_in_interval=interval.index,
                    branch_time=t_branch,
                    branch_label=branch_label,
                )
                branch_events.append(
                    BranchEvent(
                        child_profile_id=child_id,
                        parent_profile_id=node.profile_id,
                        interval_index=interval.index,
                        branch_time=t_branch,
                        branch_label=branch_label,
                    )
                )

    return RecursiveBranchingResult(intervals=intervals, final_profiles=profiles, branch_events=branch_events)


def save_recursive_branching_output(
    result: RecursiveBranchingResult,
    profiles_h5_path: Path,
    *,
    root_group_name: str,
) -> None:
    """Save final profiles under ``root_group_name`` in a shared profiles HDF5 file."""
    profiles_h5_path = Path(profiles_h5_path)
    profiles_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(profiles_h5_path, "a") as h5f:
        if root_group_name in h5f:
            del h5f[root_group_name]
        root_grp = h5f.create_group(root_group_name)
        for profile_id, node in result.final_profiles.items():
            grp = root_grp.create_group(profile_id)
            grp.create_dataset("t", data=np.asarray(node.profile.t, dtype=np.float64))
            grp.create_dataset("theta_deg", data=np.asarray(node.profile.theta_deg, dtype=np.float64))
            grp.create_dataset("v_deg_s", data=np.asarray(node.profile.v_deg_s, dtype=np.float64))
            grp.create_dataset("a_deg_s2", data=np.asarray(node.profile.a_deg_s2, dtype=np.float64))
            grp.attrs["parent_profile_id"] = "" if node.parent_profile_id is None else node.parent_profile_id
            grp.attrs["created_in_interval"] = -1 if node.created_in_interval is None else node.created_in_interval
            grp.attrs["branch_time"] = np.nan if node.branch_time is None else node.branch_time
            grp.attrs["branch_label"] = -1 if node.branch_label is None else node.branch_label
