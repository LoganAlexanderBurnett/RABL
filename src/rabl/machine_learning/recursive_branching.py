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
import importlib.util
import json
from pathlib import Path
import re
from typing import Callable, Protocol

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.io import savemat

from rabl.machine_learning.branchpoint_finder import finite_difference
from rabl.machine_learning.lstm_pipeline import build_model, rolling_forecast, _load_scaling_stats
from rabl.machine_learning.build_lstm_dataset import CONTROL_COLUMN, STATE_COLUMNS
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


@dataclass(frozen=True)
class RecursiveBranchingBatchConfig:
    model_paths: tuple[Path, ...]
    bagged_h5_path: Path
    output_dir: Path
    config_path: Path
    T: float = 200.0
    dt: float = 0.4
    Nk: int = 3
    Nb: int = 2
    Nr: int = 1
    baseline_angle_deg: float = 45.0
    seed: int = 123
    lookback: int = 12
    n_lstm: int = 1
    lstm_hidden: int = 64
    n_fc: int = 1
    fc_hidden: tuple[int, ...] = (64,)
    kernel: str = "matern52"
    ell: float = 5.0
    sigma_theta_target: float = 2.5
    nugget_v_deg2_s2: float = 0.0
    finite_difference_order: int = 4
    device: str = "cpu"


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


def _load_config_module(config_path: Path):
    spec = importlib.util.spec_from_file_location("rabl_recursive_branch_cfg", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _steady_state_rows(steady_state: dict, *, state_dim: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    control_value = steady_state.get(CONTROL_COLUMN)
    if control_value is None:
        raise ValueError(f"STEADY_STATE must contain key '{CONTROL_COLUMN}'.")
    missing = [key for key in STATE_COLUMNS if key not in steady_state]
    if missing:
        raise ValueError(f"STEADY_STATE missing required keys: {missing}")
    state_values = [steady_state[key] for key in STATE_COLUMNS]
    if state_dim > len(state_values):
        raise ValueError(f"state_dim={state_dim} exceeds available state columns.")
    state_row = np.asarray(state_values[:state_dim], dtype=np.float32)
    control_row = np.asarray([float(control_value)], dtype=np.float32)
    return (
        np.repeat(state_row[None, :], lookback, axis=0),
        np.repeat(control_row[None, :], lookback, axis=0),
    )


def _make_control_window(u_series: np.ndarray, step: int, lookback: int, control_steady: float) -> np.ndarray:
    start = max(0, step - lookback + 1)
    history = u_series[start : step + 1]
    out = np.empty(lookback, dtype=np.float32)
    out[: lookback - history.size] = np.float32(control_steady)
    out[lookback - history.size :] = history
    return out


def _build_profile_to_x_adapter(
    *,
    n_steps: int,
    lookback: int,
    n_features: int,
    state_dim: int,
    steady_state: dict,
    scaling_stats: dict,
    control_channel: int = 0,
) -> Callable[[DrumProfile], np.ndarray]:
    control_dim = n_features - state_dim
    if not (0 <= control_channel < control_dim):
        raise ValueError("control_channel out of range.")
    control_idx = state_dim + control_channel
    state_pad, control_pad = _steady_state_rows(steady_state, state_dim=state_dim, lookback=lookback)
    control_steady = float(control_pad[0, 0])

    def _adapter(profile: DrumProfile) -> np.ndarray:
        u = np.asarray(profile.theta_deg, dtype=np.float32)
        x = np.zeros((n_steps, lookback, n_features), dtype=np.float32)
        x[:, :, :state_dim] = state_pad[None, :, :]
        for step in range(n_steps):
            x[step, :, control_idx] = _make_control_window(u, step, lookback, control_steady)
        if scaling_stats["type"] == "standard":
            x = (x - scaling_stats["x"]["mean"][None, None, :]) / scaling_stats["x"]["std"][None, None, :]
        elif scaling_stats["type"] == "minmax":
            x = (x - scaling_stats["x"]["min"][None, None, :]) / scaling_stats["x"]["span"][None, None, :]
        else:
            raise ValueError(f"Unsupported scaling type: {scaling_stats['type']}")
        return x

    return _adapter


def _infer_checkpoint_io_shapes(model_path: Path) -> tuple[int, int]:
    state_dict = torch.load(Path(model_path), map_location="cpu")
    return int(state_dict["lstm.weight_ih_l0"].shape[1]), int(state_dict["output_layer.bias"].shape[0])


def _find_latest_variography_profile_index(variography_root: Path) -> int:
    if not variography_root.exists():
        return 0
    pattern = re.compile(r"^drum_profile_(\d{5})$")
    max_idx = 0
    for p in variography_root.glob("batch_*/drum_profile_*.*"):
        if p.suffix.lower() not in {".csv", ".mat"}:
            continue
        m = pattern.match(p.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def _append_manifest(
    manifest_path: Path,
    *,
    root_group_name: str,
    result: RecursiveBranchingResult,
    written_mats: list[Path],
) -> None:
    entries = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    sorted_nodes = sorted(result.final_profiles.items())
    for (profile_id, node), mat_path in zip(sorted_nodes, written_mats, strict=True):
        entries.append(
            {
                "root_group_name": root_group_name,
                "profile_id": profile_id,
                "parent_profile_id": "" if node.parent_profile_id is None else node.parent_profile_id,
                "created_in_interval": -1 if node.created_in_interval is None else int(node.created_in_interval),
                "branch_time": None if node.branch_time is None else float(node.branch_time),
                "branch_label": -1 if node.branch_label is None else int(node.branch_label),
                "mat_file": mat_path.name,
            }
        )
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


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
    profiles: dict[str, ProfileNode] = {"profile_00000": ProfileNode(profile_id="profile_00000", profile=root_profile)}
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
                child_id = f"profile_{next_profile_idx:05d}"
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


def save_profiles_as_mat_files(
    result: RecursiveBranchingResult,
    output_dir: Path,
    *,
    start_index: int = 1,
    table_name: str = "profile",
) -> list[Path]:
    """Save profile trajectories as ``drum_profile_XXXXX.mat`` files.

    Parameters
    ----------
    result:
        Recursive branching result containing profile trajectories.
    output_dir:
        Directory where MAT files are written.
    start_index:
        First global profile index to use.
    table_name:
        MAT variable name containing ``[t, theta_deg, v_deg_s, a_deg_s2]``.
    """
    if start_index < 1:
        raise ValueError(f"start_index must be >= 1, got {start_index}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    next_idx = int(start_index)
    for _profile_id, node in sorted(result.final_profiles.items()):
        table = np.column_stack(
            [
                np.asarray(node.profile.t, dtype=float),
                np.asarray(node.profile.theta_deg, dtype=float),
                np.asarray(node.profile.v_deg_s, dtype=float),
                np.asarray(node.profile.a_deg_s2, dtype=float),
            ]
        )
        out_path = output_dir / f"drum_profile_{next_idx:05d}.mat"
        savemat(str(out_path), {table_name: table})
        written.append(out_path)
        next_idx += 1
    return written


def save_profiles_lineage_graph(
    profiles_h5_path: Path,
    output_image_path: Path,
    *,
    root_group_name: str,
) -> Path:
    """Render and save a lineage graph image from a shared profiles.h5 root group."""
    profiles_h5_path = Path(profiles_h5_path)
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(profiles_h5_path, "r") as h5f:
        if root_group_name not in h5f:
            raise KeyError(f"Root group '{root_group_name}' not found in {profiles_h5_path}.")
        root_grp = h5f[root_group_name]
        profile_ids = sorted(root_grp.keys())
        if not profile_ids:
            raise ValueError(f"Root group '{root_group_name}' contains no profiles.")

        parent_by_id: dict[str, str | None] = {}
        interval_by_id: dict[str, int] = {}
        for profile_id in profile_ids:
            grp = root_grp[profile_id]
            parent_attr = str(grp.attrs.get("parent_profile_id", "")).strip()
            parent_by_id[profile_id] = parent_attr if parent_attr else None
            interval_by_id[profile_id] = int(grp.attrs.get("created_in_interval", -1))

    depth_by_id: dict[str, int] = {}
    unresolved = set(profile_ids)
    while unresolved:
        progress = False
        for profile_id in list(unresolved):
            parent_id = parent_by_id[profile_id]
            if parent_id is None or parent_id not in parent_by_id:
                depth_by_id[profile_id] = 0
                unresolved.remove(profile_id)
                progress = True
            elif parent_id in depth_by_id:
                depth_by_id[profile_id] = depth_by_id[parent_id] + 1
                unresolved.remove(profile_id)
                progress = True
        if not progress:
            for profile_id in unresolved:
                depth_by_id[profile_id] = 0
            break

    order = sorted(profile_ids, key=lambda pid: (depth_by_id[pid], pid))
    x_by_id = {profile_id: float(idx) for idx, profile_id in enumerate(order)}
    y_by_id = {profile_id: float(depth_by_id[profile_id]) for profile_id in profile_ids}

    fig, ax = plt.subplots(figsize=(len(profile_ids) * 0.3, 6.5))

    for child_id, parent_id in parent_by_id.items():
        if parent_id is None or parent_id not in x_by_id:
            continue
        ax.plot(
            [x_by_id[parent_id], x_by_id[child_id]],
            [y_by_id[parent_id], y_by_id[child_id]],
            color="0.65",
            linewidth=1.0,
            zorder=1,
            alpha=0.5
        )

    interval_palette = _interval_colors(max((interval_by_id[pid] for pid in profile_ids), default=-1) + 1)
    for profile_id in profile_ids:
        interval_idx = interval_by_id[profile_id]
        color = "black" if interval_idx < 0 else interval_palette[interval_idx % len(interval_palette)]
        ax.scatter(x_by_id[profile_id], y_by_id[profile_id], s=34, color=color, edgecolor="white", linewidth=0.4, zorder=3)

    for profile_id in profile_ids:
        ax.text(
            x_by_id[profile_id],
            y_by_id[profile_id] + 0.12,
            profile_id,
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=60,
        )

    ax.set_title(f"Profile Lineage Graph ({root_group_name})")
    ax.set_xlabel("Profiles")
    ax.set_ylabel("Generation depth")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_yticks(sorted(set(y_by_id.values())))
    ax.set_yticklabels([str(int(v)) for v in sorted(set(y_by_id.values()))])
    ax.invert_yaxis()
    max_interval_idx = max((interval_by_id[pid] for pid in profile_ids), default=-1)
    legend_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markersize=7, label="Root profile")]
    for interval_idx in range(max_interval_idx + 1):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=interval_palette[interval_idx % len(interval_palette)],
                markersize=7,
                label=f"Spawned in Interval {interval_idx + 1}",
            )
        )
    ax.legend(handles=legend_handles, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_image_path


def _interval_colors(n_intervals: int) -> list[str]:
    """Use the same interval palette as branched profile plotting."""
    palette = ["darkcyan", "aquamarine", "mediumturquoise"]
    return [palette[i % len(palette)] for i in range(max(1, n_intervals))]


def run_recursive_branching_batch(config: RecursiveBranchingBatchConfig) -> Path:
    """Run recursive branching for Nr roots and write MAT + manifest artifacts."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.Nr < 1:
        raise ValueError("Nr must be >=1.")
    if not config.model_paths:
        raise ValueError("model_paths must be non-empty.")

    inferred = [_infer_checkpoint_io_shapes(path) for path in config.model_paths]
    n_features, num_targets = inferred[0]
    if any(shape != inferred[0] for shape in inferred[1:]):
        raise ValueError("All model checkpoints must share input/output shapes.")

    scaling_stats = _load_scaling_stats(config.bagged_h5_path)
    cfg_mod = _load_config_module(config.config_path)
    steady_state = getattr(cfg_mod, "STEADY_STATE", None)
    if not isinstance(steady_state, dict):
        raise ValueError("STEADY_STATE must exist in provided config file.")

    t_grid = np.arange(0.0, config.T + config.dt, config.dt, dtype=float)
    if not np.isclose(t_grid[-1], config.T):
        t_grid = np.append(t_grid, config.T)
    n_steps = t_grid.size
    state_dim = int(num_targets)
    profile_to_x = _build_profile_to_x_adapter(
        n_steps=n_steps,
        lookback=config.lookback,
        n_features=n_features,
        state_dim=state_dim,
        steady_state=steady_state,
        scaling_stats=scaling_stats,
    )

    models = load_trained_ensemble(
        list(config.model_paths),
        timesteps=config.lookback,
        num_features=n_features,
        num_targets=num_targets,
        n_lstm=config.n_lstm,
        lstm_hidden=config.lstm_hidden,
        n_fc=config.n_fc,
        fc_hidden=config.fc_hidden,
        device=config.device,
    )
    forecaster = LSTMEnsembleForecaster(models, profile_to_x=profile_to_x, state_dim=state_dim)
    generator = DrumProfileGenerator(
        kernel=config.kernel,
        ell=float(config.ell),
        nugget_v_deg2_s2=float(config.nugget_v_deg2_s2),
    )
    generator.solve_params_for_sigma_theta(
        t_grid=t_grid,
        sigma_theta_target=float(config.sigma_theta_target),
        ell=float(config.ell),
        nugget=float(config.nugget_v_deg2_s2),
        update_instance=True,
    )
    weights = np.ones((num_targets,), dtype=float)

    profiles_h5 = config.output_dir / "profiles.h5"
    manifest_path = config.output_dir / "branched_profiles_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    next_idx = _find_latest_variography_profile_index(config.output_dir.parents[0]) + 1
    for root_idx in range(config.Nr):
        root = generate_root_profile(
            generator,
            t_grid,
            baseline_angle_deg=config.baseline_angle_deg,
            seed=config.seed + root_idx,
        )
        result = run_recursive_branching(
            forecaster=forecaster,
            generator=generator,
            root_profile=root,
            n_intervals=config.Nk,
            n_branches=config.Nb,
            weights=weights,
            finite_difference_order=config.finite_difference_order,
            seed=config.seed + root_idx,
            verbose=True,
        )
        root_group = f"root_{root_idx + 1:03d}"
        save_recursive_branching_output(result, profiles_h5, root_group_name=root_group)
        mats = save_profiles_as_mat_files(result, config.output_dir, start_index=next_idx)
        next_idx += len(mats)
        _append_manifest(
            manifest_path,
            root_group_name=root_group,
            result=result,
            written_mats=mats,
        )

    return config.output_dir
