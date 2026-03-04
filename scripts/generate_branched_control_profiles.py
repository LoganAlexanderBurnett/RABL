"""Generate branched drum-control profiles over [0, T].

Workflow implemented:
1. Build a baseline control profile u(t) from DrumVariography.
2. Initialize U_n = [u(t)].
3. Partition [0, T] into N_k subintervals I_k.
4. For each interval I_k and each profile u in U_n (snapshot per k),
   sample a random branching time t_b in I_k and branch N_b times to
   create u_k(t), then append all branches to U_n.

Notes:
- This script plots profiles and can optionally save the figure via --save_as.
- The original profile is black, each branch interval I_k has a unique color, and child lines are shown only after their branch point; branch timing can be midpoint or random.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rabl.variography.DrumVariography import DrumProfile, DrumProfileGenerator


@dataclass
class BranchedProfileRecord:
    """Container for a control profile plus branching metadata."""

    profile: DrumProfile
    is_original: bool
    interval_index: int | None
    branch_time_s: float | None
    color: str
    generation: int


def _interval_bounds(T: float, N_k: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be positive.")
    if N_k <= 0:
        raise ValueError("N_k must be a positive integer.")
    return np.linspace(0.0, T, N_k + 1)


def _interval_colors(N_k: int) -> List[str]:
    # Fixed vibrant palette requested by user; cycle through as needed.
    palette = ["darkcyan", "aquamarine", "mediumturquoise"]
    return [palette[i % len(palette)] for i in range(N_k)]


def _sample_branch_time_on_grid(
    t_grid: np.ndarray,
    interval_start: float,
    interval_end: float,
    rng: np.random.Generator,
) -> float:
    """Pick a random branching node from the time grid inside [start, end)."""
    # Prefer nodes in the half-open interval [start, end). For the final interval,
    # include the right endpoint so a node always exists.
    in_interval = (t_grid >= interval_start) & (t_grid < interval_end)
    if np.isclose(interval_end, t_grid[-1]):
        in_interval = (t_grid >= interval_start) & (t_grid <= interval_end)

    candidate_nodes = t_grid[in_interval]
    if candidate_nodes.size == 0:
        # Fallback (very coarse grids): use nearest node to interval midpoint.
        midpoint = 0.5 * (interval_start + interval_end)
        idx = int(np.argmin(np.abs(t_grid - midpoint)))
        return float(t_grid[idx])

    idx = int(rng.integers(0, candidate_nodes.size))
    return float(candidate_nodes[idx])


def _branch_time_for_interval_midpoint(
    t_grid: np.ndarray,
    interval_start: float,
    interval_end: float,
) -> float:
    """Pick the branch time at interval midpoint, snapped to nearest time-grid node."""
    midpoint = 0.5 * (interval_start + interval_end)
    idx = int(np.argmin(np.abs(t_grid - midpoint)))
    return float(t_grid[idx])


def generate_branched_profiles(
    T: float = 200.0,
    dt: float = 0.1,
    N_k: int = 3,
    N_b: int = 2,
    seed: int = 1234,
    baseline_angle_deg: float = 45.0,
    kernel: str = "matern52",
    ell: float = 5.0,
    sill_v_deg2_s2: float = 0.02,
    nugget_v_deg2_s2: float = 0.0,
    branching_time_mode: str = "midpoint",
) -> Tuple[List[BranchedProfileRecord], np.ndarray]:
    """Generate branched control profiles and return records + interval edges."""
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if N_b <= 0:
        raise ValueError("N_b must be a positive integer.")
    if branching_time_mode not in {"midpoint", "random"}:
        raise ValueError("branching_time_mode must be one of: {'midpoint', 'random'}")

    t_grid = np.arange(0.0, T + dt, dt, dtype=float)
    if not np.isclose(t_grid[-1], T):
        t_grid = np.append(t_grid, T)

    generator = DrumProfileGenerator(
        kernel=kernel,
        ell=ell,
        sill_v_deg2_s2=sill_v_deg2_s2,
        nugget_v_deg2_s2=nugget_v_deg2_s2,
    )

    base_profile = generator.generate(
        t_grid=t_grid,
        n_realizations=1,
        baseline_angle_deg=baseline_angle_deg,
        seed=seed,
    )[0]

    # Required by request: instantiate as U_n=[u(t)]
    U_n: List[BranchedProfileRecord] = [
        BranchedProfileRecord(
            profile=base_profile,
            is_original=True,
            interval_index=None,
            branch_time_s=None,
            color="black",
            generation=0,
        )
    ]

    interval_edges = _interval_bounds(T=T, N_k=N_k)
    colors = _interval_colors(N_k=N_k)
    rng = np.random.default_rng(seed)

    # Precompute one branch time per interval for midpoint mode so every profile
    # branched in the same interval I_k uses the exact same t_b.
    interval_branch_times = {
        k: _branch_time_for_interval_midpoint(
            t_grid=t_grid,
            interval_start=float(interval_edges[k]),
            interval_end=float(interval_edges[k + 1]),
        )
        for k in range(N_k)
    }

    for k in range(N_k):
        start = float(interval_edges[k])
        end = float(interval_edges[k + 1])

        # Snapshot so appends do not mutate current pass iteration.
        current_profiles = list(U_n)

        for record in current_profiles:
            if branching_time_mode == "midpoint":
                t_b = interval_branch_times[k]
            else:
                t_b = _sample_branch_time_on_grid(
                    t_grid=t_grid,
                    interval_start=start,
                    interval_end=end,
                    rng=rng,
                )

            # Unique deterministic branch seed per (k, profile, branch)
            for b in range(N_b):
                branch_seed = int(rng.integers(0, 2**31 - 1))
                branched = generator.branch(
                    original=record.profile,
                    t_branch=t_b,
                    seed=branch_seed,
                )
                U_n.append(
                    BranchedProfileRecord(
                        profile=branched,
                        is_original=False,
                        interval_index=k,
                        branch_time_s=t_b,
                        color=colors[k],
                        generation=record.generation + 1,
                    )
                )

    # Safety check: midpoint mode must yield exactly one branch time per interval.
    if branching_time_mode == "midpoint":
        for k in range(N_k):
            times_k = {
                float(rec.branch_time_s)
                for rec in U_n
                if (rec.interval_index == k and rec.branch_time_s is not None)
            }
            if len(times_k) > 1:
                raise RuntimeError(
                    f"Midpoint mode violated in interval k={k}: found {sorted(times_k)}"
                )

    return U_n, interval_edges


def plot_control_profiles(
    U_n: List[BranchedProfileRecord],
    interval_edges: np.ndarray,
    save_as: str | None = None,
    N_k: int | None = None,
    N_b: int | None = None,
    branching_time_mode: str = "midpoint",
) -> None:
    """Plot all control profiles in U_n and optionally save the figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    branch_x: List[float] = []
    branch_y: List[float] = []

    ordered_records = sorted(U_n, key=lambda rec: rec.generation, reverse=True)

    # In midpoint mode, use one canonical branch time per interval for rendering,
    # so all branch-point markers in the same I_k are vertically aligned.
    interval_canonical_branch_time: dict[int, float] = {}
    if branching_time_mode == "midpoint":
        for rec in U_n:
            if rec.interval_index is None or rec.branch_time_s is None:
                continue
            interval_canonical_branch_time.setdefault(rec.interval_index, float(rec.branch_time_s))

    for record in ordered_records:
        t = record.profile.t
        u = record.profile.theta_deg

        if record.is_original:
            ax.plot(t, u, color="black", linewidth=2.0, alpha=1.0, zorder=3)
            continue

        if record.branch_time_s is None:
            continue

        t_b = float(record.branch_time_s)
        if branching_time_mode == "midpoint" and record.interval_index is not None:
            t_b = interval_canonical_branch_time.get(record.interval_index, t_b)

        visible = t >= t_b
        if not np.any(visible):
            continue

        ax.plot(t[visible], u[visible], color=record.color, linewidth=1.6, alpha=0.95, zorder=2)

        idx = int(np.argmin(np.abs(t - t_b)))
        branch_x.append(float(t[idx]))
        branch_y.append(float(u[idx]))

    if branch_x:
        ax.scatter(branch_x, branch_y, color="black", s=14, zorder=4)

    for edge in interval_edges[1:-1]:
        ax.axvline(edge, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)

    ax.set_title("Branched control profiles u(t)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control profile u(t) [deg]")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_as:
        save_path = save_as
        if N_k is not None and N_b is not None:
            save_path = save_path.format(Nk=N_k, Nb=N_b)
        output_path = (Path(__file__).resolve().parent / save_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    plt.show()


def _ordered_interval_children(
    U_n: List[BranchedProfileRecord],
    N_k: int,
) -> List[List[BranchedProfileRecord]]:
    """Collect child profiles grouped by birth interval and sorted by generation."""
    grouped: List[List[BranchedProfileRecord]] = [[] for _ in range(N_k)]
    for record in U_n:
        if record.is_original or record.interval_index is None:
            continue
        if 0 <= record.interval_index < N_k:
            grouped[record.interval_index].append(record)

    for k in range(N_k):
        grouped[k] = sorted(grouped[k], key=lambda rec: (rec.generation, rec.branch_time_s or 0.0))

    return grouped


def save_branching_video(
    U_n: List[BranchedProfileRecord],
    interval_edges: np.ndarray,
    save_as: str,
    fps: int = 12,
    progression_frames_per_interval: int = 18,
) -> None:
    """Render a staged animation of branching progression and save it as a video."""
    N_k = len(interval_edges) - 1
    base_record = next((record for record in U_n if record.is_original), None)
    if base_record is None:
        raise RuntimeError("Unable to render video without a base profile.")

    interval_children = _ordered_interval_children(U_n=U_n, N_k=N_k)

    fig, ax = plt.subplots(figsize=(10, 6))
    t_base = base_record.profile.t
    u_base = base_record.profile.theta_deg
    x_min = float(np.min(t_base))
    x_max = float(np.max(t_base))
    y_min = min(float(np.min(rec.profile.theta_deg)) for rec in U_n)
    y_max = max(float(np.max(rec.profile.theta_deg)) for rec in U_n)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

    # Frame plan:
    # 0: base profile only
    # 1: base + interval edges
    # 2+: one progressive plotting segment per interval
    interval_frame_meta: List[Tuple[int, int]] = []
    frame_index = 2
    for k in range(N_k):
        interval_frame_meta.append((frame_index, frame_index + progression_frames_per_interval - 1))
        frame_index += progression_frames_per_interval
    total_frames = frame_index

    def _draw_interval_group(interval_idx: int, x_cutoff: float) -> None:
        for record in interval_children[interval_idx]:
            if record.branch_time_s is None:
                continue
            t = record.profile.t
            u = record.profile.theta_deg
            visible = (t >= record.branch_time_s) & (t <= x_cutoff)
            if np.any(visible):
                ax.plot(t[visible], u[visible], color=record.color, linewidth=1.6, alpha=0.95, zorder=2)

    def _update(frame: int) -> None:
        ax.clear()
        ax.plot(t_base, u_base, color="black", linewidth=2.0, alpha=1.0, zorder=3)

        if frame >= 1:
            for edge in interval_edges[1:-1]:
                ax.axvline(edge, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)

        for k, (start_frame, end_frame) in enumerate(interval_frame_meta):
            if frame < start_frame:
                continue

            if frame >= end_frame:
                x_cut = x_max
            else:
                progress = (frame - start_frame + 1) / progression_frames_per_interval
                x_cut = x_min + progress * (x_max - x_min)

            _draw_interval_group(interval_idx=k, x_cutoff=x_cut)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title("Branched control profiles u(t)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Control profile u(t) [deg]")
        ax.grid(True, alpha=0.25)

    animation = FuncAnimation(fig, _update, frames=total_frames, interval=int(1000 / max(fps, 1)))

    output_path = (Path(__file__).resolve().parent / save_as).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        animation.save(output_path, writer="pillow", fps=fps)
    else:
        # For mp4/video formats, use ffmpeg when available.
        animation.save(output_path, writer="ffmpeg", fps=fps)

    plt.close(fig)
    print(f"Saved branching video to: {output_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate branched drum-control profiles.")
    parser.add_argument("--T", type=float, default=1000.0, help="Final time horizon in seconds.")
    parser.add_argument("--dt", type=float, default=0.4, help="Time-step for control profile grid.")
    parser.add_argument("--Nk", type=int, default=3, help="Number of subintervals I_k.")
    parser.add_argument("--Nb", type=int, default=2, help="Number of branches per (I_k, profile).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--branching_time_mode",
        type=str,
        choices=["midpoint", "random"],
        default="midpoint",
        help=(
            "Branch-time strategy per interval I_k: 'midpoint' uses one shared midpoint "
            "branch time for all profiles in I_k; 'random' samples per profile as before."
        ),
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="../misc/branched_Nk{Nk}Nb{Nb}.png",
        help=(
            "Optional figure output path, relative to scripts/. Supports {Nk} and {Nb} "
            "format fields. Set to empty string to disable saving."
        ),
    )
    parser.add_argument(
        "--save_video_as",
        type=str,
        default="",
        help=(
            "Optional video output path, relative to scripts/. Supports common video formats "
            "such as .mp4 or .gif. Leave empty to disable video export."
        ),
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=12,
        help="Frames per second for video output.",
    )
    parser.add_argument(
        "--video_progression_frames",
        type=int,
        default=18,
        help="Number of progressive drawing frames allocated to each interval.",
    )
    args = parser.parse_args()

    U_n, interval_edges = generate_branched_profiles(
        T=args.T,
        dt=args.dt,
        N_k=args.Nk,
        N_b=args.Nb,
        seed=args.seed,
        branching_time_mode=args.branching_time_mode,
    )

    print(f"Generated {len(U_n)} total control profiles in U_n.")
    print(f"Interval edges: {interval_edges}")
    print("Original profile color: black")
    print("Branched profile colors are assigned uniquely by I_k interval.")
    print(f"Branching-time mode: {args.branching_time_mode}")

    save_as = args.save_as.strip() if isinstance(args.save_as, str) else None
    if save_as == "":
        save_as = None

    plot_control_profiles(
        U_n=U_n,
        interval_edges=interval_edges,
        save_as=save_as,
        N_k=args.Nk,
        N_b=args.Nb,
        branching_time_mode=args.branching_time_mode,
    )

    save_video_as = args.save_video_as.strip() if isinstance(args.save_video_as, str) else None
    if save_video_as:
        save_branching_video(
            U_n=U_n,
            interval_edges=interval_edges,
            save_as=save_video_as,
            fps=args.video_fps,
            progression_frames_per_interval=args.video_progression_frames,
        )


if __name__ == "__main__":
    main()
