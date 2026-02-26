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
- The original profile is black, and each branch interval I_k has a unique color.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

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


def _interval_bounds(T: float, N_k: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be positive.")
    if N_k <= 0:
        raise ValueError("N_k must be a positive integer.")
    return np.linspace(0.0, T, N_k + 1)


def _interval_colors(N_k: int) -> List[str]:
    cmap = colormaps["tab20"]
    return [cmap(i % cmap.N) for i in range(N_k)]


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


def generate_branched_profiles(
    T: float = 200.0,
    dt: float = 1.0,
    N_k: int = 5,
    N_b: int = 2,
    seed: int = 1234,
    baseline_angle_deg: float = 45.0,
    kernel: str = "matern52",
    ell: float = 7.0,
    sill_v_deg2_s2: float = 0.02,
    nugget_v_deg2_s2: float = 0.0,
) -> Tuple[List[BranchedProfileRecord], np.ndarray]:
    """Generate branched control profiles and return records + interval edges."""
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if N_b <= 0:
        raise ValueError("N_b must be a positive integer.")

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
        )
    ]

    interval_edges = _interval_bounds(T=T, N_k=N_k)
    colors = _interval_colors(N_k=N_k)
    rng = np.random.default_rng(seed)

    for k in range(N_k):
        start = float(interval_edges[k])
        end = float(interval_edges[k + 1])

        # Snapshot so appends do not mutate current pass iteration.
        current_profiles = list(U_n)
        for record in current_profiles:
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
                    )
                )

    return U_n, interval_edges


def plot_control_profiles(
    U_n: List[BranchedProfileRecord],
    interval_edges: np.ndarray,
    save_as: str | None = None,
    N_k: int | None = None,
    N_b: int | None = None,
) -> None:
    """Plot all control profiles in U_n and optionally save the figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    branch_x: List[float] = []
    branch_y: List[float] = []

    for record in U_n:
        t = record.profile.t
        u = record.profile.theta_deg

        if record.is_original:
            ax.plot(t, u, color="black", linewidth=2.0, alpha=1.0, zorder=3)
        else:
            ax.plot(t, u, color=record.color, linewidth=1.0, alpha=0.7, zorder=2)
            if record.branch_time_s is not None:
                idx = int(np.argmin(np.abs(t - record.branch_time_s)))
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



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate branched drum-control profiles.")
    parser.add_argument("--T", type=float, default=200.0, help="Final time horizon in seconds.")
    parser.add_argument("--dt", type=float, default=1.0, help="Time-step for control profile grid.")
    parser.add_argument("--Nk", type=int, default=5, help="Number of subintervals I_k.")
    parser.add_argument("--Nb", type=int, default=2, help="Number of branches per (I_k, profile).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--save_as",
        type=str,
        default="../misc/branched_Nk{Nk}Nb{Nb}.png",
        help=(
            "Optional figure output path, relative to scripts/. Supports {Nk} and {Nb} "
            "format fields. Set to empty string to disable saving."
        ),
    )
    args = parser.parse_args()

    U_n, interval_edges = generate_branched_profiles(
        T=args.T,
        dt=args.dt,
        N_k=args.Nk,
        N_b=args.Nb,
        seed=args.seed,
    )

    print(f"Generated {len(U_n)} total control profiles in U_n.")
    print(f"Interval edges: {interval_edges}")
    print("Original profile color: black")
    print("Branched profile colors are assigned uniquely by I_k interval.")

    save_as = args.save_as.strip() if isinstance(args.save_as, str) else None
    if save_as == "":
        save_as = None

    plot_control_profiles(
        U_n=U_n,
        interval_edges=interval_edges,
        save_as=save_as,
        N_k=args.Nk,
        N_b=args.Nb,
    )


if __name__ == "__main__":
    main()
