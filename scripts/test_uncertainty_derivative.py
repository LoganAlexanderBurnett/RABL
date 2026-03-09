"""Plot ensemble forecasts with overlaid uncertainty derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from rabl.machine_learning.bagging_ensemble import TARGET_NAMES
from rabl.machine_learning.branchpoint_finder import finite_difference

FIGSIZE_2X7 = (26, 8)


def _decode_columns(columns_attr: np.ndarray | list[object]) -> list[str]:
    out: list[str] = []
    for item in columns_attr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _plot_overlay_grid(
    *,
    t_series: np.ndarray,
    u_series: np.ndarray,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_2sigma: np.ndarray,
    dx_sigma_dt: np.ndarray,
    target_names: list[str],
    profile_name: str,
    save_path: Path,
) -> None:
    y_upper = y_mean + y_2sigma
    y_lower = y_mean - y_2sigma

    fig, axes = plt.subplots(2, 7, figsize=FIGSIZE_2X7, sharex=True)
    axes_flat = axes.flatten()

    axes_flat[0].plot(t_series, u_series, linewidth=1.5, color="black")
    axes_flat[0].set_title("drumAngleDeg")
    axes_flat[0].set_ylabel("u(t)")
    axes_flat[0].grid(True, alpha=0.3)

    derivative_axes: list[plt.Axes] = []

    for target_idx, target_name in enumerate(target_names):
        ax = axes_flat[target_idx + 1]
        ax2 = ax.twinx()
        derivative_axes.append(ax2)

        ax.plot(t_series, y_true[:, target_idx], label="Ground truth", linewidth=1.6, color="C0")
        ax.plot(t_series, y_mean[:, target_idx], label="Mean prediction", linewidth=1.6, color="C3")
        ax.plot(t_series, y_upper[:, target_idx], linestyle="--", linewidth=1.0, color="C1", label="Mean + 2σ")
        ax.plot(t_series, y_lower[:, target_idx], linestyle="--", linewidth=1.0, color="C2", label="Mean - 2σ")
        ax.fill_between(
            t_series,
            y_lower[:, target_idx],
            y_upper[:, target_idx],
            color="C1",
            alpha=0.15,
            linewidth=0,
        )

        deriv_series = dx_sigma_dt[:, target_idx]
        ax2.plot(
            t_series,
            deriv_series,
            linewidth=1.2,
            color="C4",
            linestyle=":",
            label="d(x_sigma)/dt",
        )[0]
        ax2.axhline(0.0, color="0.5", linewidth=0.9, linestyle="--", alpha=0.8)

        positive_indices = np.flatnonzero(deriv_series > 0.0)
        if positive_indices.size:
            top_count = min(3, positive_indices.size)
            ranked = positive_indices[np.argsort(deriv_series[positive_indices])[-top_count:]]
            ranked = ranked[np.argsort(deriv_series[ranked])[::-1]]
            for peak_rank, idx_peak in enumerate(ranked, start=1):
                x_peak = t_series[idx_peak]
                y_peak = deriv_series[idx_peak]
                ax2.scatter([x_peak], [y_peak], color="C4", s=14, zorder=4)
                ax2.annotate(
                    f"P{peak_rank}",
                    xy=(x_peak, y_peak),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="C4",
                    fontsize=7,
                )

        ax.set_title(target_name)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("State")
        ax2.set_ylabel("d(x_sigma)/dt")

    x_min, x_max = float(t_series[0]), float(t_series[-1])
    for ax in axes_flat:
        ax.set_xlim(x_min, x_max)

    for idx in range(7, 14):
        axes_flat[idx].set_xlabel("Time step")

    handles_left, labels_left = axes_flat[1].get_legend_handles_labels()
    handles_right, labels_right = derivative_axes[0].get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(f"Ensemble Forecast + Uncertainty Derivative - {profile_name}", y=1.06, fontsize=16)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute/plot 4th-order d(x_sigma)/dt overlaid on mean ±2σ ensemble forecast grid."
    )
    parser.add_argument("forecast_h5_path", type=Path, help="Path to rolling_forecasts.h5.")
    parser.add_argument("--profile", type=str, default=None, help="Profile name to plot. Defaults to first available profile.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/uncertainty_derivative"),
        help="Directory where the output plot is saved.",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Time-step spacing used for derivative.")
    args = parser.parse_args()

    with h5py.File(args.forecast_h5_path, "r") as h5f:
        profile_names = sorted(h5f.keys())
        if not profile_names:
            raise ValueError(f"No profiles found in {args.forecast_h5_path}.")

        profile_name = args.profile or profile_names[0]
        if profile_name not in h5f:
            raise KeyError(f"Profile '{profile_name}' not found in {args.forecast_h5_path}.")

        group = h5f[profile_name]
        table = group["data"][...].astype(np.float32)
        columns = _decode_columns(group.attrs.get("columns", []))

    target_names = list(TARGET_NAMES)
    t_series = table[:, columns.index("t")]
    u_series = table[:, columns.index("u(t)")]
    y_true = np.column_stack([table[:, columns.index(f"x_true(t)_{name}")] for name in target_names])
    y_mean = np.column_stack([table[:, columns.index(f"x_mean(t)_{name}")] for name in target_names])
    y_2sigma = np.column_stack([table[:, columns.index(f"x_2sigma(t)_{name}")] for name in target_names])

    dx_sigma_dt = finite_difference(y_2sigma, order=4, dt=args.dt)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / f"{profile_name}_ensemble_with_sigma_derivative.png"
    _plot_overlay_grid(
        t_series=t_series,
        u_series=u_series,
        y_true=y_true,
        y_mean=y_mean,
        y_2sigma=y_2sigma,
        dx_sigma_dt=dx_sigma_dt,
        target_names=target_names,
        profile_name=profile_name,
        save_path=overlay_path,
    )

    print(f"Saved overlay grid: {overlay_path}")


if __name__ == "__main__":
    main()
