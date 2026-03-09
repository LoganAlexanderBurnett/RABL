"""Plot ensemble forecast uncertainty derivatives from rolling_forecasts.h5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from rabl.machine_learning.bagging_ensemble import TARGET_NAMES, plot_ensemble_forecast_profile_grid
from rabl.machine_learning.branchpoint_finder import finite_difference


def _decode_columns(columns_attr: np.ndarray | list[object]) -> list[str]:
    out: list[str] = []
    for item in columns_attr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _plot_derivative_grid(
    *,
    t_series: np.ndarray,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    x_sigma_derivative: np.ndarray,
    target_names: list[str],
    profile_name: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 7, figsize=(26, 8), sharex=True)
    axes_flat = axes.flatten()

    axes_flat[0].axis("off")
    axes_flat[0].set_title("(unused)")

    for target_idx, target_name in enumerate(target_names):
        ax = axes_flat[target_idx + 1]
        ax.plot(t_series, y_true[:, target_idx], linewidth=1.6, color="C0", label="Ground truth")
        ax.plot(t_series, y_mean[:, target_idx], linewidth=1.6, color="C3", label="Mean prediction")
        ax.plot(
            t_series,
            x_sigma_derivative[:, target_idx],
            linewidth=1.2,
            color="C4",
            linestyle="--",
            label="d(x_sigma)/dt",
        )
        ax.set_title(target_name)
        ax.grid(True, alpha=0.3)

    for idx in range(7, 14):
        axes_flat[idx].set_xlabel("Time step")
    for idx in range(1, 14):
        axes_flat[idx].set_ylabel("State / derivative")

    handles, labels = axes_flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Ensemble Forecast + Uncertainty Derivative - {profile_name}", y=1.06, fontsize=16)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute/plot 4th-order derivative of x_sigma from rolling_forecasts.h5.")
    parser.add_argument("forecast_h5_path", type=Path, help="Path to rolling_forecasts.h5.")
    parser.add_argument("--profile", type=str, default=None, help="Profile name to plot. Defaults to first available profile.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/uncertainty_derivative"),
        help="Directory where the two output plots are saved.",
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
    y_true = np.column_stack([table[:, columns.index(f"x_true(t)_{name}")] for name in target_names])
    y_mean = np.column_stack([table[:, columns.index(f"x_mean(t)_{name}")] for name in target_names])
    x_sigma = np.column_stack([table[:, columns.index(f"x_2sigma(t)_{name}")] for name in target_names])

    # Requested: fourth-order finite differencing of x_sigma for all target variables.
    x_sigma_derivative = finite_difference(x_sigma, order=4, dt=args.dt)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = args.output_dir / f"{profile_name}_ensemble_profile_grid.png"
    plot_ensemble_forecast_profile_grid(
        args.forecast_h5_path,
        profile_name=profile_name,
        save_path=baseline_path,
        close_figure=True,
    )

    derivative_path = args.output_dir / f"{profile_name}_sigma_derivative_grid.png"
    _plot_derivative_grid(
        t_series=t_series,
        y_true=y_true,
        y_mean=y_mean,
        x_sigma_derivative=x_sigma_derivative,
        target_names=target_names,
        profile_name=profile_name,
        save_path=derivative_path,
    )

    print(f"Saved baseline ensemble grid: {baseline_path}")
    print(f"Saved derivative grid: {derivative_path}")


if __name__ == "__main__":
    main()
