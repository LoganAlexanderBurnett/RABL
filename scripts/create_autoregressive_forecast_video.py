"""Create an animation that visualizes autoregressive rolling forecasts on a 2x7 grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np

FIGSIZE_2X7 = (26, 8)


def _decode_columns(columns_attr: np.ndarray | list[object]) -> list[str]:
    out: list[str] = []
    for item in columns_attr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out




def _extract_target_names(columns: list[str]) -> list[str]:
    """Infer target names directly from HDF5 column schema."""
    prefixes = ("x(t)_", "x_true(t)_")
    target_names: list[str] = []
    for col in columns:
        for prefix in prefixes:
            if col.startswith(prefix):
                target_names.append(col[len(prefix) :])
                break
    return target_names

def _resolve_target_series(
    table: np.ndarray,
    columns: list[str],
    target_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) from either single-model or ensemble schema."""
    single_truth = [f"x(t)_{name}" for name in target_names]
    single_pred = [f"x^~(t)_{name}" for name in target_names]
    ensemble_truth = [f"x_true(t)_{name}" for name in target_names]
    ensemble_pred = [f"x_mean(t)_{name}" for name in target_names]

    if all(name in columns for name in single_truth + single_pred):
        y_true = np.column_stack([table[:, columns.index(name)] for name in single_truth])
        y_pred = np.column_stack([table[:, columns.index(name)] for name in single_pred])
        return y_true.astype(np.float32), y_pred.astype(np.float32)

    if all(name in columns for name in ensemble_truth + ensemble_pred):
        y_true = np.column_stack([table[:, columns.index(name)] for name in ensemble_truth])
        y_pred = np.column_stack([table[:, columns.index(name)] for name in ensemble_pred])
        return y_true.astype(np.float32), y_pred.astype(np.float32)

    raise KeyError(
        "Unable to resolve forecast columns. Expected either "
        "[x(t)_*, x^~(t)_*] or [x_true(t)_*, x_mean(t)_*] schemas."
    )


def _build_autoregressive_history(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Approximate the shifted state history consumed by rolling autoregressive inference."""
    history = np.empty_like(y_true)
    history[0] = y_true[0]
    if y_true.shape[0] > 1:
        history[1:] = y_pred[:-1]
    return history


def save_autoregressive_forecast_video(
    *,
    t_series: np.ndarray,
    u_series: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    profile_name: str,
    save_path: Path,
    lookback: int,
    fps: int,
    video_dpi: int,
    frame_stride: int,
) -> None:
    fig, axes = plt.subplots(2, 7, figsize=FIGSIZE_2X7, sharex=True)
    axes_flat = axes.flatten()

    input_color = "crimson"
    output_color = "royalblue"
    truth_color = "0.7"

    ar_history = _build_autoregressive_history(y_true=y_true, y_pred=y_pred)

    x_min = float(t_series[0])
    x_max = float(t_series[-1])

    # Global y-limits for stability across frames.
    control_pad = 0.06 * max(1e-6, float(np.ptp(u_series)))
    control_min = float(np.min(u_series) - control_pad)
    control_max = float(np.max(u_series) + control_pad)

    # Per-target y-limits (with per-target padding) so a single large-range variable
    # does not force every subplot into an overly wide axis range.
    state_lims: list[tuple[float, float]] = []
    for idx in range(y_true.shape[1]):
        combined = np.concatenate([y_true[:, idx], y_pred[:, idx]])
        pad = 0.06 * max(1e-6, float(np.ptp(combined)))
        y_lo = float(np.min(combined) - pad)
        y_hi = float(np.max(combined) + pad)
        state_lims.append((y_lo, y_hi))

    legend_handles = [
        Line2D([0], [0], color=truth_color, linewidth=1.0, label="Ground truth"),
        Line2D([0], [0], color=input_color, linewidth=2.2, label="Input window (AR state/control)"),
        Line2D([0], [0], color=output_color, linewidth=1.1, marker="|", markersize=10, label="Output prediction (current step)"),
    ]

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )

    def _update(frame: int) -> None:
        frame = int(frame)
        for ax in axes_flat:
            ax.clear()

        # Top-left: control signal. Input window used at this forecast step in input_color.
        ax_u = axes_flat[0]
        ax_u.plot(t_series, u_series, color="0.85", linewidth=1.3, label="Control (full profile)")
        if frame > 0:
            start = max(0, frame - lookback)
            ax_u.plot(
                t_series[start:frame],
                u_series[start:frame],
                color=input_color,
                linewidth=1.0,
                label="Input window (control)",
            )
        ax_u.set_title("drumAngleDeg")
        ax_u.set_ylabel("u(t)")
        ax_u.set_xlim(x_min, x_max)
        ax_u.set_ylim(control_min, control_max)
        ax_u.grid(True, alpha=0.3)

        # Remaining 13 target variables.
        for target_idx, target_name in enumerate(target_names):
            ax = axes_flat[target_idx + 1]
            ax.plot(t_series, y_true[:, target_idx], color=truth_color, linewidth=1.0, label="Ground truth")

            start = max(0, frame - lookback)
            if frame > 0:
                window_t = t_series[start:frame]
                window_y = ar_history[start:frame, target_idx]
                ax.plot(window_t, window_y, color=input_color, linewidth=2.2, label="Input window (AR state)")

            y_tick_center = float(y_pred[frame, target_idx])
            y_tick_span = 0.02 * max(1e-6, state_lims[target_idx][1] - state_lims[target_idx][0])
            ax.vlines(
                x=float(t_series[frame]),
                ymin=y_tick_center - y_tick_span,
                ymax=y_tick_center + y_tick_span,
                color=output_color,
                linewidth=1.1,
                zorder=5,
            )
            if frame > 0:
                ax.plot(
                    t_series[: frame + 1],
                    y_pred[: frame + 1, target_idx],
                    color=output_color,
                    linewidth=1.0,
                    alpha=0.6,
                )

            ax.set_title(target_name)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(*state_lims[target_idx])
            ax.grid(True, alpha=0.3)

        for idx in range(7, 14):
            axes_flat[idx].set_xlabel("Time step")

        fig.suptitle(
            f"Autoregressive Rolling Forecast March - {profile_name} | step={frame + 1}/{len(t_series)}",
            y=0.985,
            fontsize=15,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.88])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() != ".gif":
        raise ValueError("Video output currently supports only .gif files.")

    frame_indices = list(range(0, len(t_series), frame_stride))
    if frame_indices[-1] != len(t_series) - 1:
        frame_indices.append(len(t_series) - 1)

    animation = FuncAnimation(fig, _update, frames=frame_indices, interval=int(1000 / max(fps, 1)))

    total_frames = len(frame_indices)

    def _progress(current_frame: int, _total: int) -> None:
        # Pillow writer can appear silent for long renders; emit periodic progress.
        if current_frame == 0 or (current_frame + 1) % 25 == 0 or (current_frame + 1) == total_frames:
            print(f"Rendering frame {current_frame + 1}/{total_frames}...")

    animation.save(
        save_path,
        writer="pillow",
        fps=fps,
        dpi=max(video_dpi, 1),
        progress_callback=_progress,
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load rolling_forecasts.h5 and create a 2x7 GIF showing autoregressive "
            "forecasting with per-frame input window and output prediction coloring."
        )
    )
    parser.add_argument("forecast_h5_path", type=Path, help="Path to rolling_forecasts.h5.")
    parser.add_argument("--profile", type=str, default=None, help="Profile name. Defaults to first profile in file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../misc/autoregressive_forecast_march.gif"),
        help="GIF output path.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=12,
        help="Number of previous steps to color as model input window in each frame.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument("--video_dpi", type=int, default=100, help="GIF DPI.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth step as a frame (1 renders all steps).",
    )
    args = parser.parse_args()

    if args.lookback <= 0:
        raise ValueError("--lookback must be a positive integer.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be a positive integer.")

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

    target_names = _extract_target_names(columns)
    if not target_names:
        raise KeyError("No target columns found in forecast table.")
    t_series = table[:, columns.index("t")]
    u_series = table[:, columns.index("u(t)")]
    y_true, y_pred = _resolve_target_series(table=table, columns=columns, target_names=target_names)

    save_autoregressive_forecast_video(
        t_series=t_series,
        u_series=u_series,
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        profile_name=profile_name,
        save_path=args.output,
        lookback=args.lookback,
        fps=args.fps,
        video_dpi=args.video_dpi,
        frame_stride=args.frame_stride,
    )

    print(f"Saved autoregressive forecast video: {args.output}")


if __name__ == "__main__":
    main()
