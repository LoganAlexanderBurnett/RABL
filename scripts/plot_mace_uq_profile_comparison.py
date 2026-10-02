"""Plot simultaneous MACE/UQ interval comparisons for selected profiles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from rabl.machine_learning.profile_selection import select_profiles_by_quantile_bins

CANONICAL_METHOD_ORDER = [
    "absolute_conformal_target_trajectory",
    "absolute_conformal_target_horizon",
    "ensemble_conformal_target_trajectory",
    "ensemble_conformal_target_horizon",
    "raw_ensemble_2sigma",
]
METHOD_LABELS = {
    "absolute_conformal_target_trajectory": "Full-trajectory CP",
    "absolute_conformal_target_horizon": "Horizon CP",
    "ensemble_conformal_target_trajectory": "MACE-Trajectory",
    "ensemble_conformal_target_horizon": "MACE-Horizon",
    "raw_ensemble_2sigma": r"Raw ensemble $\\pm 2\\sigma$",
}
METHOD_COLORS = {
    "absolute_conformal_target_trajectory": "#d62728",
    "absolute_conformal_target_horizon": "#2ca02c",
    "ensemble_conformal_target_trajectory": "#1f77b4",
    "ensemble_conformal_target_horizon": "#ff7f0e",
    "raw_ensemble_2sigma": "#9467bd",
}
METHOD_LINESTYLES = {method_id: "-" for method_id in CANONICAL_METHOD_ORDER}
TARGET_ORDER = [
    "Tf", "Tm", "Thp", "TN2", "Tsg", "T_steam_out", "x_steam_out",
    "c[1]", "c[2]", "c[3]", "c[4]", "c[5]", "c[6]", "n", "rho_dollars",
]
FORECAST_DT_SECONDS = 0.4
TARGET_AXIS_LABELS = {
    "Tf": r"$T_f$ [K]",
    "Tm": r"$T_m$ [K]",
    "Thp": r"$T_{hp}$ [K]",
    "TN2": r"$T_{N_2}$ [K]",
    "Tsg": r"$T_{sg}$ [K]",
    "T_steam_out": r"$T_{\\mathrm{steam,out}}$ [K]",
    "x_steam_out": r"$x_{\\mathrm{steam,out}}$ [-]",
    "n": r"$n$ [-]",
    "rho_dollars": r"$\\rho_{\\$}$ [dollars]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simultaneous MACE/UQ profile comparisons.")
    parser.add_argument("--methods-manifest-json", type=Path, required=True)
    parser.add_argument("--selected-profiles-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--uq-methods", nargs="+", default=None, help="Subset of UQ method IDs to plot.")
    parser.add_argument("--difficulty-csv", type=Path, default=None)
    parser.add_argument("--selection-metric", default="scaled_mae")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--profiles-per-bin", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=123)
    return parser.parse_args()


def _decode_strings(value: Any) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in np.asarray(value).tolist()]


def ordered_methods(methods: list[str] | None) -> list[str]:
    requested = CANONICAL_METHOD_ORDER if methods is None else [str(m) for m in methods]
    unknown = sorted(set(requested) - set(CANONICAL_METHOD_ORDER))
    if unknown:
        raise ValueError(f"Unknown UQ method ID(s): {unknown}. Supported methods are {CANONICAL_METHOD_ORDER}.")
    requested_set = set(requested)
    return [method_id for method_id in CANONICAL_METHOD_ORDER if method_id in requested_set]


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def _load_selected_profiles(path: Path) -> tuple[list[str], dict[str, Any]]:
    data = _load_manifest(path)
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"{path} must contain a non-empty 'profiles' list.")
    return [str(profile) for profile in profiles], {"source": str(path), **data}


def _select_profiles(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if args.selected_profiles_json is not None:
        return _load_selected_profiles(args.selected_profiles_json)
    if args.difficulty_csv is None:
        raise ValueError("Provide --selected-profiles-json or fallback --difficulty-csv.")
    result = select_profiles_by_quantile_bins(
        args.difficulty_csv,
        metric=args.selection_metric,
        n_bins=args.n_bins,
        per_bin=args.profiles_per_bin,
        seed=args.selection_seed,
    )
    return result.profiles, {"source": str(args.difficulty_csv), **result.to_json_dict()}


def _load_method_h5(path: Path, selected_profiles: list[str]) -> dict[str, Any]:
    with h5py.File(path, "r") as h5f:
        target_names = _decode_strings(h5f.attrs["target_names"])
        profiles: dict[str, dict[str, np.ndarray]] = {}
        for profile in selected_profiles:
            if profile not in h5f:
                raise ValueError(f"Selected profile {profile!r} is missing from {path}.")
            group = h5f[profile]
            profiles[profile] = {key: group[key][...] for key in group.keys()}
    return {"target_names": target_names, "profiles": profiles}


def _validate_controlled_fields(method_data: dict[str, dict[str, Any]], selected_profiles: list[str]) -> None:
    first_id = next(iter(method_data))
    first = method_data[first_id]
    target_names = first["target_names"]
    for method_id, data in method_data.items():
        if data["target_names"] != target_names:
            raise ValueError(f"Target order mismatch for {method_id}.")
        for profile in selected_profiles:
            base = first["profiles"][profile]
            cur = data["profiles"][profile]
            for key in ("t", "u", "y_true", "y_pred", "y_true_scaled", "y_pred_scaled"):
                if not np.allclose(base[key], cur[key], rtol=0.0, atol=1e-7):
                    raise ValueError(f"Controlled field {key!r} differs for method {method_id}, profile {profile}.")


def _ordered_targets(target_names: list[str]) -> list[str]:
    return [name for name in TARGET_ORDER if name in target_names] + [name for name in target_names if name not in TARGET_ORDER]


def _target_axis_label(name: str) -> str:
    if name.startswith("c[") and name.endswith("]"):
        return rf"$c_{{{name[2:-1]}}}$ [-]"
    return TARGET_AXIS_LABELS.get(name, name.replace("_", r"\_"))


def _forecast_time_seconds(t: np.ndarray) -> np.ndarray:
    """Return elapsed forecast time in seconds without altering forecast values."""
    values = np.asarray(t, dtype=float)
    if values.size == 0:
        return values
    # Current ensemble audit files store t as forecast-step indices. Convert those
    # indices using the 0.4 s sampling interval used to generate the paper data.
    if values.size == 1 or np.allclose(np.diff(values), 1.0, rtol=0.0, atol=1e-7):
        return (values - values[0]) * FORECAST_DT_SECONDS
    # Preserve already-physical time arrays, but report elapsed forecast time.
    return values - values[0]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "profile"


def _plot_profile(profile: str, method_data: dict[str, dict[str, Any]], methods: list[str], out_path: Path) -> None:
    publication_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
    with plt.rc_context(publication_rc):
        first = method_data[methods[0]]["profiles"][profile]
        target_names = method_data[methods[0]]["target_names"]
        order = _ordered_targets(target_names)
        name_to_idx = {name: idx for idx, name in enumerate(target_names)}
        t_seconds = _forecast_time_seconds(first["t"])

        fig, axes = plt.subplots(4, 4, figsize=(10.0, 6.6), sharex=True)
        axes = np.asarray(axes).ravel()

        axes[0].plot(t_seconds, first["u"], color="black", linewidth=1.4)
        axes[0].set_ylabel(r"$\\theta$ [deg]", labelpad=3)
        axes[0].grid(True, alpha=0.2)

        truth_handle = None
        mean_handle = None
        method_handles: dict[str, Any] = {}

        for plot_idx, target in enumerate(order, start=1):
            ax = axes[plot_idx]
            j = name_to_idx[target]
            truth_line, = ax.plot(
                t_seconds,
                first["y_true"][:, j],
                color="black",
                linewidth=1.4,
                label="Truth",
                zorder=30,
            )
            mean_line, = ax.plot(
                t_seconds,
                first["y_pred"][:, j],
                color="#005AB5",
                linewidth=1.4,
                label="Ensemble mean",
                zorder=29,
            )
            if truth_handle is None:
                truth_handle = truth_line
                mean_handle = mean_line

            for zidx, method_id in enumerate(methods):
                entry = method_data[method_id]["profiles"][profile]
                color = METHOD_COLORS[method_id]
                label = METHOD_LABELS[method_id]
                fill = ax.fill_between(
                    t_seconds,
                    entry["lower"][:, j],
                    entry["upper"][:, j],
                    color=color,
                    alpha=0.10 + 0.015 * zidx,
                    linewidth=0.0,
                    label=label,
                    zorder=5 + zidx,
                )
                ax.plot(
                    t_seconds,
                    entry["lower"][:, j],
                    color=color,
                    linestyle=METHOD_LINESTYLES[method_id],
                    linewidth=0.8,
                    zorder=15 + zidx,
                )
                ax.plot(
                    t_seconds,
                    entry["upper"][:, j],
                    color=color,
                    linestyle=METHOD_LINESTYLES[method_id],
                    linewidth=0.8,
                    zorder=15 + zidx,
                )
                method_handles.setdefault(method_id, fill)

            ax.set_ylabel(_target_axis_label(target), labelpad=3)
            ax.grid(True, alpha=0.2)

        for ax in axes[len(order) + 1:]:
            ax.axis("off")

        # Shared x axis: show tick labels only on the bottom row and use one
        # figure-level axis label to avoid repeated text in the 4x4 layout.
        for ax in axes[:-4]:
            ax.tick_params(axis="x", labelbottom=False)
        for ax in axes[-4:]:
            ax.tick_params(axis="x", labelbottom=True)
        fig.supxlabel("Forecast time [s]", y=0.015, fontsize=9)

        legend_method_order = [
            "raw_ensemble_2sigma",
            "absolute_conformal_target_horizon",
            "absolute_conformal_target_trajectory",
            "ensemble_conformal_target_horizon",
            "ensemble_conformal_target_trajectory",
        ]
        legend_handles = [truth_handle, mean_handle]
        legend_handles.extend(
            method_handles[method_id]
            for method_id in legend_method_order
            if method_id in method_handles
        )
        legend_handles = [handle for handle in legend_handles if handle is not None]
        fig.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(len(legend_handles), 5),
            frameon=False,
            columnspacing=1.3,
            handlelength=2.2,
        )

        fig.tight_layout(rect=[0.01, 0.04, 0.995, 0.93], h_pad=0.5, w_pad=0.7)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def main() -> None:
    args = parse_args()
    methods = ordered_methods(args.uq_methods)
    profiles, selection_metadata = _select_profiles(args)
    manifest = _load_manifest(args.methods_manifest_json)
    method_entries = {entry["method_id"]: entry for entry in manifest.get("uq_methods", [])}
    missing = [method_id for method_id in methods if method_id not in method_entries]
    if missing:
        raise ValueError(f"Methods missing from manifest: {missing}")
    method_data = {
        method_id: _load_method_h5(Path(method_entries[method_id]["test_forecasts_path"]), profiles)
        for method_id in methods
    }
    _validate_controlled_fields(method_data, profiles)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selected_profiles_for_mace_uq_plots.json").write_text(
        json.dumps({"profiles": profiles, "methods": methods, "selection": selection_metadata}, indent=2),
        encoding="utf-8",
    )
    for profile in profiles:
        _plot_profile(profile, method_data, methods, args.out_dir / f"mace_uq_profile_comparison_{_safe_name(profile)}.png")
    print(f"Saved {len(profiles)} MACE/UQ comparison plot(s) to: {args.out_dir}")


if __name__ == "__main__":
    main()
