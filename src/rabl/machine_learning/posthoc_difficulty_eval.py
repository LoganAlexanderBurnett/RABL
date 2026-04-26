from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


BinMode = Literal["quantile", "fixed"]


@dataclass(frozen=True)
class BinnedSeries:
    labels: np.ndarray
    edges: np.ndarray
    label_names: list[str]


def compute_equilibrium_excursions(
    *,
    drum_angle_deg: np.ndarray,
    rho_dollars: np.ndarray,
    drum_equilibrium_deg: float,
    rho_equilibrium_dollars: float,
    dt: float,
) -> dict[str, float]:
    """Compute per-profile equilibrium-excursion descriptors.

    Returns
    -------
    dict
        Keys: ``E_theta_max``, ``E_rho_max``, ``V_theta_max``.
    """
    theta = np.asarray(drum_angle_deg, dtype=float)
    rho = np.asarray(rho_dollars, dtype=float)
    if theta.ndim != 1 or rho.ndim != 1:
        raise ValueError("drum_angle_deg and rho_dollars must be 1D arrays.")
    if theta.shape[0] != rho.shape[0]:
        raise ValueError("drum_angle_deg and rho_dollars must have same length.")
    if theta.size < 2:
        raise ValueError("Need at least 2 timesteps to compute V_theta_max.")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")

    e_theta = np.abs(theta - float(drum_equilibrium_deg))
    e_rho = np.abs(rho - float(rho_equilibrium_dollars))
    v_theta = np.gradient(theta, float(dt))

    return {
        "E_theta_max": float(np.max(e_theta)),
        "E_rho_max": float(np.max(e_rho)),
        "V_theta_max": float(np.max(np.abs(v_theta))),
    }


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.asarray(edges, dtype=float)
    # Ensure strict monotonicity for digitize by nudging ties.
    for idx in range(1, edges.size):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = np.nextafter(edges[idx - 1], np.inf)
    return edges


def bin_series(
    values: np.ndarray,
    mode: BinMode = "quantile",
    n_bins: int = 5,
    edges: np.ndarray | None = None,
) -> BinnedSeries:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if values.size == 0:
        raise ValueError("values cannot be empty.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    if mode == "quantile":
        resolved_edges = _quantile_edges(values, n_bins)
    elif mode == "fixed":
        if edges is None:
            raise ValueError("edges must be provided when mode='fixed'.")
        resolved_edges = np.asarray(edges, dtype=float)
        if resolved_edges.ndim != 1 or resolved_edges.size < 2:
            raise ValueError("edges must be a 1D array with at least 2 values.")
        if not np.all(np.diff(resolved_edges) > 0):
            raise ValueError("edges must be strictly increasing.")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    bin_ids = np.digitize(values, bins=resolved_edges[1:-1], right=False)
    label_names = [f"[{resolved_edges[i]:.6g}, {resolved_edges[i+1]:.6g})" for i in range(len(resolved_edges) - 1)]
    if label_names:
        label_names[-1] = f"[{resolved_edges[-2]:.6g}, {resolved_edges[-1]:.6g}]"

    labels = np.asarray([label_names[idx] for idx in bin_ids], dtype=object)
    return BinnedSeries(labels=labels, edges=resolved_edges, label_names=label_names)


def aggregate_metric_by_bin(
    records: list[dict[str, object]],
    metric_col: str,
    bin_col: str,
) -> list[dict[str, float | str | int]]:
    grouped: dict[str, list[float]] = {}
    for row in records:
        label = str(row[bin_col])
        grouped.setdefault(label, []).append(float(row[metric_col]))

    output: list[dict[str, float | str | int]] = []
    for label, vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        output.append(
            {
                "bin": label,
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr, ddof=0)),
            }
        )
    return output
