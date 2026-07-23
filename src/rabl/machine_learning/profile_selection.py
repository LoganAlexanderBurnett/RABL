"""Reusable profile-selection helpers for forecast and UQ plots."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProfileSelectionResult:
    profiles: list[str]
    rows: list[dict[str, Any]]
    bin_edges: list[float]
    metric: str
    n_bins: int
    profiles_per_bin: int
    seed: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "profiles": self.profiles,
            "selection_rows": self.rows,
            "bin_edges": self.bin_edges,
            "metric": self.metric,
            "n_bins": self.n_bins,
            "profiles_per_bin": self.profiles_per_bin,
            "seed": self.seed,
        }


def select_profiles_by_quantile_bins(
    difficulty_csv: Path,
    *,
    metric: str,
    n_bins: int,
    per_bin: int,
    seed: int,
    profile_column: str = "profile_id",
) -> ProfileSelectionResult:
    """Select profiles with the same quantile-bin logic used for ensemble plots."""
    difficulty_csv = Path(difficulty_csv)
    rows: list[dict[str, str]] = []
    with difficulty_csv.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rows.append(row)
    if not rows:
        return ProfileSelectionResult([], [], [], metric, int(n_bins), int(per_bin), int(seed))
    if metric not in rows[0]:
        raise ValueError(f"Metric {metric!r} is not a column in {difficulty_csv}.")
    if profile_column not in rows[0]:
        raise ValueError(f"Profile column {profile_column!r} is not a column in {difficulty_csv}.")
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    edges = np.quantile(values, np.linspace(0.0, 1.0, int(n_bins) + 1))
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    for idx in range(int(n_bins)):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (values >= lo) & (values <= hi if idx == int(n_bins) - 1 else values < hi)
        candidates = [rows[i] for i in np.flatnonzero(mask)]
        if not candidates:
            continue
        take = min(int(per_bin), len(candidates))
        chosen = rng.choice(len(candidates), size=take, replace=False)
        for chosen_idx in chosen:
            row = candidates[int(chosen_idx)]
            profile = str(row[profile_column])
            selected.append(profile)
            manifest_rows.append(
                {
                    "bin_index": int(idx),
                    "profile_id": profile,
                    metric: float(row[metric]),
                    "bin_lower": float(lo),
                    "bin_upper": float(hi),
                }
            )
    return ProfileSelectionResult(
        selected,
        manifest_rows,
        [float(edge) for edge in edges],
        metric,
        int(n_bins),
        int(per_bin),
        int(seed),
    )
