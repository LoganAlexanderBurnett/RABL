"""Focused tests for multi-method UQ comparison analysis helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

_SPEC = importlib.util.spec_from_file_location(
    "analyze_lstm_conformal_uncertainty",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_lstm_conformal_uncertainty.py",
)
assert _SPEC is not None and _SPEC.loader is not None
analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyzer)


def _method_data(center_offset: float = 0.0) -> dict:
    profile = {
        "p0": {
            "t": np.arange(2, dtype=np.float32),
            "u": np.zeros(2, dtype=np.float32),
            "y_true": np.zeros((2, 1), dtype=np.float32),
            "y_true_scaled": np.zeros((2, 1), dtype=np.float32),
            "y_pred": np.full((2, 1), center_offset, dtype=np.float32),
            "y_pred_scaled": np.full((2, 1), center_offset, dtype=np.float32),
            "lower": -np.ones((2, 1), dtype=np.float32),
            "upper": np.ones((2, 1), dtype=np.float32),
            "interval_width": 2 * np.ones((2, 1), dtype=np.float32),
            "spread_scaled": np.ones((2, 1), dtype=np.float32),
        }
    }
    return {"target_names": ["target"], "alpha": 0.05, "nominal_coverage": 0.95, "profiles": profile}


def test_multi_method_analyzer_rejects_center_mismatch() -> None:
    with pytest.raises(ValueError, match="y_pred"):
        analyzer._validate_method_comparison({"a": _method_data(0.0), "b": _method_data(0.1)})


def test_mace_plotter_method_order_and_unknown_rejection() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "plot_mace_uq_profile_comparison",
        Path(__file__).resolve().parents[1] / "scripts" / "plot_mace_uq_profile_comparison.py",
    )
    assert spec is not None and spec.loader is not None
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    assert plotter.ordered_methods([
        "raw_ensemble_2sigma",
        "ensemble_conformal_target_horizon",
        "absolute_conformal_target_trajectory",
    ]) == [
        "absolute_conformal_target_trajectory",
        "ensemble_conformal_target_horizon",
        "raw_ensemble_2sigma",
    ]
    with pytest.raises(ValueError, match="Unknown UQ method"):
        plotter.ordered_methods(["not_a_method"])


def test_profile_selection_shared_quantile_logic(tmp_path) -> None:
    from rabl.machine_learning.profile_selection import select_profiles_by_quantile_bins

    csv_path = tmp_path / "difficulty.csv"
    csv_path.write_text(
        "profile_id,scaled_mae\n"
        "p0,0.0\n"
        "p1,1.0\n"
        "p2,2.0\n"
        "p3,3.0\n",
        encoding="utf-8",
    )
    first = select_profiles_by_quantile_bins(csv_path, metric="scaled_mae", n_bins=2, per_bin=1, seed=7)
    second = select_profiles_by_quantile_bins(csv_path, metric="scaled_mae", n_bins=2, per_bin=1, seed=7)
    assert first.profiles == second.profiles
    assert len(first.rows) == 2
    assert first.bin_edges == pytest.approx([0.0, 1.5, 3.0])


def test_analyzer_primary_worst_coverage_modes() -> None:
    assert analyzer._primary_worst(
        "ensemble_conformal_target_trajectory",
        np.array([0.99, 0.98]),
        np.array([0.90, 0.80]),
        np.array([[0.99, 0.98]]),
    ) == pytest.approx(0.80)
    assert analyzer._primary_worst(
        "ensemble_conformal_target_horizon",
        np.array([0.99, 0.98]),
        np.array([0.90, 0.80]),
        np.array([[0.70, 0.98]]),
    ) == pytest.approx(0.70)


def test_analyzer_method_arrays_include_scaled_fields() -> None:
    data = _method_data(0.0)
    profiles = data["profiles"]
    profiles["p0"]["lower_scaled"] = -2 * np.ones((2, 1), dtype=np.float32)
    profiles["p0"]["upper_scaled"] = 2 * np.ones((2, 1), dtype=np.float32)
    arrays = analyzer._method_arrays(data)
    assert np.all(arrays["upper_scaled"] - arrays["lower_scaled"] == 4.0)
