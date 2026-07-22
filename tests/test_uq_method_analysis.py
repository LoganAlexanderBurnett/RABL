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
