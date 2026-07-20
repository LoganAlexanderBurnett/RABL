"""Focused tests for joint ensemble-normalized trajectory conformal utilities."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("torch")

from rabl.machine_learning import conformal_prediction as cp


def _profile(name: str, truth: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    # The patched scaled forecast uses x[..., 0, 0] as a profile selector.
    x = np.zeros((truth.shape[0], 1, 1), dtype=np.float32)
    x[..., 0, 0] = 0.0 if name == "a" else 1.0
    return name, x, truth.astype(np.float32)


def test_joint_scores_quantile_and_spread_adaptive_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    stacks = {
        0: np.array([[[0.0, 0.0], [0.0, 0.0]], [[2.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        1: np.array([[[0.0, 0.0], [0.0, 0.0]], [[4.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
    }
    def fake_forecast(_models, x, *, state_dim, ddof):
        stack = stacks[int(x[0, 0, 0])]
        return stack, stack.mean(axis=0), stack.std(axis=0, ddof=ddof)
    monkeypatch.setattr(cp, "_ensemble_scaled_forecast", fake_forecast)
    cal = [_profile("a", np.array([[2.0, 0.0], [0.0, 0.0]])), _profile("b", np.array([[4.0, 0.0], [0.0, 0.0]]))]
    result = cp.calibrate_joint_ensemble_normalized_conformal([object(), object()], cal, alpha=0.5, sigma_floor=1.0)
    # Scores are max(|2-1|/(1+1), 0) = .5 and |4-2|/(2+1) = 2/3;
    # conservative k=ceil(3*.5)=2 selects 2/3.
    assert result["calibration_scores"].shape == (2,)
    assert result["q_joint"] == pytest.approx(2.0 / 3.0)
    assert np.isscalar(result["q_joint"])
    scaling = {"scaling_type": "none"}
    # none scaling requires stats format in helper; make descaling an identity for this focused test.
    monkeypatch.setattr(cp, "_descale_targets_from_stats", lambda _stats, value: value)
    monkeypatch.setattr(cp, "_descale_feature_from_stats", lambda _stats, value, _idx: value)
    monkeypatch.setattr(cp, "_extract_control_series", lambda x, **_kwargs: x[:, 0, 0])
    forecast = cp.joint_ensemble_conformal_forecast_profile([object(), object()], "b", cal[1][1], cal[1][2], conformal_result=result, scaling_stats=scaling)
    width = forecast["upper"] - forecast["lower"]
    assert width.shape == (2, 2)
    assert width[0, 0] > width[1, 0]  # increased spread only widens the affected time/target.
    assert np.all(np.isfinite(width))


def test_joint_coverage_is_complete_trajectory_event() -> None:
    forecasts = [
        {"y_true": np.array([[0.0], [0.0]]), "lower": np.array([[-1.0], [-1.0]]), "upper": np.array([[1.0], [1.0]]), "spread_scaled": np.ones((2, 1)), "mean_scaled": np.zeros((2, 1)), "scaled": {"y_true": np.zeros((2, 1))}},
        {"y_true": np.array([[0.0], [2.0]]), "lower": np.array([[-1.0], [-1.0]]), "upper": np.array([[1.0], [1.0]]), "spread_scaled": np.ones((2, 1)), "mean_scaled": np.zeros((2, 1)), "scaled": {"y_true": np.array([[0.0], [2.0]])}},
    ]
    metrics = cp.joint_ensemble_coverage_metrics(forecasts, ["target"])
    assert metrics["primary_joint_trajectory_coverage"] == pytest.approx(0.5)
    assert metrics["targetwise_trajectory_coverage"]["target"] == pytest.approx(0.5)
    assert metrics["marginal_pointwise_coverage_overall"] == pytest.approx(0.75)
