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



def test_checkpoint_loader_reports_incompatible_architecture(tmp_path):
    torch = pytest.importorskip("torch")
    from rabl.machine_learning.bagging_ensemble import load_bagged_lstm_ensemble_checkpoints

    bad_a = tmp_path / "bad_a.pt"
    bad_b = tmp_path / "bad_b.pt"
    torch.save({"unexpected.weight": torch.zeros(1)}, bad_a)
    torch.save({"unexpected.weight": torch.zeros(1)}, bad_b)
    with pytest.raises(ValueError, match="incompatible"):
        load_bagged_lstm_ensemble_checkpoints(
            [bad_a, bad_b], timesteps=3, num_features=2, num_targets=1, device="cpu"
        )


def _shared_forecast(name, truth, mean, spread):
    return {
        "profile": name,
        "t": np.arange(truth.shape[0], dtype=np.float32),
        "u": np.zeros(truth.shape[0], dtype=np.float32),
        "y_true_scaled": truth.astype(np.float32),
        "mean_scaled": mean.astype(np.float32),
        "spread_scaled": spread.astype(np.float32),
        "member_predictions_scaled": np.stack([mean - spread, mean + spread], axis=0).astype(np.float32),
        "y_true": truth.astype(np.float32),
        "y_pred": mean.astype(np.float32),
    }


def test_five_uq_method_quantile_shapes_and_centers(tmp_path):
    scaling = {"type": "standard", "y": {"mean": np.zeros(2, dtype=np.float32), "std": np.ones(2, dtype=np.float32)}}
    cal = [
        _shared_forecast("c0", np.array([[1.0, 0.0], [0.0, 2.0]]), np.zeros((2, 2)), np.ones((2, 2))),
        _shared_forecast("c1", np.array([[2.0, 0.0], [0.0, 4.0]]), np.zeros((2, 2)), np.ones((2, 2))),
        _shared_forecast("c2", np.array([[3.0, 0.0], [0.0, 6.0]]), np.zeros((2, 2)), np.ones((2, 2))),
    ]
    test = _shared_forecast("t0", np.zeros((2, 2)), np.full((2, 2), 0.5), np.ones((2, 2)))
    ens_traj = cp.calibrate_ensemble_normalized_conformal(cal, alpha=0.25, sigma_floor=1.0, temporal_mode="trajectory")
    ens_horizon = cp.calibrate_ensemble_normalized_conformal(cal, alpha=0.25, sigma_floor=np.array([1.0, 2.0]), temporal_mode="per_horizon")
    abs_horizon = cp.calibrate_absolute_conformal(cal, alpha=0.25, temporal_mode="per_horizon")
    abs_traj = cp.calibrate_absolute_conformal(cal, alpha=0.25, temporal_mode="trajectory")
    assert ens_traj["q_by_target"].shape == (2,)
    assert ens_horizon["q_by_horizon_target"].shape == (2, 2)
    assert abs_horizon["q_by_horizon_target"].shape == (2, 2)
    assert abs_traj["q_by_target"].shape == (2,)
    assert ens_traj["q_by_target"] == pytest.approx([1.5, 3.0])
    assert abs_traj["q_by_target"] == pytest.approx([3.0, 6.0])
    methods = {
        "ensemble_conformal_target_trajectory": ens_traj,
        "ensemble_conformal_target_horizon": ens_horizon,
        "absolute_conformal_target_horizon": abs_horizon,
        "absolute_conformal_target_trajectory": abs_traj,
        "raw_ensemble_2sigma": None,
    }
    entries = []
    for method_id, calibration in methods.items():
        entry = cp.apply_uq_method(test, method_id=method_id, calibration_result=calibration, scaling_stats=scaling, alpha=0.25, sigma_floor=1.0)
        entries.append(entry)
        assert np.allclose(entry["y_pred_scaled"], test["mean_scaled"])
    centers = [entry["y_pred_scaled"] for entry in entries]
    assert all(np.array_equal(centers[0], center) for center in centers[1:])
    raw = entries[-1]
    assert np.allclose(raw["lower_scaled"], test["mean_scaled"] - 2.0 * test["spread_scaled"])
    assert np.allclose(raw["upper_scaled"], test["mean_scaled"] + 2.0 * test["spread_scaled"])


def test_per_horizon_calibration_uses_available_profiles_only():
    cal = [
        _shared_forecast("c0", np.array([[1.0], [10.0]]), np.zeros((2, 1)), np.ones((2, 1))),
        _shared_forecast("c1", np.array([[2.0]]), np.zeros((1, 1)), np.ones((1, 1))),
    ]
    result = cp.calibrate_absolute_conformal(cal, alpha=0.5, temporal_mode="per_horizon")
    assert result["q_by_horizon_target"].shape == (2, 1)
    assert result["calibration_count_by_horizon"].tolist() == [2, 1]
    assert result["q_by_horizon_target"][:, 0].tolist() == pytest.approx([2.0, 10.0])


def test_standardized_uq_hdf5_contains_required_schema(tmp_path):
    h5py = pytest.importorskip("h5py")
    entry = {
        "profile": "p0", "t": np.arange(2), "u": np.zeros(2),
        "y_true": np.zeros((2, 1)), "y_pred": np.zeros((2, 1)),
        "lower": -np.ones((2, 1)), "upper": np.ones((2, 1)), "interval_width": 2*np.ones((2, 1)),
        "y_true_scaled": np.zeros((2, 1)), "y_pred_scaled": np.zeros((2, 1)),
        "lower_scaled": -np.ones((2, 1)), "upper_scaled": np.ones((2, 1)), "spread_scaled": np.ones((2, 1)),
    }
    path = tmp_path / "uq.h5"
    cp.save_uq_forecasts_hdf5([entry], output_path=path, metadata={"method_id": "raw_ensemble_2sigma", "alpha": 0.05}, target_names=["target"])
    with h5py.File(path, "r") as h5f:
        assert h5f.attrs["method_id"] == "raw_ensemble_2sigma"
        for key in ("t", "u", "y_true", "y_pred", "lower", "upper", "interval_width", "y_true_scaled", "y_pred_scaled", "lower_scaled", "upper_scaled", "spread_scaled"):
            assert key in h5f["p0"]


def test_saved_audit_forecast_config_rejects_deprecated_fields(tmp_path):
    import scripts.run_lstm_conformal_prediction as runner

    cfg_path = tmp_path / "bad_config.json"
    cfg_path.write_text(
        '{"scaled_h5_path":"x.h5","calibration_ensemble_forecasts_audit_h5_path":"cal.h5",'
        '"test_ensemble_forecasts_audit_h5_path":"test.h5","uq_output_dir":"out",'
        '"train_manifest_path":"train.json","val_manifest_path":"val.json",'
        '"cal_manifest_path":"cal.json","test_manifest_path":"test.json",'
        '"ensemble_source":"checkpoints"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported dataset/training/checkpoint keys"):
        runner._load_cfg(cfg_path)


def test_load_saved_audit_ensemble_forecasts_schema(tmp_path):
    import h5py
    import scripts.run_lstm_conformal_prediction as runner

    scaled = tmp_path / "scaled.h5"
    with h5py.File(scaled, "w") as h5f:
        scaling = h5f.create_group("scaling")
        scaling.create_dataset("x_mean", data=np.zeros(16, dtype=np.float32))
        scaling.create_dataset("x_std", data=np.ones(16, dtype=np.float32))
        scaling.create_dataset("y_mean", data=np.array([10.0], dtype=np.float32))
        scaling.create_dataset("y_std", data=np.array([2.0], dtype=np.float32))

    forecast_h5 = tmp_path / "forecast_audit.h5"
    with h5py.File(forecast_h5, "w") as h5f:
        h5f.attrs["target_names"] = np.asarray(["target"], dtype="S")
        h5f.attrs["member_count"] = 2
        h5f.attrs["ensemble_std_ddof"] = 0
        grp = h5f.create_group("p0")
        grp.create_dataset("t", data=np.arange(2, dtype=np.float32))
        grp.create_dataset("control", data=np.zeros(2, dtype=np.float32))
        grp.create_dataset("y_true_scaled", data=np.array([[0.0], [1.0]], dtype=np.float32))
        grp.create_dataset("member_predictions_scaled", data=np.zeros((2, 2, 1), dtype=np.float32))
        grp.create_dataset("ensemble_mean_scaled", data=np.array([[0.5], [1.5]], dtype=np.float32))
        grp.create_dataset("ensemble_std_scaled", data=np.ones((2, 1), dtype=np.float32))

    forecasts = runner._load_audit_ensemble_forecasts(
        forecast_h5,
        scaling_stats=runner._load_scaling_stats(scaled),
        expected_profiles=["p0"],
        target_names=["target"],
        expected_ddof=0,
    )
    assert len(forecasts) == 1
    assert forecasts[0]["profile"] == "p0"
    assert forecasts[0]["mean_scaled"].shape == (2, 1)
    assert forecasts[0]["spread_scaled"].shape == (2, 1)
    assert forecasts[0]["member_predictions_scaled"].shape == (2, 2, 1)
    assert forecasts[0]["y_true"].tolist() == pytest.approx([[10.0], [12.0]])
    assert forecasts[0]["y_pred"].tolist() == pytest.approx([[11.0], [13.0]])


def test_uq_metrics_separate_pointwise_and_trajectory_worst_coverage() -> None:
    forecasts = []
    for idx, miss_last in enumerate([False, True]):
        y_true = np.zeros((2, 2), dtype=np.float32)
        lower = -np.ones((2, 2), dtype=np.float32)
        upper = np.ones((2, 2), dtype=np.float32)
        if miss_last:
            y_true[1, 1] = 3.0
        forecasts.append({
            "profile": f"p{idx}", "t": np.arange(2), "u": np.zeros(2),
            "y_true": y_true, "y_pred": np.zeros((2, 2), dtype=np.float32),
            "lower": lower, "upper": upper, "interval_width": upper - lower,
            "y_true_scaled": y_true, "y_pred_scaled": np.zeros((2, 2), dtype=np.float32),
            "lower_scaled": lower, "upper_scaled": upper, "spread_scaled": np.ones((2, 2), dtype=np.float32),
        })
    metrics = cp.compute_uq_coverage_metrics(
        forecasts,
        ["a", "b"],
        alpha=0.05,
        primary_coverage_type="targetwise_trajectory_coverage",
    )
    assert "worst_target_pointwise_coverage" in metrics
    assert "worst_target_trajectory_coverage" in metrics
    assert metrics["worst_target_pointwise_coverage"] == pytest.approx(0.75)
    assert metrics["worst_target_trajectory_coverage"] == pytest.approx(0.5)
    assert metrics["primary_worst_target_coverage"] == pytest.approx(0.5)
    assert "mean_scaled_interval_width_overall" in metrics
    assert "mean_interval_width_overall" not in metrics
