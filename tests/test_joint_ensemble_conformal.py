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



def _write_minimal_scaled_h5(path, *, scale_offset=0.0, leak_cal=False):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as h5f:
        scaling = h5f.create_group("scaling")
        scaling.create_dataset("x_mean", data=np.array([0.0 + scale_offset, 0.0], dtype=np.float32))
        scaling.create_dataset("x_std", data=np.array([1.0, 1.0], dtype=np.float32))
        scaling.create_dataset("y_mean", data=np.array([0.0], dtype=np.float32))
        scaling.create_dataset("y_std", data=np.array([1.0], dtype=np.float32))
        for split, names in {
            "train": ["train_a"], "val": ["val_a"], "cal": ["cal_a"], "test": ["test_a"],
        }.items():
            files = h5f.create_group(f"{split}/files")
            for name in names:
                grp = files.create_group(name)
                grp.create_dataset("X", data=np.zeros((2, 3, 2), dtype=np.float32))
                grp.create_dataset("Y", data=np.zeros((2, 1), dtype=np.float32))
        for idx in range(2):
            files = h5f.create_group(f"train/bag_{idx}/files")
            name = "cal_a" if leak_cal and idx == 0 else f"bag_train_{idx}"
            grp = files.create_group(name)
            grp.create_dataset("X", data=np.zeros((2, 3, 2), dtype=np.float32))
            grp.create_dataset("Y", data=np.zeros((2, 1), dtype=np.float32))


def test_checkpoint_metadata_validation_success_and_leakage(tmp_path):
    pytest.importorskip("torch")
    import scripts.run_lstm_conformal_prediction as runner

    current = tmp_path / "current.h5"
    bagged = tmp_path / "bagged.h5"
    leaked = tmp_path / "leaked.h5"
    _write_minimal_scaled_h5(current)
    _write_minimal_scaled_h5(bagged)
    _write_minimal_scaled_h5(leaked, leak_cal=True)
    names = runner._bag_training_profile_names(bagged, 2)
    assert names == [["bag_train_0"], ["bag_train_1"]]
    result = runner._assert_no_bag_leakage(names, cal_profile_names=["cal_a"], test_profile_names=["test_a"])
    assert result["train_calibration_overlap_detected"] is False
    assert runner._scaling_stats_equal(runner._load_scaling_stats(current), runner._load_scaling_stats(bagged))
    leaked_names = runner._bag_training_profile_names(leaked, 2)
    with pytest.raises(ValueError, match="leak"):
        runner._assert_no_bag_leakage(leaked_names, cal_profile_names=["cal_a"], test_profile_names=["test_a"])


def test_checkpoint_mode_missing_file_and_scaler_mismatch(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import scripts.run_lstm_conformal_prediction as runner

    current = tmp_path / "current.h5"
    bagged = tmp_path / "bagged.h5"
    mismatch = tmp_path / "mismatch.h5"
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"placeholder")
    checkpoint_b.write_bytes(b"placeholder")
    _write_minimal_scaled_h5(current)
    _write_minimal_scaled_h5(bagged)
    _write_minimal_scaled_h5(mismatch, scale_offset=1.0)
    datasets = {
        "sample_shape": (2, 3, 2), "target_shape": (2, 1),
        "cal_profile_names": ["cal_a"], "test_profile_names": ["test_a"],
    }
    args = runner.LSTMConformalRunConfig(
        sim_root=".", batches=["0001"], lookback=1, config_py_path="scripts/config.py",
        unscaled_out_dir=str(tmp_path), scaled_out_dir=str(tmp_path), out_dir=str(tmp_path),
        conformal_method="joint_ensemble_normalized", ensemble_source="checkpoints",
        ensemble_checkpoint_paths=[str(checkpoint_a), str(tmp_path / "missing.pt")],
        ensemble_bagged_h5_path=str(bagged),
    )
    with pytest.raises(FileNotFoundError):
        runner._load_joint_ensemble_from_checkpoints(
            args, scaled_h5_path=current, datasets=datasets, scaling_stats=runner._load_scaling_stats(current)
        )
    args = runner.LSTMConformalRunConfig(
        sim_root=".", batches=["0001"], lookback=1, config_py_path="scripts/config.py",
        unscaled_out_dir=str(tmp_path), scaled_out_dir=str(tmp_path), out_dir=str(tmp_path),
        conformal_method="joint_ensemble_normalized", ensemble_source="checkpoints",
        ensemble_checkpoint_paths=[str(checkpoint_a), str(checkpoint_b)],
        ensemble_bagged_h5_path=str(mismatch),
    )
    with pytest.raises(ValueError, match="scaling statistics"):
        runner._load_joint_ensemble_from_checkpoints(
            args, scaled_h5_path=current, datasets=datasets, scaling_stats=runner._load_scaling_stats(current)
        )


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
