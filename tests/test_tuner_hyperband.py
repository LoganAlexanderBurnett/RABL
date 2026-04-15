from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_tuner_module():
    fake_torch = types.ModuleType("torch")

    def _save(obj, path):
        Path(path).write_text(json.dumps({"saved": True}), encoding="utf-8")

    def _load(path, map_location=None):
        _ = map_location
        if Path(path).exists():
            return {"loaded": True}
        raise FileNotFoundError(path)

    fake_torch.save = _save
    fake_torch.load = _load
    sys.modules.setdefault("torch", fake_torch)

    fake_pipeline = types.ModuleType("rabl.machine_learning.lstm_pipeline")
    fake_pipeline.build_datasets = lambda **kwargs: {}
    fake_pipeline.build_model = lambda **kwargs: object()
    fake_pipeline.cleanup_cuda = lambda *args, **kwargs: None
    fake_pipeline.test_and_save_forecasts = lambda *args, **kwargs: {"mae": 0.0}
    fake_pipeline.train_with_fallback = lambda *args, **kwargs: (object(), {"val_loss": [1.0]}, "cpu")

    sys.modules.setdefault("rabl", types.ModuleType("rabl"))
    sys.modules.setdefault("rabl.machine_learning", types.ModuleType("rabl.machine_learning"))
    sys.modules["rabl.machine_learning.lstm_pipeline"] = fake_pipeline

    spec = importlib.util.spec_from_file_location(
        "tuner_under_test", Path("src/rabl/machine_learning/tuner.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["tuner_under_test"] = module
    spec.loader.exec_module(module)
    return module


tuner = _load_tuner_module()


@pytest.fixture
def base_config(tmp_path: Path):
    return tuner.GridSearchConfig(
        lookback_datasets={4: tmp_path / "dummy.h5"},
        learning_rates=[1e-3],
        batch_sizes=[16],
        n_lstm_values=[1],
        hidden_lstm_values=[8],
        hidden_fc_values=[8],
        out_dir=tmp_path / "out",
        min_epochs=1,
        max_epochs=9,
        reduction_factor=3,
        prune_strategy="successive_halving",
    )


def test_hyperband_config_validation_rejects_invalid_ranges(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_epochs must be >= 1"):
        tuner.GridSearchConfig(
            lookback_datasets={1: tmp_path / "x.h5"},
            learning_rates=[1e-3],
            batch_sizes=[8],
            n_lstm_values=[1],
            hidden_lstm_values=[8],
            hidden_fc_values=[8],
            min_epochs=0,
            max_epochs=4,
        )

    with pytest.raises(ValueError, match="max_epochs must be >= min_epochs"):
        tuner.GridSearchConfig(
            lookback_datasets={1: tmp_path / "x.h5"},
            learning_rates=[1e-3],
            batch_sizes=[8],
            n_lstm_values=[1],
            hidden_lstm_values=[8],
            hidden_fc_values=[8],
            min_epochs=3,
            max_epochs=2,
        )

    with pytest.raises(ValueError, match="reduction_factor must be > 1"):
        tuner.GridSearchConfig(
            lookback_datasets={1: tmp_path / "x.h5"},
            learning_rates=[1e-3],
            batch_sizes=[8],
            n_lstm_values=[1],
            hidden_lstm_values=[8],
            hidden_fc_values=[8],
            min_epochs=1,
            max_epochs=4,
            reduction_factor=1,
        )


def test_construct_hyperband_brackets_and_rung_budgets(base_config) -> None:
    assert base_config.preload_train_to_device is True
    assert base_config.preload_val_to_device is True
    brackets = tuner.construct_hyperband_brackets(base_config)
    assert len(brackets) == 3
    assert [b.bracket_index for b in brackets] == [2, 1, 0]
    assert brackets[0].rung_budgets == [1, 3, 9]
    assert brackets[1].rung_budgets == [3, 9]
    assert brackets[2].rung_budgets == [9]


def test_hyperband_promotes_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    h5 = tmp_path / "dummy.h5"
    h5.write_text("ok", encoding="utf-8")

    cfg = tuner.GridSearchConfig(
        lookback_datasets={4: h5},
        learning_rates=[1e-3, 1e-4],
        batch_sizes=[16],
        n_lstm_values=[1],
        hidden_lstm_values=[8],
        hidden_fc_values=[8],
        out_dir=tmp_path / "out",
        min_epochs=1,
        max_epochs=3,
        reduction_factor=2,
        early_stopping_patience=1,
    )

    monkeypatch.setattr(tuner, "build_datasets", lambda h5_path, batch_size, seed: {"h5_path": h5_path})

    calls: list[dict[str, object]] = []

    class DummyModel:
        def state_dict(self):
            return {"w": 1}

    def fake_train_with_fallback(*args, **kwargs):
        epochs = int(kwargs["epochs"])
        resume = kwargs.get("resume_from_weights")
        calls.append({"epochs": epochs, "resume": str(resume) if resume else None})
        actual = epochs if epochs == 1 else max(1, epochs - 1)
        history = {"val_loss": [1.0 / (i + 1) for i in range(actual)]}
        return DummyModel(), history, "cpu"

    monkeypatch.setattr(tuner, "train_with_fallback", fake_train_with_fallback)
    monkeypatch.setattr(tuner, "cleanup_cuda", lambda *args, **kwargs: None)

    results, best = tuner.run_hyperband_search(cfg)

    assert results
    assert best.best_val_loss <= min(item.best_val_loss for item in results)
    assert any(call["resume"] is not None for call in calls)

    payload = json.loads((cfg.out_dir / "grid_search_results.json").read_text(encoding="utf-8"))
    assert "config" in payload
    assert "results" in payload
    assert "hyperband" in payload
    assert payload["hyperband"]["config"]["min_epochs"] == 1
    assert payload["hyperband"]["brackets"]

    all_decisions = [
        rung["decision"]
        for bracket in payload["hyperband"]["brackets"]
        for trial in bracket["trials"]
        for rung in trial["rung_metrics"]
    ]
    assert any(dec in {"promoted", "pruned", "completed", "early_stopped"} for dec in all_decisions)
    timing_payload = json.loads((cfg.out_dir / "tuning_timing.json").read_text(encoding="utf-8"))
    assert timing_payload["run_type"] == "hyperband"
    assert timing_payload["total_duration_s"] >= 0.0
    assert timing_payload["num_timed_trials"] == len(timing_payload["trial_timings"])
