from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_run_single_experiment_module():
    sys.modules.setdefault("h5py", types.ModuleType("h5py"))

    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_colors = types.ModuleType("matplotlib.colors")
    fake_lines = types.ModuleType("matplotlib.lines")
    fake_colors.LogNorm = object
    fake_lines.Line2D = object
    fake_pyplot.get_cmap = lambda *_args, **_kwargs: (lambda _x: "black")
    fake_pyplot.subplots = lambda *args, **kwargs: (_DummyFigure(), _DummyAxes())
    fake_pyplot.close = lambda *args, **kwargs: None
    sys.modules.setdefault("matplotlib", fake_matplotlib)
    sys.modules["matplotlib.pyplot"] = fake_pyplot
    sys.modules["matplotlib.colors"] = fake_colors
    sys.modules["matplotlib.lines"] = fake_lines

    sys.modules.setdefault("rabl", types.ModuleType("rabl"))
    sys.modules.setdefault("rabl.machine_learning", types.ModuleType("rabl.machine_learning"))
    sys.modules.setdefault("rabl.interface", types.ModuleType("rabl.interface"))
    sys.modules.setdefault("rabl.variography", types.ModuleType("rabl.variography"))

    fake_build_dataset = types.ModuleType("rabl.machine_learning.build_lstm_dataset")
    fake_build_dataset._validate_config = lambda cfg: cfg
    fake_build_dataset._load_config = lambda path: {}
    fake_build_dataset.build_dataset = lambda **kwargs: Path("dummy.h5")
    sys.modules["rabl.machine_learning.build_lstm_dataset"] = fake_build_dataset

    fake_scaling = types.ModuleType("rabl.machine_learning.dataset_scaling")

    class _DummySplitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            return Path("scaled.h5")

    fake_scaling.LSTMDatasetScalerSplitter = _DummySplitter
    sys.modules["rabl.machine_learning.dataset_scaling"] = fake_scaling

    fake_tuner = types.ModuleType("rabl.machine_learning.tuner")

    class _GridSearchConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_tuner.GridSearchConfig = _GridSearchConfig
    fake_tuner.run_grid_search = lambda cfg: ([], types.SimpleNamespace(kind="grid", cfg=cfg))
    fake_tuner.run_hyperband_search = lambda cfg: ([], types.SimpleNamespace(kind="hyperband", cfg=cfg))
    sys.modules["rabl.machine_learning.tuner"] = fake_tuner

    fake_bagging = types.ModuleType("rabl.machine_learning.bagging_ensemble")
    fake_bagging.run_bagging_ensemble = lambda *args, **kwargs: {}
    sys.modules["rabl.machine_learning.bagging_ensemble"] = fake_bagging

    fake_lstm = types.ModuleType("rabl.machine_learning.lstm_pipeline")
    fake_lstm.save_forecast_profiles_pdf = lambda *args, **kwargs: None
    sys.modules["rabl.machine_learning.lstm_pipeline"] = fake_lstm

    fake_recursive = types.ModuleType("rabl.machine_learning.recursive_branching")

    class _RecursiveBranchingBatchConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_recursive.RecursiveBranchingBatchConfig = _RecursiveBranchingBatchConfig
    fake_recursive.run_recursive_branching_batch = lambda cfg: None
    sys.modules["rabl.machine_learning.recursive_branching"] = fake_recursive

    fake_pymola = types.ModuleType("rabl.interface.pymola")

    class _BatchConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Runner:
        def __init__(self, cfg):
            self.cfg = cfg

        def start(self):
            return None

        def run_branched_mat(self):
            return None

        def run_all(self):
            return None

        def close(self):
            return None

    fake_pymola.BatchConfig = _BatchConfig
    fake_pymola.DymolaBatchRunner = _Runner
    sys.modules["rabl.interface.pymola"] = fake_pymola

    fake_variography = types.ModuleType("rabl.variography.DrumVariography")

    class _DrumProfileGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def solve_params_for_sigma_theta(self, **kwargs):
            return None

        def generate(self, **kwargs):
            return []

    fake_variography.DrumProfileGenerator = _DrumProfileGenerator
    sys.modules["rabl.variography.DrumVariography"] = fake_variography

    spec = importlib.util.spec_from_file_location(
        "run_single_experiment_under_test",
        Path("scripts/run_single_experiment.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_single_experiment_under_test"] = module
    spec.loader.exec_module(module)
    return module


class _DummyFigure:
    def tight_layout(self):
        return None

    def savefig(self, *args, **kwargs):
        return None

    def suptitle(self, *args, **kwargs):
        return None


class _DummyAxis:
    def plot(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def set_axis_off(self):
        return None


class _DummyAxes:
    def ravel(self):
        return [_DummyAxis() for _ in range(18)]


run_single_experiment = _load_run_single_experiment_module()


def test_tune_uses_grid_search_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_grid_search(cfg):
        calls.append("grid")
        return [], types.SimpleNamespace(kind="grid", cfg=cfg)

    def _fake_hyperband_search(cfg):
        calls.append("hyperband")
        return [], types.SimpleNamespace(kind="hyperband", cfg=cfg)

    monkeypatch.setattr(run_single_experiment, "run_grid_search", _fake_grid_search)
    monkeypatch.setattr(run_single_experiment, "run_hyperband_search", _fake_hyperband_search)

    best = run_single_experiment._tune(
        scaled_h5=tmp_path / "scaled.h5",
        out_dir=tmp_path / "out",
        seed=7,
        grid={
            "lookback": 12,
            "learning_rates": [1e-4],
            "batch_sizes": [16],
            "n_lstm_values": [1],
            "hidden_lstm_values": [32],
            "hidden_fc_values": [64],
            "epochs": 3,
        },
    )

    assert calls == ["grid"]
    assert best.kind == "grid"
    assert best.cfg.min_epochs is None
    assert best.cfg.max_epochs is None


def test_tune_uses_hyperband_when_requested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_grid_search(cfg):
        calls.append("grid")
        return [], types.SimpleNamespace(kind="grid", cfg=cfg)

    def _fake_hyperband_search(cfg):
        calls.append("hyperband")
        return [], types.SimpleNamespace(kind="hyperband", cfg=cfg)

    monkeypatch.setattr(run_single_experiment, "run_grid_search", _fake_grid_search)
    monkeypatch.setattr(run_single_experiment, "run_hyperband_search", _fake_hyperband_search)

    best = run_single_experiment._tune(
        scaled_h5=tmp_path / "scaled.h5",
        out_dir=tmp_path / "out",
        seed=9,
        grid={
            "method": "hyperband",
            "lookback": 8,
            "learning_rates": [1e-4, 2e-4],
            "batch_sizes": [16],
            "n_lstm_values": [1],
            "hidden_lstm_values": [32],
            "hidden_fc_values": [64],
            "epochs": 6,
            "min_epochs": 2,
            "max_epochs": 8,
            "reduction_factor": 2,
            "early_stopping_patience": 3,
        },
    )

    assert calls == ["hyperband"]
    assert best.kind == "hyperband"
    assert best.cfg.min_epochs == 2
    assert best.cfg.max_epochs == 8
    assert best.cfg.reduction_factor == 2
    assert best.cfg.early_stopping_patience == 3


def test_tune_rejects_hyperband_without_epoch_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires hp_grid.min_epochs and hp_grid.max_epochs"):
        run_single_experiment._tune(
            scaled_h5=tmp_path / "scaled.h5",
            out_dir=tmp_path / "out",
            seed=9,
            grid={
                "method": "hyperband",
                "lookback": 8,
                "learning_rates": [1e-4],
                "batch_sizes": [16],
                "n_lstm_values": [1],
                "hidden_lstm_values": [32],
                "hidden_fc_values": [64],
                "epochs": 6,
            },
        )
