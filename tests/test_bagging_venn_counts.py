from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_bagging_module():
    sys.modules.setdefault("h5py", types.ModuleType("h5py"))
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")
    fake_torch_utils_data.DataLoader = object
    sys.modules["torch.utils"] = fake_torch_utils
    sys.modules["torch.utils.data"] = fake_torch_utils_data

    sys.modules.setdefault("rabl", types.ModuleType("rabl"))
    sys.modules.setdefault("rabl.machine_learning", types.ModuleType("rabl.machine_learning"))

    fake_branch = types.ModuleType("rabl.machine_learning.branchpoint_finder")
    fake_branch.finite_difference = lambda *args, **kwargs: None
    sys.modules["rabl.machine_learning.branchpoint_finder"] = fake_branch

    fake_lstm = types.ModuleType("rabl.machine_learning.lstm_pipeline")
    fake_lstm.STATE_DIM = 13
    fake_lstm.TARGET_NAMES = []
    fake_lstm.ProfileDataset = object
    fake_lstm.SampleDataset = object
    fake_lstm._count_samples_in_split = lambda *args, **kwargs: 0
    fake_lstm._descale_feature_from_stats = lambda *args, **kwargs: None
    fake_lstm._descale_targets_from_stats = lambda *args, **kwargs: None
    fake_lstm._extract_control_series = lambda *args, **kwargs: None
    fake_lstm._get_profile_names = lambda *args, **kwargs: []
    fake_lstm._get_profile_shapes = lambda *args, **kwargs: {}
    fake_lstm._load_scaling_stats = lambda *args, **kwargs: {}
    fake_lstm.rolling_forecast = lambda *args, **kwargs: None
    fake_lstm.train_with_fallback = lambda *args, **kwargs: (None, {"val_loss": [1.0]}, "cpu")
    sys.modules["rabl.machine_learning.lstm_pipeline"] = fake_lstm

    spec = importlib.util.spec_from_file_location(
        "rabl.machine_learning.bagging_under_test",
        Path("src/rabl/machine_learning/bagging_ensemble.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["rabl.machine_learning.bagging_under_test"] = module
    spec.loader.exec_module(module)
    return module


bagging = _load_bagging_module()


def test_venn_region_counts_unweighted() -> None:
    bag_sets = [
        {"a", "b", "c", "shared"},
        {"b", "d", "shared"},
        {"c", "e", "shared"},
    ]
    counts = bagging._venn_region_counts(bag_sets)
    assert counts == {
        "100": 1,  # a
        "010": 1,  # d
        "001": 1,  # e
        "110": 1,  # b
        "101": 1,  # c
        "011": 0,
        "111": 1,  # shared
    }


def test_venn_region_counts_weighted() -> None:
    bag_sets = [
        {"p1", "p2", "p_shared"},
        {"p2", "p3", "p_shared"},
        {"p4", "p_shared"},
    ]
    weights = {
        "p1": 10,
        "p2": 20,
        "p3": 30,
        "p4": 40,
        "p_shared": 50,
    }
    counts = bagging._venn_region_counts(bag_sets, weights=weights)
    assert counts == {
        "100": 10,  # p1
        "010": 30,  # p3
        "001": 40,  # p4
        "110": 20,  # p2
        "101": 0,
        "011": 0,
        "111": 50,  # p_shared
    }
