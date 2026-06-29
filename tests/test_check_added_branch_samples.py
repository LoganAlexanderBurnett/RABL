import csv
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

_h5py = pytest.importorskip("h5py")
if not hasattr(_h5py, "File"):
    sys.modules.pop("h5py", None)
h5py = importlib.import_module("h5py")
if not hasattr(h5py, "File"):
    pytest.skip("real h5py is required", allow_module_level=True)
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_added_branch_samples.py"


def _write_csv(path: Path, times, states, controls) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["t", "state", "u"])
        for row in zip(times, states, controls):
            writer.writerow(row)


def _write_h5(path: Path, groups: dict[str, tuple[np.ndarray, np.ndarray, dict]]) -> None:
    with h5py.File(path, "w") as h5f:
        h5f.attrs["k_lookback"] = 2
        h5f.attrs["state_feature_names"] = np.asarray(["state"], dtype="S")
        h5f.attrs["control_feature_name"] = "u"
        files = h5f.create_group("files")
        for name, (x, y, attrs) in groups.items():
            group = files.create_group(name)
            group.create_dataset("X", data=x)
            group.create_dataset("Y", data=y)
            for key, value in attrs.items():
                group.attrs[key] = value
            group.attrs["num_samples"] = x.shape[0]


def test_check_added_branch_samples_passes_for_parent_padded_branch(tmp_path: Path) -> None:
    root_csv = tmp_path / "results_drum_profile_00001.csv"
    branch_csv = tmp_path / "results_drum_profile_00002.csv"
    _write_csv(root_csv, [0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0])
    _write_csv(branch_csv, [2.5, 3.0, 3.5], [100.0, 101.0, 102.0], [200.0, 201.0, 202.0])

    root_x = np.asarray([[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]])
    root_y = np.asarray([[13.0]])
    root_attrs = {
        "source_file": str(root_csv),
        "history_rows_prepended": 3,
        "branch_root_id": "root_001",
        "branch_profile_id": "profile_000000",
        "branch_parent_profile_id": "",
        "branch_time": np.nan,
    }
    branch_x = np.asarray([
        [[11.0, 22.0], [12.0, 200.0], [100.0, 201.0]],
        [[12.0, 200.0], [100.0, 201.0], [101.0, 202.0]],
    ])
    branch_y = np.asarray([[101.0], [102.0]])
    branch_attrs = {
        "source_file": str(branch_csv),
        "history_rows_prepended": 2,
        "branch_root_id": "root_001",
        "branch_profile_id": "profile_000001",
        "branch_parent_profile_id": "profile_000000",
        "branch_time": 2.5,
        "branch_source_stem": "results_root_001__profile_000001",
    }

    before = tmp_path / "before.h5"
    after = tmp_path / "after.h5"
    _write_h5(before, {"results_drum_profile_00001": (root_x, root_y, root_attrs)})
    _write_h5(
        after,
        {
            "results_drum_profile_00001": (root_x, root_y, root_attrs),
            "results_drum_profile_00002": (branch_x, branch_y, branch_attrs),
        },
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(before), str(after)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert '"added_sample_count": 2' in result.stdout
    assert '"problems": []' in result.stdout


def test_check_added_branch_samples_fails_for_steady_state_padded_branch(tmp_path: Path) -> None:
    root_csv = tmp_path / "results_drum_profile_00001.csv"
    branch_csv = tmp_path / "results_drum_profile_00002.csv"
    _write_csv(root_csv, [0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0])
    _write_csv(branch_csv, [2.5, 3.0], [100.0, 101.0], [200.0, 201.0])

    root_x = np.asarray([[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]])
    root_y = np.asarray([[13.0]])
    root_attrs = {
        "source_file": str(root_csv),
        "history_rows_prepended": 3,
        "branch_root_id": "root_001",
        "branch_profile_id": "profile_000000",
        "branch_parent_profile_id": "",
        "branch_time": np.nan,
    }
    bad_branch_x = np.asarray([[[-999.0, -999.0], [-999.0, 200.0], [100.0, 201.0]]])
    branch_y = np.asarray([[101.0]])
    branch_attrs = {
        "source_file": str(branch_csv),
        "history_rows_prepended": 3,
        "branch_root_id": "root_001",
        "branch_profile_id": "profile_000001",
        "branch_parent_profile_id": "profile_000000",
        "branch_time": 2.5,
    }

    before = tmp_path / "before.h5"
    after = tmp_path / "after.h5"
    _write_h5(before, {"results_drum_profile_00001": (root_x, root_y, root_attrs)})
    _write_h5(
        after,
        {
            "results_drum_profile_00001": (root_x, root_y, root_attrs),
            "results_drum_profile_00002": (bad_branch_x, branch_y, branch_attrs),
        },
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(before), str(after)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "history_rows_prepended=3, expected lookback=2" in result.stdout
    assert "first X state window does not equal parent history plus first child state" in result.stdout
