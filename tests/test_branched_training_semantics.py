import csv
import importlib.util
import sys
import types
from pathlib import Path

sys.modules.setdefault("h5py", types.ModuleType("h5py"))
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if "rabl" in sys.modules and not hasattr(sys.modules["rabl"], "__path__"):
    sys.modules.pop("rabl", None)
    sys.modules.pop("rabl.machine_learning", None)
MODULE_PATH = REPO_ROOT / "src" / "rabl" / "machine_learning" / "build_lstm_dataset.py"
spec = importlib.util.spec_from_file_location("build_lstm_dataset_module", MODULE_PATH)
assert spec is not None and spec.loader is not None
build_lstm_dataset_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = build_lstm_dataset_module
spec.loader.exec_module(build_lstm_dataset_module)
_collect_csv_files = build_lstm_dataset_module._collect_csv_files


def _write_csv(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t", "drumAngleDeg"])
        for t, angle in rows:
            w.writerow([t, angle])


def test_branch_results_start_at_branch_time_and_root_starts_at_zero(tmp_path: Path) -> None:
    out_dir = tmp_path / "sim_profiles" / "batch_0001"
    branched_dir = out_dir / "branched_results"

    root_csv = branched_dir / "results_root_001__profile_000000.csv"
    child_csv = branched_dir / "results_root_001__profile_000001.csv"
    _write_csv(root_csv, [(0.0, 178.0), (1.0, 178.2), (2.0, 178.1)])
    _write_csv(child_csv, [(1.5, 178.3), (2.0, 178.4)])

    summary = out_dir / "batch_summary.csv"
    with summary.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "root_id",
            "profile_id",
            "parent_profile_id",
            "branch_time_s",
            "depth",
            "run_type",
            "profile_mat",
            "generated_profile_mat",
            "restart_source_result",
            "dymola_result_file",
            "result_csv_out",
            "result_mat_out",
            "status",
            "stop_time_s",
            "matread_s",
            "simulate_s",
            "extract_s",
            "merge_s",
            "write_s",
            "total_run_s",
            "result_base",
        ])
        w.writerow([
            "root_001",
            "profile_000000",
            "",
            "",
            0,
            "root_full",
            "",
            "",
            "",
            "",
            root_csv.name,
            root_csv.with_suffix(".mat").name,
            "OK",
            2.0,
            0,
            0,
            0,
            0,
            0,
            0,
            "root_001__profile_000000__root",
        ])
        w.writerow([
            "root_001",
            "profile_000001",
            "profile_000000",
            1.5,
            1,
            "branch_restart",
            "",
            "",
            "",
            "",
            child_csv.name,
            child_csv.with_suffix(".mat").name,
            "OK",
            2.0,
            0,
            0,
            0,
            0,
            0,
            0,
            "root_001__profile_000001__branch",
        ])

    with root_csv.open(newline="") as fp:
        root_rows = list(csv.DictReader(fp))
    with child_csv.open(newline="") as fp:
        child_rows = list(csv.DictReader(fp))

    assert float(root_rows[0]["t"]) == 0.0

    with summary.open(newline="") as fp:
        summary_rows = list(csv.DictReader(fp))
    branch_row = next(r for r in summary_rows if r["profile_id"] == "profile_000001")
    branch_time = float(branch_row["branch_time_s"])
    assert float(child_rows[0]["t"]) >= branch_time


def test_collect_csv_files_uses_batch_root_only_and_warns_on_duplicate_stems(tmp_path: Path, capsys) -> None:
    batch_0001 = tmp_path / "sim_profiles" / "batch_0001"
    batch_0002 = tmp_path / "sim_profiles" / "batch_0002"

    root_csv_1 = batch_0001 / "results_drum_profile_00001.csv"
    nested_csv = batch_0001 / "branched_results" / "results_drum_profile_00099.csv"
    root_csv_2 = batch_0002 / "results_drum_profile_00001.csv"

    _write_csv(root_csv_1, [(0.0, 178.0), (1.0, 178.1)])
    _write_csv(nested_csv, [(0.5, 178.2), (1.0, 178.3)])
    _write_csv(root_csv_2, [(0.0, 177.9), (1.0, 178.0)])

    collected = _collect_csv_files([batch_0001, batch_0002])

    assert nested_csv not in collected
    assert root_csv_1 in collected
    assert root_csv_2 in collected

    output = capsys.readouterr().out
    assert "Warning: duplicate result stems detected across batches" in output

np = build_lstm_dataset_module.np
STATE_COLUMNS = build_lstm_dataset_module.STATE_COLUMNS
CONTROL_COLUMN = build_lstm_dataset_module.CONTROL_COLUMN
BranchLineageEntry = build_lstm_dataset_module.BranchLineageEntry
_build_branch_sequences = build_lstm_dataset_module._build_branch_sequences
_build_sequences = build_lstm_dataset_module._build_sequences
_read_profile_data = build_lstm_dataset_module._read_profile_data
_steady_state_rows = build_lstm_dataset_module._steady_state_rows


def _steady_state(value: float = 0.0) -> dict[str, float]:
    out = {col: value for col in STATE_COLUMNS}
    out[CONTROL_COLUMN] = value
    return out


def _write_full_results_csv(path: Path, times: list[float], state_base: float, control_base: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    control_base = state_base if control_base is None else control_base
    with path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t", *STATE_COLUMNS, CONTROL_COLUMN])
        for idx, t in enumerate(times):
            # Make the first state column uniquely identify each physical row.
            states = [state_base + idx] + [1000.0 + state_base + idx + j for j in range(1, len(STATE_COLUMNS))]
            w.writerow([t, *states, control_base + idx])


def _profile(path: Path, lineage: object | None = None):
    return _read_profile_data(path, lineage)


def test_branch_training_window_uses_parent_history_not_equilibrium(tmp_path: Path) -> None:
    k = 3
    root_csv = tmp_path / "results_drum_profile_00001.csv"
    branch_csv = tmp_path / "results_drum_profile_00002.csv"
    _write_full_results_csv(root_csv, [0.0, 1.0, 2.0, 3.0], state_base=10.0)
    _write_full_results_csv(branch_csv, [3.0, 4.0, 5.0], state_base=100.0)

    root = _profile(root_csv, BranchLineageEntry("results_drum_profile_00001", "root_001", "profile_00000", None, None))
    branch = _profile(branch_csv, BranchLineageEntry("results_drum_profile_00002", "root_001", "profile_00001", "profile_00000", 3.0))
    state_pad, control_pad = _steady_state_rows(_steady_state(-999.0), k)

    x_seq, y_seq, history_rows = _build_branch_sequences(
        branch,
        k=k,
        by_lineage_key={("root_001", "profile_00000"): root, ("root_001", "profile_00001"): branch},
        steady_state_rows=(state_pad[:k], control_pad[:k]),
    )

    assert history_rows == k
    assert x_seq.shape[0] == 2
    # First branch window is the final 3 parent rows before t=3.0, followed by
    # the first branch-continuation row at t=3.0.
    assert x_seq[0, :, 0].tolist() == [10.0, 11.0, 12.0, 100.0]
    assert y_seq[0, 0] == 101.0
    assert -999.0 not in x_seq[0, :, 0]


def test_branch_start_time_tolerates_csv_roundoff(tmp_path: Path) -> None:
    k = 3
    root_csv = tmp_path / "results_drum_profile_00001.csv"
    branch_csv = tmp_path / "results_drum_profile_00002.csv"
    branch_time = 60.800000000000004
    rounded_first_t = 60.79999923706055
    _write_full_results_csv(root_csv, [59.2, 59.6, 60.0, 60.4], state_base=10.0)
    _write_full_results_csv(branch_csv, [rounded_first_t, 61.2, 61.6], state_base=100.0)

    root = _profile(root_csv, BranchLineageEntry("results_drum_profile_00001", "root_001", "profile_00000", None, None))
    branch = _profile(branch_csv, BranchLineageEntry("results_drum_profile_00002", "root_001", "profile_00001", "profile_00000", branch_time))
    state_pad, control_pad = _steady_state_rows(_steady_state(-999.0), k)

    x_seq, y_seq, history_rows = _build_branch_sequences(
        branch,
        k=k,
        by_lineage_key={("root_001", "profile_00000"): root, ("root_001", "profile_00001"): branch},
        steady_state_rows=(state_pad[:k], control_pad[:k]),
    )

    assert history_rows == k
    assert x_seq[0, :, 0].tolist() == [11.0, 12.0, 13.0, 100.0]
    assert y_seq[0, 0] == 101.0


def test_branch_start_time_still_fails_when_meaningfully_before_branch_time(tmp_path: Path) -> None:
    k = 3
    root_csv = tmp_path / "results_drum_profile_00001.csv"
    branch_csv = tmp_path / "results_drum_profile_00002.csv"
    _write_full_results_csv(root_csv, [59.2, 59.6, 60.0, 60.4], state_base=10.0)
    _write_full_results_csv(branch_csv, [60.79, 61.2, 61.6], state_base=100.0)

    root = _profile(root_csv, BranchLineageEntry("results_drum_profile_00001", "root_001", "profile_00000", None, None))
    branch = _profile(branch_csv, BranchLineageEntry("results_drum_profile_00002", "root_001", "profile_00001", "profile_00000", 60.8))
    state_pad, control_pad = _steady_state_rows(_steady_state(-999.0), k)

    try:
        _build_branch_sequences(
            branch,
            k=k,
            by_lineage_key={("root_001", "profile_00000"): root, ("root_001", "profile_00001"): branch},
            steady_state_rows=(state_pad[:k], control_pad[:k]),
        )
    except SystemExit as exc:
        assert "starts before branch_time" in str(exc)
    else:
        raise AssertionError("Expected branch start meaningfully before branch_time to fail loudly")


def test_recursive_branch_history_climbs_parent_lineage(tmp_path: Path) -> None:
    k = 3
    root_csv = tmp_path / "results_drum_profile_00001.csv"
    parent_branch_csv = tmp_path / "results_drum_profile_00002.csv"
    child_branch_csv = tmp_path / "results_drum_profile_00003.csv"
    _write_full_results_csv(root_csv, [0.0, 1.0, 2.0], state_base=10.0)
    _write_full_results_csv(parent_branch_csv, [2.0, 3.0], state_base=100.0)
    _write_full_results_csv(child_branch_csv, [3.5, 4.0, 5.0], state_base=200.0)

    root = _profile(root_csv, BranchLineageEntry("results_drum_profile_00001", "root_001", "profile_00000", None, None))
    parent_branch = _profile(parent_branch_csv, BranchLineageEntry("results_drum_profile_00002", "root_001", "profile_00001", "profile_00000", 2.0))
    child_branch = _profile(child_branch_csv, BranchLineageEntry("results_drum_profile_00003", "root_001", "profile_00002", "profile_00001", 3.5))
    state_pad, control_pad = _steady_state_rows(_steady_state(-999.0), k)

    by_key = {
        ("root_001", "profile_00000"): root,
        ("root_001", "profile_00001"): parent_branch,
        ("root_001", "profile_00002"): child_branch,
    }
    x_seq, y_seq, history_rows = _build_branch_sequences(
        child_branch,
        k=k,
        by_lineage_key=by_key,
        steady_state_rows=(state_pad[:k], control_pad[:k]),
    )

    assert history_rows == k
    # Only two rows are available in the immediate parent before t=3.5, so the
    # oldest row is pulled from that parent's root lineage.
    assert x_seq[0, :, 0].tolist() == [11.0, 100.0, 101.0, 200.0]
    assert y_seq[0, 0] == 201.0


def test_root_profile_padding_behavior_is_unchanged(tmp_path: Path) -> None:
    k = 3
    root_csv = tmp_path / "results_drum_profile_00001.csv"
    _write_full_results_csv(root_csv, [0.0, 1.0, 2.0], state_base=10.0)
    root = _profile(root_csv)
    state_pad, control_pad = _steady_state_rows(_steady_state(-999.0), k)

    x_seq, y_seq = _build_sequences(np.vstack([state_pad, root.states]), np.vstack([control_pad, root.control]), k)

    assert x_seq[0, :, 0].tolist() == [-999.0, -999.0, -999.0, -999.0]
    assert y_seq[0, 0] == 10.0


def test_branch_missing_parent_metadata_fails_loudly(tmp_path: Path) -> None:
    k = 3
    branch_csv = tmp_path / "results_drum_profile_00002.csv"
    _write_full_results_csv(branch_csv, [3.0, 4.0, 5.0], state_base=100.0)
    branch = _profile(branch_csv, BranchLineageEntry("results_drum_profile_00002", "root_001", "profile_00001", "profile_00000", 3.0))
    state_pad, control_pad = _steady_state_rows(_steady_state(-999.0), k)

    try:
        _build_branch_sequences(
            branch,
            k=k,
            by_lineage_key={("root_001", "profile_00001"): branch},
            steady_state_rows=(state_pad[:k], control_pad[:k]),
        )
    except SystemExit as exc:
        assert "Missing parent metadata/result" in str(exc)
    else:
        raise AssertionError("Expected missing branch parent metadata to fail loudly")
