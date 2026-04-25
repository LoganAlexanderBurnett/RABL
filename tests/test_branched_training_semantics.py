import csv
import importlib.util
import sys
import types
from pathlib import Path

sys.modules.setdefault("h5py", types.ModuleType("h5py"))
MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "rabl" / "machine_learning" / "build_lstm_dataset.py"
spec = importlib.util.spec_from_file_location("build_lstm_dataset_module", MODULE_PATH)
assert spec is not None and spec.loader is not None
build_lstm_dataset_module = importlib.util.module_from_spec(spec)
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
