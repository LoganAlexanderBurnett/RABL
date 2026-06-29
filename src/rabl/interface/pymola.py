import csv
import json
import os
import re
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time

import numpy as np
from scipy.io import loadmat, savemat
from dymola.dymola_interface import DymolaInterface

from rabl.paths import resolve_output_root


@dataclass(frozen=True)
class BatchConfig:
    # Paths RELATIVE to this script file for portability, except generated outputs.
    package_mo: str = r"../../modelica/MicroreactorPK/package.mo"
    model_name: str = "MicroreactorPK.Experiments.RunOneProfile"
    profiles_dir: str = str(resolve_output_root() / "variography_profiles" / "test_batch")
    out_dir: str = str(resolve_output_root() / "sim_profiles" / "test_batch")

    # Workflow mode
    profile_mode: str = "flat_mat"  # flat_mat | branched_mat

    # Branched artifact directories (relative to out_dir)
    generated_profile_dir: str = "generated_profiles"
    restart_results_dir: str = "restart_results"
    branched_results_dir: str = "branched_results"
    logs_dir: str = "logs"

    # Branched behaviors
    preserve_restart_artifacts: bool = True
    keep_dsfinal_for_debug: bool = False
    store_protected_vars_for_restart: bool = True
    canonical_output_interval: float | None = None
    keep_full_parent_cache: bool = True

    table_name: str = "profile"
    angle_col: int = 2
    vel_col: int = 3
    acc_col: int = 4

    output_interval: float = 0.1
    skip_existing: bool = True
    keep_last_duplicate_time: bool = True

    # CSV variables must match RunOneProfile convenience outputs
    vars_to_pull: tuple[str, ...] = (
        "t", "drumAngleDeg", "drumVelDeg_s", "drumAccDeg_s2",
        "TN2", "dTN2", "Tm", "dTm", "Thp", "dThp", "Tf", "dTf",
        "c[1]", "c[2]", "c[3]", "c[4]", "c[5]", "c[6]",
        "P_MW", "n", "dn", "rho_dollars",
        "rho_drums_dollars", "rho_fuel_dollars", "rho_moderator_dollars",
        "Tsg", "T_steam_out", "x_steam_out",
    )


@dataclass(frozen=True)
class BranchNode:
    root_id: str
    profile_id: str
    parent_profile_id: str | None
    branch_time: float | None
    branch_end_time: float | None
    created_in_interval: int | None
    branch_label: int | None
    t: np.ndarray
    theta_deg: np.ndarray
    v_deg_s: np.ndarray
    a_deg_s2: np.ndarray
    depth: int
    stop_time: float


_BRANCH_TIME_SNAP_ATOL = 1e-5
_BRANCH_TIME_SNAP_RTOL = 1e-7


def _branch_time_snap_tolerance(branch_time: float) -> float:
    return max(_BRANCH_TIME_SNAP_ATOL, abs(float(branch_time)) * _BRANCH_TIME_SNAP_RTOL)


class DymolaBatchRunner:
    def __init__(self, config: BatchConfig):
        self.cfg = config
        self.script_dir = Path(__file__).resolve().parent

        self.package_abs = (self.script_dir / self.cfg.package_mo).resolve()
        self.profiles_dir_abs = (self.script_dir / self.cfg.profiles_dir).resolve()
        self.out_dir_abs = (self.script_dir / self.cfg.out_dir).resolve()

        self.generated_profiles_abs = self.out_dir_abs / self.cfg.generated_profile_dir
        self.restart_results_abs = self.out_dir_abs / self.cfg.restart_results_dir
        self.branched_results_abs = self.out_dir_abs / self.cfg.branched_results_dir
        self.logs_abs = self.out_dir_abs / self.cfg.logs_dir

        self.dymola = None
        self._vars_verified = False

        # Timing aggregates
        self._t_startup = None
        self._t_openmodel = None
        self._t_total_wall = None
        self._sim_times = []
        self._extract_times = []
        self._write_times = []
        self._merge_times = []
        self._matread_times = []
        self._total_run_times = []

        self.summary_csv = None
    @staticmethod
    def _profile_index_from_name(profile_path: Path) -> int | None:
        m = re.search(r"drum_profile_(\d{5})", profile_path.stem)
        return int(m.group(1)) if m else None

    @staticmethod
    def results_csv_name(profile_path: Path) -> str:
        return f"results_{profile_path.stem}.csv"

    @staticmethod
    def _to_dymola_path(p: Path | str) -> str:
        return str(p).replace("\\", "/")

    def _cleanup_out_dir_independent(self) -> None:
        allowed = {".csv", ".mat", ".log"}
        for entry in self.out_dir_abs.iterdir():
            if entry.is_dir():
                continue
            if entry.suffix.lower() not in allowed:
                entry.unlink()

    def _cleanup_out_dir_branched(self) -> None:
        # Keep branch artifacts by default. Optionally remove restart MATs.
        if self.cfg.preserve_restart_artifacts:
            return
        if self.restart_results_abs.exists():
            for f in self.restart_results_abs.glob("*.mat"):
                f.unlink(missing_ok=True)

    def infer_stop_time(self, profile_mat_path: Path) -> tuple[float, float]:
        t0 = time()
        m = loadmat(str(profile_mat_path))
        tbl = np.asarray(m[self.cfg.table_name], dtype=float)
        stop_time = float(tbl[-1, 0])
        return stop_time, (time() - t0)

    @staticmethod
    def _dedupe_time_keep_last(cols: list[np.ndarray]) -> list[np.ndarray]:
        t = cols[0]
        mask = np.ones_like(t, dtype=bool)
        mask[:-1] = (t[:-1] != t[1:])
        return [col[mask] for col in cols]

    @staticmethod
    def _resample_to_uniform_grid(cols: list[np.ndarray], output_interval: float, stop_time: float) -> list[np.ndarray]:
        t_raw = np.asarray(cols[0], dtype=float)
        if t_raw.size < 2:
            return cols

        start_time = float(t_raw[0])
        stop_time = float(stop_time)
        dt = float(output_interval)

        t_uniform = np.arange(start_time, stop_time + 0.5 * dt, dt, dtype=float)
        if t_uniform.size == 0:
            t_uniform = np.array([start_time], dtype=float)
        if t_uniform[-1] < stop_time:
            t_uniform = np.append(t_uniform, stop_time)

        out = [t_uniform]
        for series in cols[1:]:
            y = np.asarray(series, dtype=float)
            out.append(np.interp(t_uniform, t_raw, y))
        return out

    def start(self):
        self.out_dir_abs.mkdir(parents=True, exist_ok=True)
        self.generated_profiles_abs.mkdir(parents=True, exist_ok=True)
        self.restart_results_abs.mkdir(parents=True, exist_ok=True)
        self.branched_results_abs.mkdir(parents=True, exist_ok=True)
        self.logs_abs.mkdir(parents=True, exist_ok=True)

        if self.summary_csv is None:
            self.summary_csv = self.out_dir_abs / "batch_summary.csv"
        else:
            self.summary_csv = self.out_dir_abs / self.summary_csv

        print("Starting Dymola instance...")
        t0 = time()
        self.dymola = DymolaInterface()
        self._t_startup = time() - t0

        t1 = time()
        self.dymola.openModel(self._to_dymola_path(self.package_abs))
        self._t_openmodel = time() - t1

        self.dymola.cd(self._to_dymola_path(self.out_dir_abs))
        self._vars_verified = False

        # Fail fast model checks before batch execution.
        if not self.dymola.checkModel(self.cfg.model_name):
            raise RuntimeError(f"checkModel failed for {self.cfg.model_name}\n{self.dymola.getLastErrorLog()}")
        if not self.dymola.translateModel(self.cfg.model_name):
            raise RuntimeError(f"translateModel failed for {self.cfg.model_name}\n{self.dymola.getLastErrorLog()}")

        if self.cfg.profile_mode == "branched_mat" and self.cfg.store_protected_vars_for_restart:
            # Preserve protected/internal block variables in result MAT so
            # importInitialResult can reconstruct CombiTimeTable internals.
            ok = self.dymola.ExecuteCommand("Advanced.StoreProtectedVariables := true;")
            if not ok:
                ok = self.dymola.ExecuteCommand("Advanced.StoreProtectedVariables=true;")
            print(f"StoreProtectedVariables enabled: {ok}")

        if not self.summary_csv.exists():
            with open(self.summary_csv, "w", newline="") as fp:
                csv.writer(fp).writerow([
                    "root_id", "profile_id", "parent_profile_id", "branch_time_s", "depth", "run_type",
                    "profile_mat", "generated_profile_mat", "restart_source_result", "dymola_result_file",
                    "result_csv_out", "result_mat_out", "status", "stop_time_s", "matread_s", "simulate_s",
                    "extract_s", "merge_s", "write_s", "total_run_s", "result_base",
                ])

    def close(self):
        if self.dymola is None:
            return
        try:
            self.dymola.close()
        finally:
            self.dymola = None

    def _append_summary(self, **row):
        with open(self.summary_csv, "a", newline="") as fp:
            w = csv.writer(fp)
            w.writerow([
                row.get("root_id", ""), row.get("profile_id", ""), row.get("parent_profile_id", ""),
                "" if row.get("branch_time_s") is None else f"{float(row['branch_time_s']):.6f}",
                row.get("depth", ""), row.get("run_type", ""), row.get("profile_mat", ""),
                row.get("generated_profile_mat", ""), row.get("restart_source_result", ""),
                row.get("dymola_result_file", ""), row.get("result_csv_out", ""),
                row.get("result_mat_out", ""), row.get("status", ""),
                "" if row.get("stop_time_s") is None else f"{float(row['stop_time_s']):.6f}",
                f"{float(row.get('matread_s', 0.0)):.6f}", f"{float(row.get('simulate_s', 0.0)):.6f}",
                f"{float(row.get('extract_s', 0.0)):.6f}", f"{float(row.get('merge_s', 0.0)):.6f}",
                f"{float(row.get('write_s', 0.0)):.6f}", f"{float(row.get('total_run_s', 0.0)):.6f}",
                row.get("result_base", ""),
            ])

    def _simulate_model_call(self, *, profile_mat_path: Path, start_time: float, stop_time: float, result_base: str) -> tuple[bool, float, str]:
        profile_rel = os.path.relpath(profile_mat_path, self.out_dir_abs).replace("\\", "/")
        model_call = (
            f'{self.cfg.model_name}(profileFile="{profile_rel}", '
            f'tableName="{self.cfg.table_name}", '
            f'angleColumn={self.cfg.angle_col}, velColumn={self.cfg.vel_col}, accColumn={self.cfg.acc_col})'
        )
        t0 = time()
        ok = self.dymola.simulateModel(
            model_call,
            startTime=float(start_time),
            stopTime=float(stop_time),
            resultFile=result_base,
            outputInterval=self.cfg.output_interval,
        )
        t_sim = time() - t0
        result_file = str(self.dymola.getLastResultFileName()) if ok else ""
        return bool(ok), t_sim, result_file

    def _simulate_continued_default_model(
        self,
        *,
        start_time: float,
        stop_time: float,
        result_base: str,
    ) -> tuple[bool, float, str, str]:
        """
        Continue from imported state using the active/default model context.
        This preserves imported dynamic states from importInitialResult.
        """
        t0 = time()
        attempt = "simulateModel(\"\")"
        ok = False
        try:
            ok = self.dymola.simulateModel(
                "",
                startTime=float(start_time),
                stopTime=float(stop_time),
                resultFile=result_base,
                outputInterval=self.cfg.output_interval,
            )
        except Exception as exc:
            attempt = f"{attempt} exception: {exc}"
            ok = False

        t_sim = time() - t0
        result_file = str(self.dymola.getLastResultFileName()) if ok else ""
        return bool(ok), t_sim, result_file, attempt

    def _read_result_columns(self, result_file: str, stop_time: float) -> tuple[list[np.ndarray], float]:
        t0 = time()
        rows = self.dymola.readTrajectorySize(result_file)
        if not self._vars_verified:
            available = set(self.dymola.readTrajectoryNames(result_file))
            missing = [v for v in self.cfg.vars_to_pull if v not in available]
            if missing:
                raise RuntimeError(f"FAIL_MISSING_VARS: {missing}")
            self._vars_verified = True

        data = self.dymola.readTrajectory(result_file, list(self.cfg.vars_to_pull), rows)
        cols = [np.asarray(col, dtype=float) for col in data]
        if self.cfg.keep_last_duplicate_time:
            cols = self._dedupe_time_keep_last(cols)
        cols = self._resample_to_uniform_grid(cols, self.cfg.output_interval, stop_time)
        return cols, (time() - t0)

    # -----------------------------
    # Independent workflow
    # -----------------------------
    def simulate_one(self, profile_path: Path) -> tuple[bool, str]:
        assert self.dymola is not None, "Call start() first."
        run_t0 = time()
        try:
            csv_name = self.results_csv_name(profile_path)
            csv_out = self.out_dir_abs / csv_name
            if self.cfg.skip_existing and csv_out.exists():
                self._append_summary(profile_mat=profile_path.name, result_csv_out=csv_out.name, status="SKIP", total_run_s=time()-run_t0)
                return True, f"SKIP (exists): {csv_out.name}"

            idx = self._profile_index_from_name(profile_path) or 0
            result_base = f"run_{idx:05d}" if idx > 0 else f"run_{profile_path.stem}"

            stop_time, t_matread = self.infer_stop_time(profile_path)
            self._matread_times.append(t_matread)

            ok, t_sim, result_file = self._simulate_model_call(
                profile_mat_path=profile_path,
                start_time=0.0,
                stop_time=stop_time,
                result_base=result_base,
            )
            self._sim_times.append(t_sim)
            if not ok:
                (self.logs_abs / f"{result_base}_dymola_error.log").write_text(str(self.dymola.getLastErrorLog()), encoding="utf-8")
                total = time() - run_t0
                self._append_summary(profile_mat=profile_path.name, result_csv_out=csv_out.name, status="FAIL", stop_time_s=stop_time,
                                     matread_s=t_matread, simulate_s=t_sim, total_run_s=total, result_base=result_base)
                return False, "FAILED"

            cols, t_extract = self._read_result_columns(result_file, stop_time)
            self._extract_times.append(t_extract)

            t_w0 = time()
            with open(csv_out, "w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(self.cfg.vars_to_pull)
                w.writerows(zip(*cols))
            savemat(str(csv_out.with_suffix(".mat")), {"table": np.column_stack(cols), "columns": np.array(self.cfg.vars_to_pull, dtype=object)})
            t_write = time() - t_w0
            self._write_times.append(t_write)

            total = time() - run_t0
            self._append_summary(profile_mat=profile_path.name, result_csv_out=csv_out.name, result_mat_out=csv_out.with_suffix('.mat').name,
                                 status="OK", stop_time_s=stop_time, matread_s=t_matread, simulate_s=t_sim, extract_s=t_extract,
                                 write_s=t_write, total_run_s=total, result_base=result_base)
            return True, str(csv_out)
        finally:
            for attempt in range(3):
                try:
                    self._cleanup_out_dir_independent()
                    break
                except PermissionError:
                    if attempt == 2:
                        break
                    sleep(2)

    def run_all(self):
        assert self.dymola is not None, "Call start() first."
        profiles = sorted(self.profiles_dir_abs.glob("drum_profile_*.mat"))
        if not profiles:
            raise RuntimeError(f"No drum_profile_*.mat found in: {self.profiles_dir_abs}")

        batch_t0 = time()
        ran = skipped = failed = 0
        for p in profiles:
            ok, msg = self.simulate_one(p)
            if ok and msg.startswith("SKIP"):
                skipped += 1
            elif ok:
                ran += 1
            else:
                failed += 1
        self._t_total_wall = time() - batch_t0
        self._print_timing_summary(ran)
        print(f"Ran={ran}, Skipped={skipped}, Failed={failed}")

    # -----------------------------
    # Branched workflow
    # -----------------------------
    def _load_branched_manifest_tree(self, batch_dir: Path) -> list[BranchNode]:
        manifest_path = batch_dir / "branched_profiles_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"branched_mat mode requires manifest: {manifest_path}")

        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"Invalid or empty branched manifest: {manifest_path}")

        nodes: list[BranchNode] = []
        for entry in entries:
            root_id = str(entry["root_group_name"])
            profile_id = str(entry["profile_id"])
            mat_path = batch_dir / str(entry["mat_file"])
            if not mat_path.exists():
                raise RuntimeError(f"Manifest MAT file not found: {mat_path}")
            m = loadmat(str(mat_path))
            if self.cfg.table_name not in m:
                raise RuntimeError(f"MAT file missing table '{self.cfg.table_name}': {mat_path}")
            table = np.asarray(m[self.cfg.table_name], dtype=float)
            if table.ndim != 2 or table.shape[1] < 4:
                raise RuntimeError(f"Unexpected profile table shape in {mat_path}: {table.shape}")

            parent_attr = str(entry.get("parent_profile_id", "")).strip()
            parent_profile_id = parent_attr or None
            branch_time_raw = entry.get("branch_time", None)
            branch_end_time_raw = entry.get("branch_end_time", None)
            created_in_interval = int(entry.get("created_in_interval", -1))
            branch_label = int(entry.get("branch_label", -1))

            t = np.asarray(table[:, 0], dtype=float)
            theta_deg = np.asarray(table[:, 1], dtype=float)
            v_deg_s = np.asarray(table[:, 2], dtype=float)
            a_deg_s2 = np.asarray(table[:, 3], dtype=float)
            branch_end_time = None
            if branch_end_time_raw not in (None, ""):
                branch_end_time_value = float(branch_end_time_raw)
                branch_end_time = branch_end_time_value if np.isfinite(branch_end_time_value) else None
            if parent_profile_id is None:
                stop_time = float(t[-1])
            elif branch_end_time is None:
                stop_time = float(t[-1])
            else:
                stop_time = branch_end_time

            nodes.append(BranchNode(
                root_id=root_id,
                profile_id=profile_id,
                parent_profile_id=parent_profile_id,
                branch_time=None if branch_time_raw is None else float(branch_time_raw),
                branch_end_time=branch_end_time,
                created_in_interval=None if created_in_interval < 0 else created_in_interval,
                branch_label=None if branch_label < 0 else branch_label,
                t=t,
                theta_deg=theta_deg,
                v_deg_s=v_deg_s,
                a_deg_s2=a_deg_s2,
                depth=0,
                stop_time=stop_time,
            ))
        return self._validate_branch_tree(nodes)

    def _validate_branch_tree(self, nodes: list[BranchNode]) -> list[BranchNode]:
        by_root: dict[str, dict[str, BranchNode]] = defaultdict(dict)
        for n in nodes:
            by_root[n.root_id][n.profile_id] = n

        out: list[BranchNode] = []
        for root_id, mapping in by_root.items():
            depths: dict[str, int] = {}

            def depth(pid: str) -> int:
                if pid in depths:
                    return depths[pid]
                node = mapping[pid]
                if node.parent_profile_id is None:
                    depths[pid] = 0
                else:
                    if node.parent_profile_id not in mapping:
                        raise ValueError(f"Missing parent {node.parent_profile_id} for {root_id}/{pid}")
                    if node.branch_time is None:
                        raise ValueError(f"Child node {root_id}/{pid} missing branch_time")
                    branch_tol = _branch_time_snap_tolerance(float(node.branch_time))
                    stop_tol = _branch_time_snap_tolerance(float(node.stop_time))
                    node_start = float(node.t[0])
                    node_end = float(node.t[-1])
                    if node.branch_time < node_start - branch_tol or node.branch_time > node_end + branch_tol:
                        raise ValueError(
                            f"Invalid branch_time for {root_id}/{pid}: "
                            f"branch_time={node.branch_time}, profile_range=[{node_start}, {node_end}], tol={branch_tol}"
                        )
                    if node.stop_time < node.branch_time - stop_tol or node.stop_time > node_end + stop_tol:
                        raise ValueError(
                            f"Invalid branch_end_time/stop_time for {root_id}/{pid}: "
                            f"branch_time={node.branch_time}, stop_time={node.stop_time}, "
                            f"profile_end={node_end}, tol={stop_tol}"
                        )
                    parent = mapping[node.parent_profile_id]
                    parent_start = float(parent.t[0])
                    parent_end = float(parent.t[-1])
                    if node.branch_time < parent_start - branch_tol or node.branch_time > parent_end + branch_tol:
                        raise ValueError(
                            f"branch_time out of parent range for {root_id}/{pid}: "
                            f"branch_time={node.branch_time}, parent_range=[{parent_start}, {parent_end}], "
                            f"tol={branch_tol}"
                        )
                    depths[pid] = depth(node.parent_profile_id) + 1
                return depths[pid]

            for pid in mapping:
                d = depth(pid)
                out.append(BranchNode(**{**mapping[pid].__dict__, "depth": d}))
        return out

    def _toposort_branch_jobs(self, nodes: list[BranchNode]) -> list[BranchNode]:
        by_key = {(n.root_id, n.profile_id): n for n in nodes}
        children: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        indeg: dict[tuple[str, str], int] = {k: 0 for k in by_key}
        for n in nodes:
            k = (n.root_id, n.profile_id)
            if n.parent_profile_id is None:
                continue
            pk = (n.root_id, n.parent_profile_id)
            children[pk].append(k)
            indeg[k] += 1

        q = deque(sorted([k for k, d in indeg.items() if d == 0]))
        out = []
        while q:
            k = q.popleft()
            out.append(by_key[k])
            for ck in sorted(children.get(k, [])):
                indeg[ck] -= 1
                if indeg[ck] == 0:
                    q.append(ck)
        if len(out) != len(nodes):
            raise RuntimeError("Cycle detected in branch DAG")
        return out

    def _write_root_profile_mat(self, job: BranchNode) -> Path:
        out = self.generated_profiles_abs / f"{job.root_id}__{job.profile_id}__full.mat"
        table = np.column_stack([job.t, job.theta_deg, job.v_deg_s, job.a_deg_s2])
        savemat(str(out), {self.cfg.table_name: table})
        return out

    def _write_full_profile_mat(self, job: BranchNode) -> Path:
        out = self.generated_profiles_abs / f"{job.root_id}__{job.profile_id}__full.mat"
        table = np.column_stack([job.t, job.theta_deg, job.v_deg_s, job.a_deg_s2])
        savemat(str(out), {self.cfg.table_name: table})
        return out

    def _simulate_root_job(self, job: BranchNode) -> tuple[bool, dict]:
        profile_mat = self._write_root_profile_mat(job)
        result_base = f"{job.root_id}__{job.profile_id}__root"
        ok, t_sim, result_file = self._simulate_model_call(profile_mat_path=profile_mat, start_time=0.0, stop_time=job.stop_time, result_base=result_base)
        if not ok:
            return False, {"status": "FAIL_SIMULATE_ROOT", "simulate_s": t_sim, "result_base": result_base, "generated_profile_mat": str(profile_mat)}
        return True, {"status": "OK", "simulate_s": t_sim, "result_base": result_base, "generated_profile_mat": str(profile_mat), "dymola_result_file": result_file}

    def _simulate_branch_job(
        self,
        job: BranchNode,
        parent_result_file: str,
        parent_generated_profile_mat: str | None = None,
    ) -> tuple[bool, dict]:
        child_full_mat = self._write_full_profile_mat(job)
        print(
            f"[BRANCH] root={job.root_id} profile={job.profile_id} parent={job.parent_profile_id} "
            f"branch_time={job.branch_time:.6f} child_profile_mat={child_full_mat.name}"
        )
        parent_path = Path(parent_result_file)
        if not parent_path.is_absolute():
            parent_path = self.out_dir_abs / parent_path
        parent_exists = parent_path.exists()
        child_profile_exists = child_full_mat.exists()
        print(f"[BRANCH] parent_result_exists={parent_exists} child_profile_mat_exists={child_profile_exists}")
        import_ok = self.dymola.importInitialResult(self._to_dymola_path(parent_path), float(job.branch_time))
        if not import_ok:
            return False, {"status": "FAIL_IMPORT_INITIAL_RESULT", "generated_profile_mat": str(child_full_mat)}
        print("[BRANCH] importInitialResult=OK")

        if self.cfg.keep_dsfinal_for_debug:
            dsin = self.out_dir_abs / "dsin.txt"
            if dsin.exists():
                shutil.copy2(dsin, self.logs_abs / f"{job.root_id}__{job.profile_id}__after_import_dsin.txt")

        child_profile_rel = os.path.relpath(child_full_mat, self.out_dir_abs).replace("\\", "/")
        context_log = self.logs_abs / f"{job.root_id}__{job.profile_id}__branch_context.log"
        context_log.write_text(
            "\n".join([
                f"parent_result_file={parent_result_file}",
                f"parent_result_path_resolved={parent_path}",
                f"parent_result_exists={parent_exists}",
                f"child_profile_mat={child_full_mat}",
                f"child_profile_mat_exists={child_profile_exists}",
                f"child_profile_rel={child_profile_rel}",
                f"branch_time={job.branch_time}",
                "profile_rebind_mode=mat_copy_only",
            ]),
            encoding="utf-8",
        )
        effective_profile_mat = str(child_full_mat)

        # Standard/only profile rebind path: keep imported dynamic state
        # (simulateModel("")) and update the already-bound MAT in-place.
        if parent_generated_profile_mat:
            fallback_target = Path(parent_generated_profile_mat)
            try:
                parent_input_backup = self.generated_profiles_abs / (
                    f"{job.root_id}__{job.profile_id}__parent_input_before_rebind.mat"
                )
                if fallback_target.exists():
                    shutil.copy2(fallback_target, parent_input_backup)
                    print(
                        f"[BRANCH] preserved prior parent-bound input MAT: "
                        f"{fallback_target.name} -> {parent_input_backup.name}"
                    )
                shutil.copy2(child_full_mat, fallback_target)
                effective_profile_mat = str(fallback_target)
                print(f"[BRANCH] profile rebind via MAT copy: {child_full_mat.name} -> {fallback_target.name}")
            except Exception as exc:
                (self.logs_abs / f"{job.root_id}__{job.profile_id}__setvars_error.log").write_text(
                    f"MAT-copy rebind failed: copy2({child_full_mat}, {fallback_target})\n\n{exc}",
                    encoding="utf-8",
                )
                return False, {
                    "status": "FAIL_SET_PROFILE_VARIABLES",
                    "generated_profile_mat": str(child_full_mat),
                }
        else:
            (self.logs_abs / f"{job.root_id}__{job.profile_id}__setvars_error.log").write_text(
                "No parent_generated_profile_mat was provided for MAT-copy rebind.",
                encoding="utf-8",
            )
            return False, {
                "status": "FAIL_SET_PROFILE_VARIABLES",
                "generated_profile_mat": str(child_full_mat),
            }

        if self.cfg.keep_dsfinal_for_debug:
            dsin = self.out_dir_abs / "dsin.txt"
            if dsin.exists():
                shutil.copy2(dsin, self.logs_abs / f"{job.root_id}__{job.profile_id}__before_continue_dsin.txt")

        result_base = f"{job.root_id}__{job.profile_id}__branch"
        ok, t_sim, result_file, sim_attempt = self._simulate_continued_default_model(
            start_time=float(job.branch_time),
            stop_time=float(job.stop_time),
            result_base=result_base,
        )
        if not ok:
            (self.logs_abs / f"{job.root_id}__{job.profile_id}__setvars_error.log").write_text(
                f"Continuation attempt failed: {sim_attempt}\n\n{self.dymola.getLastErrorLog()}",
                encoding="utf-8",
            )
            return False, {"status": "FAIL_SIMULATE_BRANCH", "simulate_s": t_sim, "result_base": result_base, "generated_profile_mat": str(child_full_mat)}

        return True, {
            "status": "OK", "simulate_s": t_sim, "result_base": result_base,
            "generated_profile_mat": effective_profile_mat, "restart_source_result": parent_result_file,
            "dymola_result_file": result_file,
        }

    def _save_branch_outputs(self, job: BranchNode, result_cols: list[np.ndarray]) -> tuple[Path, Path, float]:
        t0 = time()
        csv_out = self.branched_results_abs / f"results_{job.root_id}__{job.profile_id}.csv"
        output_cols = [np.asarray(col, dtype=float).copy() for col in result_cols]
        if output_cols and job.branch_time is not None:
            t = output_cols[0]
            if t.size:
                branch_time = float(job.branch_time)
                delta = abs(float(t[0]) - branch_time)
                tol = _branch_time_snap_tolerance(branch_time)
                if delta <= tol:
                    t[0] = branch_time
                else:
                    raise ValueError(
                        f"Branched result {job.root_id}/{job.profile_id} starts at t={float(t[0])}, "
                        f"which differs from branch_time={branch_time} by {delta} (> {tol})."
                    )
        with open(csv_out, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(self.cfg.vars_to_pull)
            w.writerows(zip(*output_cols))
        mat_out = csv_out.with_suffix(".mat")
        savemat(str(mat_out), {"table": np.column_stack(output_cols), "columns": np.array(self.cfg.vars_to_pull, dtype=object)})
        return csv_out, mat_out, (time() - t0)

    def _log_branch_failure(self, job: BranchNode, status: str, *, detail: str = "") -> None:
        """Emit immediate console + file diagnostics for branch failures."""
        msg = f"[BRANCH-FAIL] {job.root_id}/{job.profile_id} status={status}"
        if detail:
            msg += f" detail={detail}"
        print(msg)
        try:
            err = str(self.dymola.getLastErrorLog())
        except Exception:
            err = ""
        log_path = self.logs_abs / f"{job.root_id}__{job.profile_id}__{status.lower()}.log"
        payload = msg + ("\n\n" + err if err else "")
        log_path.write_text(payload, encoding="utf-8")

    def run_branched_mat(self):
        assert self.dymola is not None, "Call start() first."
        nodes = self._load_branched_manifest_tree(self.profiles_dir_abs)
        jobs = self._toposort_branch_jobs(nodes)
        jobs_by_root: dict[str, list[BranchNode]] = defaultdict(list)
        for job in jobs:
            jobs_by_root[job.root_id].append(job)

        batch_t0 = time()
        root_ids = sorted(jobs_by_root.keys())
        for i, root_id in enumerate(root_ids):
            if i > 0:
                print(f"[ROOT-RESET] restarting Dymola session for root={root_id}")
                self.close()
                self.start()

            result_by_key: dict[tuple[str, str], str] = {}
            generated_profile_by_key: dict[tuple[str, str], str] = {}
            for job in jobs_by_root[root_id]:
                run_t0 = time()
                key = (job.root_id, job.profile_id)

                if job.parent_profile_id is None:
                    ok, rec = self._simulate_root_job(job)
                    if not ok:
                        print(f"[ROOT-FAIL] {job.root_id}/{job.profile_id} status={rec.get('status', 'FAIL')}")
                        self._append_summary(root_id=job.root_id, profile_id=job.profile_id, parent_profile_id="", depth=job.depth,
                                             run_type="root_full", stop_time_s=job.stop_time, total_run_s=time()-run_t0, **rec)
                        continue
                    cols, t_extract = self._read_result_columns(rec["dymola_result_file"], job.stop_time)
                    csv_out, mat_out, t_write = self._save_branch_outputs(job, cols)
                    result_by_key[key] = rec["dymola_result_file"]
                    generated_profile_by_key[key] = rec.get("generated_profile_mat", "")
                    self._append_summary(root_id=job.root_id, profile_id=job.profile_id, parent_profile_id="", depth=job.depth,
                                         run_type="root_full", branch_time_s=None, stop_time_s=job.stop_time, extract_s=t_extract,
                                         write_s=t_write, total_run_s=time()-run_t0, result_csv_out=csv_out.name,
                                         result_mat_out=mat_out.name, **rec)
                    continue

                parent_key = (job.root_id, job.parent_profile_id)
                if parent_key not in result_by_key:
                    self._log_branch_failure(job, "FAIL_PARENT_RESULT_MISSING", detail=f"missing parent result for {parent_key}")
                    self._append_summary(root_id=job.root_id, profile_id=job.profile_id, parent_profile_id=job.parent_profile_id,
                                         branch_time_s=job.branch_time, depth=job.depth, run_type="branch_restart", status="FAIL_PARENT_RESULT_MISSING",
                                         stop_time_s=job.stop_time, total_run_s=time()-run_t0)
                    continue

                ok, rec = self._simulate_branch_job(
                    job,
                    result_by_key[parent_key],
                    parent_generated_profile_mat=generated_profile_by_key.get(parent_key),
                )
                if not ok:
                    self._log_branch_failure(job, rec.get("status", "FAIL_BRANCH"))
                    self._append_summary(root_id=job.root_id, profile_id=job.profile_id, parent_profile_id=job.parent_profile_id,
                                         branch_time_s=job.branch_time, depth=job.depth, run_type="branch_restart", stop_time_s=job.stop_time,
                                         total_run_s=time()-run_t0, **rec)
                    continue

                try:
                    cols_child, t_extract = self._read_result_columns(rec["dymola_result_file"], job.stop_time)
                except Exception as exc:
                    status = "FAIL_MISSING_VARS"
                    self._log_branch_failure(job, status, detail=str(exc))
                    self._append_summary(
                        root_id=job.root_id,
                        profile_id=job.profile_id,
                        parent_profile_id=job.parent_profile_id,
                        branch_time_s=job.branch_time,
                        depth=job.depth,
                        run_type="branch_restart",
                        stop_time_s=job.stop_time,
                        total_run_s=time() - run_t0,
                        status=status,
                        generated_profile_mat=rec.get("generated_profile_mat", ""),
                        restart_source_result=rec.get("restart_source_result", ""),
                        dymola_result_file=rec.get("dymola_result_file", ""),
                        result_base=rec.get("result_base", ""),
                    )
                    continue
                csv_out, mat_out, t_write = self._save_branch_outputs(job, cols_child)
                result_by_key[key] = rec["dymola_result_file"]
                generated_profile_by_key[key] = rec.get("generated_profile_mat", "")

                self._append_summary(root_id=job.root_id, profile_id=job.profile_id, parent_profile_id=job.parent_profile_id,
                                     branch_time_s=job.branch_time, depth=job.depth, run_type="branch_restart", stop_time_s=job.stop_time,
                                     extract_s=t_extract, merge_s=0.0, write_s=t_write, total_run_s=time()-run_t0,
                                     result_csv_out=csv_out.name, result_mat_out=mat_out.name, **rec)

        self._t_total_wall = time() - batch_t0
        self._cleanup_out_dir_branched()

    # Backward-compatible alias for callers still using the old method name.
    def run_branched_hdf5(self):
        self.run_branched_mat()

    def _print_timing_summary(self, ran_count: int):
        def stats(x):
            return (float(np.mean(x)), float(np.min(x)), float(np.max(x))) if x else (0.0, 0.0, 0.0)

        sim_avg, sim_min, sim_max = stats(self._sim_times)
        ext_avg, ext_min, ext_max = stats(self._extract_times)
        wr_avg, wr_min, wr_max = stats(self._write_times)
        merge_avg, merge_min, merge_max = stats(self._merge_times)
        print("--- Timing summary ---")
        print(f"startup={self._t_startup:.2f}s openModel={self._t_openmodel:.2f}s wall={self._t_total_wall:.2f}s")
        if ran_count > 0 and self._t_total_wall > 0:
            print(f"throughput={60.0 * ran_count / self._t_total_wall:.2f} runs/min")
        print(f"simulate avg/min/max: {sim_avg:.4f}/{sim_min:.4f}/{sim_max:.4f}")
        print(f"extract  avg/min/max: {ext_avg:.4f}/{ext_min:.4f}/{ext_max:.4f}")
        print(f"merge    avg/min/max: {merge_avg:.4f}/{merge_min:.4f}/{merge_max:.4f}")
        print(f"write    avg/min/max: {wr_avg:.4f}/{wr_min:.4f}/{wr_max:.4f}")


if __name__ == "__main__":
    cfg = BatchConfig(
        profiles_dir=r"../../../tests/test_batch",
        out_dir=r"../../../tests/test_results",
        output_interval=0.1,
        skip_existing=True,
    )

    runner = DymolaBatchRunner(cfg)
    runner.start()
    try:
        if cfg.profile_mode == "branched_mat":
            runner.run_branched_mat()
        else:
            runner.run_all()
    finally:
        runner.close()
