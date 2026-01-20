import os
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from time import time

import numpy as np
from scipy.io import loadmat
from dymola.dymola_interface import DymolaInterface


@dataclass(frozen=True)
class BatchConfig:
    # Paths RELATIVE to this script file for portability
    package_mo: str = r"../../modelica/MicroreactorPK/package.mo"
    model_name: str = "MicroreactorPK.Experiments.RunOneProfile"
    profiles_dir: str = r"../../../outputs/variography/test_batch"
    out_dir: str = r"../../../outputs/sim/test_batch"

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
        "dc[1]", "dc[2]", "dc[3]", "dc[4]", "dc[5]", "dc[6]",
        "P_MW", "n", "dn", "rho", "rho_dollars",
        "m_dot_steam", "Q_to_steam",
    )


class DymolaBatchRunner:
    def __init__(self, config: BatchConfig):
        self.cfg = config
        self.script_dir = Path(__file__).resolve().parent
        
        self.package_abs = (self.script_dir / self.cfg.package_mo).resolve()
        self.profiles_dir_abs = (self.script_dir / self.cfg.profiles_dir).resolve()
        self.out_dir_abs = (self.script_dir / self.cfg.out_dir).resolve()

        self.dymola = None
        self._vars_verified = False

        # Timing aggregates
        self._t_startup = None
        self._t_openmodel = None
        self._t_total_wall = None
        self._sim_times = []
        self._extract_times = []
        self._write_times = []
        self._matread_times = []
        self._total_run_times = []

        self.summary_csv = None

    # -----------------------------
    # Helpers
    # -----------------------------
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

    def _cleanup_out_dir(self) -> None:
        allowed = {".csv", ".mat", ".log"}
        for entry in self.out_dir_abs.iterdir():
            if entry.is_dir():
                continue
            if entry.suffix.lower() not in allowed:
                entry.unlink()

    def infer_stop_time(self, profile_mat_path: Path) -> tuple[float, float]:
        """
        Returns: (stop_time, mat_read_seconds)
        """
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

    # -----------------------------
    # Dymola lifecycle
    # -----------------------------
    def start(self):
        self.out_dir_abs.mkdir(parents=True, exist_ok=True)
        self.summary_csv = self.out_dir_abs / "batch_summary.csv"

        print("Starting Dymola instance...")
        t0 = time()
        self.dymola = DymolaInterface()
        self._t_startup = time() - t0
        print(f"Dymola active (startup {self._t_startup:.2f} s)")

        print("Loading package:", self.package_abs)
        t1 = time()
        self.dymola.openModel(self._to_dymola_path(self.package_abs))
        self._t_openmodel = time() - t1
        print(f"openModel time: {self._t_openmodel:.2f} s")

        # Force output location + relative profile paths from here
        self.dymola.cd(self._to_dymola_path(self.out_dir_abs))
        self._vars_verified = False

        # Create summary file header if it doesn't exist
        if not self.summary_csv.exists():
            with open(self.summary_csv, "w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow([
                    "profile_mat",
                    "csv_out",
                    "status",
                    "stop_time_s",
                    "matread_s",
                    "simulate_s",
                    "extract_s",
                    "write_s",
                    "total_run_s",
                    "result_base"
                ])

    def close(self):
        if self.dymola is None:
            return
        try:
            self.dymola.close()
        finally:
            self.dymola = None


    # -----------------------------
    # Run one profile
    # -----------------------------
    def simulate_one(self, profile_path: Path) -> tuple[bool, str]:
        assert self.dymola is not None, "Call start() first."

        run_t0 = time()

        try:
            csv_name = self.results_csv_name(profile_path)
            csv_out = self.out_dir_abs / csv_name

            # Skip existing
            if self.cfg.skip_existing and csv_out.exists():
                self._append_summary(profile_path, csv_out, "SKIP", None, 0.0, 0.0, 0.0, 0.0, time()-run_t0, "")
                return True, f"SKIP (exists): {csv_out.name}"

            idx = self._profile_index_from_name(profile_path)
            if idx is None:
                idx = 0

            result_base = f"run_{idx:05d}" if idx > 0 else f"run_{profile_path.stem}"

            # profileFile must be relative to OUT_DIR (Dymola cwd)
            profile_rel = os.path.relpath(profile_path, self.out_dir_abs).replace("\\", "/")

            # Stop time from MAT
            stop_time, t_matread = self.infer_stop_time(profile_path)
            self._matread_times.append(t_matread)

            model_call = (
                f'{self.cfg.model_name}('
                f'profileFile="{profile_rel}", '
                f'tableName="{self.cfg.table_name}", '
                f'angleColumn={self.cfg.angle_col}, '
                f'velColumn={self.cfg.vel_col}, '
                f'accColumn={self.cfg.acc_col}'
                f')'
            )

            # Simulate
            t_sim0 = time()
            ok = self.dymola.simulateModel(
                model_call,
                startTime=0.0,
                stopTime=stop_time,
                resultFile=result_base,
                outputInterval=self.cfg.output_interval
            )
            t_sim = time() - t_sim0
            self._sim_times.append(t_sim)

            if not ok:
                log = self.dymola.getLastErrorLog()
                err_path = self.out_dir_abs / f"{result_base}_dymola_error.log"
                err_path.write_text(str(log), encoding="utf-8")
                total = time() - run_t0
                self._total_run_times.append(total)
                self._append_summary(profile_path, csv_out, "FAIL", stop_time, t_matread, t_sim, 0.0, 0.0, total, result_base)
                return False, f"FAILED: wrote {err_path.name}"

            res = self.dymola.getLastResultFileName()
            rows = self.dymola.readTrajectorySize(res)

            # Verify variables once
            if not self._vars_verified:
                available = set(self.dymola.readTrajectoryNames(res))
                missing = [v for v in self.cfg.vars_to_pull if v not in available]
                if missing:
                    total = time() - run_t0
                    self._total_run_times.append(total)
                    self._append_summary(profile_path, csv_out, "FAIL_MISSING_VARS", stop_time, t_matread, t_sim, 0.0, 0.0, total, result_base)
                    return False, f"FAILED: Missing variables in result: {missing}"
                self._vars_verified = True

            # Extract
            t_ext0 = time()
            data = self.dymola.readTrajectory(res, list(self.cfg.vars_to_pull), rows)
            cols = [np.asarray(col, dtype=float) for col in data]

            if self.cfg.keep_last_duplicate_time:
                cols = self._dedupe_time_keep_last(cols)

            t_extract = time() - t_ext0
            self._extract_times.append(t_extract)

            # Write
            t_write0 = time()
            with open(csv_out, "w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(self.cfg.vars_to_pull)
                w.writerows(zip(*cols))
            t_write = time() - t_write0
            self._write_times.append(t_write)

            total = time() - run_t0
            self._total_run_times.append(total)

            self._append_summary(profile_path, csv_out, "OK", stop_time, t_matread, t_sim, t_extract, t_write, total, result_base)

            return True, str(csv_out)
        finally:
            self._cleanup_out_dir()

    def _append_summary(self, profile_path, csv_out, status, stop_time, t_matread, t_sim, t_extract, t_write, total, result_base):
        with open(self.summary_csv, "a", newline="") as fp:
            w = csv.writer(fp)
            w.writerow([
                profile_path.name,
                csv_out.name,
                status,
                "" if stop_time is None else f"{stop_time:.6f}",
                f"{t_matread:.6f}",
                f"{t_sim:.6f}",
                f"{t_extract:.6f}",
                f"{t_write:.6f}",
                f"{total:.6f}",
                result_base
            ])

    # -----------------------------
    # Batch run
    # -----------------------------
    def run_all(self):
        assert self.dymola is not None, "Call start() first."

        profiles = sorted(self.profiles_dir_abs.glob("drum_profile_*.mat"))
        if not profiles:
            raise RuntimeError(f"No drum_profile_*.mat found in: {self.profiles_dir_abs}")

        print(f"Profiles found: {len(profiles)}")
        print(f"Output dir: {self.out_dir_abs}")
        print(f"Summary CSV: {self.summary_csv.name}")

        batch_t0 = time()
        ran = skipped = failed = 0

        for i, p in enumerate(profiles, start=1):
            ok, msg = self.simulate_one(p)

            if ok and msg.startswith("SKIP"):
                skipped += 1
                print(f"[{i}/{len(profiles)}] {msg}")
                continue

            if ok:
                ran += 1
                print(f"[{i}/{len(profiles)}] OK -> {Path(msg).name}")
            else:
                failed += 1
                print(f"[{i}/{len(profiles)}] {msg}")

        self._t_total_wall = time() - batch_t0

        print("\nDONE")
        print(f"Ran:     {ran}")
        print(f"Skipped: {skipped}")
        print(f"Failed:  {failed}")

        self._print_timing_summary(ran)

    def _print_timing_summary(self, ran_count: int):
        def stats(x):
            if not x:
                return (0.0, 0.0, 0.0)
            return (float(np.mean(x)), float(np.min(x)), float(np.max(x)))

        sim_avg, sim_min, sim_max = stats(self._sim_times)
        ext_avg, ext_min, ext_max = stats(self._extract_times)
        wr_avg,  wr_min,  wr_max  = stats(self._write_times)
        mr_avg,  mr_min,  mr_max  = stats(self._matread_times)
        tot_avg, tot_min, tot_max = stats(self._total_run_times)

        print("\n--- Timing summary ---")
        print(f"Dymola startup:   {self._t_startup:.2f} s")
        print(f"openModel:        {self._t_openmodel:.2f} s")
        print(f"Total wall time:  {self._t_total_wall:.2f} s")

        if ran_count > 0:
            runs_per_min = 60.0 * ran_count / self._t_total_wall if self._t_total_wall > 0 else 0.0
            print(f"Throughput:       {runs_per_min:.2f} runs/min")

        print("\nPer-run (seconds): avg / min / max")
        print(f"  MAT read:   {mr_avg:.4f} / {mr_min:.4f} / {mr_max:.4f}")
        print(f"  simulate:   {sim_avg:.4f} / {sim_min:.4f} / {sim_max:.4f}")
        print(f"  extract:    {ext_avg:.4f} / {ext_min:.4f} / {ext_max:.4f}")
        print(f"  write CSV:  {wr_avg:.4f} / {wr_min:.4f} / {wr_max:.4f}")
        print(f"  total run:  {tot_avg:.4f} / {tot_min:.4f} / {tot_max:.4f}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    cfg = BatchConfig(
        profiles_dir=r"../../../outputs/variography/batch_0000/",
        out_dir=r"../../../outputs/sim/test_batch",
        output_interval=0.1,
        skip_existing=True,
    )

    runner = DymolaBatchRunner(cfg)
    runner.start()
    try:
        runner.run_all()
    finally:
        runner.close()
