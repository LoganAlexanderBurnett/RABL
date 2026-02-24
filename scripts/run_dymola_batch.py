import importlib.util
import re
from pathlib import Path
from time import time

import numpy as np
from rabl.interface import BatchConfig, DymolaBatchRunner
from rabl.variography.DrumVariography import DrumProfileGenerator



def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"Missing config file: {config_path}")

    spec = importlib.util.spec_from_file_location("variography_batch_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load config file: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return {
        "baseline_angle_deg": getattr(config_module, "BASELINE_ANGLE_DEG", None),
        "t_grid_duration": getattr(config_module, "T_GRID_DURATION", None),
        "t_grid_intervals": getattr(config_module, "T_GRID_INTERVALS", None),
        "kernel": getattr(config_module, "KERNEL", None),
        "ell": getattr(config_module, "ELL", None),
        "sigma_theta_target": getattr(config_module, "SIGMA_THETA_TARGET", None),
        "nugget_v_deg2_s2": getattr(config_module, "NUGGET_V_DEG2_S2", None),
        "batch_number": getattr(config_module, "BATCH_NUMBER", None),
        "num_profiles": getattr(config_module, "NUM_PROFILES", None),
        "seed": getattr(config_module, "SEED", None),
        "dymola_output_interval": getattr(config_module, "DYMOLA_OUTPUT_INTERVAL", 0.1),
    }


def _validate_config(config: dict) -> dict:
    if not isinstance(config["t_grid_duration"], (int, float)) or config["t_grid_duration"] <= 0:
        raise SystemExit("T_GRID_DURATION must be a positive number in config.py.")
    if not isinstance(config["t_grid_intervals"], int) or config["t_grid_intervals"] <= 0:
        raise SystemExit("T_GRID_INTERVALS must be a positive integer in config.py.")
    if not isinstance(config["kernel"], str):
        raise SystemExit("KERNEL must be a string in config.py.")
    if not isinstance(config["ell"], (int, float)) or config["ell"] <= 0:
        raise SystemExit("ELL must be a positive number in config.py.")
    if not isinstance(config["sigma_theta_target"], (int, float)) or config["sigma_theta_target"] <= 0:
        raise SystemExit("SIGMA_THETA_TARGET must be a positive number in config.py.")
    if not isinstance(config["nugget_v_deg2_s2"], (int, float)) or config["nugget_v_deg2_s2"] < 0:
        raise SystemExit("NUGGET_V_DEG2_S2 must be a non-negative number in config.py.")
    if not isinstance(config["batch_number"], int) or config["batch_number"] < 0:
        raise SystemExit("BATCH_NUMBER must be a non-negative integer in config.py.")
    if not isinstance(config["num_profiles"], int) or config["num_profiles"] <= 0:
        raise SystemExit("NUM_PROFILES must be a positive integer in config.py.")
    if not isinstance(config["baseline_angle_deg"], (int, float)):
        raise SystemExit("BASELINE_ANGLE_DEG must be a number in config.py.")
    if not (0.0 < float(config["baseline_angle_deg"]) < 180.0):
        raise SystemExit("BASELINE_ANGLE_DEG must be > 0 and < 180 in config.py.")
    if not isinstance(config["seed"], int):
        raise SystemExit("SEED must be an integer in config.py.")
    if not isinstance(config["dymola_output_interval"], (int, float)) or config["dymola_output_interval"] <= 0:
        raise SystemExit("DYMOLA_OUTPUT_INTERVAL must be a positive number in config.py.")

    return config


def _find_latest_profile_index(variography_root: Path) -> int:
    if not variography_root.exists():
        return 0

    pattern = re.compile(r"drum_profile_(\d{5})")
    indices = set()

    for batch_dir in variography_root.iterdir():
        if not batch_dir.is_dir():
            continue
        for profile in batch_dir.glob("drum_profile_*.csv"):
            match = pattern.search(profile.stem)
            if match:
                indices.add(int(match.group(1)))
        for profile in batch_dir.glob("drum_profile_*.mat"):
            match = pattern.search(profile.stem)
            if match:
                indices.add(int(match.group(1)))

    return max(indices, default=0)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.py"
    config = _validate_config(_load_config(config_path))

    repo_root = script_dir.parent
    variography_root = repo_root / "outputs" / "variography_profiles"
    sim_root = repo_root / "outputs" / "sim_profiles"

    batch_name = f"batch_{config['batch_number']:04d}"
    variography_dir = variography_root / batch_name
    sim_dir = sim_root / batch_name

    if variography_dir.exists() or sim_dir.exists():
        raise SystemExit(f"Batch folder already exists: {batch_name}. Use a new batch number.")

    start_index = _find_latest_profile_index(variography_root) + 1

    t_grid = np.linspace(
        0.0,
        config["t_grid_duration"],
        config["t_grid_intervals"] + 1,
    )

    variography_dir.mkdir(parents=True, exist_ok=False)
    sim_dir.mkdir(parents=True, exist_ok=False)

    print("Defining variogram...")
    generator = DrumProfileGenerator(
        kernel=config["kernel"],
        ell=config["ell"],
        sill_v_deg2_s2=1.0,
        nugget_v_deg2_s2=config["nugget_v_deg2_s2"],
        jitter_frac=1e-10,
        cond_jitter=1e-10,
    )
    generator.solve_params_for_sigma_theta(
        t_grid=t_grid,
        sigma_theta_target=config["sigma_theta_target"],
        ell=config["ell"],
        nugget=config["nugget_v_deg2_s2"],
        update_instance=True,
    )

    num_profiles = config["num_profiles"]
    print(f"Generating {num_profiles} profiles...")
    start_gen = time()
    profiles = generator.generate(
        t_grid,
        n_realizations=num_profiles,
        baseline_angle_deg=config["baseline_angle_deg"],
        seed=config["seed"],
    )
    end_gen = time()
    avg_gen_time = (end_gen - start_gen) / num_profiles
    print(
        f"Generated {num_profiles} in {end_gen - start_gen:.3f} seconds "
        f"(Average {avg_gen_time:.3f} sec/profile)"
    )

    for idx, profile in enumerate(profiles, start=start_index):
        csv_path = variography_dir / f"drum_profile_{idx:05d}.csv"
        mat_path = variography_dir / f"drum_profile_{idx:05d}.mat"
        profile.save_csv(csv_path)
        profile.save_mat(mat_path)

    print(f"Saved generated profiles in {variography_dir}")

    cfg = BatchConfig(
        profiles_dir=str(variography_dir),
        out_dir=str(sim_dir),
        output_interval=float(config["dymola_output_interval"]),
    )

    runner = DymolaBatchRunner(cfg)
    runner.summary_csv = f"batch_summary_sigmatheta{config['sigma_theta_target']}-ell{config['ell']}-seed{config['seed']}.csv"
    runner.start()
    try:
        runner.run_all()
    finally:
        runner.close()

if __name__ == "__main__":
    main()
