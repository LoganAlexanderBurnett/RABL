import argparse
import importlib.util
import re
from pathlib import Path
from time import time

from scipy.io import savemat


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"Missing config file: {config_path}")

    spec = importlib.util.spec_from_file_location("variography_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load config file: {config_path}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return {
        "baseline_angle_deg": getattr(config_module, "BASELINE_ANGLE_DEG", None),
        "ell": getattr(config_module, "ELL", None),
        "grid_intervals": getattr(config_module, "GRID_INTERVALS", None),
        "grid_length": getattr(config_module, "GRID_LENGTH", None),
        "kernel": getattr(config_module, "KERNEL", None),
        "nugget_v_deg2_s2": getattr(config_module, "NUGGET_V_DEG2_S2", None),
        "num_profiles": getattr(config_module, "NUM_PROFILES", None),
        "seed": getattr(config_module, "SEED", None),
    }


def _validate_config(config: dict) -> dict:
    if not isinstance(config["num_profiles"], int) or config["num_profiles"] <= 0:
        raise SystemExit("NUM_PROFILES must be a positive integer in config.py.")
    if not isinstance(config["kernel"], str):
        raise SystemExit("KERNEL must be a string in config.py.")
    if not isinstance(config["ell"], (int, float)) or config["ell"] <= 0:
        raise SystemExit("ELL must be a positive number in config.py.")
    if not isinstance(config["nugget_v_deg2_s2"], (int, float)) or config["nugget_v_deg2_s2"] < 0:
        raise SystemExit("NUGGET_V_DEG2_S2 must be a non-negative number in config.py.")
    if not isinstance(config["baseline_angle_deg"], (int, float)):
        raise SystemExit("BASELINE_ANGLE_DEG must be a number in config.py.")
    if not (0.0 < float(config["baseline_angle_deg"]) < 180.0):
        raise SystemExit("BASELINE_ANGLE_DEG must be > 0 and < 180 in config.py.")
    if not isinstance(config["grid_length"], (int, float)) or config["grid_length"] <= 0:
        raise SystemExit("GRID_LENGTH must be a positive number in config.py.")
    if not isinstance(config["grid_intervals"], int) or config["grid_intervals"] <= 0:
        raise SystemExit("GRID_INTERVALS must be a positive integer in config.py.")
    if not isinstance(config["seed"], int):
        raise SystemExit("SEED must be an integer in config.py.")
    return config


def _next_batch_dir(output_root: Path) -> Path:
    batch_pattern = re.compile(r"batch_(\d{4})$")
    batch_indices = []
    for entry in output_root.iterdir():
        if not entry.is_dir():
            continue
        match = batch_pattern.match(entry.name)
        if match:
            batch_indices.append(int(match.group(1)))

    next_index = max(batch_indices, default=-1) + 1
    return output_root / f"batch_{next_index:04d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate drum profiles and save CSV/MAT output.")
    parser.add_argument(
        "--num-profiles",
        type=int,
        default=None,
        help="Override NUM_PROFILES from config.py.",
    )
    return parser.parse_args()


def main() -> None:
    import numpy as np

    from microreactor_dynamics.variography.DrumVariography import DrumProfileGenerator

    args = _parse_args()

    config_path = Path(__file__).resolve().parent / "config.py"
    config = _validate_config(_load_config(config_path))
    num_profiles = args.num_profiles if args.num_profiles is not None else config["num_profiles"]
    if not isinstance(num_profiles, int) or num_profiles <= 0:
        raise SystemExit("--num-profiles must be a positive integer.")

    t_grid = np.linspace(0.0, config["grid_length"], config["grid_intervals"] + 1)

    output_root = Path(__file__).resolve().parent / "generated_profiles"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = _next_batch_dir(output_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Defining variogram...")
    generator = DrumProfileGenerator(
        kernel=config["kernel"],
        ell=config["ell"],
        sill_v_deg2_s2=0.1,
        nugget_v_deg2_s2=config["nugget_v_deg2_s2"],
        jitter_frac=1e-10,
        cond_jitter=1e-10,
    )

    print(f"Generating {num_profiles} profiles...")
    start_gen = time()
    profiles = generator.generate(
        t_grid,
        n_realizations=num_profiles,
        baseline_angle_deg=config["baseline_angle_deg"],
        seed=config["seed"],
    )
    end_gen = time()
    avg_gen_time = (end_gen-start_gen)/num_profiles
    print(f"Generated {num_profiles} in {end_gen-start_gen:.3f} seconds (Average {avg_gen_time:.3f} sec/profile)")

    for idx, profile in enumerate(profiles, start=1):
        csv_path = output_dir / f"drum_profile_{idx:05d}.csv"
        mat_path = output_dir / f"drum_profile_{idx:05d}.mat"
        profile.save_csv(csv_path)
    
        # Ensure 1D vectors (no Nx1 arrays)
        t     = np.asarray(profile.t).squeeze()
        theta = np.asarray(profile.theta_deg).squeeze()
        v     = np.asarray(profile.v_deg_s).squeeze()
        a     = np.asarray(profile.a_deg_s2).squeeze()
    
        # Dymola CombiTimeTable expects: col 1 time, col 2 angle, etc.
        table = np.column_stack([t, theta, v, a])
    
        savemat(
            mat_path,
            {
                "profile": table
            },
            format="4"  # <-- MATLAB v4 most compatible with Dymola tables
        )
    print(f"Saved generated profiles as .CSV and .MAT in {output_dir}")


if __name__ == "__main__":
    main()
