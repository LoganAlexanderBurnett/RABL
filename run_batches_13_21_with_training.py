from pathlib import Path
import re
import shutil
import subprocess
import sys

from rabl.paths import resolve_output_root


# =============================================================================
# User settings
# =============================================================================

START_BATCH = 13
END_BATCH = 21
LOOKBACK = 12

START_CUMULATIVE_BATCH = 1

DT = "0.4"
N_BINS = "10"

BATCH_SEEDS = {
    13: 888,
    14: 1234,
    15: 2345,
    16: 3456,
    17: 4567,
    18: 5678,
    19: 6789,
    20: 7890,
    21: 8901,
}

# Training settings copied from train_lstm_cumulative_batches.py defaults
TRAIN_BATCH_SIZE = 128
TRAIN_EPOCHS = 50
TRAIN_SEED = 123
TRAIN_MAX_PLOTS = 5

TRAIN_N_LSTM = 1
TRAIN_LSTM_HIDDEN = 64
TRAIN_LSTM_DROPOUT = 0.0

TRAIN_N_FC = 1
TRAIN_FC_HIDDEN = (384,)

TRAIN_LEARNING_RATE = 3e-4
PRELOAD_TRAINING_DATA = True


# =============================================================================
# Helper functions
# =============================================================================

def get_repo_root() -> Path:
    """
    Assumes this script is either:
    1. in the repo root, or
    2. inside the scripts/ directory.
    """
    here = Path(__file__).resolve().parent

    if (here / "scripts" / "config.py").exists():
        return here

    if here.name == "scripts" and (here / "config.py").exists():
        return here.parent

    raise FileNotFoundError(
        "Could not find scripts/config.py. "
        "Place this script in the repo root or inside scripts/."
    )


def ensure_python_import_paths(repo_root: Path) -> None:
    """
    Make sure local package imports work whether this runner is placed in the
    repo root or scripts/ directory.
    """
    candidate_paths = [
        repo_root,
        repo_root / "src",
        repo_root / "scripts",
    ]

    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def update_config(config_path: Path, batch_number: int, seed: int) -> None:
    """
    Update only BATCH_NUMBER, PLOT_BATCH_NUMBERS, and SEED in scripts/config.py.
    """
    text = config_path.read_text()

    text, n_batch = re.subn(
        r"^BATCH_NUMBER\s*=\s*\d+",
        f"BATCH_NUMBER       = {batch_number}",
        text,
        flags=re.MULTILINE,
    )

    text, n_plot = re.subn(
        r"^PLOT_BATCH_NUMBERS\s*=\s*\[[^\]]*\]",
        f"PLOT_BATCH_NUMBERS = [{batch_number}]",
        text,
        flags=re.MULTILINE,
    )

    text, n_seed = re.subn(
        r"^SEED\s*=\s*\d+",
        f"SEED               = {seed}",
        text,
        flags=re.MULTILINE,
    )

    if n_batch != 1:
        raise RuntimeError(
            f"Expected to replace exactly one BATCH_NUMBER entry, replaced {n_batch}."
        )

    if n_plot != 1:
        raise RuntimeError(
            f"Expected to replace exactly one PLOT_BATCH_NUMBERS entry, replaced {n_plot}."
        )

    if n_seed != 1:
        raise RuntimeError(
            f"Expected to replace exactly one SEED entry, replaced {n_seed}."
        )

    config_path.write_text(text)


def format_cumulative_batch_window(start_batch: int, end_batch: int) -> str:
    """
    Example:
        start_batch=1, end_batch=4
        -> 0001-0002-0003-0004
    """
    return "-".join(f"{b:04d}" for b in range(start_batch, end_batch + 1))


def padded_batch_numbers(start_batch: int, end_batch: int) -> list[str]:
    """
    Example:
        start_batch=1, end_batch=4
        -> ["0001", "0002", "0003", "0004"]
    """
    return [f"{i:04d}" for i in range(start_batch, end_batch + 1)]


def run_command(cmd: list[str], cwd: Path) -> None:
    """
    Run a command and stop immediately if it fails.
    """
    print("\n" + "=" * 100)
    print("Running command:")
    print(" ".join(str(c) for c in cmd))
    print("=" * 100 + "\n")

    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}:\n"
            f"{' '.join(str(c) for c in cmd)}"
        )


def find_scaled_h5(output_root: Path, batch_window: str, lookback: int) -> Path:
    """
    Find the scaled HDF5 file created by scale_lstm_dataset.py.

    This avoids hard-coding the train/val/test profile-count suffix, which changes
    as more cumulative batches are added.

    Example matched filenames:
        lstm_merged_batches_0001-...-0012_k12_minmax_train1100profiles_val1000profiles_test2000profiles.h5
        lstm_merged_batches_0001-...-0013_k12_minmax_train1200profiles_val1000profiles_test2000profiles.h5
    """
    scaled_dir = output_root / "datasets" / "scaled_split"
    pattern = f"lstm_merged_batches_{batch_window}_k{lookback}_minmax*.h5"

    matches = sorted(scaled_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No scaled HDF5 file found matching pattern:\n"
            f"{scaled_dir / pattern}"
        )

    if len(matches) > 1:
        newest = max(matches, key=lambda p: p.stat().st_mtime)

        print("\nWarning: multiple scaled files matched:")
        for match in matches:
            print(f"  {match}")
        print(f"Using most recently modified file:\n  {newest}")

        return newest

    return matches[0]


def train_cumulative_lstm_directly(
    h5_path: Path,
    out_dir: Path,
    batch_window: str,
) -> Path:
    """
    Integrated replacement for scripts/train_lstm_cumulative_batches.py.

    Unlike the standalone script, this function does not construct the HDF5 path
    internally. It uses the actual dynamically resolved scaled_h5 path produced
    by scale_lstm_dataset.py.
    """
    from rabl.machine_learning.lstm_pipeline import (
        LSTMPipeline,
        LSTMPipelineConfig,
        test_and_save_forecasts,
        clear_cuda_cache,
        compute_and_save_rolling_forecast_metrics,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"Training cumulative batch window: {batch_window}")
    print(f"h5_path:        {h5_path}")
    print(f"h5_path exists: {h5_path.exists()}")
    print(f"out_dir:        {out_dir}")
    print("=" * 100 + "\n")

    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find HDF5 file: {h5_path}")

    config = LSTMPipelineConfig(
        h5_path=h5_path,
        batch_size=TRAIN_BATCH_SIZE,
        seed=TRAIN_SEED,
        n_lstm=TRAIN_N_LSTM,
        lstm_hidden=TRAIN_LSTM_HIDDEN,
        lstm_dropout=TRAIN_LSTM_DROPOUT,
        n_fc=TRAIN_N_FC,
        fc_hidden=tuple(TRAIN_FC_HIDDEN),
        learning_rate=TRAIN_LEARNING_RATE,
        use_tqdm=False,
    )

    pipeline = LSTMPipeline(config)

    datasets = pipeline.build()
    pipeline.inspect()

    model, history, used_device = pipeline.train(
        epochs=TRAIN_EPOCHS,
        out_dir=out_dir,
        preload_train_to_device=PRELOAD_TRAINING_DATA,
        early_stopping_min_delta=1e-7,
        early_stopping_patience=10,
        restore_best_weights=True,
        step_lr_step_size=30,
        step_lr_gamma=0.5,
    )

    print(f"Finished training using device: {used_device}")

    test_and_save_forecasts(
        model,
        datasets["test_profile_ds"],
        out_dir=out_dir,
        state_dim=pipeline.config.state_dim,
        control_channel=pipeline.config.control_channel,
        target_names=pipeline.config.target_names,
        max_plots=TRAIN_MAX_PLOTS,
        plot_callback=pipeline.plot,
        h5_path=h5_path,
    )

    model_path = out_dir / "model.pt"
    pipeline.save_model_pt(model, model_path)
    print(f"Saved model weights to: {model_path}")

    if "cuda" in str(used_device).lower():
        print(f"Clearing {used_device}...")
        del model
        clear_cuda_cache()

    forecast_h5_path = out_dir / "rolling_forecasts.h5"
    output_json_path = out_dir / "rolling_forecasts.json"

    compute_and_save_rolling_forecast_metrics(
        forecast_h5_path=forecast_h5_path,
        output_json_path=output_json_path,
    )

    print(f"Finished cumulative training workflow for: {batch_window}")

    return model_path


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    repo_root = get_repo_root()
    ensure_python_import_paths(repo_root)
    output_root = resolve_output_root()

    scripts_dir = repo_root / "scripts"
    config_path = scripts_dir / "config.py"

    backup_path = config_path.with_suffix(".py.bak")
    shutil.copy2(config_path, backup_path)

    print(f"Using repo root: {repo_root}")
    print(f"Using config:    {config_path}")
    print(f"Backup saved:    {backup_path}")

    for batch in range(START_BATCH, END_BATCH + 1):
        if batch not in BATCH_SEEDS:
            raise KeyError(f"No seed provided for batch {batch:04d}")

        seed = BATCH_SEEDS[batch]
        batch_padded = f"{batch:04d}"

        print("\n" + "#" * 100)
        print(f"Starting full workflow for batch {batch_padded}")
        print(f"Using seed: {seed}")
        print("#" * 100)

        # ---------------------------------------------------------------------
        # 1. Update config.py for current batch and seed
        # ---------------------------------------------------------------------
        update_config(
            config_path=config_path,
            batch_number=batch,
            seed=seed,
        )

        # ---------------------------------------------------------------------
        # 2. Run Dymola batch simulation
        # ---------------------------------------------------------------------
        run_command(
            [
                sys.executable,
                str(scripts_dir / "run_dymola_batch.py"),
            ],
            cwd=repo_root,
        )

        # ---------------------------------------------------------------------
        # 3. Plot simulation batches
        # ---------------------------------------------------------------------
        run_command(
            [
                sys.executable,
                str(scripts_dir / "plot_sim_batches.py"),
            ],
            cwd=repo_root,
        )

        # ---------------------------------------------------------------------
        # 4. Build cumulative LSTM dataset from 0001 through current batch
        # ---------------------------------------------------------------------
        batch_args = padded_batch_numbers(
            start_batch=START_CUMULATIVE_BATCH,
            end_batch=batch,
        )

        batch_window = format_cumulative_batch_window(
            start_batch=START_CUMULATIVE_BATCH,
            end_batch=batch,
        )

        unscaled_h5 = (
            output_root
            / "datasets"
            / "unscaled_unsplit"
            / f"lstm_merged_batches_{batch_window}_k{LOOKBACK}.h5"
        )

        run_command(
            [
                sys.executable,
                str(scripts_dir / "build_lstm_dataset.py"),
                "--lookback",
                str(LOOKBACK),
                "--batches",
                *batch_args,
            ],
            cwd=repo_root,
        )

        if not unscaled_h5.exists():
            raise FileNotFoundError(
                f"Dataset build finished, but expected unscaled file was not found:\n"
                f"{unscaled_h5}"
            )

        # ---------------------------------------------------------------------
        # 5. Scale cumulative LSTM dataset
        # ---------------------------------------------------------------------
        run_command(
            [
                sys.executable,
                str(scripts_dir / "scale_lstm_dataset.py"),
                str(unscaled_h5),
                "--scaling-type",
                "minmax",
                "--split-mode",
                "profile",
                "--test-manifest",
                str(scripts_dir / "test_profiles_1001_3000.json"),
                "--val-manifest",
                str(scripts_dir / "val_profiles_0001_1000.json"),
            ],
            cwd=repo_root,
        )

        scaled_h5 = find_scaled_h5(
            output_root=output_root,
            batch_window=batch_window,
            lookback=LOOKBACK,
        )

        print(f"\nResolved scaled dataset:\n  {scaled_h5}")

        # ---------------------------------------------------------------------
        # 6. Train cumulative LSTM model directly
        # ---------------------------------------------------------------------
        out_dir = (
            output_root
            / "ml_results"
            / "training_playground"
            / f"Batch{batch_window}_k{LOOKBACK}_minmax"
        )

        model_path = train_cumulative_lstm_directly(
            h5_path=scaled_h5,
            out_dir=out_dir,
            batch_window=batch_window,
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Training finished, but model was not found at:\n"
                f"{model_path}"
            )

        # ---------------------------------------------------------------------
        # 7. Evaluate test-set difficulty
        # ---------------------------------------------------------------------
        run_command(
            [
                sys.executable,
                str(scripts_dir / "evaluate_testset_difficulty.py"),
                "--scaled-h5",
                str(scaled_h5),
                "--model-path",
                str(model_path),
                "--out-dir",
                str(out_dir),
                "--dt",
                DT,
                "--n-bins",
                N_BINS,
                "--include-per-target",
            ],
            cwd=repo_root,
        )

        print("\n" + "#" * 100)
        print(f"Finished full workflow for batch {batch_padded}")
        print("#" * 100)

    print("\nAll batch workflows completed successfully.")


if __name__ == "__main__":
    main()
