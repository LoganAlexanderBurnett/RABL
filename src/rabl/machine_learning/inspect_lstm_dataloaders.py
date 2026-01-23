import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


STATE_DIM = 13
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100


def _get_profile_names(h5_path: Path, split: str) -> list[str]:
    with h5py.File(h5_path, "r") as h5f:
        return sorted(h5f[split]["files"].keys())


def _get_profile_shapes(h5_path: Path, split: str, profile_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with h5py.File(h5_path, "r") as h5f:
        group = h5f[split]["files"][profile_name]
        return tuple(group["X"].shape), tuple(group["Y"].shape)


def _train_sample_generator(
    h5_path: Path, profile_names: list[str], split: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for profile_name in profile_names:
            group = files_group[profile_name]
            x_data = group["X"][...].astype(np.float32)
            y_data = group["Y"][...].astype(np.float32)
            indices = np.arange(x_data.shape[0])
            rng.shuffle(indices)
            for idx in indices:
                yield x_data[idx], y_data[idx]


def _sample_generator(h5_path: Path, profile_names: list[str], split: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for profile_name in profile_names:
            group = files_group[profile_name]
            x_data = group["X"][...].astype(np.float32)
            y_data = group["Y"][...].astype(np.float32)
            for idx in range(x_data.shape[0]):
                yield x_data[idx], y_data[idx]


def _profile_generator(h5_path: Path, profile_names: list[str], split: str) -> tuple[str, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5f:
        files_group = h5f[split]["files"]
        for profile_name in profile_names:
            group = files_group[profile_name]
            x_data = group["X"][...].astype(np.float32)
            y_data = group["Y"][...].astype(np.float32)
            yield profile_name, x_data, y_data


def build_datasets(h5_path: Path, batch_size: int, seed: int) -> dict[str, tf.data.Dataset]:
    train_profiles = _get_profile_names(h5_path, "train")
    val_profiles = _get_profile_names(h5_path, "val")
    test_profiles = _get_profile_names(h5_path, "test")

    if not train_profiles:
        raise ValueError("No training profiles found in HDF5.")

    x_shape, y_shape = _get_profile_shapes(h5_path, "train", train_profiles[0])

    train_signature = (
        tf.TensorSpec(shape=x_shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=y_shape[1:], dtype=tf.float32),
    )
    train_ds = tf.data.Dataset.from_generator(
        lambda: _train_sample_generator(h5_path, train_profiles, "train", seed),
        output_signature=train_signature,
    )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_sample_ds = tf.data.Dataset.from_generator(
        lambda: _sample_generator(h5_path, val_profiles, "val"),
        output_signature=train_signature,
    )
    val_sample_ds = val_sample_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    profile_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(None, *x_shape[1:]), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *y_shape[1:]), dtype=tf.float32),
    )

    val_profile_ds = tf.data.Dataset.from_generator(
        lambda: _profile_generator(h5_path, val_profiles, "val"),
        output_signature=profile_signature,
    )
    test_profile_ds = tf.data.Dataset.from_generator(
        lambda: _profile_generator(h5_path, test_profiles, "test"),
        output_signature=profile_signature,
    )

    return {
        "train": train_ds,
        "val_samples": val_sample_ds,
        "val_profile_ds": val_profile_ds,
        "test_profile_ds": test_profile_ds,
        "train_profile_names": train_profiles,
        "val_profile_names": val_profiles,
        "test_profile_names": test_profiles,
        "sample_shape": x_shape,
        "target_shape": y_shape,
    }


def build_model(timesteps: int, num_features: int, num_targets: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, num_features)),
            tf.keras.layers.LSTM(64, return_sequences=False, stateful=False),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_targets),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


def rolling_forecast(model: tf.keras.Model, x_profile: np.ndarray) -> np.ndarray:
    timesteps, num_features = x_profile.shape[1:]
    if num_features <= STATE_DIM:
        raise ValueError("Expected control features appended to state features.")
    control_dim = num_features - STATE_DIM

    window_states = x_profile[0, :, :STATE_DIM].copy()
    preds = []
    for step in range(x_profile.shape[0]):
        control_window = x_profile[step, :, STATE_DIM : STATE_DIM + control_dim]
        input_window = np.concatenate([window_states, control_window], axis=1)
        pred = model.predict(input_window[None, ...], verbose=0)[0]
        preds.append(pred)
        if step + 1 < x_profile.shape[0]:
            window_states = np.vstack([window_states[1:], pred])

    return np.asarray(preds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect LSTM HDF5 datasets and build model.")
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to the merged LSTM HDF5 dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for shuffling training samples within each profile.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional path to save the training/validation curve plot.",
    )
    parser.add_argument(
        "--demo-rolling",
        action="store_true",
        help="Run a rolling forecast on the first validation profile and print the output shape.",
    )
    args = parser.parse_args(argv)

    datasets = build_datasets(args.h5, args.batch_size, args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu_name = gpus[0].name
        print(f"Using GPU: {gpu_name}")
        training_device = "/GPU:0"
    else:
        print("No GPU detected; training on CPU.")
        training_device = "/CPU:0"

    print("Dataset summary:")
    print(f"  Train profiles: {len(datasets['train_profile_names'])}")
    print(f"  Val profiles:   {len(datasets['val_profile_names'])}")
    print(f"  Test profiles:  {len(datasets['test_profile_names'])}")
    print(f"  Sample X shape: {datasets['sample_shape']}")
    print(f"  Sample Y shape: {datasets['target_shape']}")

    train_batch = next(iter(datasets["train"]))
    print(f"Train batch X shape: {train_batch[0].shape}")
    print(f"Train batch Y shape: {train_batch[1].shape}")

    for split_name, dataset_key in (("val", "val_profile_ds"), ("test", "test_profile_ds")):
        profile_name, x_profile, y_profile = next(iter(datasets[dataset_key]))
        print(f"{split_name.capitalize()} profile: {profile_name.numpy().decode()}")
        print(f"  X profile shape: {x_profile.shape}")
        print(f"  Y profile shape: {y_profile.shape}")

    timesteps = datasets["sample_shape"][1]
    num_features = datasets["sample_shape"][2]
    num_targets = datasets["target_shape"][1]
    with tf.device(training_device):
        model = build_model(timesteps, num_features, num_targets)
        model.summary()

        history = model.fit(
            datasets["train"],
            validation_data=datasets["val_samples"],
            epochs=args.epochs,
        )

    plot_path = args.plot_path
    if plot_path is None:
        plot_path = Path(__file__).resolve().parents[3] / "outputs" / "plots" / "lstm_training_curves.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history.history["loss"], label="Train Loss (MSE)")
    plt.plot(epochs_range, history.history["val_loss"], label="Val Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training curves to {plot_path}")

    if args.demo_rolling:
        _, x_profile, _ = next(iter(datasets["val_profile_ds"]))
        preds = rolling_forecast(model, x_profile.numpy())
        print(f"Rolling forecast output shape: {preds.shape}")


if __name__ == "__main__":
    main()
