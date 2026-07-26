from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import train_test_split

SIM_ROOT = Path("../outputs/sim_profiles")
OUTPUT_DIR = Path("manifests")
SPLIT_SEED = 12345

BATCHES = {
    "batch_0001": {"expected_count": 3000, "control_sampling_seed": 456},
    "batch_0002": {"expected_count": 1000, "control_sampling_seed": 789},
    "batch_0003": {"expected_count": 5000, "control_sampling_seed": 123},
}

# Adjust this glob only if profile files use a more specific naming pattern.
profile_names = []
batch_labels = []

for batch_name, metadata in BATCHES.items():
    batch_dir = SIM_ROOT / batch_name
    paths = sorted(batch_dir.glob("results_drum_profile*.csv"))

    if len(paths) != metadata["expected_count"]:
        raise ValueError(
            f"{batch_name}: expected {metadata['expected_count']} CSVs, "
            f"found {len(paths)}."
        )

    profile_names.extend(path.stem for path in paths)
    batch_labels.extend([batch_name] * len(paths))

if len(profile_names) != len(set(profile_names)):
    raise ValueError("Duplicate profile names were found across batches.")

profiles = np.asarray(profile_names)
batches = np.asarray(batch_labels)

# Exact global counts, approximately preserving each batch's proportions.
train, remaining, _, remaining_batches = train_test_split(
    profiles,
    batches,
    train_size=5000,
    test_size=4000,
    random_state=SPLIT_SEED,
    shuffle=True,
    stratify=batches,
)

val, remaining, _, remaining_batches = train_test_split(
    remaining,
    remaining_batches,
    train_size=1000,
    test_size=3000,
    random_state=SPLIT_SEED + 1,
    shuffle=True,
    stratify=remaining_batches,
)

cal, test, _, _ = train_test_split(
    remaining,
    remaining_batches,
    train_size=1000,
    test_size=2000,
    random_state=SPLIT_SEED + 2,
    shuffle=True,
    stratify=remaining_batches,
)

splits = {
    "train": sorted(train.tolist()),
    "val": sorted(val.tolist()),
    "cal": sorted(cal.tolist()),
    "test": sorted(test.tolist()),
}

# Final leakage and count checks.
expected_counts = {"train": 5000, "val": 1000, "cal": 1000, "test": 2000}

for split, expected in expected_counts.items():
    if len(splits[split]) != expected:
        raise AssertionError(
            f"{split}: expected {expected}, got {len(splits[split])}."
        )

for left, left_names in splits.items():
    for right, right_names in splits.items():
        if left < right and set(left_names) & set(right_names):
            raise AssertionError(f"Profile leakage between {left} and {right}.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for split, names in splits.items():
    output_path = OUTPUT_DIR / f"{split}_manifest.json"
    output_path.write_text(
        json.dumps({f"{split}_profiles": names}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(names):,} profiles to {output_path}")

summary = {
    "manifest_generation_seed": SPLIT_SEED,
    "total_profiles": len(profile_names),
    "source_batches": BATCHES,
    "split_counts": {name: len(values) for name, values in splits.items()},
}

(OUTPUT_DIR / "manifest_generation_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)

print("Manifest generation complete.")