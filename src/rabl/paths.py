"""Shared path helpers for RABL runtime artifacts."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT_ENV_VAR = "RABL_OUTPUT_ROOT"


def resolve_output_root(explicit: Path | str | None = None) -> Path:
    """Resolve the root directory for generated RABL outputs.

    Resolution order:
    1. An explicit CLI/config value.
    2. The ``RABL_OUTPUT_ROOT`` environment variable.
    3. The repository-local ``outputs`` directory.
    """
    raw = explicit if explicit is not None else os.environ.get(OUTPUT_ROOT_ENV_VAR)
    if raw is None or str(raw).strip() == "":
        return (REPO_ROOT / "outputs").resolve()
    return Path(raw).expanduser().resolve()
