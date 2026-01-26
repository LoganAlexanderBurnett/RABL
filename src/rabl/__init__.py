"""RABL (Microreactor Dynamics) Python package."""

try:
    from rabl.interface.pymola import BatchConfig, DymolaBatchRunner
except ModuleNotFoundError:
    # Dymola is an optional dependency (not needed for ML on HPC)
    BatchConfig = None
    DymolaBatchRunner = None

from rabl.variography.DrumVariography import DrumProfileGenerator, DrumProfile

__all__ = [
    "BatchConfig",
    "DymolaBatchRunner",
    "DrumProfile",
    "DrumProfileGenerator",
]
