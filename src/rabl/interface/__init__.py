"""Interface utilities."""

try:
    from rabl.interface.pymola import BatchConfig, DymolaBatchRunner
except ModuleNotFoundError:
    # Dymola is an optional dependency (not needed for ML/datasets)
    BatchConfig = None
    DymolaBatchRunner = None

__all__ = ["BatchConfig", "DymolaBatchRunner"]
