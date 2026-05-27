"""Module for utility functions in XtalPaint."""

from functools import lru_cache

from xtalpaint.data import BatchedStructures


@lru_cache(maxsize=1)
def is_aiida_installed() -> bool:
    """Check if AiiDA is installed and available.

    Returns:
        bool: True if AiiDA is installed, False otherwise.
    """
    try:
        import aiida  # noqa: F401

        return True
    except ImportError:
        return False


def _is_batched_structure(obj) -> bool:
    """Check if object is a batched structure type.

    Args:
        obj: The object to check.

    Returns:
        bool: True if obj is a BatchedStructures or BatchedStructuresData.
    """
    if isinstance(obj, BatchedStructures):
        return True
    if is_aiida_installed():
        from xtalpaint.aiida.data import BatchedStructuresData

        return isinstance(obj, BatchedStructuresData)
    return False
