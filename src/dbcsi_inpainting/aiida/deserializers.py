"""Deserializers for AiiDA data types in DBCSI inpainting."""

from dbcsi_inpainting.aiida.data import (
    BatchedStructures,
    BatchedStructuresData,
)


def batched_structures_data_to_batched_structures(
    batched_structures: BatchedStructuresData,
) -> BatchedStructures:
    """Convert BatchedStructures to BatchedStructuresData."""
    return BatchedStructures(
        batched_structures.get_structures(strct_type="pymatgen")
    )
