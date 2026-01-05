"""Serializers for converting python datatypes to AiiDA Data nodes."""

from typing import TYPE_CHECKING

from aiida import orm

from xtalpaint.aiida.data import (
    BatchedStructures,
    BatchedStructuresData,
    PandasDataFrameData,
)

if TYPE_CHECKING:
    import pandas as pd
    from pymatgen.core.structure import Structure


def pymatgen_to_structure_data(structure: "Structure") -> orm.StructureData:
    """Convert a pymatgen Structure to an AiiDA StructureData node."""
    return orm.StructureData(pymatgen=structure)


def pymatgen_traj_to_aiida_traj(trajectory):
    """Convert a pymatgen trajectory to an AiiDA TrajectoryData node.

    :param trajectory: A pymatgen trajectory object.
    :return: An AiiDA TrajectoryData node, or None if trajectory is empty.
    """
    if not trajectory:
        # AiiDA TrajectoryData doesn't support empty trajectories
        return None

    aiida_structures = []
    for structure in trajectory:
        aiida_structure = pymatgen_to_structure_data(structure)
        aiida_structures.append(aiida_structure)
    return orm.TrajectoryData(structurelist=aiida_structures)


def batched_structures_to_batched_structures_data(
    batched_structures: BatchedStructures,
) -> BatchedStructuresData:
    """Convert BatchedStructures to BatchedStructuresData."""
    return BatchedStructuresData.from_batched_structures(batched_structures)


def pandas_dataframe_to_pandas_dataframe_data(df: "pd.DataFrame") -> orm.Data:
    """Convert a pandas DataFrame to a custom AiiDA Data node."""
    return PandasDataFrameData(value=df)
