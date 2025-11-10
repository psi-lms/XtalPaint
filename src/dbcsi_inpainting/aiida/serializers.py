from aiida import orm
from dbcsi_inpainting.aiida.data import (
    BatchedStructures,
    BatchedStructuresData,
)


def pymatgen_to_structure_data(structure):
    return orm.StructureData(pymatgen=structure)


def pymatgen_traj_to_aiida_traj(trajectory):
    """
    Convert a pymatgen trajectory to an AiiDA StructureData node.

    :param trajectory: A pymatgen trajectory object.
    :return: A list of AiiDA StructureData nodes.
    """
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
