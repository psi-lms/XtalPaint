"""Module defining custom datatypes for inpainting of structures."""

from collections.abc import Mapping

import ase
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

__all__ = (
    "BatchedStructures",
    "convert_structure",
)


def convert_structure(
    structure: ase.Atoms | Structure, strct_type: str = "ase"
) -> ase.Atoms | Structure:
    """Convert a structure to the specified type."""
    if strct_type not in ["ase", "pymatgen"]:
        raise ValueError(
            f"Unknown structure type: {strct_type}. Available types are 'ase' "
            "and 'pymatgen'."
        )

    if strct_type == "ase":
        if isinstance(structure, ase.Atoms):
            return structure
        elif isinstance(structure, Structure):
            return AseAtomsAdaptor.get_atoms(structure)
    elif strct_type == "pymatgen":
        if isinstance(structure, Structure):
            return structure
        elif isinstance(structure, ase.Atoms):
            return AseAtomsAdaptor.get_structure(structure)


class BatchedStructures(Mapping):
    """Class to store multiple structures in a batched format."""

    def __init__(self, structures: Mapping[str, ase.Atoms | Structure]):
        """Initialize the BatchedStructures with structures."""
        if not all([isinstance(key, str) for key in structures.keys()]):
            raise ValueError("All keys in structures must be strings.")
        self._structures = dict(structures)

    def __getitem__(self, key):
        """Return the structure corresponding to the given key."""
        return self._structures[key]

    def __iter__(self):
        """Return an iterator over the keys of the structures."""
        return iter(self._structures)

    def __len__(self):
        """Return the number of structures."""
        return len(self._structures)

    @property
    def structures(self):
        """Return the stored structures."""
        return self._structures

    def get_structure(
        self, key: str, strct_type: str = "ase"
    ) -> ase.Atoms | Structure:
        """Return a single structure by key."""
        if key not in self:
            raise ValueError(
                f"Key '{key}' not found in the available structures."
            )

        return convert_structure(self._structures[key], strct_type=strct_type)

    def get_structures(
        self, strct_type: str = "ase"
    ) -> dict[str, ase.Atoms | Structure]:
        """Return all structures as a dictionary."""
        return {
            key: convert_structure(
                self._structures[key], strct_type=strct_type
            )
            for key in self
        }
