import numpy as np
from aiida.orm import Data
from pymatgen.core import Structure
import ase
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor
import io


__all__ = (
    "InpaintingStructure",
    "BatchedStructures",
    "BatchedStructuresData",
    "InpaintingResults",
    "InpaintingResultsData",
    "convert_structure",
)


class InpaintingStructure(Data):
    """AiiDA node to store a pymatgen Structure that may contain nan coordinates."""

    def __init__(self, value=None, **kwargs):
        """
        :param structure: pymatgen Structure instance to initialize from
        """
        structure = value
        super().__init__(**kwargs)

        if structure is None:
            return
        # serialize Structure to dict and replace nan for JSON compatibility
        data = structure.as_dict()
        data = self._replace_nan_to_str_coords(data)
        keys = list(data.keys())
        self.base.attributes.set_many(data)
        self.base.attributes.set("keys", keys)

    @staticmethod
    def _replace_nan_to_str_coords(obj):
        """Convert only site coordinate NaNs to the string 'nan' for JSON serialization."""
        for site in obj.get("sites", []):
            for key in ("abc", "xyz"):
                site[key] = [
                    None if np.isnan(coord) else coord
                    for coord in site.get(key, [])
                ]
        return obj

    @staticmethod
    def _replace_str_to_nan_coords(obj):
        """Convert only 'nan' strings in site coordinates back to np.nan."""
        for site in obj.get("sites", []):
            for key in ("abc", "xyz"):
                site[key] = [
                    np.nan if coord is None else coord
                    for coord in site.get(key, [])
                ]
        return obj

    @property
    def value(self):
        """Return the stored Structure, converting 'nan' strings back to np.nan"""
        keys = self.base.attributes.get("keys")
        values = self.base.attributes.get_many(keys)
        data = dict(zip(keys, values))
        data = self._replace_str_to_nan_coords(data)
        return Structure.from_dict(data)

    @property
    def structure(self):
        """Alias for value property to maintain compatibility with pymatgen."""
        return self.value


def convert_structure(
    structure: ase.Atoms | Structure, strct_type: str = "ase"
) -> ase.Atoms | Structure:
    """Convert a structure to the specified type."""
    if strct_type not in ["ase", "pymatgen"]:
        raise ValueError(
            f"Unknown structure type: {strct_type}. Available types are 'ase' and 'pymatgen'."
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


class BatchedStructures:
    def __init__(self, structures: dict):
        """Initialize the BatchedStructures with structures."""
        if not all([isinstance(key, str) for key in structures.keys()]):
            raise ValueError("All keys in structures must be strings.")
        self._keys = tuple(structures.keys())

        self._structures = structures

    @property
    def keys(self):
        """Return the keys of the structures."""
        return self._keys

    @property
    def structures(self):
        """Return the original structures."""
        return self._structures

    def get_structure(
        self, key: str, strct_type: str = "ase"
    ) -> ase.Atoms | Structure:
        """Return a single structure by key."""
        if key not in self.keys:
            raise ValueError(
                f"Key '{key}' not found in the available structures."
            )

        return convert_structure(self.structures[key], strct_type=strct_type)

    def get_structures(
        self, strct_type: str = "ase"
    ) -> dict[str, ase.Atoms | Structure]:
        """Return all structures as a dictionary."""
        return {
            key: convert_structure(self.structures[key], strct_type=strct_type)
            for key in self.keys
        }


class BatchedStructuresData(Data):
    def __init__(self, structures: dict, **kwargs):
        """Initialize the BatchedStructuresData with structures."""
        super().__init__(**kwargs)

        self.base.attributes.set("keys", list(structures.keys()).copy())
        self.base.attributes.set("filename", "structures.extxyz")

        ase_structures = self.structures_to_atoms(structures)

        self.structures_to_file(ase_structures, "structures")

    @property
    def keys(self):
        """Return the keys of the structures."""
        return self.base.attributes.get("keys")

    @property
    def file_name(self):
        """Return the name of the file where structures are stored."""
        return self.base.attributes.get("filename")

    @property
    def value(self):
        """Return the stored structures as a dictionary."""
        return BatchedStructures(self.get_structures(strct_type="pymatgen"))

    @classmethod
    def from_batched_structures(cls, batched_structures: BatchedStructures):
        """Convert BatchedStructures to AiiDA data."""
        return cls(structures=batched_structures.structures)

    def structures_to_atoms(self, structures) -> list:
        """Convert structures to ASE Atoms objects."""
        ase_atoms = []
        for key in self.keys:
            ase_atom = structures[key]
            if isinstance(ase_atom, Structure):
                ase_atom = AseAtomsAdaptor.get_atoms(ase_atom)
            ase_atom.info["key"] = key
            ase_atoms.append(ase_atom)

        return ase_atoms

    def structures_to_file(self, structures, name) -> None:
        import tempfile

        # Write the array to a temporary file, and then add it to the repository of the node
        with tempfile.NamedTemporaryFile(mode="w+", suffix="extxyz") as handle:
            ase.io.write(handle, structures, format="extxyz")

            # Flush and rewind the handle, otherwise the command to store it in the repo will write an empty file
            handle.flush()
            handle.seek(0)

            self.base.repository.put_object_from_filelike(
                handle, f"{name}.extxyz"
            )

    def get_structure(
        self, key: str, strct_type: str = "ase"
    ) -> ase.Atoms | Structure:
        """Return a single structure by key."""
        if self.file_name not in self.base.repository.list_object_names():
            raise ValueError("No structures found in the repository.")
        if key not in self.keys:
            raise ValueError(
                f"Key '{key}' not found in the available structures."
            )

        with self.base.repository.open(self.file_name, mode="rb") as handle:
            structure = ase.io.read(
                io.TextIOWrapper(handle),
                index=self.keys.index(key),
                format="extxyz",
            )

        if strct_type == "ase":
            return structure
        elif strct_type == "pymatgen":
            return AseAtomsAdaptor.get_structure(structure)
        else:
            raise ValueError(
                f"Unknown structure type: {strct_type}. Available types are 'ase' and 'pymatgen'."
            )

    def get_structures(
        self, strct_type: str = "ase"
    ) -> dict[str, ase.Atoms | Structure]:
        """Return all structures as a dictionary."""
        if self.file_name not in self.base.repository.list_object_names():
            raise ValueError("No structures found in the repository.")

        with self.base.repository.open(self.file_name, mode="rb") as handle:
            ase_structures = ase.io.read(
                io.TextIOWrapper(handle), index=":", format="extxyz"
            )

        if strct_type == "ase":
            return {
                str(ase_atom.info["key"]): ase_atom
                for ase_atom in ase_structures
            }
        elif strct_type == "pymatgen":
            return {
                str(ase_atom.info["key"]): AseAtomsAdaptor.get_structure(
                    ase_atom
                )
                for ase_atom in ase_structures
            }
        else:
            raise ValueError(
                f"Unknown structure type: {strct_type}. Available types are 'ase' and 'pymatgen'."
            )


class InpaintingResults:
    def __init__(self, structures: dict, relaxed_structures: dict = None):
        """Initialize the BatchedStructures with structures and optional relaxed structures."""

        if relaxed_structures:
            if set(structures.keys()) != set(relaxed_structures.keys()):
                raise ValueError(
                    "Keys of structures and relaxed_structures must match."
                )

        if not all([isinstance(key, str) for key in structures.keys()]):
            raise ValueError("All keys in structures must be strings.")
        self._keys = tuple(structures.keys())

        self._structures = structures
        self._relaxed_structures = (
            relaxed_structures if relaxed_structures else {}
        )

    @property
    def keys(self):
        """Return the keys of the structures."""
        return self._keys

    @property
    def structures(self):
        """Return the original structures."""
        return self._structures

    @property
    def relaxed_structures(self):
        """Return the relaxed structures."""
        return self._relaxed_structures

    def get_structure(
        self, key: str, relaxed: bool = False, strct_type: str = "ase"
    ) -> ase.Atoms | Structure:
        """Return a single structure by key."""
        if key not in self.keys:
            raise ValueError(
                f"Key '{key}' not found in the available structures."
            )
        if relaxed:
            if not self.relaxed_structures:
                raise ValueError("No relaxed structures available.")
            return convert_structure(
                self.relaxed_structures[key], strct_type=strct_type
            )
        else:
            return convert_structure(
                self.structures[key], strct_type=strct_type
            )

    def get_structures(
        self, relaxed: bool = False, strct_type: str = "ase"
    ) -> dict[str, ase.Atoms | Structure]:
        """Return all structures as a dictionary."""
        if relaxed:
            if not self.relaxed_structures:
                raise ValueError("No relaxed structures available.")
            return {
                key: convert_structure(
                    self.relaxed_structures[key], strct_type=strct_type
                )
                for key in self.keys
            }
        else:
            return {
                key: convert_structure(
                    self.structures[key], strct_type=strct_type
                )
                for key in self.keys
            }


class InpaintingResultsData(Data):
    def __init__(
        self, structures: dict, relaxed_structures: dict = None, **kwargs
    ):
        """Initialize the InpaintingResultsData with structures and optional relaxed structures."""
        super().__init__(**kwargs)

        if relaxed_structures is not None:
            if set(structures.keys()) != set(relaxed_structures.keys()):
                raise ValueError(
                    "Keys of structures and relaxed_structures must match."
                )

        self.base.attributes.set("keys", list(structures.keys()).copy())

        ase_structures = self.structures_to_atoms(structures)

        self.structures_to_file(ase_structures, "structures")

        if relaxed_structures is not None:
            ase_relaxed_structures = self.structures_to_atoms(
                relaxed_structures
            )
            self.structures_to_file(
                ase_relaxed_structures, "relaxed_structures"
            )

    @property
    def keys(self):
        """Return the keys of the structures."""
        return self.base.attributes.get("keys")

    @classmethod
    def from_inpainting_results(cls, inpainting_results: InpaintingResults):
        """Convert inpainting results to AiiDA data."""
        return cls(
            structures=inpainting_results.structures,
            relaxed_structures=inpainting_results.relaxed_structures,
        )

    def structures_to_atoms(self, structures) -> list:
        """Convert structures to ASE Atoms objects."""
        ase_atoms = []
        for key in self.keys:
            ase_atom = structures[key]
            if isinstance(ase_atom, Structure):
                ase_atom = AseAtomsAdaptor.get_atoms(ase_atom)
            ase_atom.info["key"] = key
            ase_atoms.append(ase_atom)

        return ase_atoms

    def structures_to_file(self, structures, name) -> None:
        import tempfile

        # Write the array to a temporary file, and then add it to the repository of the node
        with tempfile.NamedTemporaryFile(mode="w+", suffix="extxyz") as handle:
            ase.io.write(handle, structures, format="extxyz")

            # Flush and rewind the handle, otherwise the command to store it in the repo will write an empty file
            handle.flush()
            handle.seek(0)

            self.base.repository.put_object_from_filelike(
                handle, f"{name}.extxyz"
            )

    def get_structure(
        self, key: str, relaxed: bool = False, strct_type: str = "ase"
    ) -> ase.Atoms | Structure:
        """Return a single structure by key."""
        file_name = (
            "relaxed_structures.extxyz" if relaxed else "structures.extxyz"
        )
        if file_name not in self.base.repository.list_object_names():
            raise ValueError(
                f"No {'relaxed ' if relaxed else ''}structures found in the repository."
            )
        if key not in self.keys:
            raise ValueError(
                f"Key '{key}' not found in the available structures."
            )

        with self.base.repository.open(file_name, mode="rb") as handle:
            structure = ase.io.read(
                io.TextIOWrapper(handle),
                index=self.keys.index(key),
                format="extxyz",
            )

        if strct_type == "ase":
            return structure
        elif strct_type == "pymatgen":
            return AseAtomsAdaptor.get_structure(structure)
        else:
            raise ValueError(
                f"Unknown structure type: {strct_type}. Available types are 'ase' and 'pymatgen'."
            )

    def get_structures(
        self, relaxed: bool = False, strct_type: str = "ase"
    ) -> dict[str, ase.Atoms | Structure]:
        """Return all structures as a dictionary."""
        file_name = (
            "relaxed_structures.extxyz" if relaxed else "structures.extxyz"
        )
        if file_name not in self.base.repository.list_object_names():
            raise ValueError(
                f"No {'relaxed ' if relaxed else ''}structures found in the repository."
            )

        with self.base.repository.open(file_name, mode="rb") as handle:
            ase_structures = ase.io.read(
                io.TextIOWrapper(handle), index=":", format="extxyz"
            )

        if strct_type == "ase":
            return {
                str(ase_atom.info["key"]): ase_atom
                for ase_atom in ase_structures
            }
        elif strct_type == "pymatgen":
            return {
                str(ase_atom.info["key"]): AseAtomsAdaptor.get_structure(
                    ase_atom
                )
                for ase_atom in ase_structures
            }
        else:
            raise ValueError(
                f"Unknown structure type: {strct_type}. Available types are 'ase' and 'pymatgen'."
            )
