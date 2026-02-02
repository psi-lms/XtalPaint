"""Module defining custom AiiDA datatypes for inpainting of structures."""

import io
import tempfile
import typing as t

import ase
import ase.io
import numpy as np
import pandas as pd
from aiida.common import NotExistent
from aiida.orm import Data, ProcessNode, QueryBuilder, StructureData
from aiida_pythonjob import pyfunction, spec
from disk_objectstore.utils import PackedObjectReader
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from xtalpaint.data import BatchedStructures

__all__ = (
    "InpaintingStructureData",
    "BatchedStructuresData",
)


@pyfunction(
    outputs=spec.namespace(structures=t.Any),
)
def extract_from_batched_structures(
    batched_structures: "BatchedStructuresData", keys: list[str]
) -> dict[str, StructureData]:
    outputs = {}
    for key in keys:
        atoms = batched_structures.get_structure(key)
        outputs[key] = StructureData(ase=atoms)
    return {"structures": outputs}


class InpaintingStructureData(Data):
    """AiiDA node to store a pymatgen Structure containing nan coordinates."""

    def __init__(self, value=None, **kwargs):
        """Initialize the InpaintingStructureData with a Structure."""
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
        """Convert site coordinate NaNs to the string 'nan' for JSON format."""
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
    def value(self) -> Structure:
        """Return the structure, converting 'nan' strings back to np.nan."""
        keys = self.base.attributes.get("keys")
        values = self.base.attributes.get_many(keys)
        data = dict(zip(keys, values))
        data = self._replace_str_to_nan_coords(data)
        return Structure.from_dict(data)

    @property
    def structure(self) -> Structure:
        """Alias for value property to maintain compatibility with pymatgen."""
        return self.value


class BatchedStructuresData(Data):
    """AiiDA node to store multiple structures in a batched format."""

    def __init__(self, value=None, **kwargs):
        """Initialize the BatchedStructuresData with structures."""
        structures = value or {}
        super().__init__(**kwargs)
        if isinstance(structures, BatchedStructures):
            structures = structures.structures

        self.base.attributes.set("keys", list(structures.keys()).copy())
        self.base.attributes.set("filename", "structures.extxyz")

        ase_structures = self.structures_to_atoms(structures)

        self._structures_to_file(ase_structures, "structures")

    def keys(self):
        """Return the keys of the structures."""
        return self.base.attributes.get("keys")

    @property
    def file_name(self) -> str:
        """Return the name of the file where structures are stored."""
        return self.base.attributes.get("filename")

    @property
    def value(self) -> BatchedStructures:
        """Return the stored structures as a dictionary."""
        return BatchedStructures(self.get_structures(strct_type="pymatgen"))

    @classmethod
    def from_batched_structures(
        cls, batched_structures: BatchedStructures
    ) -> "BatchedStructuresData":
        """Convert BatchedStructures to AiiDA data."""
        return cls(value=batched_structures.structures)

    def structures_to_atoms(self, structures) -> list:
        """Convert structures to ASE Atoms objects."""
        ase_atoms = []
        for key in self.keys():
            ase_atom = structures[key]
            if isinstance(ase_atom, Structure):
                ase_atom = AseAtomsAdaptor.get_atoms(ase_atom)
            ase_atom.info["key"] = key
            ase_atoms.append(ase_atom)

        return ase_atoms

    def _structures_to_file(self, structures, name) -> None:
        """Store the structures as an extxyz file in the repository."""
        # Write the array to a temporary file, and then add it to the
        # repository of the node
        with tempfile.NamedTemporaryFile(mode="w+", suffix="extxyz") as handle:
            ase.io.write(handle, structures, format="extxyz")

            # Flush and rewind the handle, otherwise the command to store it
            # in the repo will write an empty file
            handle.flush()
            handle.seek(0)

            self.base.repository.put_object_from_filelike(
                handle, f"{name}.extxyz"
            )

    def _get_structures_from_file(self, keys=None) -> list[ase.Atoms]:
        """Return the stored structures from file."""
        all_keys = self.keys()

        if self.file_name not in self.base.repository.list_object_names():
            raise ValueError("No structures found in the repository.")
        if keys is not None and not isinstance(keys, (str, list)):
            raise ValueError("Keys must be a string or a list of strings.")
        elif isinstance(keys, list) and not set(keys).issubset(all_keys):
            raise ValueError(
                f"Keys `{[k for k in keys if k not in all_keys]}` not found "
                "in the available structures."
            )

        with self.base.repository.open(self.file_name, mode="rb") as handle:
            wrapped_handle = (
                io.TextIOWrapper(io.BytesIO(handle.read()))
                if isinstance(handle, PackedObjectReader)
                else io.TextIOWrapper(handle)
            )

            if keys is None:
                all_ase_structures = ase.io.read(
                    wrapped_handle, index=":", format="extxyz"
                )
            else:
                indices = [all_keys.index(k) for k in keys]
                index_to_ase_strct = {}

                for idx, ase_strct in enumerate(
                    ase.io.iread(wrapped_handle, index=":", format="extxyz")
                ):
                    if idx in indices:
                        index_to_ase_strct[idx] = ase_strct

                all_ase_structures = [index_to_ase_strct[i] for i in indices]

        return all_ase_structures

    def get_structures(
        self,
        keys: str | list[str] | None = None,
        strct_type: str = "ase",
        use_existing_aiida_nodes: bool = True,
    ) -> dict[str, ase.Atoms | Structure | StructureData]:
        """Return a single structure by key.

        key: If None, return all structures as a list. If provided,
            return the structure(s) corresponding to the key(s).
        strct_type: 'ase', 'pymatgen', or 'aiida' (returns StructureData node).
        """
        if keys is not None and not isinstance(keys, (str, list)):
            raise ValueError(
                "Keys must be a string, list of strings, or `None`."
            )
        if isinstance(keys, str):
            keys = [keys]

        structures = self._get_structures_from_file(keys=keys)

        if keys is None:
            keys = self.keys()

        if strct_type == "ase":
            return {
                str(ase_atom.info["key"]): ase_atom for ase_atom in structures
            }
        elif strct_type == "pymatgen":
            return {
                str(ase_atom.info["key"]): AseAtomsAdaptor.get_structure(
                    ase_atom
                )
                for ase_atom in structures
            }
        elif strct_type == "aiida":
            edge_labels_to_filter = [f"structures__{k}" for k in keys]

            # Check if StructureData node for this structure already exists as
            # a child of this node
            qb = QueryBuilder()
            qb.append(type(self), filters={"uuid": self.uuid}, tag="parent")
            qb.append(
                ProcessNode,
                filters={
                    "attributes.process_label": (
                        "extract_from_batched_structures"
                    )
                },
                with_incoming="parent",
                tag="process",
            )
            qb.append(
                StructureData,
                with_incoming="process",
                edge_filters={"label": {"in": edge_labels_to_filter}},
                project="*",
                edge_project="label",
            )
            result = qb.all(flat=False)

            if len(result) > len(edge_labels_to_filter):
                raise ValueError(
                    "Warning: More StructureData nodes found than requested!"
                )

            missing_keys = set(keys)
            existing_aiida_structures = {}
            if result is not None and use_existing_aiida_nodes:
                print("Reusing existing StructureData nodes.")

                existing_aiida_structures = {
                    edge_label.split("__")[-1]: struct
                    for struct, edge_label in result
                }

                if set(existing_aiida_structures.keys()) == set(keys):
                    return existing_aiida_structures

                missing_keys -= set(existing_aiida_structures.keys())

            result = extract_from_batched_structures(self, list(missing_keys))

            aiida_structures = {
                **existing_aiida_structures,
                **result["structures"],
            }
            return aiida_structures
        else:
            raise ValueError(
                f"Unknown structure type: {strct_type}. Available types are "
                "'ase', 'pymatgen', and 'aiida'."
            )


class PandasDataFrameData(Data):
    """AiiDA node to store a pandas DataFrame as a .parquet file."""

    def __init__(self, value=None, **kwargs):
        """Initialize the PandasDataFrameData with a DataFrame."""
        df = value
        super().__init__(**kwargs)

        self.base.attributes.set("filename", "dataframe.parquet")
        self._df_to_file(df)

    def _df_to_file(self, df: pd.DataFrame) -> None:
        """Store the DataFrame as a .parquet file in the repository."""
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".parquet"
        ) as handle:
            df.to_parquet(handle, index=True)

            handle.flush()
            handle.seek(0)

            self.base.repository.put_object_from_filelike(
                handle, self.base.attributes.get("filename")
            )

    @property
    def value(self) -> pd.DataFrame:
        """Return the stored DataFrame."""
        if "dataframe.parquet" not in self.base.repository.list_object_names():
            raise NotExistent("No dataframe found in the repository.")

        with self.base.repository.open(
            self.base.attributes.get("filename"), mode="rb"
        ) as handle:
            df = pd.read_parquet(handle)

        return df
