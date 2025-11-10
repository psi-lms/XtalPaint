import numpy as np
import pytest
from aiida.orm import StructureData
from ase import Atoms
from pymatgen.core import Lattice, Structure

from dbcsi_inpainting.aiida.data import (
    BatchedStructures,
    InpaintingStructure,
    BatchedStructuresData,
    convert_structure,
    extract_from_batched_structures,
)


@pytest.fixture
def simple_structure_pmg():
    lattice = Lattice.cubic(3.0)
    return Structure(lattice, ["Li"], [[0.0, 0.0, 0.0]])


@pytest.fixture
def simple_atoms():
    atoms = Atoms("Li", positions=[[0.0, 0.0, 0.0]])
    atoms.set_cell([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    atoms.set_pbc(True)
    return atoms


def test_convert_structure_identity(simple_atoms, simple_structure_pmg):
    assert convert_structure(simple_atoms, "ase") is simple_atoms
    assert (
        convert_structure(simple_structure_pmg, "pymatgen")
        is simple_structure_pmg
    )


def test_convert_structure_between_formats(simple_atoms, simple_structure_pmg):
    converted_atoms = convert_structure(simple_structure_pmg, "ase")
    assert isinstance(converted_atoms, Atoms)
    converted_structure = convert_structure(simple_atoms, "pymatgen")
    assert isinstance(converted_structure, Structure)
    assert str(converted_structure.formula) == str(
        simple_structure_pmg.formula
    )


@pytest.mark.parametrize("invalid_type", ["unknown", 123])
def test_convert_structure_invalid_type(simple_atoms, invalid_type):
    with pytest.raises(ValueError):
        convert_structure(simple_atoms, invalid_type)  # type: ignore[arg-type]


def test_batched_structures_basic(simple_atoms, simple_structure_pmg):
    batched = BatchedStructures(
        {"atoms": simple_atoms, "pmg": simple_structure_pmg}
    )

    assert batched.keys() == ("atoms", "pmg")
    assert batched.structures["atoms"] is simple_atoms

    pmg_struct = batched.get_structure("atoms", strct_type="pymatgen")
    assert isinstance(pmg_struct, Structure)

    ase_structs = batched.get_structures(strct_type="ase")
    assert set(ase_structs.keys()) == {"atoms", "pmg"}
    assert all(isinstance(v, Atoms) for v in ase_structs.values())


def test_batched_structures_invalid_key(simple_atoms):
    batched = BatchedStructures({"atoms": simple_atoms})
    with pytest.raises(ValueError):
        batched.get_structure("missing")


def test_batched_structures_non_string_key(simple_atoms):
    with pytest.raises(ValueError):
        BatchedStructures({1: simple_atoms})  # type: ignore[arg-type]


def test_inpainting_structure_roundtrip_with_nan(
    aiida_profile, simple_structure_pmg
):
    coords = simple_structure_pmg.frac_coords.copy()
    coords[0][0] = float("nan")
    structure_with_nan = Structure(
        simple_structure_pmg.lattice,
        simple_structure_pmg.species,
        coords.tolist(),
    )

    node = InpaintingStructure(value=structure_with_nan)

    sites_attr = node.base.attributes.get("sites")
    assert sites_attr[0]["abc"][0] is None

    recovered = node.value
    assert recovered.formula == structure_with_nan.formula
    assert np.isnan(recovered.frac_coords[0][0])
    assert np.isnan(node.structure.frac_coords[0][0])


def test_inpainting_structure_without_value(aiida_profile):
    node = InpaintingStructure()
    assert list(node.base.attributes.keys()) == []


def test_batched_structures_data_roundtrip(
    aiida_profile, simple_atoms, simple_structure_pmg
):
    data_node = BatchedStructuresData(
        value={"atoms": simple_atoms, "pmg": simple_structure_pmg}
    )

    assert data_node.keys() == ["atoms", "pmg"]
    assert data_node.file_name == "structures.extxyz"

    ase_structs = data_node.get_structures(strct_type="ase")
    assert set(ase_structs.keys()) == {"atoms", "pmg"}
    assert all(isinstance(v, Atoms) for v in ase_structs.values())

    pmg_structs = data_node.get_structures(keys="pmg", strct_type="pymatgen")
    assert set(pmg_structs.keys()) == {"pmg"}
    np.testing.assert_allclose(
        pmg_structs["pmg"].frac_coords, simple_structure_pmg.frac_coords
    )


def test_batched_structures_data_from_batched(
    aiida_profile, simple_atoms, simple_structure_pmg
):
    batched = BatchedStructures(
        {"atoms": simple_atoms, "pmg": simple_structure_pmg}
    )
    data_node = BatchedStructuresData.from_batched_structures(batched)

    value = data_node.value
    assert isinstance(value, BatchedStructures)
    assert value.keys() == batched.keys()


def test_batched_structures_data_invalid_key(aiida_profile, simple_atoms):
    data_node = BatchedStructuresData(value={"atoms": simple_atoms})
    with pytest.raises(ValueError):
        data_node.get_structures(keys=["missing"])


def test_extract_from_batched_structures_returns_aiida(
    aiida_profile, simple_atoms
):
    data_node = BatchedStructuresData(value={"atoms": simple_atoms})
    result = extract_from_batched_structures(data_node, ["atoms"])

    structures = result["structures"]
    assert set(structures.keys()) == {"atoms"}
    assert isinstance(structures["atoms"], StructureData)


def test_batched_structures_data_get_structures_aiida(
    aiida_profile, simple_atoms
):
    data_node = BatchedStructuresData(value={"atoms": simple_atoms})

    structures = data_node.get_structures(
        strct_type="aiida", use_existing_aiida_nodes=False
    )
    assert set(structures.keys()) == {"atoms"}
    assert isinstance(structures["atoms"], StructureData)

    with pytest.raises(ValueError):
        data_node.get_structures(keys=object())
