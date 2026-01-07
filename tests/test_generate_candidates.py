"""Tests for the generate_candidates module."""

import pytest
import numpy as np
from pymatgen.core import Structure, Lattice
from aiida.orm import StructureData
from xtalpaint.inpainting.generate_candidates import (
    _add_inpainting_sites,
    _structures_to_pymatgen,
    _prepare_inpainting_inputs,
    structure_to_inpainting_candidates,
    generate_inpainting_candidates,
)


@pytest.fixture
def simple_structure():
    """Create a simple structure for testing."""
    lattice = Lattice.cubic(5.0)
    species = ["Si", "Si", "O", "O"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    return Structure(lattice, species, coords)


@pytest.fixture
def structure_with_properties():
    """Create a structure with properties for testing."""
    lattice = Lattice.cubic(5.0)
    species = ["Si", "O"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, species, coords)
    structure.properties["test_prop"] = "test_value"
    return structure


@pytest.fixture
def simple_structure_data(aiida_profile, simple_structure):
    """Create a simple AiiDA StructureData for testing."""
    return StructureData(pymatgen=simple_structure)


class TestAddInpaintingSites:
    """Tests for _add_inpainting_sites function."""

    def test_add_single_site(self, simple_structure):
        """Test adding a single inpainting site."""
        original_len = len(simple_structure)
        result = _add_inpainting_sites(simple_structure, 1, "Si")

        assert len(result) == original_len + 1
        assert result[-1].specie.symbol == "Si"
        assert np.all(np.isnan(result[-1].frac_coords))

    def test_add_multiple_sites(self, simple_structure):
        """Test adding multiple inpainting sites."""
        original_len = len(simple_structure)
        n_sites = 3
        result = _add_inpainting_sites(simple_structure, n_sites, "O")

        assert len(result) == original_len + n_sites
        for i in range(n_sites):
            assert result[-(i+1)].specie.symbol == "O"
            assert np.all(np.isnan(result[-(i+1)].frac_coords))

    def test_does_not_modify_original(self, simple_structure):
        """Test that original structure is not modified."""
        original_len = len(simple_structure)
        _add_inpainting_sites(simple_structure, 2, "Si")

        assert len(simple_structure) == original_len

    def test_add_zero_sites(self, simple_structure):
        """Test adding zero sites."""
        original_len = len(simple_structure)
        result = _add_inpainting_sites(simple_structure, 0, "Si")

        assert len(result) == original_len


class TestStructuresToPymatgen:
    """Tests for _structures_to_pymatgen function."""

    def test_list_of_structures(self, simple_structure):
        """Test conversion of list of Structure objects."""
        structures = [simple_structure, simple_structure.copy()]
        result = _structures_to_pymatgen(structures)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "0" in result and "1" in result
        assert all(isinstance(s, Structure) for s in result.values())

    def test_dict_of_structures(self, simple_structure):
        """Test conversion of dict of Structure objects."""
        structures = {"strct1": simple_structure, "strct2": simple_structure.copy()}
        result = _structures_to_pymatgen(structures)

        assert result == structures
        assert "strct1" in result and "strct2" in result

    def test_dict_of_structure_data(self, aiida_profile, simple_structure):
        """Test conversion of dict of StructureData objects."""
        structure_data1 = StructureData(pymatgen=simple_structure)
        strct2 = simple_structure.copy()
        structure_data2 = StructureData(pymatgen=strct2)

        structures = {"strct1": structure_data1, "strct2": structure_data2}
        result = _structures_to_pymatgen(structures)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert result["strct1"].properties["uuid"] == structure_data1.uuid
        assert result["strct2"].properties["uuid"] == structure_data2.uuid
        assert isinstance(result["strct1"], Structure)
        assert isinstance(result["strct2"], Structure)

    def test_list_of_structure_data(self, aiida_profile, simple_structure):
        """Test conversion of list of StructureData objects."""
        structure_data1 = StructureData(pymatgen=simple_structure)
        structure_data2 = StructureData(pymatgen=simple_structure.copy())

        structures = [structure_data1, structure_data2]
        result = _structures_to_pymatgen(structures)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "0" in result and "1" in result
        assert result["0"].properties["uuid"] == structure_data1.uuid
        assert result["1"].properties["uuid"] == structure_data2.uuid

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with pytest.raises(TypeError, match="Input must be a list or dictionary"):
            _structures_to_pymatgen("invalid")


class TestPrepareInpaintingInputs:
    """Tests for _prepare_inpainting_inputs function."""

    def test_single_structure_single_element(self, simple_structure):
        """Test with single structure and element."""
        structures, n_inp, element = _prepare_inpainting_inputs(
            simple_structure, 2, "Si"
        )

        assert isinstance(structures, dict)
        assert "0" in structures
        assert element == {"0": "Si"}
        assert n_inp == {"0": 2}

    def test_list_of_structures(self, simple_structure):
        """Test with list of structures."""
        structures, n_inp, element = _prepare_inpainting_inputs(
            [simple_structure, simple_structure.copy()], 3, "O"
        )

        assert len(structures) == 2
        assert element == {"0": "O", "1": "O"}
        assert n_inp == {"0": 3, "1": 3}

    def test_dict_element(self, simple_structure):
        """Test with dict of elements."""
        structures, n_inp, element = _prepare_inpainting_inputs(
            {"s1": simple_structure}, {"s1": 2}, {"s1": "Si"}
        )

        assert element == {"s1": "Si"}

    def test_tuple_n_inp(self, simple_structure):
        """Test with tuple for n_inp."""
        structures, n_inp, element = _prepare_inpainting_inputs(
            simple_structure, (1, 3), "Si"
        )

        assert n_inp == {"0": (1, 3)}

    def test_invalid_element_type(self, simple_structure):
        """Test with invalid element type."""
        with pytest.raises(ValueError, match="element must be a str or a dict"):
            _prepare_inpainting_inputs(simple_structure, 2, ["Si", "O"])

    def test_invalid_n_inp_type(self, simple_structure):
        """Test with invalid n_inp type."""
        with pytest.raises(ValueError, match="n_inp must be an int, tuple, or dict"):
            _prepare_inpainting_inputs(simple_structure, [1, 2], "Si")

    def test_invalid_n_inp_tuple_length(self, simple_structure):
        """Test with invalid tuple length for n_inp."""
        with pytest.raises(ValueError, match="n_inp must be an int or a list of two ints"):
            _prepare_inpainting_inputs({"s1": simple_structure}, {"s1": (1, 2, 3)}, "Si")


class TestStructureToInpaintingCandidates:
    """Tests for structure_to_inpainting_candidates function."""

    def test_single_num_sites(self, simple_structure):
        """Test with single number of inpainting sites."""
        result = structure_to_inpainting_candidates(
            simple_structure, "test", 2, "O", num_samples=1
        )

        assert len(result) == 1
        assert "test" in result
        structure = result["test"]
        # Original has 4 sites, removing 2 O atoms leaves 2, adding 2 gives 4
        assert len(structure) == len(simple_structure)
        assert structure.properties["material_id"] == "test"

    def test_range_num_sites(self, simple_structure):
        """Test with range of inpainting sites."""
        result = structure_to_inpainting_candidates(
            simple_structure, "test", (1, 3), "O", num_samples=1
        )

        assert len(result) == 3  # 1, 2, 3 sites
        assert "test_n_inp_1" in result
        assert "test_n_inp_2" in result
        assert "test_n_inp_3" in result

    def test_multiple_samples(self, simple_structure):
        """Test with multiple samples."""
        result = structure_to_inpainting_candidates(
            simple_structure, "test", 2, "O", num_samples=3
        )

        assert len(result) == 3
        assert "test_sample_0" in result
        assert "test_sample_1" in result
        assert "test_sample_2" in result
        assert len(result["test_sample_1"]) == 4

        # Test case when keeping existing sites
        result = structure_to_inpainting_candidates(
            simple_structure, "test", 2, "O", num_samples=1,
            remove_existing_sites=False
        )
        assert len(result["test"]) == 6

    def test_range_and_samples(self, simple_structure):
        """Test with both range and multiple samples."""
        result = structure_to_inpainting_candidates(
            simple_structure, "test", (1, 2), "O", num_samples=2
        )

        assert len(result) == 4  # 2 ranges * 2 samples
        assert "test_n_inp_1_sample_0" in result
        assert "test_n_inp_2_sample_1" in result

    def test_removes_correct_species(self, simple_structure):
        """Test that correct species are removed."""
        original_o_count = sum(1 for site in simple_structure if site.specie.symbol == "O")
        result = structure_to_inpainting_candidates(
            simple_structure, "test", 1, "O", num_samples=1
        )

        structure = result["test"]
        # All O should be removed, then 1 added with NaN coords
        o_sites = [site for site in structure if site.specie.symbol == "O"]
        assert len(o_sites) == 1
        assert np.all(np.isnan(o_sites[0].frac_coords))

    def test_invalid_tuple_length(self, simple_structure):
        """Test with invalid tuple length."""
        with pytest.raises(ValueError, match="num_inpaint_sites must be an int or a tuple"):
            structure_to_inpainting_candidates(
                simple_structure, "test", (1, 2, 3), "O"
            )


class TestGenerateInpaintingCandidates:
    """Tests for generate_inpainting_candidates function."""

    def test_single_structure(self, simple_structure):
        """Test with single structure."""
        result = generate_inpainting_candidates(simple_structure, 2, "O")

        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_list_of_structures(self, simple_structure):
        """Test with list of structures."""
        structures = [simple_structure, simple_structure.copy()]
        result = generate_inpainting_candidates(structures, 2, "Si")

        assert len(result) >= 2

    def test_dict_of_structures(self, simple_structure):
        """Test with dict of structures."""
        structures = {"s1": simple_structure, "s2": simple_structure.copy()}
        result = generate_inpainting_candidates(structures, 1, "O")

        assert "s1" in result or any("s1" in key for key in result.keys())
        assert "s2" in result or any("s2" in key for key in result.keys())

    def test_range_n_inp(self, simple_structure):
        """Test with range for n_inp."""
        result = generate_inpainting_candidates(simple_structure, (1, 3), "O")

        assert len(result) == 3  # 1, 2, 3 sites

    def test_multiple_samples(self, simple_structure):
        """Test with multiple samples."""
        result = generate_inpainting_candidates(
            simple_structure, 2, "O", num_samples=5
        )

        assert len(result) == 5

    def test_dict_n_inp(self, simple_structure):
        """Test with dict for n_inp."""
        structures = {"s1": simple_structure, "s2": simple_structure.copy()}
        n_inp = {"s1": 1, "s2": 2}
        result = generate_inpainting_candidates(structures, n_inp, "O")

        assert len(result) == 2

    def test_dict_element(self, simple_structure):
        """Test with dict for element."""
        structures = {"s1": simple_structure, "s2": simple_structure.copy()}
        element = {"s1": "Si", "s2": "O"}
        result = generate_inpainting_candidates(structures, 1, element)

        assert len(result) == 2

    def test_preserves_properties(self, structure_with_properties):
        """Test that structure properties are preserved."""
        result = generate_inpainting_candidates(structure_with_properties, 1, "O")

        # At least one result should have the material_id property
        assert any("material_id" in s.properties for s in result.values())
