"""Tests for the eval module."""

import numpy as np
import pytest
from pymatgen.core.structure import Structure

from dbcsi_inpainting.eval import (
    _check_for_nan,
    evaluate_inpainting,
    get_structure_keys,
)


@pytest.fixture
def simple_structure():
    """Create a simple cubic structure for testing."""
    lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    species = ["Si", "Si"]
    coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


@pytest.fixture
def perturbed_structure(simple_structure):
    """Create a slightly perturbed version of simple_structure."""
    structure = simple_structure.copy()
    for site in structure.sites:
        site.frac_coords += np.random.normal(scale=0.1, size=3)
        site.frac_coords = site.frac_coords % 1.0  # Ensure coordinates are within [0, 1)

    return structure


@pytest.fixture
def different_structure():
    """Create a different structure."""
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    species = ["Al", "Al"]
    coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


@pytest.fixture
def nan_structure():
    """Create a structure with NaN coordinates."""
    lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    species = ["Si", "Si"]
    coords = [[np.nan, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


class TestCheckForNan:
    """Test the _check_for_nan function."""

    def test_no_nan(self, simple_structure):
        """Test structure without NaN values."""
        assert not _check_for_nan(simple_structure)

    def test_with_nan(self, nan_structure):
        """Test structure with NaN values."""
        assert _check_for_nan(nan_structure)

class TestGetStructureKeys:
    """Test the get_structure_keys function."""

    def test_no_samples(self):
        """Test with keys without sample indices."""
        structures = {
            "structure_1": None,
            "structure_2": None,
        }
        keys, indices = get_structure_keys(structures)
        assert keys == ["structure_1", "structure_2"]
        assert indices == [None, None]

    def test_with_samples(self):
        """Test with keys containing sample indices."""
        structures = {
            "structure_1_sample_0": None,
            "structure_1_sample_1": None,
            "structure_2_sample_0": None,
        }
        keys, indices = get_structure_keys(structures)
        assert keys == ["structure_1", "structure_1", "structure_2"]
        assert indices == ["0", "1", "0"]

    def test_mixed_samples(self):
        """Test with mixed keys (some with samples, some without)."""
        structures = {
            "structure_1": None,
            "structure_2_sample_0": None,
            "structure_2_sample_1": None,
        }
        keys, indices = get_structure_keys(structures)
        assert keys == ["structure_1", "structure_2", "structure_2"]
        assert indices == [None, "0", "1"]


class TestEvaluateInpainting:
    """Test the evaluate_inpainting function."""

    def test_matching_structures(self, simple_structure):
        """Test evaluation with matching structures."""
        inpainted = {"strct_1": simple_structure}
        reference = {"strct_1": simple_structure}

        agg, individual = evaluate_inpainting(
            inpainted, reference, metric="match", max_workers=1
        )

        assert "strct_1" in agg
        assert agg["strct_1"] == [True]
        assert individual["strct_1"] is True

    def test_non_matching_structures(
        self, simple_structure, different_structure
    ):
        """Test evaluation with non-matching structures."""
        inpainted = {"strct_1": different_structure}
        reference = {"strct_1": simple_structure}

        agg, individual = evaluate_inpainting(
            inpainted, reference, metric="match", max_workers=1
        )

        assert "strct_1" in agg
        assert agg["strct_1"] == [False]
        assert individual["strct_1"] is False

    def test_rmsd_metric(self, simple_structure, perturbed_structure):
        """Test evaluation using RMSD metric."""
        inpainted = {"strct_1": perturbed_structure}
        reference = {"strct_1": simple_structure}

        agg, individual = evaluate_inpainting(
            inpainted, reference, metric="rmsd", max_workers=1
        )

        assert "strct_1" in agg
        assert isinstance(agg["strct_1"][0], float)
        assert agg["strct_1"][0] > 0.0

    def test_multiple_samples(self, simple_structure, perturbed_structure):
        """Test evaluation with multiple samples per structure."""
        inpainted = {
            "strct_1_sample_0": simple_structure,
            "strct_1_sample_1": perturbed_structure,
        }
        reference = {"strct_1": simple_structure}

        agg, individual = evaluate_inpainting(
            inpainted, reference, metric="match", max_workers=1
        )

        assert "strct_1" in agg
        assert len(agg["strct_1"]) == 2
        assert "strct_1_sample_0" in individual
        assert "strct_1_sample_1" in individual

    def test_nan_structure(self, simple_structure, nan_structure):
        """Test evaluation with NaN structure."""
        inpainted = {"strct_1": nan_structure}
        reference = {"strct_1": simple_structure}

        agg, individual = evaluate_inpainting(
            inpainted, reference, metric="match", max_workers=1
        )

        assert "strct_1" in agg
        assert agg["strct_1"][0] is None
        assert individual["strct_1"] is None

    def test_mismatched_keys(self, simple_structure):
        """Test that mismatched keys raise ValueError."""
        inpainted = {"strct_1": simple_structure}
        reference = {"strct_2": simple_structure}

        with pytest.raises(ValueError, match="keys of inpainted structures"):
            evaluate_inpainting(inpainted, reference, metric="match")
