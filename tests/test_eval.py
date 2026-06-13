"""Tests for the eval module."""

import numpy as np
import pandas as pd
import pytest
from ase.io import read
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from xtalpaint.eval import (
    evaluate_inpainting,
)

@pytest.fixture
def reference_structures():
    """Load reference structures from extxyz file."""
    atoms_list = read(
        "tests/data/reference_structures.extxyz",
        index=":",
    )
    adaptor = AseAtomsAdaptor()
    structures = {}
    for atoms in atoms_list:
        key = atoms.info.get("key")
        if key:
            structures[key] = adaptor.get_structure(atoms)
    return structures


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
        site.frac_coords += np.random.normal(scale=0.7, size=3)
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

class TestEvaluateInpainting:
    """Test the evaluate_inpainting function."""

    def test_matching_structures(self, simple_structure):
        """Test evaluation with matching structures."""
        inpainted = {"strct_1": simple_structure}
        reference = {"strct_1": simple_structure}

        df = evaluate_inpainting(
            inpainted, reference, metric="match", max_workers=1
        )

        assert isinstance(df, pd.DataFrame)
        assert "strct_1" in df.index
        assert df.loc["strct_1", "match"].item() is True

    def test_non_matching_structures(
        self, simple_structure, different_structure
    ):
        """Test evaluation with non-matching structures."""
        inpainted = {"strct_1": different_structure}
        reference = {"strct_1": simple_structure}

        df = evaluate_inpainting(
            inpainted, reference, metric="match", max_workers=1
        )

        assert isinstance(df, pd.DataFrame)
        assert "strct_1" in df.index
        assert df.loc["strct_1", "match"].item() is False

    def test_rmsd_metric(self, simple_structure, perturbed_structure):
        """Test evaluation using RMSD metric."""
        inpainted = {"strct_1": perturbed_structure}
        reference = {"strct_1": simple_structure}

        df = evaluate_inpainting(
            inpainted, reference, metric="rmsd", max_workers=1
        )

        assert isinstance(df, pd.DataFrame)
        assert "strct_1" in df.index
        assert isinstance(df.loc["strct_1", "rmsd"], float)
        assert df.loc["strct_1", "rmsd"] > 0.0

    def test_multiple_samples(self, simple_structure, perturbed_structure):
        """Test evaluation with multiple samples per structure."""
        inpainted = {
            "strct_1_sample_0": simple_structure,
            "strct_1_sample_1": perturbed_structure,
        }
        reference = {"strct_1": simple_structure}

        df = evaluate_inpainting(
            inpainted, reference, metric="match", max_workers=1
        )

        assert isinstance(df, pd.DataFrame)
        assert "strct_1_sample_0" in df.index
        assert "strct_1_sample_1" in df.index
        assert len(df) == 2

    def test_nan_structure(self, simple_structure, nan_structure):
        """Test evaluation with NaN structure."""
        inpainted = {"strct_1": nan_structure}
        reference = {"strct_1": simple_structure}

        df = evaluate_inpainting(
            inpainted, reference, metric="match", max_workers=1
        )

        assert isinstance(df, pd.DataFrame)
        assert "strct_1" in df.index
        assert pd.isna(df.loc["strct_1", "match"])

    def test_mismatched_keys(self, simple_structure):
        """Test that mismatched keys raise ValueError."""
        inpainted = {"strct_1": simple_structure}
        reference = {"strct_2": simple_structure}

        with pytest.raises(ValueError, match="keys of inpainted structures"):
            evaluate_inpainting(inpainted, reference, metric="match")

    def test_real_structures_subset(self, reference_structures):
        """Test with a subset of real structures."""
        # Take first 5 structures
        subset_keys = list(reference_structures.keys())[:5]
        reference = {k: reference_structures[k] for k in subset_keys}

        # Remove sample suffix to create base keys
        base_keys = [k.rsplit("_sample_", 1)[0] for k in subset_keys]
        inpainted = {base_keys[i]: reference_structures[k] for i, k in enumerate(subset_keys)}

        df = evaluate_inpainting(
            inpainted,
            {base_keys[i]: reference_structures[k] for i, k in enumerate(subset_keys)},
            metric="match",
            max_workers=2,
        )

        assert isinstance(df, pd.DataFrame)
        # All should match since they're identical
        for key in base_keys:
            assert df.loc[key, "match"].item() is True

    def test_real_structures_rmsd(self, reference_structures):
        """Test RMSD metric with real structures."""
        # Take first 3 structures
        subset_keys = list(reference_structures.keys())[:3]
        base_keys = [k.rsplit("_sample_", 1)[0] for k in subset_keys]

        reference = {base_keys[i]: reference_structures[k] for i, k in enumerate(subset_keys)}
        inpainted = {base_keys[i]: reference_structures[k] for i, k in enumerate(subset_keys)}

        df = evaluate_inpainting(
            inpainted,
            reference,
            metric="rmsd",
            max_workers=1,
        )

        assert isinstance(df, pd.DataFrame)
        # RMSD should be close to 0 for identical structures
        for key in base_keys:
            assert df.loc[key, "rmsd"] == pytest.approx(0.0, abs=1e-3)

    def test_parallel_workers(self, reference_structures):
        """Test evaluation with multiple workers."""
        subset_keys = list(reference_structures.keys())[:10]
        base_keys = [k.rsplit("_sample_", 1)[0] for k in subset_keys]

        reference = {base_keys[i]: reference_structures[k] for i, k in enumerate(subset_keys)}
        inpainted = {base_keys[i]: reference_structures[k] for i, k in enumerate(subset_keys)}

        df = evaluate_inpainting(
            inpainted,
            reference,
            metric="match",
            max_workers=4,
            chunksize=3,
        )

        assert isinstance(df, pd.DataFrame)
        # All should match
        assert len(df) == len(base_keys)
        for key in base_keys:
            assert df.loc[key, "match"].item() is True
