"""Tests for relaxation_utils module."""

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core import Lattice, Structure

# Try to import dependencies, skip all tests if not available
pytest.importorskip("mattersim", reason="MatterSim not available")

from xtalpaint.utils.relaxation_utils import (
    _load_calculator,
    _relax_atoms_mlip,
    relax_structures,
    relax_atoms_mattersim_batched,
)


@pytest.fixture
def sample_structure():
    """Create a simple pymatgen Structure for testing."""
    lattice = Lattice.cubic(4.0)
    species = ["Si", "Si"]
    coords = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    return Structure(lattice, species, coords)


@pytest.fixture
def sample_atoms():
    """Create a simple ASE Atoms object for testing."""
    positions = [[0, 0, 0], [1, 1, 1]]
    cell = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    return Atoms("Si2", positions=positions, cell=cell, pbc=True)


class TestRelaxAtomsMatterSim:
    """Tests for relax_atoms_mattersim_batched function."""

    def test_relax_single_atoms(self, sample_atoms):
        """Test relaxing a single Atoms object."""
        relaxed_atoms, energies = relax_atoms_mattersim_batched(
            atoms=[sample_atoms],
            device="cpu",
            max_n_steps=5,
            fmax=0.5,
        )

        assert len(relaxed_atoms) == 1
        assert isinstance(relaxed_atoms[0], Atoms)
        assert len(energies) == 1
        assert isinstance(energies[0], (float, np.floating))

    def test_relax_multiple_atoms(self, sample_atoms):
        """Test relaxing multiple Atoms objects."""
        atoms_list = [sample_atoms, sample_atoms.copy()]

        relaxed_atoms, energies = relax_atoms_mattersim_batched(
            atoms=atoms_list,
            device="cpu",
            max_n_steps=5,
            fmax=0.5,
        )

        assert len(relaxed_atoms) == 2
        assert all(isinstance(a, Atoms) for a in relaxed_atoms)
        assert len(energies) == 2


class TestLoadCalculator:
    """Tests for _load_calculator function."""

    def test_unsupported_mlip_raises_error(self):
        """Test that unsupported MLIP raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported mlip"):
            _load_calculator(
                mlip="unsupported_mlip",
                device="cpu",
                load_path=None,
                default_dtype="float32",
            )


class TestRelaxAtomsMlip:
    """Tests for _relax_atoms_mlip function."""

    def test_unsupported_optimizer_raises_error(self, sample_atoms):
        """Test that unsupported optimizer raises ValueError."""
        # Note: We don't set a calculator here since we're testing optimizer validation
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            _relax_atoms_mlip(
                atoms=sample_atoms,
                fmax=0.1,
                steps=10,
                optimizer="invalid_optimizer",
            )

    def test_filter_not_implemented(self, sample_atoms):
        """Test that filter parameter raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Filter not implemented"):
            _relax_atoms_mlip(
                atoms=sample_atoms,
                fmax=0.1,
                steps=10,
                optimizer="bfgs",
                filter="some_filter",
            )


class TestRelaxStructures:
    """Tests for relax_structures function."""

    def test_relax_single_structure(self, sample_structure):
        """Test relaxing a single structure."""
        relaxed, energies, _, _, _ = relax_structures(
            structures=sample_structure,
            device="cpu",
            mlip="mattersim",
            max_n_steps=5,
            fmax=0.5,
        )

        assert len(relaxed) == 1
        assert isinstance(relaxed[0], Structure)
        assert len(energies) == 1
        assert isinstance(energies[0], (float, np.floating))

    def test_relax_multiple_structures(self, sample_structure):
        """Test relaxing multiple structures."""
        structures = [sample_structure, sample_structure.copy()]

        relaxed, energies, _, _, _ = relax_structures(
            structures=structures,
            device="cpu",
            mlip="mattersim",
            max_n_steps=5,
            fmax=0.5,
        )

        assert len(relaxed) == 2
        assert all(isinstance(s, Structure) for s in relaxed)
        assert len(energies) == 2

    def test_relax_with_constraints(self, sample_structure):
        """Test relaxation with element constraints."""
        relaxed, energies, _, _, _ = relax_structures(
            structures=sample_structure,
            device="cpu",
            mlip="mattersim",
            elements_to_relax=["Si"],
            max_n_steps=5,
            fmax=0.5,
        )

        assert len(relaxed) == 1
        assert isinstance(relaxed[0], Structure)

    def test_mattersim_batched_return_metrics_raises_error(self, sample_structure):
        """Test that MatterSim raises error when requesting initial/final metrics."""
        with pytest.raises(ValueError, match="does not support"):
            relax_structures(
                structures=sample_structure,
                device="cpu",
                mlip="mattersim-batched",
                return_initial_energies=True,
                max_n_steps=5,
                fmax=0.5,
            )

        with pytest.raises(ValueError, match="does not support"):
            relax_structures(
                structures=sample_structure,
                device="cpu",
                mlip="mattersim-batched",
                return_initial_forces=True,
                max_n_steps=5,
                fmax=0.5,
            )

        with pytest.raises(ValueError, match="does not support"):
            relax_structures(
                structures=sample_structure,
                device="cpu",
                mlip="mattersim-batched",
                return_final_forces=True,
                max_n_steps=5,
                fmax=0.5,
            )

    def test_custom_optimizer_params(self, sample_structure):
        """Test relaxation with custom optimizer parameters."""
        relaxed, energies, _, _, _ = relax_structures(
            structures=sample_structure,
            device="cpu",
            mlip="mattersim",
            optimizer="fire",
            max_n_steps=10,
            fmax=0.5,
        )

        assert len(relaxed) == 1
        assert len(energies) == 1
