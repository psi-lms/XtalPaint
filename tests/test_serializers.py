"""Tests for the serializers module."""

import pytest
import pandas as pd
import numpy as np
from pymatgen.core import Structure, Lattice
from aiida.orm import StructureData, TrajectoryData

from xtalpaint.aiida.serializers import (
    pymatgen_to_structure_data,
    pymatgen_traj_to_aiida_traj,
    batched_structures_to_batched_structures_data,
    pandas_dataframe_to_pandas_dataframe_data,
)
from xtalpaint.aiida.data import (
    BatchedStructures,
    BatchedStructuresData,
    PandasDataFrameData,
)


@pytest.fixture
def simple_structure():
    """Create a simple pymatgen structure for testing."""
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
    structure.properties["material_id"] = "test-001"
    structure.properties["energy"] = -10.5
    return structure


@pytest.fixture
def pymatgen_trajectory(simple_structure):
    """Create a trajectory of pymatgen structures."""
    structures = []
    for i in range(3):
        s = simple_structure.copy()
        # Slightly perturb the structure
        s.translate_sites(range(len(s)), [0.01 * i, 0, 0])
        structures.append(s)
    return structures


@pytest.fixture
def simple_dataframe():
    """Create a simple pandas DataFrame for testing."""
    return pd.DataFrame({
        "material_id": ["mat-001", "mat-002", "mat-003"],
        "energy": [-10.5, -12.3, -8.7],
        "n_atoms": [4, 6, 8],
    })


@pytest.fixture
def batched_structures(simple_structure):
    """Create a BatchedStructures object for testing."""
    structures = {
        "struct_1": simple_structure,
        "struct_2": simple_structure.copy(),
        "struct_3": simple_structure.copy(),
    }
    return BatchedStructures(structures=structures)


class TestPymatgenToStructureData:
    """Tests for pymatgen_to_structure_data function."""

    def test_basic_conversion(self, aiida_profile, simple_structure):
        """Test basic conversion of pymatgen Structure to StructureData."""
        result = pymatgen_to_structure_data(simple_structure)

        assert isinstance(result, StructureData)
        assert len(result.sites) == len(simple_structure)

    def test_preserves_structure_data(self, aiida_profile, simple_structure):
        """Test that structure data is correctly preserved."""
        result = pymatgen_to_structure_data(simple_structure)

        # Convert back to pymatgen and compare
        recovered = result.get_pymatgen_structure()
        assert len(recovered) == len(simple_structure)
        assert recovered.lattice.a == simple_structure.lattice.a
        assert recovered.lattice.b == simple_structure.lattice.b
        assert recovered.lattice.c == simple_structure.lattice.c

    def test_preserves_species(self, aiida_profile, simple_structure):
        """Test that species are correctly preserved."""
        result = pymatgen_to_structure_data(simple_structure)
        recovered = result.get_pymatgen_structure()

        original_species = [site.specie.symbol for site in simple_structure]
        recovered_species = [site.specie.symbol for site in recovered]
        assert original_species == recovered_species

    def test_with_properties(self, aiida_profile, structure_with_properties):
        """Test conversion of structure with properties."""
        result = pymatgen_to_structure_data(structure_with_properties)

        assert isinstance(result, StructureData)
        # Note: AiiDA StructureData doesn't automatically preserve pymatgen properties
        # The properties are on the input structure, not necessarily the StructureData extras
        assert len(result.sites) == len(structure_with_properties)


class TestPymatgenTrajToAiidaTraj:
    """Tests for pymatgen_traj_to_aiida_traj function."""

    def test_basic_trajectory_conversion(self, aiida_profile, pymatgen_trajectory):
        """Test basic conversion of trajectory."""
        result = pymatgen_traj_to_aiida_traj(pymatgen_trajectory)

        assert isinstance(result, TrajectoryData)
        assert result.numsteps == len(pymatgen_trajectory)

    def test_empty_trajectory(self, aiida_profile):
        """Test conversion of empty trajectory."""
        result = pymatgen_traj_to_aiida_traj([])

        # AiiDA TrajectoryData doesn't support empty trajectories, so None is returned
        assert result is None

    def test_single_structure_trajectory(self, aiida_profile, simple_structure):
        """Test trajectory with single structure."""
        result = pymatgen_traj_to_aiida_traj([simple_structure])

        assert isinstance(result, TrajectoryData)
        assert result.numsteps == 1

    def test_trajectory_preserves_structures(self, aiida_profile, pymatgen_trajectory):
        """Test that structures are correctly preserved in trajectory."""
        result = pymatgen_traj_to_aiida_traj(pymatgen_trajectory)

        assert result.numsteps == len(pymatgen_trajectory)
        # Each structure should be a StructureData node
        for i in range(result.numsteps):
            step_structure = result.get_step_structure(i)
            assert isinstance(step_structure, StructureData)


class TestBatchedStructuresToBatchedStructuresData:
    """Tests for batched_structures_to_batched_structures_data function."""

    def test_basic_conversion(self, aiida_profile, batched_structures):
        """Test basic conversion of BatchedStructures."""
        result = batched_structures_to_batched_structures_data(batched_structures)

        assert isinstance(result, BatchedStructuresData)

    def test_preserves_structure_count(self, aiida_profile, batched_structures):
        """Test that all structures are preserved."""
        result = batched_structures_to_batched_structures_data(batched_structures)

        original_structures = batched_structures.get_structures()
        recovered_batch = result.value
        recovered_structures = recovered_batch.get_structures()

        assert len(recovered_structures) == len(original_structures)

    def test_preserves_structure_keys(self, aiida_profile, batched_structures):
        """Test that structure keys are preserved."""
        result = batched_structures_to_batched_structures_data(batched_structures)

        original_keys = set(batched_structures.get_structures().keys())
        recovered_batch = result.value
        recovered_keys = set(recovered_batch.get_structures().keys())

        assert original_keys == recovered_keys

    def test_empty_batched_structures(self, aiida_profile):
        """Test conversion of empty BatchedStructures."""
        empty_batch = BatchedStructures(structures={})
        result = batched_structures_to_batched_structures_data(empty_batch)

        assert isinstance(result, BatchedStructuresData)
        recovered_batch = result.value
        recovered = recovered_batch.get_structures()
        assert len(recovered) == 0


class TestPandasDataFrameToPandasDataFrameData:
    """Tests for pandas_dataframe_to_pandas_dataframe_data function."""

    def test_basic_conversion(self, aiida_profile, simple_dataframe):
        """Test basic conversion of pandas DataFrame."""
        result = pandas_dataframe_to_pandas_dataframe_data(simple_dataframe)

        assert isinstance(result, PandasDataFrameData)

    def test_preserves_data(self, aiida_profile, simple_dataframe):
        """Test that DataFrame data is correctly preserved."""
        result = pandas_dataframe_to_pandas_dataframe_data(simple_dataframe)
        recovered = result.value

        pd.testing.assert_frame_equal(simple_dataframe, recovered)

    def test_preserves_columns(self, aiida_profile, simple_dataframe):
        """Test that DataFrame columns are preserved."""
        result = pandas_dataframe_to_pandas_dataframe_data(simple_dataframe)
        recovered = result.value

        assert list(recovered.columns) == list(simple_dataframe.columns)

    def test_preserves_dtypes(self, aiida_profile, simple_dataframe):
        """Test that DataFrame dtypes are preserved."""
        result = pandas_dataframe_to_pandas_dataframe_data(simple_dataframe)
        recovered = result.value

        assert recovered.dtypes.to_dict() == simple_dataframe.dtypes.to_dict()

    def test_empty_dataframe(self, aiida_profile):
        """Test conversion of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = pandas_dataframe_to_pandas_dataframe_data(empty_df)

        assert isinstance(result, PandasDataFrameData)
        recovered = result.value
        assert len(recovered) == 0

    def test_dataframe_with_nan(self, aiida_profile):
        """Test DataFrame with NaN values."""
        df_with_nan = pd.DataFrame({
            "a": [1, 2, np.nan],
            "b": [4.0, np.nan, 6.0],
        })
        result = pandas_dataframe_to_pandas_dataframe_data(df_with_nan)
        recovered = result.value

        pd.testing.assert_frame_equal(df_with_nan, recovered)

    def test_dataframe_with_mixed_types(self, aiida_profile):
        """Test DataFrame with mixed column types."""
        mixed_df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        result = pandas_dataframe_to_pandas_dataframe_data(mixed_df)
        recovered = result.value

        pd.testing.assert_frame_equal(mixed_df, recovered)
