"""Tests for relaxation_graph and filter_unique_structures."""

import sys

from aiida import orm
import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice
from pymatgen.core.structure import Structure

from xtalpaint.aiida.data import BatchedStructuresData
from xtalpaint.aiida.workgraphs.relaxation import relaxation_graph
from xtalpaint.data import BatchedStructures
from xtalpaint.eval import filter_unique_structures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bcc_si():
    """BCC silicon — space group Im-3m (229)."""
    return Structure(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture
def fcc_al():
    """FCC aluminium — space group Fm-3m (225)."""
    a = 4.05
    return Structure(
        [[a, 0, 0], [0, a, 0], [0, 0, a]],
        ["Al", "Al", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


@pytest.fixture
def nan_si(bcc_si):
    """BCC silicon with a NaN fractional coordinate."""
    s = bcc_si.copy()
    s[0].frac_coords[0] = np.nan
    return s


# ---------------------------------------------------------------------------
# relaxation_graph: graph-structure tests (no AiiDA runtime required)
# ---------------------------------------------------------------------------


# graph_inputs / graph_outputs / graph_ctx are always present in every built
# WorkGraph; they are infrastructure nodes, not user tasks.
_BUILTIN_TASKS = {"graph_inputs", "graph_outputs", "graph_ctx"}


def _user_tasks(wg) -> set:
    return {t.name for t in wg.tasks if t.name not in _BUILTIN_TASKS}


def _structures_source(wg) -> str:
    """Return the task name that the graph-level ``structures`` output links from."""
    return wg.outputs["structures"]._links[0].from_task.name


class TestRelaxationGraphStructure:
    """Verify that relaxation_graph wires the correct tasks for each flag combination.

    These tests call ``.build()`` to materialise the inner WorkGraph without
    running any AiiDA processes.  They check which tasks exist, which output
    sockets are declared, and that the ``structures`` output socket is wired to
    the last active task in the chain.
    """

    def _build(self, structures, **kwargs):
        return relaxation_graph.build(
            structures=structures,
            relax_inputs={},
            **kwargs,
        )

    def test_base_contains_only_relaxation_task(self, bcc_si):
        wg = self._build(BatchedStructures({"s": bcc_si}))
        assert _user_tasks(wg) == {"relaxation_task"}

    def test_refine_flag_adds_refinement_task(self, bcc_si):
        wg = self._build(BatchedStructures({"s": bcc_si}), refine=True)
        assert _user_tasks(wg) == {"relaxation_task", "refine_structures_task"}

    def test_filter_unique_flag_adds_uniqueness_task(self, bcc_si):
        wg = self._build(BatchedStructures({"s": bcc_si}), filter_unique=True)
        assert _user_tasks(wg) == {"relaxation_task", "uniqueness_filter_task"}

    def test_both_flags_produce_full_chain(self, bcc_si):
        wg = self._build(
            BatchedStructures({"s": bcc_si}), refine=True, filter_unique=True
        )
        assert _user_tasks(wg) == {
            "relaxation_task",
            "refine_structures_task",
            "uniqueness_filter_task",
        }

    @pytest.mark.parametrize(
        "refine,filter_unique",
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_structures_and_energies_always_in_outputs(
        self, bcc_si, refine, filter_unique
    ):
        wg = self._build(
            BatchedStructures({"s": bcc_si}),
            refine=refine,
            filter_unique=filter_unique,
        )
        assert "structures" in wg.outputs
        assert "final_energies" in wg.outputs

    def test_optional_force_energy_sockets_declared(self, bcc_si):
        """initial_energies / initial_forces / final_forces are declared as
        optional sockets even though they are only populated at runtime when
        requested via relax_inputs."""
        wg = self._build(BatchedStructures({"s": bcc_si}))
        for socket in ("initial_energies", "initial_forces", "final_forces"):
            assert socket in wg.outputs

    @pytest.mark.parametrize(
        "refine,filter_unique,expected_src",
        [
            (False, False, "relaxation_task"),
            (True, False, "refine_structures_task"),
            (False, True, "uniqueness_filter_task"),
            (True, True, "uniqueness_filter_task"),
        ],
    )
    def test_structures_output_linked_to_last_active_task(
        self, bcc_si, refine, filter_unique, expected_src
    ):
        """The graph-level ``structures`` output must be wired to the final
        step in the active chain, not hardcoded to the relaxation task."""
        wg = self._build(
            BatchedStructures({"s": bcc_si}),
            refine=refine,
            filter_unique=filter_unique,
        )
        assert _structures_source(wg) == expected_src


# ---------------------------------------------------------------------------
# filter_unique_structures: functional tests
# ---------------------------------------------------------------------------


class TestFilterUniqueStructures:
    """Tests for the pure-Python filter_unique_structures function."""

    def test_identical_samples_collapsed_to_one(self, bcc_si):
        """Two identical samples of the same parent become one representative."""
        structures = BatchedStructures(
            {"s_sample_0": bcc_si, "s_sample_1": bcc_si}
        )
        result = filter_unique_structures(structures)
        assert len(result.keys()) == 1

    def test_distinct_compositions_both_kept(self, bcc_si, fcc_al):
        """Samples of different compositions can never be StructureMatcher-equal."""
        structures = BatchedStructures(
            {
                "a_sample_0": bcc_si,
                "a_sample_1": fcc_al,
            }
        )
        result = filter_unique_structures(structures)
        assert len(result.keys()) == 2

    def test_different_space_groups_both_kept(self, bcc_si, fcc_al):
        """Samples assigned to different space groups are never merged, even
        if StructureMatcher would call them equal (it won't across SG groups)."""
        structures = BatchedStructures(
            {"p_sample_0": bcc_si, "p_sample_1": fcc_al}
        )
        result = filter_unique_structures(structures)
        # bcc_si (SG 229) and fcc_al (SG 225) land in different bins → both kept
        assert len(result.keys()) == 2

    def test_nan_structures_are_excluded(self, bcc_si, nan_si):
        """NaN-coordinate structures are dropped entirely."""
        structures = BatchedStructures(
            {"s_sample_0": nan_si, "s_sample_1": bcc_si}
        )
        result = filter_unique_structures(structures)
        assert len(result.keys()) == 1
        assert "s_sample_0" not in result.keys()

    def test_all_nan_returns_empty(self, nan_si):
        structures = BatchedStructures(
            {"s_sample_0": nan_si, "s_sample_1": nan_si}
        )
        result = filter_unique_structures(structures)
        assert len(result.keys()) == 0

    def test_multiple_parents_filtered_independently(self, bcc_si, fcc_al):
        """Each parent key group is deduplicated independently."""
        structures = BatchedStructures(
            {
                "a_sample_0": bcc_si,
                "a_sample_1": bcc_si,  # duplicate of a_sample_0
                "b_sample_0": fcc_al,
                "b_sample_1": fcc_al,  # duplicate of b_sample_0
            }
        )
        result = filter_unique_structures(structures)
        keys = result.keys()
        # One unique per parent
        assert len(keys) == 2
        a_keys = [k for k in keys if k.startswith("a_")]
        b_keys = [k for k in keys if k.startswith("b_")]
        assert len(a_keys) == 1
        assert len(b_keys) == 1

    def test_accepts_plain_dict(self, bcc_si):
        """filter_unique_structures works with a plain dict, not just BatchedStructures."""
        result = filter_unique_structures({"s": bcc_si})
        assert len(result.keys()) == 1

    def test_returns_batched_structures(self, bcc_si):
        result = filter_unique_structures({"s": bcc_si})
        assert isinstance(result, BatchedStructures)


# ---------------------------------------------------------------------------
# Fixtures for execution tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diamond_si():
    """Diamond-cubic silicon (Fd-3m, SG 227) — 8-atom conventional cell."""
    a = 5.43
    return Structure(
        Lattice.cubic(a),
        ["Si"] * 8,
        [
            [0.000, 0.000, 0.000],
            [0.250, 0.250, 0.250],
            [0.500, 0.500, 0.000],
            [0.750, 0.750, 0.250],
            [0.500, 0.000, 0.500],
            [0.750, 0.250, 0.750],
            [0.000, 0.500, 0.500],
            [0.250, 0.750, 0.750],
        ],
    )


@pytest.fixture(scope="module")
def strained_si(diamond_si):
    """Diamond silicon compressed to 98 % of equilibrium volume.

    MatterSim will relax this back to the same diamond-cubic minimum as
    ``diamond_si``, which lets us verify that the uniqueness filter correctly
    identifies the two relaxed structures as equivalent.
    """
    s = diamond_si.copy()
    s.apply_strain(-0.02)
    return s


@pytest.fixture(scope="module")
def fcc_al_conventional():
    """FCC aluminium (Fm-3m, SG 225) — 4-atom conventional cell."""
    a = 4.05
    return Structure(
        Lattice.cubic(a),
        ["Al"] * 4,
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
    )


# MatterSim relax settings kept minimal so the tests run quickly on CPU.
_MATTERSIM_RELAX_INPUTS = {
    "device": "cpu",
    "mlip": "mattersim",
    "fmax": 0.1,
    "max_n_steps": 100,
}


# ---------------------------------------------------------------------------
# TestRelaxationGraphExecution
# ---------------------------------------------------------------------------


class TestRelaxationGraphExecution:
    """Integration tests: actually execute the WorkGraph with MatterSim.

    These tests require a temporary AiiDA profile (``aiida_profile``) and a
    configured localhost computer (``aiida_localhost``).  The WorkGraph tasks
    are pythonjob tasks that run in a subprocess using the current Python
    executable so that all installed packages (including MatterSim) are
    available.

    Each test method requests both ``aiida_profile`` and ``aiida_localhost``
    as fixtures to ensure the AiiDA backend and transport layer are ready
    before the WorkGraph is submitted.
    """

    def _build_and_run(self, structures: dict, **graph_kwargs):
        wg = relaxation_graph.build(
            structures=BatchedStructures(structures),
            relax_inputs=_MATTERSIM_RELAX_INPUTS,
            command_info={"filepath_executable": sys.executable},
            **graph_kwargs,
        )
        wg.run()

        assert wg.process.is_finished_ok, "WorkGraph did not finish successfully"

        return wg

    def test_basic_relaxation_returns_structures_and_energies(
        self, aiida_profile, aiida_localhost, diamond_si, fcc_al_conventional
    ):
        """WorkGraph runs to completion and returns one relaxed structure and
        one energy row per input."""
        wg = self._build_and_run(
            {"si": diamond_si, "al": fcc_al_conventional},
        )

        assert wg.state == "FINISHED"

        # --- structures output ---
        out_node = wg.tasks.relaxation_task.outputs.structures.value
        assert isinstance(out_node, BatchedStructuresData)
        out_keys = set(out_node.value.keys())
        assert out_keys == {"si", "al"}

        # --- final_energies output ---
        energies_node = wg.tasks.relaxation_task.outputs.final_energies.value
        energies_df: pd.DataFrame = energies_node.value
        assert isinstance(energies_df, pd.DataFrame)
        assert set(energies_df.index) == {"si", "al"}
        assert (energies_df["final_energy"] < 0).all(), (
            "Energies from MatterSim should be negative (eV)"
        )

    def test_filter_unique_deduplicates_identical_relaxed_copies(
        self,
        aiida_profile,
        aiida_localhost,
        diamond_si,
        strained_si,
        fcc_al_conventional,
    ):
        """Three Si samples (two identical, one 2 %-strained) and two Al
        samples are passed through the full relaxation + uniqueness-filter
        pipeline.  After relaxation all three Si structures converge to the
        same diamond-cubic minimum, so the filter should retain exactly one Si
        representative.  The two identical Al copies should likewise collapse
        to one, leaving two unique structures in total."""
        structures = {
            # Three Si samples with the same parent key "si" — all should
            # relax to diamond-cubic Si and be deduplicated to one.
            "si_sample_0": diamond_si,
            "si_sample_1": diamond_si,   # exact copy
            "si_sample_2": strained_si,  # 2 % compressed, same basin
            # Two Al samples with parent key "al" — both relax to FCC Al.
            "al_sample_0": fcc_al_conventional,
            "al_sample_1": fcc_al_conventional,  # exact copy
        }

        wg = self._build_and_run(structures, filter_unique=True)

        assert wg.state == "FINISHED"

        unique_node = (
            wg.tasks.uniqueness_filter_task.outputs.unique_structures.value
        )
        assert isinstance(unique_node, BatchedStructuresData)
        unique_keys = list(unique_node.value.keys())

        si_keys = [k for k in unique_keys if k.startswith("si_")]
        al_keys = [k for k in unique_keys if k.startswith("al_")]

        assert len(si_keys) == 1, (
            f"Expected 1 unique Si structure after deduplication, "
            f"got {len(si_keys)}: {si_keys}"
        )
        assert len(al_keys) == 1, (
            f"Expected 1 unique Al structure after deduplication, "
            f"got {len(al_keys)}: {al_keys}"
        )
