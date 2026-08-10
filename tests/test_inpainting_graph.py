"""Tests for the InpaintingWorkGraph: build-only wiring and execution."""

import sys
from pathlib import Path

import numpy as np
import pytest
from aiida_workgraph.utils import get_or_create_code
from pymatgen.core.structure import Structure

from xtalpaint.aiida.data import BatchedStructuresData
from xtalpaint.aiida.workgraphs.inpainting import InpaintingWorkGraph
from xtalpaint.data import BatchedStructures
from xtalpaint.inpainting.config_schema import (
    AiiDAOptions,
    CandidateGenerationConfig,
    InpaintingRelaxationConfig,
    RefinementConfig,
    RelaxationAiiDAOptions,
    XtalPaintConfig,
)

_BUILTIN_TASKS = {"graph_inputs", "graph_outputs", "graph_ctx"}

_RELAX_PASSES = {
    "inpainted_constrained_relaxation",
    "pre_relaxed_inpainted_full_relaxation",
    "unrelaxed_inpainted_full_relaxation",
}

# Use the sampling config bundled with XtalPaint: mattergen's built-in default
# resolves relative to its install location and does not exist for wheel
# installs (e.g. in CI).
_SAMPLING_CONF = str(
    (Path(__file__).parent.parent / "src" / "xtalpaint" / "sampling_conf").resolve()
)

_INPAINTING = dict(
    pretrained_name="mattergen_base",
    predictor_corrector="baseline",
    N_steps=5,
    coordinates_snr=0.2,
    n_corrector_steps=1,
    batch_size=16,
    sampling_config_path=_SAMPLING_CONF,
)


@pytest.fixture
def bcc_si():
    return Structure(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _relaxation(constrained=True, full=False, full_direct=False, **params):
    base_params = dict(
        mlip="mattersim",
        optimizer="BFGS",
        load_path="MatterSim-v1.0.0-1M",
        elements_to_relax=["Si", "H"],
    )
    base_params.update(params)
    return InpaintingRelaxationConfig(
        constrained=constrained,
        full=full,
        full_direct=full_direct,
        relax_config=dict(
            params=base_params,
            aiida=RelaxationAiiDAOptions(relax_code_label="python3@localhost"),
        ),
    )


def _build(structure, **overrides):
    config = dict(
        candidate_generation=CandidateGenerationConfig(n_inp=2, element="H"),
        inpainting=_INPAINTING,
        aiida=AiiDAOptions(),
    )
    config.update(overrides)
    return InpaintingWorkGraph.build(
        structures=BatchedStructures({"s": structure}),
        inputs=XtalPaintConfig(**config),
    )


def _user_tasks(wg) -> set:
    return {t.name for t in wg.tasks if t.name not in _BUILTIN_TASKS}


def _source(wg, output: str) -> str:
    """Name of the task that the graph-level *output* socket links from."""
    return wg.outputs[output]._links[0].from_task.name


class TestInpaintingWorkGraphStructure:
    def test_minimal_graph(self, bcc_si):
        wg = _build(bcc_si)
        assert _user_tasks(wg) == {"generate_inpainting_candidates", "inpainting"}
        assert _source(wg, "inpainted_structures") == "inpainting"
        assert _source(wg, "inpainting_candidates") == "generate_inpainting_candidates"

    def test_run_inpainting_false_passes_structures_through(self, bcc_si):
        wg = _build(bcc_si, run_inpainting=False, candidate_generation=None)
        assert _user_tasks(wg) == set()
        assert _source(wg, "inpainted_structures") == "graph_inputs"

    def test_pre_refinement_adds_refine_task(self, bcc_si):
        wg = _build(bcc_si, pre_refinement=RefinementConfig())
        assert "refine_structures" in _user_tasks(wg)
        assert _source(wg, "inpainted_structures") == "refine_structures"

    def test_trajectories_output_only_linked_when_recorded(self, bcc_si):
        wg = _build(bcc_si)
        assert not wg.outputs["inpainted_trajectories"]._links
        wg = _build(bcc_si, inpainting=_INPAINTING | {"record_trajectories": True})
        assert _source(wg, "inpainted_trajectories") == "inpainting"

    @pytest.mark.parametrize(
        "constrained,full,full_direct,expected",
        [
            (True, False, False, {"inpainted_constrained_relaxation"}),
            (False, False, True, {"unrelaxed_inpainted_full_relaxation"}),
            (
                True,
                True,
                False,
                {
                    "inpainted_constrained_relaxation",
                    "pre_relaxed_inpainted_full_relaxation",
                },
            ),
            (True, True, True, _RELAX_PASSES),
        ],
    )
    def test_relaxation_passes_add_matching_subgraphs(
        self, bcc_si, constrained, full, full_direct, expected
    ):
        wg = _build(bcc_si, relaxation=_relaxation(constrained, full, full_direct))
        assert _user_tasks(wg) & _RELAX_PASSES == expected
        # Each active pass wires its output namespace; inactive ones stay unlinked.
        for name in _RELAX_PASSES:
            assert bool(wg.outputs[name]["structures"]._links) == (name in expected)

    def test_full_passes_drop_elements_to_relax(self, bcc_si):
        wg = _build(bcc_si, relaxation=_relaxation(True, True, True))

        def params(task_name):
            return wg.tasks[task_name].inputs.relax_config.params._value

        assert params("inpainted_constrained_relaxation")[
            "elements_to_relax"
        ] == ["Si", "H"]
        assert "elements_to_relax" not in params(
            "unrelaxed_inpainted_full_relaxation"
        )
        assert "elements_to_relax" not in params(
            "pre_relaxed_inpainted_full_relaxation"
        )

    @pytest.mark.parametrize(
        "flags",
        [
            dict(constrained=False, full=False, full_direct=False),
            dict(constrained=False, full=True),  # full requires constrained
        ],
    )
    def test_invalid_pass_combinations_rejected(self, flags):
        with pytest.raises(ValueError):
            _relaxation(**flags)


# ---------------------------------------------------------------------------
# Execution test: runs MatterGen inpainting + MatterSim relaxation
# ---------------------------------------------------------------------------


@pytest.fixture
def python3_code(aiida_localhost):
    """``python3@localhost`` code running the current interpreter."""
    return get_or_create_code(code_path=sys.executable)


class TestInpaintingWorkGraphExecution:
    """Integration test: execute the full WorkGraph on localhost.

    Covers candidate generation → diffusion inpainting → constrained
    relaxation, with all pythonjob tasks running through the
    ``python3@localhost`` code.
    """

    def test_full_pipeline(self, aiida_profile, python3_code, bcc_si):
        wg = _build(
            bcc_si,
            candidate_generation=CandidateGenerationConfig(
                n_inp=2, element="H", num_samples=2
            ),
            relaxation=_relaxation(
                elements_to_relax=["H"], fmax=0.1, max_n_steps=100
            ),
            aiida=AiiDAOptions(default_code_label="python3@localhost"),
        )
        wg.run()
        assert wg.process.is_finished_ok, "WorkGraph did not finish successfully"

        sample_keys = {"s_sample_0", "s_sample_1"}

        # Inpainted structures: one per sample, each with two H sites added.
        inpainted = wg.outputs.inpainted_structures.value
        assert isinstance(inpainted, BatchedStructuresData)
        structures = inpainted.value.get_structures("pymatgen")
        assert set(structures.keys()) == sample_keys
        for s in structures.values():
            assert s.composition["H"] == 2
            assert len(s) == len(bcc_si) + 2

        # Constrained relaxation: structures and finite energies per sample.
        relaxed = wg.outputs.inpainted_constrained_relaxation.structures.value
        assert set(relaxed.value.keys()) == sample_keys
        energies = (
            wg.outputs.inpainted_constrained_relaxation.final_energies.value.value
        )
        assert set(energies.index) == sample_keys
        assert np.isfinite(energies["final_energy"]).all()
