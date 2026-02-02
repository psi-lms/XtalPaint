"""AiiDA WorkGraph tasks for structure relaxation and inpainting."""

import typing as t

import pandas as pd
from aiida_workgraph import spec, task
from aiida_workgraph.socket_spec import meta
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from xtalpaint.aiida.data import (
    BatchedStructures,
    BatchedStructuresData,
)
from xtalpaint.aiida.tasks.relax_parallel_utils import (
    _relax_mpi_parallel,
)
from xtalpaint.eval import evaluate_inpainting
from xtalpaint.inpainting.generate_candidates import (
    generate_inpainting_candidates,
)
from xtalpaint.inpainting.inpainting_process import (
    run_inpainting_pipeline,
    run_mpi_parallel_inpainting_pipeline,
)
from xtalpaint.utils.relaxation_utils import relax_structures


@task.pythonjob(
    outputs=spec.namespace(candidates=t.Any),
)
def _generate_inpainting_candidates_task(
    structures: t.Union[Structure, t.Iterable[Structure]] | BatchedStructures,
    n_inp: t.Union[
        int, t.Tuple[int, int], t.List[int], t.List[t.Tuple[int, int]]
    ],
    element: t.Union[str, t.List[str]],
    num_samples: int = 1,
) -> BatchedStructures:
    if isinstance(structures, BatchedStructures):
        structures = structures.get_structures("pymatgen")
    candidates = generate_inpainting_candidates(
        structures=structures,
        n_inp=n_inp,
        element=element,
        num_samples=num_samples,
    )
    return {"candidates": BatchedStructures(candidates)}


@task.pythonjob(
    outputs=spec.namespace(structures=t.Any),
)
def _refine_structures_task(
    structures: t.Union[Structure, t.Iterable[Structure]] | BatchedStructures,
    refinement_symprec: float,
) -> BatchedStructures:
    """Refine structures to standard conventional cells."""
    if isinstance(structures, BatchedStructures):
        structures = structures.get_structures("pymatgen")

    refined_structures = {}
    for k, s in structures.items():
        analyzer = SpacegroupAnalyzer(s, symprec=refinement_symprec)
        try:
            refined_structure = analyzer.get_refined_structure()
        except Exception:
            refined_structure = s
        refined_structures[k] = refined_structure

    return {"structures": BatchedStructures(refined_structures)}


@task.pythonjob(
    outputs=spec.namespace(
        structures=t.Any,
        relaxed_structures=spec.socket(t.Any, required=False),
        trajectories=t.Annotated[
            dict, spec.dynamic(t.Any), meta(required=False)
        ],
        mean_trajectories=t.Annotated[
            dict, spec.dynamic(t.Any), meta(required=False)
        ],
    )
)
def _inpainting_pipeline_task(
    structures: t.Union[Structure, t.Iterable[Structure]] | BatchedStructures,
    config: dict,
    usempi: bool = False,
):
    if usempi:
        return run_mpi_parallel_inpainting_pipeline(structures, config)

    return run_inpainting_pipeline(structures, config)


_evaluate_inpainting_task = task.pythonjob(
    outputs=spec.namespace(
        metric_results=t.Any,
    ),
)(evaluate_inpainting)


@task.pythonjob(
    outputs=spec.namespace(
        structures=t.Any,
        final_energies=t.Any,
        initial_energies=spec.socket(t.Any, required=False),
        initial_forces=spec.socket(t.Any, required=False),
        final_forces=spec.socket(t.Any, required=False),
    )
)
def _relaxation_task(
    structures: t.Union[
        dict[str, Structure], BatchedStructuresData, BatchedStructures
    ],
    relax_inputs: dict,
    usempi: bool = False,
) -> dict:
    """Wrapper for the relaxation function to be used in a WorkGraph."""
    if isinstance(structures, (BatchedStructuresData, BatchedStructures)):
        structures = structures.get_structures(strct_type="pymatgen")
    keys = list(structures.keys())
    structures = [structures[k] for k in keys]

    if usempi:
        (
            relaxed_structures,
            relaxed_energies,
            initial_energies,
            initial_forces,
            final_forces,
        ) = _relax_mpi_parallel(structures, keys, relax_inputs)

        if relaxed_structures is None or relaxed_energies is None:
            return {
                "structures": None,
                "energies": None,
            }
    else:
        (
            relaxed_structures,
            relaxed_energies,
            initial_energies,
            initial_forces,
            final_forces,
        ) = relax_structures(structures, **relax_inputs)

        relaxed_structures = {
            key: relaxed_structure
            for key, relaxed_structure in zip(keys, relaxed_structures)
        }
        relaxed_energies = {
            key: relaxed_energy
            for key, relaxed_energy in zip(keys, relaxed_energies)
        }
        if initial_energies:
            initial_energies = {
                key: initial_energy
                for key, initial_energy in zip(keys, initial_energies)
            }
        if initial_forces:
            initial_forces = {
                key: initial_force
                for key, initial_force in zip(keys, initial_forces)
            }
        if final_forces:
            final_forces = {
                key: final_force
                for key, final_force in zip(keys, final_forces)
            }
    print(f"Relaxed {len(relaxed_structures)} structures. Returning results.")

    outputs = {
        "structures": BatchedStructures(relaxed_structures),
        "final_energies": pd.DataFrame(
            relaxed_energies.items(), columns=["keys", "final_energy"]
        ).set_index("keys"),
    }
    if initial_energies:
        outputs["initial_energies"] = pd.DataFrame(
            initial_energies.items(), columns=["keys", "initial_energy"]
        ).set_index("keys")
    if initial_forces:
        outputs["initial_forces"] = pd.DataFrame(
            initial_forces.items(), columns=["keys", "initial_force"]
        ).set_index("keys")
    if final_forces:
        outputs["final_forces"] = pd.DataFrame(
            final_forces.items(), columns=["keys", "final_force"]
        ).set_index("keys")

    return outputs
