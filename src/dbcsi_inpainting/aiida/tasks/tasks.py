"""AiiDA WorkGraph tasks for structure relaxation and inpainting."""

from typing import Iterable, List, Tuple, Union

import pandas as pd
from aiida_workgraph import task
from pymatgen.core.structure import Structure

from dbcsi_inpainting.aiida.data import (
    BatchedStructures,
    BatchedStructuresData,
)
from dbcsi_inpainting.aiida.tasks.relax_parallel_utils import (
    _relax_mpi_parallel,
)
from dbcsi_inpainting.eval import evaluate_inpainting
from dbcsi_inpainting.inpainting.generate_candidates import (
    generate_inpainting_candidates,
)
from dbcsi_inpainting.inpainting.inpainting_process import (
    run_inpainting_pipeline,
    run_mpi_parallel_inpainting_pipeline,
)
from dbcsi_inpainting.utils.relaxation_utils import relax_structures


@task(
    inputs=[
        {
            "name": "structures",
        }
    ],
    outputs=[
        {
            "name": "candidates",
        }
    ],
)
def _aiida_generate_inpainting_candidates(
    structures: Union[Structure, Iterable[Structure]] | BatchedStructures,
    n_inp: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]],
    element: Union[str, List[str]],
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


@task(
    inputs=[
        {
            "name": "structures",
        }
    ],
    outputs=[
        {"name": "structures"},
        {
            "name": "structures_relaxed",
            "metadata": {"required": False},
        },
        {
            "name": "trajectories",
            "identifier": "workgraph.namespace",
            "metadata": {"dynamic": True, "required": False},
        },
        {
            "name": "mean_trajectories",
            "identifier": "workgraph.namespace",
            "metadata": {"dynamic": True, "required": False},
        },
    ],
)
def _inpainting_pipeline_task(
    structures,
    config,
    usempi: bool = False,
):
    if usempi:
        return run_mpi_parallel_inpainting_pipeline(structures, config)

    return run_inpainting_pipeline(structures, config)


_evaluate_inpainting_task = task(
    inputs=[
        {
            "name": "inpainted_structures",
        },
        {
            "name": "reference_structures",
        },
    ],
    outputs=[{"name": "metric_agg"}, {"name": "metric_individual"}],
)(evaluate_inpainting)


@task(
    outputs=[
        {"name": "structures"},
        {"name": "final_energies"},
        {
            "name": "initial_energies",
            "metadata": {"required": False},
        },
        {
            "name": "initial_forces",
            "metadata": {"required": False},
        },
        {
            "name": "final_forces",
            "metadata": {"required": False},
        },
    ]
)
def _relaxation_task(
    structures: (
        dict[str, Structure] | BatchedStructuresData | BatchedStructures
    ),
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
