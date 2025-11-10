"""AiiDA WorkGraph for inpainting of crystal structures."""

from aiida import orm
from dbcsi_inpainting.aiida.inpainting_process import run_inpainting_pipeline
from dbcsi_inpainting.aiida.generate_candidates import (
    _aiida_generate_inpainting_candidates,
)
from aiida_workgraph import WorkGraph, task
from dbcsi_inpainting.aiida.data import (
    BatchedStructuresData,
    BatchedStructures,
)
from pymatgen.core.structure import Structure

from dbcsi_inpainting.utils.relaxation_utils import relax_structures
from copy import deepcopy
from dbcsi_inpainting.aiida.config_schema import InpaintingWorkGraphConfig

_run_inpainting_pipeline_task = task(
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
    ],
)(run_inpainting_pipeline)


def get_inpainting_wg(
    inputs: InpaintingWorkGraphConfig,
) -> WorkGraph:
    """Create a WorkGraph for inpainting of crystal structures."""
    wg = WorkGraph()

    inpainting_candidates = inputs.structures
    if not inputs.is_inpainting_structures:
        wg.add_task(
            "workgraph.pythonjob",
            function=_aiida_generate_inpainting_candidates,
            structures=inpainting_candidates,
            n_inp=inputs.gen_inpainting_candidates_params.n_inp,
            element=inputs.gen_inpainting_candidates_params.element,
            num_samples=inputs.gen_inpainting_candidates_params.num_samples,
            name="generate_inpainting_candidates",
            metadata={
                "options": (
                    inputs.gen_inpainting_candidates_options or inputs.options
                )
            },
            deserializers={
                "aiida.orm.nodes.data.structure.StructureData": (
                    "aiida_pythonjob.data.deserializer."
                    "structure_data_to_pymatgen"
                )
            },
            serializers={
                "pymatgen.core.structure.Structure": (
                    "dbcsi_inpainting.aiida.data.InpaintingStructure"
                ),
                "dbcsi_inpainting.aiida.data.BatchedStructures": (
                    "dbcsi_inpainting.aiida.serializers."
                    "batched_structures_to_batched_structures_data"
                ),
            },
        )
        inpainting_candidates = wg.tasks[
            "generate_inpainting_candidates"
        ].outputs["candidates"]
        wg.outputs.inpainting_candidates = inpainting_candidates

    if inputs.code_label is None:
        code = None
    else:
        code = orm.load_code(inputs.code_label)

    wg.add_task(
        "workgraph.pythonjob",
        function=_run_inpainting_pipeline_task,
        structures=inpainting_candidates,
        config=inputs.inpainting_pipeline_params.model_dump(exclude_none=True),
        name="inpainting",
        metadata={
            "options": (inputs.inpainting_pipeline_options or inputs.options),
        },
        deserializers={
            "aiida.orm.nodes.data.structure.StructureData": (
                "aiida_pythonjob.data.deserializer.structure_data_to_pymatgen"
            ),
        },
        serializers={
            "pymatgen.core.structure.Structure": (
                "dbcsi_inpainting.aiida.serializers.pymatgen_to_structure_data"
            ),
            "dbcsi_inpainting.aiida.data.BatchedStructures": (
                "dbcsi_inpainting.aiida.serializers."
                "batched_structures_to_batched_structures_data"
            ),
            "pymatgen.core.trajectory.Trajectory": (
                "dbcsi_inpainting.aiida.serializers."
                "pymatgen_traj_to_aiida_traj"
            ),
        },
        code=code,
    )
    wg.outputs.inpainted_structures = wg.tasks["inpainting"].outputs[
        "structures"
    ]
    if inputs.inpainting_pipeline_params.record_trajectories:
        wg.outputs.inpainted_trajectories = wg.tasks["inpainting"].outputs[
            "trajectories"
        ]

    relax_kwargs = deepcopy(inputs.relax_kwargs.model_dump())
    if inputs.relax:
        wg = _add_full_relax_task(
            wg=wg,
            structures=wg.tasks["inpainting"].outputs["structures"],
            relax_inputs=relax_kwargs,
            task_name="inpainted_constrained_relaxation",
            options=inputs.relax_options or inputs.options,
            code=code,
            as_graph_outputs=True,
        )

    if inputs.full_relax:
        relax_kwargs.pop("fix_elements", None)

        wg = _add_full_relax_task(
            wg=wg,
            structures=wg.tasks["inpainting"].outputs["structures"],
            relax_inputs=relax_kwargs,
            task_name="unrelaxed_inpainted_full_relaxation",
            options=inputs.relax_options or inputs.options,
            code=code,
            as_graph_outputs=True,
        )
        if inputs.relax:
            wg = _add_full_relax_task(
                wg=wg,
                structures=wg.tasks[
                    "inpainted_constrained_relaxation"
                ].outputs["structures"],
                relax_inputs=relax_kwargs,
                task_name="pre_relaxed_inpainted_full_relaxation",
                options=inputs.relax_options or inputs.options,
                code=code,
                as_graph_outputs=True,
            )

    return wg


def _add_full_relax_task(
    wg: WorkGraph,
    structures: (
        dict[str, Structure] | BatchedStructuresData | BatchedStructures
    ),
    relax_inputs: dict,
    task_name: str = "full_relaxation",
    options: dict = None,
    code: orm.Code | None = None,
    as_graph_outputs: bool = False,
):
    """Add a full relaxation task to the workgraph."""
    wg.add_task(
        "workgraph.pythonjob",
        function=aiida_relax,
        structures=structures,
        relax_inputs=relax_inputs,
        name=task_name,
        metadata={
            "options": options or {},
        },
        deserializers={
            "aiida.orm.nodes.data.structure.StructureData": (
                "aiida_pythonjob.data.deserializer.structure_data_to_pymatgen"
            ),
        },
        serializers={
            "pymatgen.core.structure.Structure": (
                "dbcsi_inpainting.aiida.serializers.pymatgen_to_structure_data"
            ),
            "dbcsi_inpainting.aiida.data.BatchedStructures": (
                "dbcsi_inpainting.aiida.serializers."
                "batched_structures_to_batched_structures_data"
            ),
        },
        code=code,
    )
    if as_graph_outputs:
        outputs = {
            f"{task_name}.structures": wg.tasks[task_name].outputs[
                "structures"
            ],
            f"{task_name}.energies": wg.tasks[task_name].outputs["energies"],
        }
        wg.outputs = outputs
        # ToDo: Wait for aiida-workgraph release to
        # support __setitem__ for outputs
        # # wg.outputs[f"{task_name}.structures"] = wg.tasks[task_name].outputs[
        # #     "structures"
        # # ]
        # # wg.outputs[f"{task_name}.energies"] = wg.tasks[task_name].outputs[
        # #     "energies"
        # # ]

    return wg


@task(outputs=[{"name": "structures"}, {"name": "energies"}])
def aiida_relax(
    structures: (
        dict[str, Structure] | BatchedStructuresData | BatchedStructures
    ),
    relax_inputs: dict,
):
    """Wrapper for the relaxation function to be used in a WorkGraph."""
    if isinstance(structures, (BatchedStructuresData, BatchedStructures)):
        structures = structures.get_structures(strct_type="pymatgen")
    keys = structures.keys()
    structures = [structures[k] for k in keys]

    relaxed_structures, relaxed_energies = relax_structures(
        structures, **relax_inputs
    )
    relaxed_structures = {
        key: relaxed_structure
        for key, relaxed_structure in zip(keys, relaxed_structures)
    }
    relaxed_energies = {
        key: relaxed_energy
        for key, relaxed_energy in zip(keys, relaxed_energies)
    }

    return {
        "structures": BatchedStructures(relaxed_structures),
        "energies": relaxed_energies,
    }
