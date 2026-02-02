"""AiiDA WorkGraph for inpainting of crystal structures."""

from copy import deepcopy

from aiida import orm
from aiida_workgraph import WorkGraph
from pymatgen.core.structure import Structure

from xtalpaint.aiida.data import (
    BatchedStructuresData,
)
from xtalpaint.aiida.tasks.tasks import (
    _evaluate_inpainting_task,
    _generate_inpainting_candidates_task,
    _inpainting_pipeline_task,
    _refine_structures_task,
    _relaxation_task,
)
from xtalpaint.data import BatchedStructures
from xtalpaint.inpainting.config_schema import InpaintingWorkGraphConfig


def setup_inpainting_wg(
    inputs: InpaintingWorkGraphConfig,
) -> WorkGraph:
    """Create a WorkGraph for inpainting of crystal structures."""
    possible_relaxation_tasks = {
        "inpainted_constrained_relaxation": inputs.relax,
        "unrelaxed_inpainted_full_relaxation": inputs.full_relax
        and inputs.full_relax_wo_pre_relax,
        "pre_relaxed_inpainted_full_relaxation": inputs.full_relax
        and inputs.relax,
    }

    wg = WorkGraph()

    if not inputs.is_inpainting_structures and inputs.run_inpainting:
        _add_inpainting_candidates_generation(wg, inputs)

    if inputs.run_inpainting:
        _add_inpainting_pipeline(wg, inputs)
        inpainted_structures = wg.tasks["inpainting"].outputs["structures"]
    else:
        inpainted_structures = inputs.structures

    if inputs.refine_structures:
        _add_refinement_task(
            wg,
            structures=inpainted_structures,
            refinement_symprec=inputs.refinement_symprec,
            inputs=inputs,
            task_name="refine_structures",
        )
        inpainted_structures = wg.tasks["refine_structures"].outputs[
            "structures"
        ]

    wg.outputs.inpainted_structures = inpainted_structures

    if inputs.relax or inputs.full_relax:
        _add_relaxation_tasks(wg, inpainted_structures, inputs)

    if inputs.evaluate:
        relaxation_tasks = {
            k: k for k, v in possible_relaxation_tasks.items() if v
        }
        _add_evaluation_tasks(wg, inputs, relaxation_tasks)

    return wg


def _add_inpainting_candidates_generation(
    wg: WorkGraph,
    inputs: InpaintingWorkGraphConfig,
) -> None:
    """Add inpainting candidates generation task to the workgraph."""
    wg.add_task(
        _generate_inpainting_candidates_task,
        structures=inputs.structures,
        n_inp=inputs.gen_inpainting_candidates_params.n_inp,
        element=inputs.gen_inpainting_candidates_params.element,
        num_samples=inputs.gen_inpainting_candidates_params.num_samples,
        name="generate_inpainting_candidates",
        metadata={
            "options": (
                inputs.gen_inpainting_candidates_options or inputs.options
            )
        },
    )

    wg.outputs.inpainting_candidates = wg.tasks[
        "generate_inpainting_candidates"
    ].outputs["candidates"]


def _add_refinement_task(
    wg: WorkGraph,
    structures: BatchedStructures | BatchedStructuresData,
    refinement_symprec: float,
    inputs: InpaintingWorkGraphConfig,
    task_name: str = "refine_structures",
) -> None:
    """Add structure refinement task to the workgraph."""
    wg.add_task(
        _refine_structures_task,
        structures=structures,
        refinement_symprec=refinement_symprec,
        name=task_name,
        metadata={
            "options": inputs.options or {},
        },
    )


def _add_inpainting_pipeline(
    wg: WorkGraph,
    inputs: InpaintingWorkGraphConfig,
) -> None:
    """Add inpainting pipeline task to the workgraph."""
    inpainting_candidates = (
        wg.tasks["generate_inpainting_candidates"].outputs["candidates"]
        if not inputs.is_inpainting_structures and inputs.run_inpainting
        else inputs.structures
    )

    code_label = inputs.inpainting_code_label or inputs.code_label

    wg.add_task(
        _inpainting_pipeline_task,
        structures=inpainting_candidates,
        config=inputs.inpainting_pipeline_params.model_dump(exclude_none=True),
        usempi=(
            inputs.inpainting_pipeline_options.get("withmpi", False)
            if inputs.inpainting_pipeline_options
            else False
        ),
        name="inpainting",
        metadata={
            "options": (inputs.inpainting_pipeline_options or inputs.options),
        },
        # serializers={
        #     "pymatgen.core.trajectory.Trajectory": (
        #         "xtalpaint.aiida.serializers.pymatgen_traj_to_aiida_traj"
        #     ),
        # },
        code=orm.load_code(code_label) if code_label else None,
    )

    if inputs.inpainting_pipeline_params.record_trajectories:
        wg.outputs.inpainted_trajectories = wg.tasks["inpainting"].outputs[
            "trajectories"
        ]
        if "mean_trajectories" in wg.tasks["inpainting"].outputs:
            wg.outputs.inpainted_mean_trajectories = wg.tasks[
                "inpainting"
            ].outputs["mean_trajectories"]


def _add_relaxation_tasks(
    wg: WorkGraph,
    structures: BatchedStructures | BatchedStructuresData,
    inputs: InpaintingWorkGraphConfig,
) -> None:
    """Add relaxation tasks to the workgraph."""
    code_label = inputs.relax_code_label or inputs.code_label
    relax_kwargs = deepcopy(inputs.relax_kwargs.model_dump())

    if inputs.relax:
        wg = _add_full_relax_task(
            wg=wg,
            structures=structures,
            relax_inputs=relax_kwargs,
            task_name="inpainted_constrained_relaxation",
            options=inputs.relax_options or inputs.options,
            code=orm.load_code(code_label) if code_label else None,
            as_graph_outputs=True,
        )

    if inputs.full_relax:
        relax_kwargs.pop("elements_to_relax", None)
        if inputs.full_relax_wo_pre_relax:
            wg = _add_full_relax_task(
                wg=wg,
                structures=structures,
                relax_inputs=relax_kwargs,
                task_name="unrelaxed_inpainted_full_relaxation",
                options=inputs.relax_options or inputs.options,
                code=orm.load_code(code_label) if code_label else None,
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
                code=orm.load_code(code_label) if code_label else None,
                as_graph_outputs=True,
            )


def _add_evaluation_tasks(
    wg: WorkGraph,
    inputs: InpaintingWorkGraphConfig,
    relaxation_tasks: dict[str, str],
) -> None:
    """Add evaluation tasks to the workgraph."""
    code_label = inputs.evaluate_params.code_label or inputs.code_label

    evaluation_results = {}
    metrics = (
        inputs.evaluate_params.metrics
        if isinstance(inputs.evaluate_params.metrics, list)
        else [inputs.evaluate_params.metrics]
    )
    tasks_to_evaluate = {}
    if inputs.run_inpainting:
        tasks_to_evaluate["inpainting"] = "inpainting"
        if inputs.refine_structures:
            tasks_to_evaluate["inpainting"] = "refine_structures"

    tasks_to_evaluate.update(relaxation_tasks)

    for metric in metrics:
        for label, task_name in tasks_to_evaluate.items():
            wg.add_task(
                _evaluate_inpainting_task,
                inpainted_structures=wg.tasks[task_name].outputs["structures"],
                reference_structures=inputs.structures,
                metric=metric,
                max_workers=inputs.evaluate_params.max_workers,
                name=f"evaluate_inpainting_{metric}_{label}",
                metadata={
                    "options": inputs.options or {},
                },
                code=orm.load_code(code_label) if code_label else None,
            )
            evaluation_results.setdefault(label, {}).update(
                {
                    f"{metric}": wg.tasks[
                        f"evaluate_inpainting_{metric}_{label}"
                    ].outputs["metric_results"],
                }
            )
    wg.outputs.evaluation_results = evaluation_results


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
        _relaxation_task,
        structures=structures,
        relax_inputs=relax_inputs,
        usempi=options.get("withmpi", False),
        name=task_name,
        metadata={
            "options": options or {},
        },
        # serializers={
        #     "pymatgen.core.structure.Structure": (
        #         "xtalpaint.aiida.serializers.pymatgen_to_structure_data"
        #     ),
        # },
        code=code,
    )
    if as_graph_outputs:
        outputs = {
            f"{task_name}.structures": wg.tasks[task_name].outputs[
                "structures"
            ],
            f"{task_name}.final_energies": wg.tasks[task_name].outputs[
                "final_energies"
            ],
        }

        if relax_inputs.get("return_initial_energies", False):
            outputs[f"{task_name}.initial_energies"] = wg.tasks[
                task_name
            ].outputs["initial_energies"]
        if relax_inputs.get("return_initial_forces", False):
            outputs[f"{task_name}.initial_forces"] = wg.tasks[
                task_name
            ].outputs["initial_forces"]
        if relax_inputs.get("return_final_forces", False):
            outputs[f"{task_name}.final_forces"] = wg.tasks[task_name].outputs[
                "final_forces"
            ]

        wg.outputs = outputs
        # ToDo: Wait for aiida-workgraph release to
        # ToDo: Update: It's released, just have to update the repo
        # ToDo: to the new version
        # support __setitem__ for outputs
        # It's already released now, just have to update the general WorkGraphs
        # then

        # # wg.outputs[f"{task_name}.structures"] = (
        # # wg.tasks[task_name].outputs[
        # #     "structures"
        # # ]
        # # wg.outputs[f"{task_name}.energies"] = wg.tasks[task_name].outputs[
        # #     "energies"
        # # ]

    return wg
