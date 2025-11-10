"""AiiDA WorkGraph for inpainting of crystal structures."""

from aiida import orm
from aiida_workgraph import WorkGraph
from dbcsi_inpainting.aiida.data import (
    BatchedStructuresData,
    BatchedStructures,
)
from pymatgen.core.structure import Structure


from copy import deepcopy
from dbcsi_inpainting.aiida.config_schema import InpaintingWorkGraphConfig
from dbcsi_inpainting.aiida.tasks.tasks import (
    _relaxation_task,
    _aiida_generate_inpainting_candidates,
    _inpainting_pipeline_task,
    _evaluate_inpainting_task,
)


def get_inpainting_wg(
    inputs: InpaintingWorkGraphConfig,
) -> WorkGraph:
    """Create a WorkGraph for inpainting of crystal structures."""
    wg = WorkGraph()

    inpainting_candidates = inputs.structures
    if not inputs.is_inpainting_structures and inputs.run_inpainting:
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
            # ToDo: These serializers/deserializers can probably be removed as
            # they are also registered as entry-points
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

    code_label = inputs.inpainting_code_label or inputs.code_label

    if inputs.run_inpainting:
        wg.add_task(
            "workgraph.pythonjob",
            function=_inpainting_pipeline_task,
            structures=inpainting_candidates,
            config=inputs.inpainting_pipeline_params.model_dump(
                exclude_none=True
            ),
            usempi=(
                inputs.inpainting_pipeline_options.get("withmpi", False)
                if inputs.inpainting_pipeline_options
                else False
            ),
            name="inpainting",
            metadata={
                "options": (
                    inputs.inpainting_pipeline_options or inputs.options
                ),
            },
            # ToDo: These serializers/deserializers can probably be removed as they
            # are also registered as entry-points
            deserializers={
                "aiida.orm.nodes.data.structure.StructureData": (
                    "aiida_pythonjob.data.deserializer.structure_data_to_pymatgen"
                ),
            },
            serializers={
                "pymatgen.core.structure.Structure": (
                    "dbcsi_inpainting.aiida.serializers."
                    "pymatgen_to_structure_data"
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
            code=orm.load_code(code_label) if code_label else None,
        )
        wg.outputs.inpainted_structures = wg.tasks["inpainting"].outputs[
            "structures"
        ]
        if inputs.inpainting_pipeline_params.record_trajectories:
            wg.outputs.inpainted_trajectories = wg.tasks["inpainting"].outputs[
                "trajectories"
            ]
            if "mean_trajectories" in wg.tasks["inpainting"].outputs:
                wg.outputs.inpainted_mean_trajectories = wg.tasks[
                    "inpainting"
                ].outputs["mean_trajectories"]

    code_label = inputs.relax_code_label or inputs.code_label
    relax_kwargs = deepcopy(inputs.relax_kwargs.model_dump())
    relaxation_tasks = []
    if inputs.relax:
        wg = _add_full_relax_task(
            wg=wg,
            structures=wg.tasks["inpainting"].outputs["structures"]
            if inputs.run_inpainting
            else inputs.structures,
            relax_inputs=relax_kwargs,
            task_name="inpainted_constrained_relaxation",
            options=inputs.relax_options or inputs.options,
            code=orm.load_code(code_label) if code_label else None,
            as_graph_outputs=True,
        )
        relaxation_tasks.append("inpainted_constrained_relaxation")

    if inputs.full_relax:
        relax_kwargs.pop("elements_to_relax", None)
        if inputs.full_relax_wo_pre_relax:
            wg = _add_full_relax_task(
                wg=wg,
                structures=wg.tasks["inpainting"].outputs["structures"]
                if inputs.run_inpainting
                else inputs.structures,
                relax_inputs=relax_kwargs,
                task_name="unrelaxed_inpainted_full_relaxation",
                options=inputs.relax_options or inputs.options,
                code=orm.load_code(code_label) if code_label else None,
                as_graph_outputs=True,
            )
            relaxation_tasks.append("unrelaxed_inpainted_full_relaxation")

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
            relaxation_tasks.append("pre_relaxed_inpainted_full_relaxation")

    if inputs.evaluate:
        code_label = inputs.evaluate_params.code_label or inputs.code_label

        evaluation_results = {}
        metrics = (
            inputs.evaluate_params.metrics
            if isinstance(inputs.evaluate_params.metrics, list)
            else [inputs.evaluate_params.metrics]
        )
        for metric in metrics:
            for task_name in ["inpainting"] + relaxation_tasks:
                wg.add_task(
                    "workgraph.pythonjob",
                    function=_evaluate_inpainting_task,
                    inpainted_structures=wg.tasks[task_name].outputs[
                        "structures"
                    ],
                    reference_structures=inputs.structures,
                    metric=metric,
                    max_workers=inputs.evaluate_params.max_workers,
                    name=f"evaluate_{metric}_inpainting_{task_name}",
                    metadata={
                        "options": inputs.options or {},
                    },
                    code=orm.load_code(code_label) if code_label else None,
                )
                evaluation_results[task_name].update(
                    {
                        f"{metric}_agg": wg.tasks[
                            f"evaluate_{metric}_inpainting_{task_name}"
                        ].outputs["metric_agg"],
                        f"{metric}_individual": wg.tasks[
                            f"evaluate_{metric}_inpainting_{task_name}"
                        ].outputs["metric_individual"],
                    }
                )
        wg.outputs.evaluation_results = evaluation_results

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
        function=_relaxation_task,
        structures=structures,
        relax_inputs=relax_inputs,
        usempi=options.get("withmpi", False),
        name=task_name,
        metadata={
            "options": options or {},
        },
        # ToDo: These serializers/deserializers can probably be removed as they
        # are also registered as entry-points
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
            "pandas.core.frame.DataFrame": (
                "dbcsi_inpainting.aiida.serializers."
                "pandas_dataframe_to_pandas_dataframe_data"
            ),
        },
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
        # ToDo: Update: It's released, just have to update the repo to the new version
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
