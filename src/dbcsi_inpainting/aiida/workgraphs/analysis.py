"""Analysis WorkGraph for Inpainting Results."""

from aiida_workgraph import WorkGraph, task
from dbcsi_inpainting.aiida.config_schema import InpaintingWorkGraphConfig
from dbcsi_inpainting.aiida.data import (
    BatchedStructures,
    BatchedStructuresData,
)

from dbcsi_inpainting.aiida.tasks import _evaluate_inpainting_task
from aiida import orm


@task.graph_builder
def setup_analysis_wg(
    structures_to_compare: dict[
        str, BatchedStructures | BatchedStructuresData
    ],
    reference_structures: BatchedStructures,
    inputs: InpaintingWorkGraphConfig,
    options: dict = None,
    name: str = None,
) -> WorkGraph:
    code_label = inputs.code_label or inputs.code_label
    name = name or "analyze-inpainting"

    with WorkGraph(name) as wg:
        evaluation_results = {}
        metrics = (
            inputs.metrics
            if isinstance(inputs.metrics, list)
            else [inputs.metrics]
        )
        for metric in metrics:
            for label, structures in structures_to_compare.items():
                wg.add_task(
                    "workgraph.pythonjob",
                    function=_evaluate_inpainting_task,
                    inpainted_structures=structures,
                    reference_structures=reference_structures,
                    metric=metric,
                    max_workers=inputs.max_workers,
                    name=f"evaluate_inpainting_{metric}_{label}",
                    metadata={
                        "options": options or {},
                    },
                    code=orm.load_code(code_label) if code_label else None,
                )
                evaluation_results.setdefault(label, {}).update(
                    {
                        f"{metric}_agg": wg.tasks[
                            f"evaluate_inpainting_{metric}_{label}"
                        ].outputs["metric_agg"],
                        f"{metric}_individual": wg.tasks[
                            f"evaluate_inpainting_{metric}_{label}"
                        ].outputs["metric_individual"],
                    }
                )

        wg.outputs.evaluation = evaluation_results

    return wg
