"""Analysis WorkGraph for Inpainting Results."""

from aiida import orm
from aiida_workgraph import WorkGraph, task

from dbcsi_inpainting.aiida.data import (
    BatchedStructures,
    BatchedStructuresData,
)
from dbcsi_inpainting.aiida.tasks.tasks import _evaluate_inpainting_task
from dbcsi_inpainting.inpainting.config_schema import InpaintingWorkGraphConfig


@task.graph_builder
def setup_analysis_wg(
    structures_to_compare: dict[
        str, BatchedStructures | BatchedStructuresData
    ],
    reference_structures: BatchedStructures,
    inputs: InpaintingWorkGraphConfig,
    options: dict = None,
    name: str = None,
    rmsd_normalization_element: str | None = None,
) -> WorkGraph:
    """Setup AiiDA WorkGraph to compare structure sets against references.

    Args:
        structures_to_compare: Dictionary of labels to structures to compare.
        reference_structures: Reference structures for comparison.
        inputs: Configuration for the evaluation task.
        options: Optional dictionary of (PythonJob) options. Defaults to None.
        name: Optional name for the workgraph. Defaults to None.
        rmsd_normalization_element: Optional element for RMSD normalization.
            Defaults to None.

    Returns:
        WorkGraph: AiiDA WorkGraph for analysis.
    """
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
            additional_kwargs = {}
            if metric == "rmsd" and rmsd_normalization_element is not None:
                additional_kwargs = {
                    "normalization_element": rmsd_normalization_element
                }
            for label, structures in structures_to_compare.items():
                wg.add_task(
                    "workgraph.pythonjob",
                    function=_evaluate_inpainting_task,
                    inpainted_structures=structures,
                    reference_structures=reference_structures,
                    metric=metric,
                    max_workers=inputs.max_workers,
                    **additional_kwargs,
                    name=f"evaluate_inpainting_{metric}_{label}",
                    metadata={
                        "options": options or {},
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

        wg.outputs.evaluation = evaluation_results

    return wg
