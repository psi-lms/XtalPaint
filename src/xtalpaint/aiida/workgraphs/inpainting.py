"""AiiDA WorkGraph for inpainting of crystal structures."""

import typing as t

from aiida import orm
from aiida_workgraph import WorkGraph, spec, task
from aiida_workgraph.socket_spec import meta
from pymatgen.core import Structure

from xtalpaint.aiida.tasks import tasks
from xtalpaint.aiida.workgraphs.relaxation import relaxation_graph
from xtalpaint.data import BatchedStructures
from xtalpaint.inpainting.config_schema import (
    AiiDAOptions,
    InpaintingRelaxationConfig,
    XtalPaintConfig,
)


def _relax_outputs(
    prefix: str, out, relax: InpaintingRelaxationConfig
) -> dict:
    """Build a prefixed output dict from a relaxation_graph result."""
    outputs = {
        f"{prefix}.structures": out.structures,
        f"{prefix}.final_energies": out.final_energies,
    }
    if relax.params.return_initial_energies:
        outputs[f"{prefix}.initial_energies"] = out.initial_energies
    if relax.params.return_initial_forces:
        outputs[f"{prefix}.initial_forces"] = out.initial_forces
    if relax.params.return_final_forces:
        outputs[f"{prefix}.final_forces"] = out.final_forces
    return outputs


RELAXATION_OUTPUTS_SPEC = spec.namespace(
    structures=t.Any,
    final_energies=t.Any,
    initial_energies=spec.socket(t.Any, required=False),
    initial_forces=spec.socket(t.Any, required=False),
    final_forces=spec.socket(t.Any, required=False),
    required=False,
)


@task.graph(
    outputs=spec.namespace(
        inpainted_structures=t.Any,
        inpainted_trajectories=t.Annotated[
            dict, spec.dynamic(t.Any), meta(required=False)
        ],
        inpainting_candidates=spec.socket(t.Any, required=False),
        inpainted_constrained_relaxation=RELAXATION_OUTPUTS_SPEC,
        unrelaxed_inpainted_full_relaxation=RELAXATION_OUTPUTS_SPEC,
        pre_relaxed_inpainted_full_relaxation=RELAXATION_OUTPUTS_SPEC,
    )
)
def InpaintingWorkGraph(
    structures: BatchedStructures | dict[str, Structure],
    inputs: spec.Leaf[XtalPaintConfig],
):
    """WorkGraph for inpainting of crystal structures."""
    graph_outputs = {}

    _aiida: AiiDAOptions = inputs.aiida or AiiDAOptions()

    # --- Generate inpainting candidates ---
    if inputs.run_inpainting:
        cand_opts = _aiida.candidate_generation_options
        cand_code_label = _aiida.get_code_label(
            _aiida.candidate_generation_code_label
        )
        gen_out = tasks.generate_inpainting_candidates_task(
            structures=structures,
            **inputs.candidate_generation.model_dump(),
            metadata={
                "call_link_label": "generate_inpainting_candidates",
                "options": cand_opts,
            },
            code=orm.load_code(cand_code_label) if cand_code_label else None,
        )
        inpainting_candidates = gen_out.candidates
        graph_outputs["inpainting_candidates"] = inpainting_candidates
    else:
        inpainting_candidates = structures

    # --- Inpainting pipeline ---
    if inputs.run_inpainting:
        inp_opts = _aiida.inpainting_options
        inp_code_label = _aiida.get_code_label(_aiida.inpainting_code_label)
        inp_out = tasks.inpainting_pipeline_task(
            structures=inpainting_candidates,
            config=inputs.inpainting.model_dump(),
            usempi=inp_opts["withmpi"],
            metadata={
                "call_link_label": "inpainting",
                "options": inp_opts,
            },
            code=orm.load_code(inp_code_label) if inp_code_label else None,
        )
        inpainted_structures = inp_out.structures

        if inputs.inpainting.record_trajectories:
            graph_outputs["inpainted_trajectories"] = inp_out.trajectories
    else:
        inpainted_structures = structures

    # --- Pre-refinement (before relaxation) ---
    if inputs.pre_refinement is not None:
        pre_ref_opts = _aiida.pre_refinement_options
        pre_ref_code_label = _aiida.get_code_label(
            _aiida.pre_refinement_code_label
        )
        ref_out = tasks.refine_structures_task(
            structures=inpainted_structures,
            symprec=inputs.pre_refinement.symprec,
            primitive=inputs.pre_refinement.primitive,
            metadata={
                "call_link_label": "refine_structures",
                "options": pre_ref_opts,
            },
            code=orm.load_code(pre_ref_code_label)
            if pre_ref_code_label
            else None,
        )
        inpainted_structures = ref_out.structures

    graph_outputs["inpainted_structures"] = inpainted_structures

    # --- Relaxation ---
    # AiiDA options for relaxation are embedded in inputs.relaxation.aiida
    if inputs.relaxation is not None:
        relax = inputs.relaxation

        cr_out = None
        if relax.constrained:
            cr_out = relaxation_graph(
                structures=inpainted_structures,
                relax_config=relax.relax_config.model_dump(),
                metadata={
                    "call_link_label": "inpainted_constrained_relaxation"
                },
            )
            graph_outputs |= _relax_outputs(
                "inpainted_constrained_relaxation", cr_out, relax.relax_config
            )

        if relax.full_direct or relax.full:
            full_relax = relax.relax_config.model_dump(
                exclude={"elements_to_relax"}
            )

        if relax.full_direct:
            ufr_out = relaxation_graph(
                structures=inpainted_structures,
                relax_config=full_relax,
                metadata={
                    "call_link_label": "unrelaxed_inpainted_full_relaxation"
                },
            )
            graph_outputs |= _relax_outputs(
                "unrelaxed_inpainted_full_relaxation",
                ufr_out,
                relax.relax_config,
            )

        if relax.full and cr_out is not None:
            pfr_out = relaxation_graph(
                structures=cr_out.structures,
                relax_config=full_relax,
                metadata={
                    "call_link_label": "pre_relaxed_inpainted_full_relaxation"
                },
            )
            graph_outputs |= _relax_outputs(
                "pre_relaxed_inpainted_full_relaxation",
                pfr_out,
                relax.relax_config,
            )

    return graph_outputs


def setup_inpainting_wg(inputs: XtalPaintConfig) -> WorkGraph:
    """Create a WorkGraph for inpainting of crystal structures."""
    return InpaintingWorkGraph.build(inputs=inputs)
