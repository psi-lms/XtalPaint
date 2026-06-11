"""AiiDA WorkGraph for inpainting of crystal structures."""

from aiida import orm
from aiida_workgraph import WorkGraph, task

from xtalpaint.aiida.tasks import tasks
from xtalpaint.aiida.workgraphs.relaxation import relaxation_graph
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


@task.graph
def InpaintingWorkGraph(inputs: XtalPaintConfig):
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
            structures=inputs.structures,
            **inputs.candidate_generation.model_dump(),
            metadata={
                "call_link_label": "generate_inpainting_candidates",
                "options": cand_opts.model_dump(exclude_none=True),
            },
            code=orm.load_code(cand_code_label) if cand_code_label else None,
        )
        inpainting_candidates = gen_out.candidates
        graph_outputs["inpainting_candidates"] = inpainting_candidates
    else:
        inpainting_candidates = inputs.structures

    # --- Inpainting pipeline ---
    if inputs.run_inpainting:
        inp_opts = _aiida.inpainting_options
        inp_code_label = _aiida.get_code_label(_aiida.inpainting_code_label)
        inp_out = tasks.inpainting_pipeline_task(
            structures=inpainting_candidates,
            config=inputs.inpainting.model_dump(exclude_none=True),
            usempi=inp_opts.withmpi,
            metadata={
                "call_link_label": "inpainting",
                "options": inp_opts.model_dump(exclude_none=True),
            },
            code=orm.load_code(inp_code_label) if inp_code_label else None,
        )
        inpainted_structures = inp_out.structures

        if inputs.inpainting.record_trajectories:
            graph_outputs["inpainted_trajectories"] = inp_out.trajectories
    else:
        inpainted_structures = inputs.structures

    # --- Pre-refinement (before relaxation) ---
    if inputs.pre_refinement is not None:
        pre_ref_opts = _aiida.pre_refinement_options
        pre_ref_code_label = _aiida.get_code_label(
            _aiida.pre_refinement_code_label
        )
        ref_out = tasks.refine_structures_task(
            structures=inpainted_structures,
            refinement_symprec=inputs.pre_refinement.symprec,
            primitive=inputs.pre_refinement.primitive,
            usempi=pre_ref_opts.withmpi,
            metadata={
                "call_link_label": "refine_structures",
                "options": pre_ref_opts.model_dump(exclude_none=True),
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
                relax_config=relax,
                constrained=True,
                metadata={
                    "call_link_label": "inpainted_constrained_relaxation"
                },
            )
            graph_outputs |= _relax_outputs(
                "inpainted_constrained_relaxation", cr_out, relax
            )

        if relax.full_direct:
            ufr_out = relaxation_graph(
                structures=inpainted_structures,
                relax_config=relax,
                constrained=False,
                metadata={
                    "call_link_label": "unrelaxed_inpainted_full_relaxation"
                },
            )
            graph_outputs |= _relax_outputs(
                "unrelaxed_inpainted_full_relaxation", ufr_out, relax
            )

        if relax.full and cr_out is not None:
            pfr_out = relaxation_graph(
                structures=cr_out.structures,
                relax_config=relax,
                constrained=False,
                metadata={
                    "call_link_label": "pre_relaxed_inpainted_full_relaxation"
                },
            )
            graph_outputs |= _relax_outputs(
                "pre_relaxed_inpainted_full_relaxation", pfr_out, relax
            )

    return graph_outputs


def setup_inpainting_wg(inputs: XtalPaintConfig) -> WorkGraph:
    """Create a WorkGraph for inpainting of crystal structures."""
    return InpaintingWorkGraph.build(inputs=inputs)
