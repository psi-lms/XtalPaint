"""AiiDA WorkGraph for inpainting of crystal structures."""

from aiida import orm
from aiida_workgraph import WorkGraph, task

from xtalpaint.aiida.tasks import tasks
from xtalpaint.aiida.workgraphs.relaxation import relaxation_graph
from xtalpaint.inpainting.config_schema import (
    AiiDAOptions,
    RelaxationGraphConfig,
    XtalPaintConfig,
)


def _relax_outputs(prefix: str, out, relax: RelaxationGraphConfig) -> dict:
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
        gen_out = tasks.generate_inpainting_candidates_task(
            structures=inputs.structures,
            **inputs.candidate_generation.model_dump(),
            metadata={
                "call_link_label": "generate_inpainting_candidates",
                "options": cand_opts.model_dump(exclude={"withmpi"}),
            },
        )
        inpainting_candidates = gen_out.candidates
        graph_outputs["inpainting_candidates"] = inpainting_candidates
    else:
        inpainting_candidates = inputs.structures

    # --- Inpainting pipeline ---
    if inputs.run_inpainting:
        inp_opts = _aiida.inpainting_options
        code_label = _aiida.get_code_label(_aiida.inpainting_code_label)
        inp_out = tasks.inpainting_pipeline_task(
            structures=inpainting_candidates,
            config=inputs.inpainting.model_dump(exclude_none=True),
            usempi=inp_opts.withmpi,
            metadata={
                "call_link_label": "inpainting",
                "options": inp_opts.model_dump(exclude={"withmpi"}),
            },
            code=orm.load_code(code_label) if code_label else None,
        )
        inpainted_structures = inp_out.structures

        if inputs.inpainting.record_trajectories:
            graph_outputs["inpainted_trajectories"] = inp_out.trajectories
    else:
        inpainted_structures = inputs.structures

    # --- Pre-refinement (before relaxation) ---
    if inputs.pre_refinement is not None:
        ref_out = tasks.refine_structures_task(
            structures=inpainted_structures,
            refinement_symprec=inputs.pre_refinement.symprec,
            primitive=inputs.pre_refinement.primitive,
            metadata={
                "call_link_label": "refine_structures",
                "options": {},
            },
        )
        inpainted_structures = ref_out.structures

    graph_outputs["inpainted_structures"] = inpainted_structures

    # --- Relaxation ---
    if inputs.relaxation is not None:
        relax = inputs.relaxation
        relax_opts = _aiida.relax_options
        relax_code_label = _aiida.get_code_label(_aiida.relax_code_label)

        cr_out = None
        if relax.constrained:
            cr_out = relaxation_graph(
                structures=inpainted_structures,
                relax_config=relax,
                aiida_options=relax_opts,
                code_label=relax_code_label,
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
                aiida_options=relax_opts,
                code_label=relax_code_label,
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
                aiida_options=relax_opts,
                code_label=relax_code_label,
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
