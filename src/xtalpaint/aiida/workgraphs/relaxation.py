"""Relaxation WorkGraph with optional refinement and uniqueness filtering."""

import typing as t

from aiida import orm
from aiida_workgraph import spec, task

from xtalpaint.aiida.tasks import tasks
from xtalpaint.inpainting.config_schema import (
    AiiDATaskOptions,
    RelaxationGraphConfig,
)


@task.graph(
    outputs=spec.namespace(
        structures=t.Any,
        final_energies=t.Any,
        initial_energies=spec.socket(t.Any, required=False),
        initial_forces=spec.socket(t.Any, required=False),
        final_forces=spec.socket(t.Any, required=False),
    )
)
def relaxation_graph(
    structures: t.Any,
    relax_config: RelaxationGraphConfig,
    aiida_options: AiiDATaskOptions = None,
    code_label: str = None,
    command_info: dict = None,
    constrained: bool = True,
):
    """Relaxation WG with optional symmetry refinement and deduplication.

    Runs ``_relaxation_task`` and then optionally:

    1. ``_refine_structures_task`` — symmetry-refine the relaxed structures.
    2. ``_uniqueness_filter_task`` — keep one representative per unique
       (space-group, StructureMatcher equivalence class) group.

    The ``structures`` output always points to the last active step, so
    downstream tasks see a consistent socket name regardless of which optional
    steps are enabled.

    ``relax_config.refine`` and ``relax_config.filter_unique`` are evaluated
    at graph build-time (when the WorkGraph is materialised), so they must
    resolve to plain Python ``bool`` values; passing AiiDA nodes wired from
    another task's output is not supported for these flags.

    Args:
        structures: Input structures to relax.
        relax_config: Relaxation and post-processing configuration.
            ``relax_config.params`` is forwarded to ``relax_structures`` as
            ``relax_inputs``.  ``relax_config.refine`` and
            ``relax_config.filter_unique`` control the optional steps.
        aiida_options: AiiDA scheduler/resource options forwarded to all inner
            tasks.  If ``None``, default options are used (no resource limits,
            no MPI).
        code_label: AiiDA code label for all inner pythonjob tasks.  If
            ``None``, aiida-pythonjob locates ``python3`` automatically.
        command_info: Passed as ``command_info`` to every inner pythonjob task
            (e.g. ``{"filepath_executable": "/path/to/python"}``).  Overrides
            automatic executable detection when set.
        constrained: If ``True`` (default), ``elements_to_relax`` from
            ``relax_config.params`` is included in the relax call so that only
            those elements are relaxed.  Pass ``False`` for full relaxation
            of all atoms.

    Returns:
        dict with ``structures`` (relaxed, and optionally refined/filtered),
        ``final_energies``, and optionally ``initial_energies``,
        ``initial_forces``, ``final_forces`` when requested via
        ``relax_config.params``.
    """
    _aiida = aiida_options or AiiDATaskOptions()
    _options = _aiida.model_dump(exclude={"withmpi"}, exclude_none=True)
    _code = orm.load_code(code_label) if code_label else None
    _metadata = {"options": _options}
    _command_info = command_info or {}

    relaxed = tasks.relaxation_task(
        structures=structures,
        relax_inputs=relax_config.relax_inputs(constrained=constrained),
        usempi=_aiida.withmpi,
        metadata=_metadata,
        code=_code,
        command_info=_command_info,
    )

    current_structures = relaxed.structures

    if relax_config.refine:
        refined = tasks.refine_structures_task(
            structures=current_structures,
            refinement_symprec=relax_config.refinement_symprec,
            primitive=relax_config.refinement_primitive,
            metadata=_metadata,
            code=_code,
            command_info=_command_info,
        )
        current_structures = refined.structures

    if relax_config.filter_unique:
        filtered = tasks.uniqueness_filter_task(
            structures=current_structures,
            symprec=relax_config.uniqueness.symprec,
            ltol=relax_config.uniqueness.ltol,
            stol=relax_config.uniqueness.stol,
            angle_tol=relax_config.uniqueness.angle_tol,
            metadata=_metadata,
            code=_code,
            command_info=_command_info,
        )
        current_structures = filtered.unique_structures

    outputs = {
        "structures": current_structures,
        "final_energies": relaxed.final_energies,
    }
    if relax_config.params.return_initial_energies:
        outputs["initial_energies"] = relaxed.initial_energies
    if relax_config.params.return_initial_forces:
        outputs["initial_forces"] = relaxed.initial_forces
    if relax_config.params.return_final_forces:
        outputs["final_forces"] = relaxed.final_forces

    return outputs
