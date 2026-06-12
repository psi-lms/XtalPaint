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
    inputs=spec.namespace(
        structures=t.Any,
        relax_config=RelaxationGraphConfig,
        command_info=spec.socket(t.Any, required=False),
    ),
    outputs=spec.namespace(
        structures=t.Any,
        final_energies=t.Any,
        initial_energies=spec.socket(t.Any, required=False),
        initial_forces=spec.socket(t.Any, required=False),
        final_forces=spec.socket(t.Any, required=False),
    ),
)
def relaxation_graph(
    structures: t.Any,
    relax_config: dict,
    command_info: dict = None,
):
    """Relaxation WG with optional symmetry refinement and deduplication.

    Runs ``_relaxation_task`` and then optionally:

    1. ``_refine_structures_task`` — symmetry-refine the relaxed structures
       when ``relax_config.refinement`` is not ``None``.
    2. ``_uniqueness_filter_task`` — keep one representative per unique
       (space-group, StructureMatcher equivalence class) group when
       ``relax_config.uniqueness`` is not ``None``.

    Args:
        structures: Input structures to relax.
        relax_config: Self-contained relaxation config.
            ``relax_config.params`` is forwarded to ``relax_structures`` as
            ``relax_inputs``. ``relax_config.refinement`` and
            ``relax_config.uniqueness`` control the optional post-relaxation
            steps.  AiiDA options live in ``relax_config.aiida``.  For an
            unconstrained (full) relaxation, pass a config whose
            ``params.elements_to_relax`` is ``None``.
        command_info: Passed as ``command_info`` to every inner pythonjob task
            (e.g. ``{"filepath_executable": "/path/to/python"}``).

    Returns:
        dict with ``structures`` (relaxed, and optionally refined/filtered),
        ``final_energies``, and optionally ``initial_energies``,
        ``initial_forces``, ``final_forces`` when requested via
        ``relax_config.params``.
    """
    _command_info = command_info or {}

    _aiida = relax_config["aiida"]

    def _resolve_metadata(
        options: AiiDATaskOptions,
        code_label: str | None,
    ) -> tuple[dict, t.Any, bool]:
        """Resolve ``(metadata, code, withmpi)`` for one inner task."""
        return (
            {"options": {k: v for k, v in options.items()}},
            orm.load_code(code_label) if code_label else None,
            options["withmpi"],
        )

    _metadata, _code, _usempi = _resolve_metadata(
        _aiida["relax_options"], _aiida["relax_code_label"]
    )

    relaxed = tasks.relaxation_task(
        structures=structures,
        relax_inputs=relax_config["params"],
        usempi=_usempi,
        metadata=_metadata,
        code=_code,
        command_info=_command_info,
    )

    current_structures = relaxed.structures

    if relax_config["refinement"]["include_task"]:
        _metadata, _code, _usempi = _resolve_metadata(
            _aiida["refinement_options"], _aiida["refinement_code_label"]
        )

        refined = tasks.refine_structures_task(
            structures=current_structures,
            symprec=relax_config["refinement"]["symprec"],
            primitive=relax_config["refinement"]["primitive"],
            metadata=_metadata,
            code=_code,
            command_info=_command_info,
        )
        current_structures = refined.structures

    if relax_config["uniqueness"]["include_task"]:
        _metadata, _code, _usempi = _resolve_metadata(
            _aiida["uniqueness_options"], _aiida["uniqueness_code_label"]
        )

        filtered = tasks.uniqueness_filter_task(
            structures=current_structures,
            symprec=relax_config["uniqueness"]["symprec"],
            ltol=relax_config["uniqueness"]["ltol"],
            stol=relax_config["uniqueness"]["stol"],
            angle_tol=relax_config["uniqueness"]["angle_tol"],
            metadata=_metadata,
            code=_code,
            command_info=_command_info,
        )
        current_structures = filtered.unique_structures

    outputs = {
        "structures": current_structures,
        "final_energies": relaxed.final_energies,
    }
    if (
        relax_config["params"]["return_initial_energies"]
        or relax_config["params"]["return_initial_forces"]
    ):
        outputs["initial_energies"] = relaxed.initial_energies
    if relax_config["params"]["return_initial_forces"]:
        outputs["initial_forces"] = relaxed.initial_forces
    if relax_config["params"]["return_final_forces"]:
        outputs["final_forces"] = relaxed.final_forces

    return outputs
