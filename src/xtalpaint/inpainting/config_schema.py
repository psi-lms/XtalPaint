"""Pydantic schemas for configuring XtalPaint inpainting workflows."""

from typing import Optional

from ase import Atoms
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.config import ConfigDict
from pymatgen.core import Structure

from xtalpaint.data import BatchedStructures
from xtalpaint.utils import _is_batched_structure, is_aiida_installed


def _is_valid_structure_type(obj) -> bool:
    """Check if object is a valid structure type."""
    if isinstance(obj, (Structure, Atoms)):
        return True
    if is_aiida_installed():
        from aiida.orm import StructureData

        from xtalpaint.aiida.data import InpaintingStructureData

        return isinstance(obj, (StructureData, InpaintingStructureData))
    return False


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------


class CandidateGenerationConfig(BaseModel):
    """Configuration for generating inpainting candidates."""

    n_inp: int | dict[str, int]
    element: str | dict[str, str]
    num_samples: int = 1


class InpaintingConfig(BaseModel):
    """Configuration for the diffusion inpainting stage.

    Sampling parameters are kept flat (no nested ModelParams sub-object)
    so they can be specified in a single block and passed directly as a
    dict to the inpainting pipeline.
    """

    # Model — exactly one of these must be provided
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None

    # Diffusion sampling
    predictor_corrector: str
    N_steps: int
    coordinates_snr: float
    n_corrector_steps: int
    batch_size: int
    fix_cell: bool = True
    record_trajectories: bool = False
    sampling_config_path: Optional[str] = None

    # Repaint-specific (required when predictor_corrector contains 'repaint')
    n_resample_steps: Optional[int] = None
    jump_length: Optional[int] = None

    @field_validator("predictor_corrector")
    @classmethod
    def validate_predictor_corrector(cls, v):
        """Validate that predictor_corrector is one of the allowed options."""
        from xtalpaint.inpainting.inpainting_process import (
            GUIDED_PREDICTOR_CORRECTOR_MAPPING,
        )

        if v not in GUIDED_PREDICTOR_CORRECTOR_MAPPING:
            allowed = list(GUIDED_PREDICTOR_CORRECTOR_MAPPING.keys())
            raise ValueError(
                f"predictor_corrector must be one of {allowed}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    @classmethod
    def check_pretrained_model_exclusive(cls, cfg):
        """Validate model specification."""
        if (
            cfg.pretrained_name is not None and cfg.model_path is not None
        ) or (cfg.pretrained_name is None and cfg.model_path is None):
            raise ValueError(
                "`pretrained_name` and `model_path` are mutually exclusive; "
                "provide only one."
            )
        return cfg

    @model_validator(mode="after")
    @classmethod
    def check_repaint_requires_resample_and_jump(cls, cfg):
        """Validate RePaint-specific parameters."""
        if "repaint" in cfg.predictor_corrector.lower():
            if cfg.n_resample_steps is None or cfg.jump_length is None:
                raise ValueError(
                    "When 'predictor_corrector' contains 'repaint', "
                    "both 'n_resample_steps' and 'jump_length' must be set."
                )
        return cfg


class RefinementConfig(BaseModel):
    """Symmetry refinement stage."""

    symprec: float = 0.01
    primitive: bool = False


class UniquenessConfig(BaseModel):
    """Parameters for post-relaxation uniqueness/deduplication filtering."""

    symprec: float = 0.01
    ltol: float = 0.2
    stol: float = 0.3
    angle_tol: float = 5.0


class RelaxationParams(BaseModel):
    """Core relaxation parameters forwarded to ``relax_structures()``.

    These are the settings that control *how* a single relaxation is run
    (MLIP, optimiser, convergence criteria, etc.).  They are kept separate
    from the inpainting-workflow-level controls in ``RelaxationConfig``.
    """

    mlip: str
    optimizer: str
    fmax: float = 0.05
    max_n_steps: int = 500
    max_natoms_per_batch: int = 512
    device: str = "cpu"
    filter: Optional[str] = None
    load_path: Optional[str] = None
    elements_to_relax: Optional[list[str]] = None
    return_initial_energies: bool = False
    return_initial_forces: bool = False
    return_final_forces: bool = False


class RelaxationGraphConfig(BaseModel):
    """Configuration for a single ``relaxation_graph`` call.

    Bundles the core relaxation parameters with the optional post-relaxation
    processing steps (symmetry refinement and uniqueness filtering) that
    ``relaxation_graph`` can apply after each pass.

    This class is the direct input type for ``relaxation_graph``.
    ``RelaxationConfig`` extends it with multi-pass orchestration flags for
    use inside the inpainting WorkGraph.
    """

    # Core relaxation parameters (forwarded as relax_inputs
    # to relax_structures)
    params: RelaxationParams

    # Post-relaxation steps
    refine: bool = False
    refinement_symprec: float = 0.01
    refinement_primitive: bool = False
    filter_unique: bool = False
    uniqueness: UniquenessConfig = Field(default_factory=UniquenessConfig)

    def relax_inputs(self, constrained: bool = True) -> dict:
        """Build the ``relax_inputs`` kwargs dict for ``relax_structures()``.

        Args:
            constrained: If True, include ``elements_to_relax`` so that only
                those elements are relaxed.  Pass False for full-relax passes
                where all atoms move freely.
        """
        if constrained:
            return self.params.model_dump()
        return self.params.model_dump(exclude={"elements_to_relax"})


class RelaxationConfig(RelaxationGraphConfig):
    """Configuration for the relaxation stage in the inpainting workflow.

    Extends ``RelaxationGraphConfig`` with multi-pass orchestration flags
    that are specific to the inpainting WorkGraph.  The three passes share
    the same ``params`` and post-relaxation settings.

    Pass names and their semantics
    --------------------------------
    constrained
        Relax only the atoms listed in ``params.elements_to_relax``.  Requires
        ``params.elements_to_relax`` to be set.  Labelled
        ``inpainted_constrained_relaxation`` in the WorkGraph.
    full
        Run a full (all-atom) relaxation on the output of the constrained pass.
        Requires ``constrained=True``.  Labelled
        ``pre_relaxed_inpainted_full_relaxation``.
    full_direct
        Run a full relaxation directly on the inpainted structures, bypassing
        the constrained pre-relax step (useful for comparison).  Labelled
        ``unrelaxed_inpainted_full_relaxation``.
    """

    # Which relaxation passes to run (inpainting-WG-specific)
    constrained: bool = True
    full: bool = False
    full_direct: bool = False

    @model_validator(mode="after")
    @classmethod
    def validate_passes(cls, cfg):
        """Validate relaxation modes."""
        if not any([cfg.constrained, cfg.full, cfg.full_direct]):
            raise ValueError(
                "At least one of 'constrained', 'full', or 'full_direct' "
                "must be True."
            )
        if cfg.constrained and cfg.params.elements_to_relax is None:
            raise ValueError(
                "'params.elements_to_relax' must be set when "
                "'constrained=True'."
            )
        if cfg.full and not cfg.constrained:
            raise ValueError(
                "'full=True' requires 'constrained=True': the full-relax pass "
                "runs on the output of the constrained pass."
            )
        return cfg


# ---------------------------------------------------------------------------
# AiiDA-specific options (ignored outside AiiDA execution)
# ---------------------------------------------------------------------------


class AiiDATaskOptions(BaseModel):
    """AiiDA scheduler and resource options for a single task.

    Replaces the raw ``Optional[dict]`` options fields.  ``withmpi``
    controls MPI-parallel execution and is kept here (infrastructure concern)
    rather than in the pipeline config.
    """

    resources: dict = Field(default_factory=dict)
    max_wallclock_seconds: Optional[int] = None
    queue_name: Optional[str] = None
    withmpi: bool = False


class AiiDAOptions(BaseModel):
    """AiiDA-specific settings: code labels and per-task scheduler options.

    Place this in ``XtalPaintConfig.aiida``; it is ignored entirely in
    non-AiiDA (plain Python) execution.
    """

    default_code_label: Optional[str] = None
    inpainting_code_label: Optional[str] = None
    relax_code_label: Optional[str] = None
    candidate_generation_code_label: Optional[str] = None

    inpainting_options: AiiDATaskOptions = Field(
        default_factory=AiiDATaskOptions
    )
    relax_options: AiiDATaskOptions = Field(default_factory=AiiDATaskOptions)
    candidate_generation_options: AiiDATaskOptions = Field(
        default_factory=AiiDATaskOptions
    )

    def get_code_label(self, specific: Optional[str] = None) -> Optional[str]:
        """Return *specific* code label, falling back to the default."""
        return specific or self.default_code_label


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class XtalPaintConfig(BaseModel):
    """Complete configuration for the XtalPaint inpainting workflow.

    Works for both AiiDA-based (WorkGraph) and plain-Python execution.
    AiiDA-specific settings live in the optional ``aiida`` block and are
    ignored in non-AiiDA runs.

    Pipeline stages are controlled by presence/absence of their config
    objects — no boolean flags required:

    * ``candidate_generation`` — omit if structures are already
      ``InpaintingStructureData`` objects.
    * ``pre_refinement`` — symmetry-refine structures before relaxation;
      omit to skip.
    * ``relaxation`` — geometry optimisation; omit to skip.

    Example (minimal)::

        XtalPaintConfig(
            structures={"si_001": structure},
            inpainting=InpaintingConfig(
                pretrained_name="mattergen_base",
                predictor_corrector="baseline",
                N_steps=5, coordinates_snr=0.2,
                n_corrector_steps=1, batch_size=1000,
            ),
        )

    Example (with relaxation + deduplication on AiiDA)::

        XtalPaintConfig(
            structures=...,
            candidate_generation=CandidateGenerationConfig(
                n_inp={"H": 2}, element="H"
            ),
            inpainting=InpaintingConfig(...),
            pre_refinement=RefinementConfig(symprec=0.01),
            relaxation=RelaxationConfig(
                params=RelaxationParams(
                    mlip="mattersim",
                    optimizer="BFGS",
                    elements_to_relax=["H"],
                    fmax=0.01,
                ),
                full=True,
                filter_unique=True,
            ),
            aiida=AiiDAOptions(
                default_code_label="xtalpaint@localhost",
                relax_code_label="relax@hpc",
                relax_options=AiiDATaskOptions(
                    resources={"num_machines": 1},
                    withmpi=True,
                ),
            ),
        )
    """

    structures: BatchedStructures | dict[str, Structure]
    run_inpainting: bool = True
    candidate_generation: Optional[CandidateGenerationConfig] = None
    pre_refinement: Optional[RefinementConfig] = None
    inpainting: InpaintingConfig
    relaxation: Optional[RelaxationConfig] = None
    aiida: Optional[AiiDAOptions] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("structures")
    @classmethod
    def validate_structures(cls, v):
        """Validate `structures` type."""
        structures = v
        if _is_batched_structure(v):
            structures = v.get_structures(strct_type="pymatgen")
        elif not isinstance(v, dict):
            raise TypeError(
                f"Expected a dictionary or BatchedStructures, got {type(v)}"
            )
        if not all(isinstance(k, str) for k in structures.keys()):
            raise TypeError("All keys in the dictionary must be strings")
        if not all(_is_valid_structure_type(s) for s in structures.values()):
            raise TypeError(
                "All values in the dictionary must be of type StructureData, "
                "Structure, ase.Atoms, or InpaintingStructureData"
            )
        types = {type(s) for s in structures.values()}
        if len(types) > 1:
            raise TypeError(
                "All values in the dictionary must be of the same type"
            )
        return v


# ---------------------------------------------------------------------------
# Evaluation parameters (standalone — not part of the inpainting workflow)
# ---------------------------------------------------------------------------


class EvalParameters(BaseModel):
    """Evaluation parameters for generated structures."""

    max_workers: int = 6
    chunksize: int = 50
    metrics: str | list[str] = "match"
    code_label: Optional[str] = None
