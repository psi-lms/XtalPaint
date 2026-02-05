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


def _is_inpainting_structure(obj) -> bool:
    """Check if object is an InpaintingStructureData (requires AiiDA)."""
    if is_aiida_installed():
        from xtalpaint.aiida.data import InpaintingStructureData

        return isinstance(obj, InpaintingStructureData)
    return False


class RelaxParameters(BaseModel):
    """Configuration for the relaxation stage."""

    load_path: str | None = None
    fmax: float = 0.05
    elements_to_relax: Optional[list[str]] = Field(
        default=None,
        description="List of elements to relax during optimization.",
    )
    max_natoms_per_batch: int = 512
    max_n_steps: int = 500
    device: str = "cpu"
    filter: Optional[str] = None
    optimizer: str
    mlip: str
    return_initial_energies: bool = False
    return_initial_forces: bool = False
    return_final_forces: bool = False


class InpaintingModelParams(BaseModel):
    """Diffusion sampling parameters for the inpainting model."""

    N_steps: int
    coordinates_snr: float
    n_corrector_steps: int
    batch_size: int
    n_resample_steps: Optional[int] = None
    jump_length: Optional[int] = None


class InpaintingPipelineParams(BaseModel):
    """Settings for constructing an inpainting pipeline."""

    predictor_corrector: str
    fix_cell: bool = True
    inpainting_model_params: InpaintingModelParams
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None
    record_trajectories: Optional[bool] = False
    sampling_config_path: Optional[str] = None

    @field_validator("predictor_corrector")
    @classmethod
    def validate_predictor_corrector(cls, v):
        """Validator to ensure 'predictor_corrector' is a supported key."""
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
        """Validate model specification.

        Ensure that either 'pretrained_name' or 'model_path' is provided,
        but not both.
        """
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
        """Validate 'n_resample_steps' and 'jump_length'.

        If 'predictor_corrector' contains 'repaint', both parameters must be
        set in 'inpainting_model_params'.
        """
        if "repaint" in cfg.predictor_corrector.lower():
            params = cfg.inpainting_model_params
            if params.n_resample_steps is None or params.jump_length is None:
                raise ValueError(
                    "When 'predictor_corrector' contains 'repaint', "
                    "inpainting_model_params must set both 'n_resample_steps' "
                    "and 'jump_length'."
                )
        return cfg


class GenInpaintingCandidatesParams(BaseModel):
    """Configuration for generating inpainting candidates."""

    n_inp: int | dict[str, int]
    element: str | dict[str, str]
    num_samples: int = 1


class EvalParameters(BaseModel):
    """Evaluation parameters for generated structures."""

    max_workers: int = 6
    chunksize: int = 50
    metrics: str | list[str] = "match"
    code_label: Optional[str] = None


class InpaintingWorkflowConfig(BaseModel):
    """Top-level configuration for a XtalPaint inpainting workflow.

    This config can be used for both AiiDA-based workflows (WorkGraphs)
    and regular Python-based workflows.
    """

    structures: BatchedStructures | dict[str, Structure]
    run_inpainting: bool = True
    inpainting_pipeline_params: InpaintingPipelineParams
    gen_inpainting_candidates_params: Optional[
        GenInpaintingCandidatesParams
    ] = None
    code_label: Optional[str] = None
    relax_code_label: Optional[str] = None
    inpainting_code_label: Optional[str] = None
    relax: Optional[bool] = False
    relax_kwargs: Optional[RelaxParameters] = {}
    full_relax: Optional[bool] = False
    full_relax_wo_pre_relax: Optional[bool] = False
    options: Optional[dict] = {}
    relax_options: Optional[dict] = {}
    gen_inpainting_candidates_options: Optional[dict] = {}
    inpainting_pipeline_options: Optional[dict] = {}
    evaluate: Optional[bool] = False
    evaluate_params: Optional[EvalParameters] = None
    refine_structures: bool = False
    refine_structures_after_relax: bool = False
    refinement_symprec: float = 0.01

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("structures")
    @classmethod
    def validate_structures(cls, v):
        """Validate input structures.

        Ensure 'structures' is a dictionary with string keys and values of
        uniform, supported types.

        Raises:
            TypeError: If the structure mapping is not valid.
        """
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

    @model_validator(mode="after")
    @classmethod
    def check_n_inp_for_structures(cls, cfg):
        """Validate inputs for inpainting candidates.

        Ensure that 'gen_inpainting_candidates_params' is provided when
        structures are not already inpainting structure instances.
        """
        values = (
            list(cfg.structures.values())
            if isinstance(cfg.structures, dict)
            else cfg.structures.get_structures(strct_type="pymatgen")
        )
        if not all(
            _is_inpainting_structure(s) or isinstance(s, Structure)
            for s in values
        ):
            if cfg.gen_inpainting_candidates_params is None:
                raise ValueError(
                    "If structures are not InpaintingStructure objects, "
                    "gen_inpainting_candidates_params must be provided."
                )
        return cfg

    @model_validator(mode="after")
    @classmethod
    def check_evaluate_inpainting_structures(cls, cfg):
        """Check if structures are already InpaintingStructure objects."""
        if cfg.evaluate and cfg.is_inpainting_structures:
            raise ValueError(
                "If 'evaluate' is True, structures must not be "
                "InpaintingStructure objects. We need the original structures "
                "to compare against inpainted structures."
            )
        return cfg

    @property
    def is_inpainting_structures(self) -> bool:
        """Check if structures are already InpaintingStructure objects."""
        structures = (
            self.structures.values()
            if isinstance(self.structures, dict)
            else self.structures.get_structures(strct_type="pymatgen")
        )
        return all(_is_inpainting_structure(s) for s in structures)
