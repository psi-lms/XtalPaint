"""Configuration schemas for the DBCSI AiiDA inpainting integration.

This module defines Pydantic BaseModel classes and validators for
inpainting workflows, pipelines, and their configuration parameters.
"""

from typing import Optional

from aiida import orm
from ase import Atoms
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.config import ConfigDict
from pymatgen.core import Structure

from dbcsi_inpainting.aiida.data import (
    BatchedStructures,
    InpaintingStructure,
)


class RelaxParameters(BaseModel):
    """Keywords for relaxation step."""

    load_path: str = None
    """:param load_path: Path to the model checkpoint for relaxation."""
    fmax: float = 0.05
    """:param fmax: Maximum force for relaxation."""
    fix_elements: Optional[list[str]] = Field(
        default=None, description="List of elements to fix during relaxation."
    )
    """:param fix_elements: Elements to fix during relaxation."""
    device: str = "cpu"
    """:param device: Device to run relaxation on."""
    filter: Optional[str] = None
    """:param filter: Filter to apply during relaxation."""


class InpaintingModelParams(BaseModel):
    """Parameters for diffusion sampling.

    Attributes:
        N_steps (int): Number of diffusion steps.
        coordinates_snr (float): SNR for coordinate diffusion.
        n_corrector_steps (int): Number of corrector steps in diffusion.
        batch_size (int): Number of samples per batch.
        N_samples_per_structure (int): Number of samples to generate
            per structure.
        n_resample_steps (Optional[int]): Number of resampling steps when using
            'repaint'.
        jump_length (Optional[float]): Jump length parameter for 'repaint'.
    """

    N_steps: int
    coordinates_snr: float
    n_corrector_steps: int
    batch_size: int
    N_samples_per_structure: int
    n_resample_steps: Optional[int] = None
    jump_length: Optional[float] = None


class InpaintingPipelineParams(BaseModel):
    """Parameters for inpainting pipeline.

    Attributes:
        predictor_corrector (str): Name of the predictor-corrector scheme.
        fix_cell (bool): Whether to fix the unit cell during diffusion.
        inpainting_model_params (InpaintingModelParams): Parameters for the
            diffusion model.
        pretrained_name (Optional[str]): Name of the pretrained model to use.
        model_path (Optional[str]): Path to the model checkpoint.
        record_trajectories (Optional[bool]): Whether to record trajectories
            during sampling.
    """

    predictor_corrector: str
    fix_cell: bool = True
    inpainting_model_params: InpaintingModelParams
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None
    record_trajectories: Optional[bool] = False

    @field_validator("predictor_corrector")
    @classmethod
    def validate_predictor_corrector(cls, v):
        """Validator to ensure 'predictor_corrector' is a supported key."""
        from dbcsi_inpainting.inpainting.inpainting_process import (
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
    """Parameters for generating inpainting candidates.

    Attributes:
        n_inp (int | dict[str, int]): Number of inpainting regions
            per structure.
        element (str | dict[str, str]): Element(s) to inpaint.
        num_samples (int): Number of samples to generate per region.
    """

    n_inp: int | dict[str, int]
    element: str | dict[str, str]
    num_samples: int = 1


class InpaintingWorkGraphConfig(BaseModel):
    """Configuration schema for the inpainting experiment.

    Attributes:
        structures (
            dict[str, Structure | InpaintingStructure] | BatchedStructures
            ): Mapping from labels to structures or batched structures.
        inpainting_pipeline_params (InpaintingPipelineParams):
            Pipeline parameters for inpainting.
        gen_inpainting_candidates_params (
            Optional[GenInpaintingCandidatesParams]]
            ): Parameters for generating inpainting candidates.
        code_label (Optional[str]): Optional label for the code plugin.
        relax (Optional[bool]): Whether to perform a relaxation step after
            inpainting.
        relax_kwargs (Optional[RelaxKwargs]): Keyword arguments for the
            relaxation step.
        full_relax (Optional[bool]): Whether to perform a full relaxation
            ignoring cell constraints.
        options (Optional[dict]): Additional execution options.
    """

    structures: dict[str, Structure | InpaintingStructure] | BatchedStructures
    inpainting_pipeline_params: InpaintingPipelineParams
    gen_inpainting_candidates_params: Optional[
        GenInpaintingCandidatesParams
    ] = None
    code_label: Optional[str] = None
    relax: Optional[bool] = False
    relax_kwargs: Optional[RelaxParameters] = {}
    full_relax: Optional[bool] = False
    options: Optional[dict] = {}
    relax_options: Optional[dict] = {}
    gen_inpainting_candidates_options: Optional[dict] = {}
    inpainting_pipeline_options: Optional[dict] = {}

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
        if not isinstance(v, dict):
            raise TypeError(
                "Expected a dictionary of StructureData objects, "
                f"got {type(v)}"
            )
        if not all(isinstance(k, str) for k in v.keys()):
            raise TypeError("All keys in the dictionary must be strings")
        if not all(
            isinstance(
                s, (orm.StructureData, Structure, Atoms, InpaintingStructure)
            )
            for s in v.values()
        ):
            raise TypeError(
                "All values in the dictionary must be of type StructureData, "
                "Structure, ase.Atoms, or InpaintingStructure"
            )

        types = {type(s) for s in v.values()}
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
        structures are not already InpaintingStructure instances.
        """
        values = list(cfg.structures.values())
        if not all(isinstance(s, InpaintingStructure) for s in values):
            if cfg.gen_inpainting_candidates_params is None:
                raise ValueError(
                    "If structures are not InpaintingStructure objects, "
                    "gen_inpainting_candidates_params must be provided."
                )
        return cfg

    @property
    def is_inpainting_structures(self) -> bool:
        """Check if structures are already InpaintingStructure objects."""
        return all(
            isinstance(s, InpaintingStructure)
            for s in self.structures.values()
        )
