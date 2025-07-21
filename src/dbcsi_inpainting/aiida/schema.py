"""Configuration schema validation for DBCSI inpainting experiments."""

from typing import Any, Dict, List, Optional, Union
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


class ParamGridConfig(BaseModel):
    """Configuration for parameter grid in experiments."""
    N_structures: int = Field(..., description="Number of structures to process")
    N_steps: int = Field(..., gt=0, description="Number of steps")
    coordinates_snr: List[float] = Field(..., description="Coordinates signal-to-noise ratio values")
    n_corrector_steps: List[int] = Field(..., description="Number of corrector steps")
    batch_size: int = Field(..., gt=0, description="Batch size")
    N_samples_per_structure: int = Field(..., gt=0, description="Number of samples per structure")
    n_resample_steps: Optional[List[int]] = Field(None, description="Number of resample steps")
    jump_length: Optional[List[int]] = Field(None, description="Jump length values")

    @field_validator('N_structures')
    @classmethod
    def validate_n_structures(cls, v):
        if v < -1 or v == 0:
            raise ValueError("N_structures must be positive or -1 for all structures")
        return v

    @field_validator('coordinates_snr')
    @classmethod
    def validate_coordinates_snr(cls, v):
        for snr in v:
            if not 0 < snr <= 1:
                raise ValueError(f"coordinates_snr values must be between 0 and 1, got {snr}")
        return v

    @field_validator('n_corrector_steps')
    @classmethod
    def validate_n_corrector_steps(cls, v):
        for steps in v:
            if steps <= 0:
                raise ValueError(f"n_corrector_steps values must be positive, got {steps}")
        return v


class RelaxKwargsConfig(BaseModel):
    """Configuration for relaxation parameters."""
    load_path: str = Field(..., min_length=1, description="Load path for relaxation")
    fmax: float = Field(..., gt=0, description="Maximum force threshold")


class InpaintingConfig(BaseModel):
    """Main configuration schema for DBCSI inpainting experiments."""
    name: str = Field(..., min_length=1, description="Experiment name")
    fix_cell: bool = Field(..., description="Whether to fix cell parameters")
    max_num_atoms: int = Field(..., gt=0, description="Maximum number of atoms")
    predictor_corrector: str = Field(..., min_length=1, description="Predictor-corrector method")
    param_grid: ParamGridConfig = Field(..., description="Parameter grid configuration")
    relax_kwargs: RelaxKwargsConfig = Field(..., description="Relaxation parameters")
    save_prefix: str = Field(..., min_length=1, description="Save prefix for outputs")
    
    # Model specification - exactly one must be provided
    pretrained_name: Optional[str] = Field(None, description="Name of pretrained model")
    model_path: Optional[str] = Field(None, description="Path to custom model")
    
    # Optional parameters
    mattergen_path: Optional[str] = Field(None, description="Path to MatterGen model")

    @model_validator(mode='after')
    def validate_model_specification(self):
        """Ensure exactly one of pretrained_name or model_path is specified."""
        model_specs = [self.pretrained_name, self.model_path]
        non_none_specs = [spec for spec in model_specs if spec is not None]
        
        if len(non_none_specs) == 0:
            raise ValueError("Either 'pretrained_name' or 'model_path' must be specified")
        elif len(non_none_specs) > 1:
            raise ValueError("Only one of 'pretrained_name' or 'model_path' can be specified, not both")
        
        return self


class ConfigValidator:
    """Utility class for loading and validating configuration files."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> InpaintingConfig:
        """Load and validate a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Validated InpaintingConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file has invalid YAML syntax
            ValueError: If configuration validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {e}")
        
        if config_dict is None:
            raise ValueError(f"Configuration file {config_path} is empty")
        
        return ConfigValidator._dict_to_config(config_dict)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> InpaintingConfig:
        """Convert a dictionary to InpaintingConfig with validation.
        
        Args:
            config_dict: Dictionary representation of configuration
            
        Returns:
            Validated InpaintingConfig object
        """
        # With pydantic, we can create the model directly from the dict
        # and it will handle nested model creation automatically
        try:
            return InpaintingConfig(**config_dict)
        except ValidationError as e:
            # Convert pydantic ValidationError to ValueError for consistent API
            raise ValueError(str(e))
    
    @staticmethod
    def validate_config_dict(config_dict: Dict[str, Any]) -> InpaintingConfig:
        """Validate a configuration dictionary.
        
        Args:
            config_dict: Dictionary representation of configuration
            
        Returns:
            Validated InpaintingConfig object
            
        Raises:
            ValueError: If configuration validation fails
        """
        return ConfigValidator._dict_to_config(config_dict)