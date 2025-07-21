"""Configuration schema validation for DBCSI inpainting experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import yaml
from pathlib import Path


@dataclass
class ParamGridConfig:
    """Configuration for parameter grid in experiments."""
    N_structures: int
    N_steps: int
    coordinates_snr: List[float]
    n_corrector_steps: List[int]
    batch_size: int
    N_samples_per_structure: int
    n_resample_steps: Optional[List[int]] = None
    jump_length: Optional[List[int]] = None


@dataclass
class RelaxKwargsConfig:
    """Configuration for relaxation parameters."""
    load_path: str
    fmax: float


@dataclass
class InpaintingConfig:
    """Main configuration schema for DBCSI inpainting experiments."""
    name: str
    fix_cell: bool
    max_num_atoms: int
    predictor_corrector: str
    param_grid: ParamGridConfig
    relax_kwargs: RelaxKwargsConfig
    save_prefix: str
    
    # Model specification - exactly one must be provided
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None
    
    # Optional parameters
    mattergen_path: Optional[str] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        self._validate_model_specification()
        self._validate_types()
        self._validate_values()

    def _validate_model_specification(self):
        """Ensure exactly one of pretrained_name or model_path is specified."""
        model_specs = [self.pretrained_name, self.model_path]
        non_none_specs = [spec for spec in model_specs if spec is not None]
        
        if len(non_none_specs) == 0:
            raise ValueError("Either 'pretrained_name' or 'model_path' must be specified")
        elif len(non_none_specs) > 1:
            raise ValueError("Only one of 'pretrained_name' or 'model_path' can be specified, not both")

    def _validate_types(self):
        """Validate that all fields have the correct types."""
        # Type validation is mostly handled by dataclass type hints
        # Additional custom validation can be added here
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("'name' must be a non-empty string")
        
        if not isinstance(self.save_prefix, str) or not self.save_prefix.strip():
            raise ValueError("'save_prefix' must be a non-empty string")
        
        if not isinstance(self.predictor_corrector, str) or not self.predictor_corrector.strip():
            raise ValueError("'predictor_corrector' must be a non-empty string")

    def _validate_values(self):
        """Validate that values are within reasonable ranges."""
        if self.max_num_atoms <= 0:
            raise ValueError("'max_num_atoms' must be positive")
        
        if self.param_grid.N_structures < -1 or self.param_grid.N_structures == 0:
            raise ValueError("'N_structures' must be positive or -1 for all structures")
        
        if self.param_grid.N_steps <= 0:
            raise ValueError("'N_steps' must be positive")
        
        if self.param_grid.batch_size <= 0:
            raise ValueError("'batch_size' must be positive")
        
        if self.param_grid.N_samples_per_structure <= 0:
            raise ValueError("'N_samples_per_structure' must be positive")
        
        # Validate coordinates_snr values
        for snr in self.param_grid.coordinates_snr:
            if not 0 < snr <= 1:
                raise ValueError(f"coordinates_snr values must be between 0 and 1, got {snr}")
        
        # Validate n_corrector_steps values
        for steps in self.param_grid.n_corrector_steps:
            if steps <= 0:
                raise ValueError(f"n_corrector_steps values must be positive, got {steps}")
        
        # Validate relax_kwargs
        if self.relax_kwargs.fmax <= 0:
            raise ValueError("relax_kwargs.fmax must be positive")


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
        # Extract param_grid
        param_grid_dict = config_dict.get('param_grid', {})
        param_grid = ParamGridConfig(**param_grid_dict)
        
        # Extract relax_kwargs
        relax_kwargs_dict = config_dict.get('relax_kwargs', {})
        relax_kwargs = RelaxKwargsConfig(**relax_kwargs_dict)
        
        # Create main config
        config_dict_copy = config_dict.copy()
        config_dict_copy['param_grid'] = param_grid
        config_dict_copy['relax_kwargs'] = relax_kwargs
        
        return InpaintingConfig(**config_dict_copy)
    
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