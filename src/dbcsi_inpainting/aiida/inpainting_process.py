"""Input validation utilities for DBCSI inpainting configuration."""

from pathlib import Path
from typing import Any, Dict

try:
    from .schema import ConfigValidator, InpaintingConfig
except ImportError:
    # Fallback for direct execution
    from schema import ConfigValidator, InpaintingConfig


def validate_config_file(config_path: str) -> InpaintingConfig:
    """Validate a configuration file and return the parsed config.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Validated InpaintingConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    return ConfigValidator.load_config(config_path)


def validate_config_dict(config_dict: Dict[str, Any]) -> InpaintingConfig:
    """Validate a configuration dictionary and return the parsed config.
    
    Args:
        config_dict: Dictionary representation of configuration
        
    Returns:
        Validated InpaintingConfig object
        
    Raises:
        ValueError: If configuration is invalid
    """
    return ConfigValidator.validate_config_dict(config_dict)