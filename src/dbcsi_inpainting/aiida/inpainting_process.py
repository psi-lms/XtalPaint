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


if __name__ == "__main__":
    # Example usage for validation only
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inpainting_process.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Validate configuration
    try:
        config = validate_config_file(config_path)
        print(f"✓ Configuration '{config.name}' is valid")
        print(f"  Model: {config.pretrained_name or config.model_path}")
        print(f"  Predictor-corrector: {config.predictor_corrector}")
        print(f"  Max atoms: {config.max_num_atoms}")
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)