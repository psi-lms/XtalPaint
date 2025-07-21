"""DBCSI Inpainting AiiDA integration package.

This package provides configuration schema validation and process management
for diffusion-based crystal structure inpainting experiments.
"""

from .schema import InpaintingConfig, ConfigValidator

# Lazy imports to avoid dependency issues
def get_inpainting_process():
    """Lazy import for InpaintingProcess to avoid numpy dependency issues."""
    from .inpainting_process import InpaintingProcess, run_experiment_from_config, validate_config_file
    return InpaintingProcess, run_experiment_from_config, validate_config_file

__all__ = [
    'InpaintingConfig',
    'ConfigValidator', 
    'get_inpainting_process'
]