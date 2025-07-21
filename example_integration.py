#!/usr/bin/env python3
"""
Integration example showing how to use the configuration schema
with the existing DBCSI inpainting codebase.
"""

import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dbcsi_inpainting.aiida import ConfigValidator
from src.dbcsi_inpainting.aiida.inpainting_process import InpaintingProcess


def main():
    """Example integration with existing codebase."""
    
    # Example 1: Validate configuration file
    print("=== Configuration Validation Example ===")
    config_path = "src/dbcsi_inpainting/aiida/config.yaml"
    
    try:
        config = ConfigValidator.load_config(config_path)
        print(f"✓ Configuration '{config.name}' is valid")
        print(f"  Model specification: {config.model_path or config.pretrained_name}")
        print(f"  Predictor-corrector: {config.predictor_corrector}")
        print(f"  Max atoms: {config.max_num_atoms}")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return
    
    # Example 2: Extract parameters for existing functions
    print("\n=== Parameter Extraction Example ===")
    process = InpaintingProcess(config_path)
    
    # These parameters can be passed to existing functions
    param_grid = process.get_experiment_parameters()
    relax_kwargs = process.get_relax_kwargs()
    
    print("Parameters for existing run_inpainting.main():")
    print(f"  param_grid: {param_grid}")
    print(f"  predictor_corrector: {config.predictor_corrector}")
    print(f"  fix_cell: {config.fix_cell}")
    print(f"  relax_kwargs: {relax_kwargs}")
    print(f"  max_num_atoms: {config.max_num_atoms}")
    
    # Example 3: Show how this integrates with existing code
    print("\n=== Integration with Existing Code ===")
    print("This is how you would integrate with the existing codebase:")
    print("""
    # In your experiment script:
    from dbcsi_inpainting.aiida import InpaintingProcess
    from dbcsi_inpainting.run_inpainting import main
    
    # Load and validate configuration
    process = InpaintingProcess('config.yaml')
    
    # Extract validated parameters
    param_grid = process.get_experiment_parameters()
    relax_kwargs = process.get_relax_kwargs()
    
    # Call existing function with validated parameters
    main(
        param_grid=param_grid,
        predictor_corrector=process.config.predictor_corrector,
        fix_cell=process.config.fix_cell,
        relax_kwargs=relax_kwargs,
        max_num_atoms=process.config.max_num_atoms
    )
    """)
    
    # Example 4: Validate multiple configurations
    print("\n=== Multiple Configuration Validation ===")
    config_files = [
        "src/dbcsi_inpainting/aiida/config.yaml",
        "src/dbcsi_inpainting/aiida/config_baseline.yaml", 
        "src/dbcsi_inpainting/aiida/config_repaint.yaml"
    ]
    
    for config_file in config_files:
        try:
            config = ConfigValidator.load_config(config_file)
            model_type = "pretrained" if config.pretrained_name else "custom"
            print(f"✓ {config.name} ({model_type} model) - valid")
        except Exception as e:
            print(f"✗ {config_file} - invalid: {e}")


if __name__ == "__main__":
    main()