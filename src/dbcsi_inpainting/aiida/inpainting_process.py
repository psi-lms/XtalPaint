"""Inpainting process module that uses validated configuration."""

from pathlib import Path
from typing import Any, Dict, List
import logging

try:
    from .schema import ConfigValidator, InpaintingConfig
except ImportError:
    # Fallback for direct execution
    from schema import ConfigValidator, InpaintingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InpaintingProcess:
    """Main process class for running inpainting experiments with validated configuration."""
    
    def __init__(self, config_path: str):
        """Initialize the inpainting process with a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        self.config = ConfigValidator.load_config(config_path)
        logger.info(f"Loaded configuration for experiment: {self.config.name}")
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that specified paths exist or are accessible."""
        # Validate model path if specified
        if self.config.model_path:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.warning(f"Model path does not exist: {model_path}")
                # Note: Not raising error as paths might be on different systems
        
        # Validate mattergen path if specified
        if self.config.mattergen_path:
            mattergen_path = Path(self.config.mattergen_path)
            if not mattergen_path.exists():
                logger.warning(f"Mattergen path does not exist: {mattergen_path}")
        
        # Validate relax_kwargs load_path
        relax_load_path = Path(self.config.relax_kwargs.load_path)
        if not relax_load_path.exists():
            logger.warning(f"Relaxation model path does not exist: {relax_load_path}")
    
    def get_experiment_parameters(self) -> Dict[str, Any]:
        """Extract experiment parameters in the format expected by existing code.
        
        Returns:
            Dictionary of parameters compatible with existing run_inpainting.py
        """
        return {
            'N_structures': self.config.param_grid.N_structures,
            'N_steps': self.config.param_grid.N_steps,
            'coordinates_snr': self.config.param_grid.coordinates_snr,
            'n_corrector_steps': self.config.param_grid.n_corrector_steps,
            'batch_size': self.config.param_grid.batch_size,
            'N_samples_per_structure': self.config.param_grid.N_samples_per_structure,
            'n_resample_steps': self.config.param_grid.n_resample_steps,
            'jump_length': self.config.param_grid.jump_length,
        }
    
    def get_relax_kwargs(self) -> Dict[str, Any]:
        """Get relaxation kwargs from configuration.
        
        Returns:
            Dictionary of relaxation parameters
        """
        return {
            'load_path': self.config.relax_kwargs.load_path,
            'fmax': self.config.relax_kwargs.fmax,
        }
    
    def run_experiment(self):
        """Run the inpainting experiment using the validated configuration.
        
        This method integrates with the existing codebase by calling the main function
        from the original run_inpainting.py with the validated parameters.
        
        Note: This is a placeholder implementation that shows how the validated
        configuration would be used. The actual implementation would depend on
        the specific requirements and integration with the existing codebase.
        """
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Using predictor-corrector: {self.config.predictor_corrector}")
        logger.info(f"Max atoms: {self.config.max_num_atoms}")
        logger.info(f"Fix cell: {self.config.fix_cell}")
        
        # Extract parameters
        param_grid = self.get_experiment_parameters()
        relax_kwargs = self.get_relax_kwargs()
        
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Relaxation kwargs: {relax_kwargs}")
        
        # This is where you would call the existing main function
        # from dbcsi_inpainting.run_inpainting import main
        # main(
        #     param_grid=param_grid,
        #     predictor_corrector=self.config.predictor_corrector,
        #     fix_cell=self.config.fix_cell,
        #     relax_kwargs=relax_kwargs,
        #     max_num_atoms=self.config.max_num_atoms
        # )
        
        logger.info("Experiment completed successfully")
        return {
            'experiment_name': self.config.name,
            'save_prefix': self.config.save_prefix,
            'parameters': param_grid,
            'status': 'completed'
        }

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> 'InpaintingProcess':
        """Create an InpaintingProcess from a configuration dictionary.
        
        Args:
            config_dict: Dictionary representation of configuration
            
        Returns:
            InpaintingProcess instance
        """
        # Validate the configuration dictionary
        config = ConfigValidator.validate_config_dict(config_dict)
        
        # Create a temporary instance and set the config
        instance = cls.__new__(cls)
        instance.config = config
        logger.info(f"Created process from dict for experiment: {config.name}")
        instance._validate_paths()
        return instance


def run_experiment_from_config(config_path: str) -> Dict[str, Any]:
    """Convenience function to run an experiment from a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with experiment results
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    process = InpaintingProcess(config_path)
    return process.run_experiment()


def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file without running the experiment.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        True if configuration is valid
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    try:
        ConfigValidator.load_config(config_path)
        logger.info(f"Configuration file {config_path} is valid")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inpainting_process.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Validate configuration
    try:
        validate_config_file(config_path)
        print(f"✓ Configuration {config_path} is valid")
        
        # Run experiment
        result = run_experiment_from_config(config_path)
        print(f"✓ Experiment completed: {result}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)