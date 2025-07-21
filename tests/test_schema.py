"""Tests for the DBCSI inpainting configuration schema validation."""

import unittest
import tempfile
import yaml
from pathlib import Path
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dbcsi_inpainting.aiida.schema import ConfigValidator, InpaintingConfig


class TestConfigurationSchema(unittest.TestCase):
    """Test cases for configuration schema validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = {
            'name': 'test-experiment',
            'fix_cell': True,
            'max_num_atoms': 20,
            'predictor_corrector': 'new-timesteps',
            'model_path': '/path/to/model',
            'save_prefix': 'test-prefix',
            'param_grid': {
                'N_structures': 10,
                'N_steps': 50,
                'coordinates_snr': [0.2],
                'n_corrector_steps': [1],
                'batch_size': 100,
                'N_samples_per_structure': 1
            },
            'relax_kwargs': {
                'load_path': 'test.pth',
                'fmax': 0.01
            }
        }
    
    def test_valid_configuration(self):
        """Test that a valid configuration passes validation."""
        config = ConfigValidator.validate_config_dict(self.valid_config)
        self.assertEqual(config.name, 'test-experiment')
        self.assertEqual(config.model_path, '/path/to/model')
        self.assertIsNone(config.pretrained_name)
    
    def test_mutual_exclusivity_both_specified(self):
        """Test that specifying both pretrained_name and model_path fails."""
        invalid_config = self.valid_config.copy()
        invalid_config['pretrained_name'] = 'test_model'
        
        with self.assertRaises(ValueError) as context:
            ConfigValidator.validate_config_dict(invalid_config)
        
        self.assertIn('Only one of', str(context.exception))
    
    def test_mutual_exclusivity_neither_specified(self):
        """Test that specifying neither pretrained_name nor model_path fails."""
        invalid_config = self.valid_config.copy()
        del invalid_config['model_path']
        
        with self.assertRaises(ValueError) as context:
            ConfigValidator.validate_config_dict(invalid_config)
        
        self.assertIn('Either', str(context.exception))
        self.assertIn('must be specified', str(context.exception))
    
    def test_pretrained_name_only(self):
        """Test that specifying only pretrained_name works."""
        config_dict = self.valid_config.copy()
        del config_dict['model_path']
        config_dict['pretrained_name'] = 'mattergen_base'
        
        config = ConfigValidator.validate_config_dict(config_dict)
        self.assertEqual(config.pretrained_name, 'mattergen_base')
        self.assertIsNone(config.model_path)
    
    def test_invalid_coordinates_snr(self):
        """Test that invalid coordinates_snr values are rejected."""
        invalid_config = self.valid_config.copy()
        invalid_config['param_grid']['coordinates_snr'] = [1.5]  # > 1
        
        with self.assertRaises(ValueError) as context:
            ConfigValidator.validate_config_dict(invalid_config)
        
        self.assertIn('coordinates_snr', str(context.exception))
        self.assertIn('between 0 and 1', str(context.exception))
    
    def test_negative_values(self):
        """Test that negative values are rejected."""
        test_cases = [
            ('max_num_atoms', -5),
            ('param_grid.N_steps', -10),
            ('param_grid.batch_size', -100),
            ('param_grid.N_samples_per_structure', -1),
            ('relax_kwargs.fmax', -0.01)
        ]
        
        for field_path, invalid_value in test_cases:
            with self.subTest(field=field_path, value=invalid_value):
                invalid_config = self.valid_config.copy()
                
                # Set the nested value
                parts = field_path.split('.')
                target = invalid_config
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = invalid_value
                
                with self.assertRaises(ValueError):
                    ConfigValidator.validate_config_dict(invalid_config)
    
    def test_empty_strings(self):
        """Test that empty strings are rejected for required fields."""
        test_fields = ['name', 'save_prefix', 'predictor_corrector']
        
        for field in test_fields:
            with self.subTest(field=field):
                invalid_config = self.valid_config.copy()
                invalid_config[field] = ''
                
                with self.assertRaises(ValueError) as context:
                    ConfigValidator.validate_config_dict(invalid_config)
                
                self.assertIn(field, str(context.exception))
                self.assertIn('non-empty', str(context.exception))
    
    def test_yaml_file_loading(self):
        """Test loading configuration from a YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            temp_path = f.name
        
        try:
            config = ConfigValidator.load_config(temp_path)
            self.assertEqual(config.name, 'test-experiment')
        finally:
            Path(temp_path).unlink()
    
    def test_nonexistent_file(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ConfigValidator.load_config('/nonexistent/path/config.yaml')
    
    def test_invalid_yaml_syntax(self):
        """Test that invalid YAML syntax raises YAMLError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: syntax: [')  # Invalid YAML
            temp_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                ConfigValidator.load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        config_dict = self.valid_config.copy()
        config_dict['mattergen_path'] = '/optional/path'
        config_dict['param_grid']['n_resample_steps'] = [3]
        config_dict['param_grid']['jump_length'] = [10]
        
        config = ConfigValidator.validate_config_dict(config_dict)
        self.assertEqual(config.mattergen_path, '/optional/path')
        self.assertEqual(config.param_grid.n_resample_steps, [3])
        self.assertEqual(config.param_grid.jump_length, [10])


if __name__ == '__main__':
    unittest.main()