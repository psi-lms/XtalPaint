"""Tests for the DBCSI inpainting configuration schema validation."""

import tempfile
import yaml
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dbcsi_inpainting.aiida.schema import ConfigValidator, InpaintingConfig
from pydantic import ValidationError


# Test fixtures
def get_valid_config():
    """Get a valid configuration for testing."""
    return {
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


def test_valid_configuration():
    """Test that a valid configuration passes validation."""
    valid_config = get_valid_config()
    config = ConfigValidator.validate_config_dict(valid_config)
    assert config.name == 'test-experiment'
    assert config.model_path == '/path/to/model'
    assert config.pretrained_name is None


def test_mutual_exclusivity_both_specified():
    """Test that specifying both pretrained_name and model_path fails."""
    invalid_config = get_valid_config()
    invalid_config['pretrained_name'] = 'test_model'
    
    try:
        ConfigValidator.validate_config_dict(invalid_config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Only one of' in str(e)


def test_mutual_exclusivity_neither_specified():
    """Test that specifying neither pretrained_name nor model_path fails."""
    invalid_config = get_valid_config()
    del invalid_config['model_path']
    
    try:
        ConfigValidator.validate_config_dict(invalid_config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Either' in str(e)
        assert 'must be specified' in str(e)


def test_pretrained_name_only():
    """Test that specifying only pretrained_name works."""
    config_dict = get_valid_config()
    del config_dict['model_path']
    config_dict['pretrained_name'] = 'mattergen_base'
    
    config = ConfigValidator.validate_config_dict(config_dict)
    assert config.pretrained_name == 'mattergen_base'
    assert config.model_path is None


def test_invalid_coordinates_snr():
    """Test that invalid coordinates_snr values are rejected."""
    invalid_config = get_valid_config()
    invalid_config['param_grid']['coordinates_snr'] = [1.5]  # > 1
    
    try:
        ConfigValidator.validate_config_dict(invalid_config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'coordinates_snr' in str(e)
        assert 'between 0 and 1' in str(e)


def test_negative_values():
    """Test that negative values are rejected."""
    test_cases = [
        ('max_num_atoms', -5),
        ('param_grid.N_steps', -10),
        ('param_grid.batch_size', -100),
        ('param_grid.N_samples_per_structure', -1),
        ('relax_kwargs.fmax', -0.01)
    ]
    
    for field_path, invalid_value in test_cases:
        invalid_config = get_valid_config()
        
        # Set the nested value
        parts = field_path.split('.')
        target = invalid_config
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = invalid_value
        
        try:
            ConfigValidator.validate_config_dict(invalid_config)
            assert False, f"Should have raised ValueError for {field_path} = {invalid_value}"
        except ValueError:
            pass  # Expected


def test_empty_strings():
    """Test that empty strings are rejected for required fields."""
    test_fields = ['name', 'save_prefix', 'predictor_corrector']
    
    for field in test_fields:
        invalid_config = get_valid_config()
        invalid_config[field] = ''
        
        try:
            ConfigValidator.validate_config_dict(invalid_config)
            assert False, f"Should have raised ValueError for empty {field}"
        except ValueError as e:
            # Pydantic uses "String should have at least 1 character" for empty strings
            error_message = str(e)
            assert 'String should have at least 1 character' in error_message or 'non-empty' in error_message


def test_yaml_file_loading():
    """Test loading configuration from a YAML file."""
    valid_config = get_valid_config()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config, f)
        temp_path = f.name
    
    try:
        config = ConfigValidator.load_config(temp_path)
        assert config.name == 'test-experiment'
    finally:
        Path(temp_path).unlink()


def test_nonexistent_file():
    """Test that loading a nonexistent file raises FileNotFoundError."""
    try:
        ConfigValidator.load_config('/nonexistent/path/config.yaml')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # Expected


def test_invalid_yaml_syntax():
    """Test that invalid YAML syntax raises YAMLError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('invalid: yaml: syntax: [')  # Invalid YAML
        temp_path = f.name
    
    try:
        ConfigValidator.load_config(temp_path)
        assert False, "Should have raised YAMLError"
    except yaml.YAMLError:
        pass  # Expected
    finally:
        Path(temp_path).unlink()


def test_optional_fields():
    """Test that optional fields work correctly."""
    config_dict = get_valid_config()
    config_dict['mattergen_path'] = '/optional/path'
    config_dict['param_grid']['n_resample_steps'] = [3]
    config_dict['param_grid']['jump_length'] = [10]
    
    config = ConfigValidator.validate_config_dict(config_dict)
    assert config.mattergen_path == '/optional/path'
    assert config.param_grid.n_resample_steps == [3]
    assert config.param_grid.jump_length == [10]


# If running directly, run all test functions
if __name__ == '__main__':
    test_functions = [
        test_valid_configuration,
        test_mutual_exclusivity_both_specified,
        test_mutual_exclusivity_neither_specified,
        test_pretrained_name_only,
        test_invalid_coordinates_snr,
        test_negative_values,
        test_empty_strings,
        test_yaml_file_loading,
        test_nonexistent_file,
        test_invalid_yaml_syntax,
        test_optional_fields
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
    
    print("All tests completed.")