# DBCSI Inpainting Configuration Schema

This module provides configuration schema validation for diffusion-based crystal structure inpainting experiments.

## Features

- **Configuration validation**: Ensures all required parameters are provided with correct types
- **Mutual exclusivity validation**: Validates that only one of `pretrained_name` or `model_path` is specified
- **Range validation**: Validates that numeric parameters are within reasonable ranges
- **YAML support**: Load and validate configuration from YAML files
- **Type safety**: Full type checking using Python dataclasses

## Usage

### Basic Validation

```python
from dbcsi_inpainting.aiida import ConfigValidator
from dbcsi_inpainting.aiida.inpainting_process import validate_config_file, validate_config_dict

# Load and validate configuration from YAML file
config = ConfigValidator.load_config('config.yaml')

# Validate a configuration dictionary
config_dict = {...}  # your config as dict
config = validate_config_dict(config_dict)

# Simple validation function
config = validate_config_file('config.yaml')
```

### Command Line Validation

```bash
# Validate a configuration file
python -m dbcsi_inpainting.aiida.inpainting_process config.yaml

# Or from the aiida directory
cd src/dbcsi_inpainting/aiida/
python inpainting_process.py config.yaml
```

### Configuration File Format

The configuration file should be a YAML file with the following structure:

```yaml
name: "experiment-name"                    # Required: experiment name
fix_cell: true                            # Required: whether to fix cell
max_num_atoms: 20                         # Required: maximum number of atoms
predictor_corrector: "new-timesteps"      # Required: predictor-corrector method
save_prefix: "experiment-prefix"          # Required: save prefix

# Model specification - exactly one must be provided
pretrained_name: "mattergen_base"         # Option 1: use pretrained model
# model_path: "/path/to/custom/model"      # Option 2: use custom model path

# Optional: path to mattergen installation
mattergen_path: "/path/to/mattergen/"

# Required: parameter grid for experiments
param_grid:
  N_structures: -1                         # Number of structures (-1 for all)
  N_steps: 50                             # Number of diffusion steps
  coordinates_snr: [0.2]                 # Signal-to-noise ratio (0 < value <= 1)
  n_corrector_steps: [1]                 # Number of corrector steps (positive)
  batch_size: 500                        # Batch size (positive)
  N_samples_per_structure: 1             # Samples per structure (positive)
  
  # Optional parameters (for specific methods)
  n_resample_steps: [3]                  # Number of resample steps
  jump_length: [10]                      # Jump length

# Required: relaxation parameters
relax_kwargs:
  load_path: "MatterSim-v1.0.0-5M.pth"   # Path to relaxation model
  fmax: 0.0025                            # Force threshold (positive)
```

## Validation Rules

### Required Fields
- `name`: Non-empty string
- `fix_cell`: Boolean
- `max_num_atoms`: Positive integer
- `predictor_corrector`: Non-empty string
- `save_prefix`: Non-empty string
- `param_grid`: Object with required sub-fields
- `relax_kwargs`: Object with required sub-fields

### Model Specification (Mutual Exclusivity)
Exactly one of the following must be specified:
- `pretrained_name`: Name of a pretrained model
- `model_path`: Path to a custom model file

### Parameter Validation
- `max_num_atoms`: Must be positive
- `param_grid.N_structures`: Must be positive or -1 (for all structures)
- `param_grid.N_steps`: Must be positive
- `param_grid.coordinates_snr`: All values must be between 0 and 1
- `param_grid.n_corrector_steps`: All values must be positive
- `param_grid.batch_size`: Must be positive
- `param_grid.N_samples_per_structure`: Must be positive
- `relax_kwargs.fmax`: Must be positive

## Example Configurations

### Example 1: Using Pretrained Model
```yaml
name: "baseline-experiment"
fix_cell: true
max_num_atoms: 30
predictor_corrector: "baseline"
pretrained_name: "mattergen_base"
save_prefix: "baseline-exp"
param_grid:
  N_structures: 100
  N_steps: 200
  coordinates_snr: [0.2, 0.4]
  n_corrector_steps: [1, 5]
  batch_size: 256
  N_samples_per_structure: 5
relax_kwargs:
  load_path: "MatterSim-v1.0.0-5M.pth"
  fmax: 0.01
```

### Example 2: Using Custom Model with RePaint
```yaml
name: "repaint-experiment"
fix_cell: false
max_num_atoms: 50
predictor_corrector: "repaint-v2"
model_path: "/path/to/repaint/model"
save_prefix: "repaint-exp"
param_grid:
  N_structures: 250
  N_steps: 100
  coordinates_snr: [0.3]
  n_corrector_steps: [3]
  n_resample_steps: [3]
  jump_length: [10]
  batch_size: 512
  N_samples_per_structure: 1
relax_kwargs:
  load_path: "MatterSim-v1.0.0-5M.pth"
  fmax: 0.005
```

## API Reference

### ConfigValidator

#### Methods
- `load_config(config_path)`: Load and validate configuration from YAML file
- `validate_config_dict(config_dict)`: Validate configuration from dictionary

### Standalone Validation Functions
- `validate_config_file(config_path)`: Load and validate configuration from YAML file
- `validate_config_dict(config_dict)`: Validate configuration from dictionary

## Error Handling

The validation system provides clear error messages for common configuration issues:

- **Missing required fields**: "Field 'name' is required"
- **Invalid types**: "'max_num_atoms' must be positive"
- **Mutual exclusivity violations**: "Only one of 'pretrained_name' or 'model_path' can be specified"
- **Range violations**: "coordinates_snr values must be between 0 and 1"
- **File issues**: "Configuration file not found: config.yaml"

## Testing

Run the test suite to verify the validation system:

```bash
python -m unittest test_schema.py -v
```

All tests should pass, covering:
- Valid configuration loading
- Mutual exclusivity validation
- Type and range validation
- Error handling
- File I/O operations