# diffusion-based-crystal-structure-inpainting

## Development Setup

This repository uses pre-commit hooks to ensure code quality and consistency.

### Setting up pre-commit

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. (Optional) Run pre-commit on all files:
   ```bash
   pre-commit run --all-files
   ```

### Code Quality Tools

The pre-commit configuration includes:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting and style checks
- **pydocstyle**: Docstring style checking (numpy convention)
- **Basic checks**: Trailing whitespace, end-of-file, YAML/TOML/JSON validation

These tools will run automatically on git commits. You can also run them manually:
```bash
black src/
isort src/
flake8 src/
pydocstyle src/
```
