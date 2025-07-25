# diffusion-based-crystal-structure-inpainting

[![CI](https://github.com/t-reents/diffusion-based-crystal-structure-inpainting/actions/workflows/ci.yml/badge.svg)](https://github.com/t-reents/diffusion-based-crystal-structure-inpainting/actions/workflows/ci.yml)

## Development Setup

This repository uses pre-commit hooks and CI to ensure code quality and consistency. 

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

The pre-commit configuration and CI include:
- **Black**: Code formatting
- **isort**: Import sorting 
- **flake8**: Linting and style checks

These tools will run automatically on git commits and in CI. You can also run them manually:
```bash
black src/
isort src/
flake8 --max-line-length=100 --extend-ignore=E203,W503,E501 src/
```

### Continuous Integration

The repository includes GitHub Actions CI that automatically runs:
- Code formatting checks (Black)
- Import sorting checks (isort) 
- Linting checks (flake8)

CI runs on all pushes to `main` and pull requests targeting `main`.