<p align="left">
  <img src="docs/assets//XtalPaint-logo-horizontal.png" alt="XtalPaint logo" width="540">
</p>


# XtalPaint – A framework for crystal structure inpainting based on diffusion models

## Installation

## Example

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

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@misc{reents_2026_inpainting,
      title={Score-based diffusion models for accurate crystal-structure inpainting and reconstruction of hydrogen positions}, 
      author={Timo Reents and Arianna Cantarella and Marnik Bercx and Pietro Bonfà and Giovanni Pizzi},
      year={2026},
      eprint={2601.01959},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2601.01959}, 
}
```

## Acknowledgements

The initial version of this project relies on and implements extensions to [`Mattergen`](https://github.com/microsoft/mattergen). Please also consider citing the corresponding publication:

```bibtex
@article{MatterGen2025,
  author  = {Zeni, Claudio and Pinsler, Robert and Z{\"u}gner, Daniel and Fowler, Andrew and Horton, Matthew and Fu, Xiang and Wang, Zilong and Shysheya, Aliaksandra and Crabb{\'e}, Jonathan and Ueda, Shoko and Sordillo, Roberto and Sun, Lixin and Smith, Jake and Nguyen, Bichlien and Schulz, Hannes and Lewis, Sarah and Huang, Chin-Wei and Lu, Ziheng and Zhou, Yichi and Yang, Han and Hao, Hongxia and Li, Jielan and Yang, Chunlei and Li, Wenjie and Tomioka, Ryota and Xie, Tian},
  journal = {Nature},
  title   = {A generative model for inorganic materials design},
  year    = {2025},
  doi     = {10.1038/s41586-025-08628-5},
}
```

All modules that are reused or adapted from MatterGen are clearly marked in the codebase.
