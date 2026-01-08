<p align="left">
  <img src="docs/assets//XtalPaint-logo-horizontal.png" alt="XtalPaint logo" width="540">
</p>


# XtalPaint – A framework for crystal structure inpainting based on diffusion models

## Installation
The recommended way of installing the package is using uv: https://docs.astral.sh/uv/getting-started/installation/

```bash
git clone https://github.com/t-reents/XtalPaint.git
cd XtalPaint/

uv pip install .
```

This will install the default version. If you want to use it in combination with [AiiDA](https://aiida.readthedocs.io/projects/aiida-core/en/stable/), please also install the optional dependencies:

```bash
uv pip install .[aiida]
```

## Example

<img src="./docs/assets/XtalPaint-example.gif"/>



In the following, we show one example of how to use the inpainting models.
More details can be found in the [documentation](https://t-reents.github.io/XtalPaint/) and the corresponding examples, including the relaxation steps and the evaluation of inpainted structures against references.

```python
from ase.build import bulk

host_structures = {
  'Al': bulk('Al', cubic=True),
  'Si': bulk('Si', cubic=True),
}

from xtalpaint.inpainting.generate_candidates import (
    generate_inpainting_candidates,
)

# Inpaint 3 H sites to the Al structure and 2 O sites to the Si structure
inpainting_candidates = generate_inpainting_candidates(
    structures=host_structures,
    n_inp={'Al': 3, 'Si': 2},
    element={'Al': 'H', 'Si': 'O'},
    num_samples=1, # Number of samples per structure
)

from xtalpaint.inpainting.inpainting_process import (
    run_inpainting_pipeline,
)

# Parameters for the inpainting, please adjust to reasonable values
param_grid = {
    "N_steps": 50,
    "coordinates_snr": 0.2,
    "n_corrector_steps": 1,
    "batch_size": 100,
}

inpainted_structures = run_inpainting_pipeline(
    structures=inpainting_candidates,
    config={
        "record_trajectories": False,
        "predictor_corrector": "baseline",
        "inpainting_model_params": param_grid,
        "pretrained_name": "mattergen_base",
        "sampling_config_path": None, # Potentially change it to point to the mattergen sampling config directory
    }
)

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
