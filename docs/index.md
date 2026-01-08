# XtalPaint – A framework for crystal structure inpainting based on diffusion models


Welcome to the `XtalPaint` Documentation.

## Overview

`XtalPaint` is a Python package that provides tools to perform crystal structure inpainting, i.e. adding atomic sites to a given host structure, using score-based diffusion models. Here, we provide retrained versions of the [`Mattergen`](https://github.com/microsoft/mattergen) architecture and the building blocks to set up the inpainting workflows. The initial application in our latest work: [Score-based diffusion models for accurate crystal-structure inpainting and reconstruction of hydrogen positions](https://doi.org/10.48550/arXiv.2601.01959), focuses on adding missing hydrogen sites to inorganic crystal structures, but the framework can be adapted to other inpainting tasks as well, i.e. general crystal structure prediction based on given host structures (see other interesting works in the field, e.g. by [Zhong _et al._](https://pubs.rsc.org/en/content/articlehtml/2025/mh/d5mh00774g)).

## Features

- Inpainting pipeline for crystal structures
- Integration with AiiDA workflow management
- Support for various relaxation methods
- Evaluation metrics for inpainting quality

## Getting Started

Check out the examples on to run the inpainting pipeline:

* [With AiiDA integration](examples/running-with-AiiDA.ipynb)
* [Without AiiDA integration](examples/running-wo-AiiDA.ipynb)

## Installation

```bash
git clone https://github.com/psi-lms/XtalPaint.git
cd XtalPaint/

uv pip install .
```

This will install the default version. If you want to use it in combination with [AiiDA](https://aiida.readthedocs.io/projects/aiida-core/en/stable/), please also install the optional dependencies:

```bash
uv pip install .[aiida]
```

### Model checkpoints for retrained versions of MatterGen

Model checkpoints for the retrained versions of MatterGen used in our work can be downloaded from [Hugging Face](https://huggingface.co/t-reents/XtalPaint). Currently, the repository contains the `pos-only` and `TD-pos-only` models discussed in the paper.


## Acknowledgements

This project is developed to perform crystal structure inpainting, currently on top of Microsoft's [MatterGen](https://github.com/microsoft/mattergen). Some parts of the codebase are adapted from MatterGen's implementation (as highlighted in the respective files).
