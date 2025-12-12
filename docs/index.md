# DBCSI Documentation

Welcome to the DBCSI Documentation.

## Overview

This project provides tools for crystal structure inpainting using machine learning models.

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
git clone https://github.com/t-reents/DBCSI.git
uv pip install .
```

## Acknowledgements

This project is developed to perform crystal structure inpainting, currently on top of Microsoft's [MatterGen](https://github.com/microsoft/mattergen). Some parts of the codebase are adapted from MatterGen's implementation (as highlighted in the respective files).
