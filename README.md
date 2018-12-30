[![Build Status](https://travis-ci.com/dennissergeev/octant.svg?branch=master)](https://travis-ci.com/dennissergeev/octant)
[![Documentation Status](https://readthedocs.org/projects/octant-docs/badge/?version=latest)](https://octant-docs.readthedocs.io/en/latest/?badge=latest)
[![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6,3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Anaconda-Server Badge](https://anaconda.org/dennissergeev/octant/badges/version.svg)](https://anaconda.org/dennissergeev/octant)
[![LICENSE](https://anaconda.org/dennissergeev/octant/badges/license.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1313078.svg)](https://doi.org/10.5281/zenodo.1313078)


# octant
Objective Cyclone Tracking ANalysis Tools

## Installation
`octant` depends on the following packages:
  - cython
  - matplotlib
  - numpy
  - pandas
  - xarray

Plotting examples also require the `cartopy` package

### With conda
```bash
conda install -c dennissergeev octant
```

### From source
```bash
git clone https://github.com/dennissergeev/octant.git

cd octant

python setup.py install
```
