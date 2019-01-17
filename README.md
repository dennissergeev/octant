[![Build Status](https://travis-ci.com/dennissergeev/octant.svg?branch=master)](https://travis-ci.com/dennissergeev/octant)
[![Documentation Status](https://readthedocs.org/projects/octant-docs/badge/?version=latest)](https://octant-docs.readthedocs.io/en/latest/?badge=latest)
[![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6,3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Anaconda-Server Badge](https://anaconda.org/dennissergeev/octant/badges/version.svg)](https://anaconda.org/dennissergeev/octant)
[![LICENSE](https://anaconda.org/dennissergeev/octant/badges/license.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1313078.svg)](https://doi.org/10.5281/zenodo.1313078)


# octant
Objective Cyclone Tracking ANalysis Tools

The package is developed as a companion to [PMCTRACK](https://github.com/dennissergeev/pmctrack) - vorticity-based cyclone tracking algorithm.

The documentation is available [here](https://octant-docs.readthedocs.io/en/latest/).

Contributions are welcome.

## Installation
`octant` depends on the following packages:
  - cython
  - matplotlib
  - numpy
  - pandas
  - pytables
  - xarray

Plotting examples also require the `cartopy` package.

### With conda (recommended)
```bash
conda install -c dennissergeev octant
```

### From source
After the dependencies are installed:
```bash
git clone https://github.com/dennissergeev/octant.git

cd octant

python setup.py install
```
