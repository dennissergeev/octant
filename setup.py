#!/usr/bin/env python
"""Package build and install script."""
import os
import sys

import Cython.Build

import numpy as np

from setuptools import Extension, find_packages, setup


# Publish the library to PyPI.
if 'publish' in sys.argv[-1]:
    os.system('python setup.py sdist upload')
    sys.exit()


def get_version():
    """
    Load the version from version.py, without importing it.

    This function assumes that the last line in the file contains a variable
    defining the version string with single quotes.
    """
    with open('octant/version.py', 'r') as f:
        return f.read().split('=')[-1].replace('\'', '').strip()


def get_readme():
    """Load README for display on PyPI."""
    with open('README.md') as f:
        return f.read()


setup(
    name='octant',
    version=get_version(),
    description='Objective Cyclone Tracking ANalysis Tools',
    long_description=get_readme(),
    author='Denis Sergeev',
    author_email='dennis.sergeev@gmail.com',
    url='https://github.com/dennissergeev/octant',
    cmdclass={'build_ext': Cython.Build.build_ext},
    package_dir={'octant': 'octant'},
    packages=find_packages(),
    ext_modules=[Extension(
        'octant.utils',
        sources=['octant/utils.pyx'],
        include_dirs=[np.get_include()],
    )],
    zip_safe=False,
    setup_requires=['numpy', 'cython'],
    install_requires=['numpy', 'pytest', 'cython', 'pandas', 'xarray'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
    ],
)
