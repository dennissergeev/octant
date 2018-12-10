#!/usr/bin/env python
"""Package build and install script."""
import os
import sys

import Cython.Build

import numpy as np

from setuptools import Extension, find_packages, setup

import versioneer


# Publish the library to PyPI.
if 'publish' in sys.argv[-1]:
    os.system('python setup.py sdist upload')
    sys.exit()


def get_readme():
    """Load README for display on PyPI."""
    with open('README.md') as f:
        return f.read()


CMDCLASS = versioneer.get_cmdclass({'build_ext': Cython.Build.build_ext})


setup(
    name='octant',
    version=versioneer.get_version(),
    cmdclass=CMDCLASS,
    description='Objective Cyclone Tracking ANalysis Tools',
    long_description=get_readme(),
    author='Denis Sergeev',
    author_email='dennis.sergeev@gmail.com',
    url='https://github.com/dennissergeev/octant',
    package_dir={'octant': 'octant'},
    packages=find_packages(),
    ext_modules=[Extension(
        'octant.utils',
        sources=['octant/utils.pyx'],
        include_dirs=[np.get_include()],
    )],
    zip_safe=False,
    setup_requires=['numpy>=1.7', 'cython>=0.24.1'],
    install_requires=['numpy>=1.7', 'pytest>=3.3', 'cython>=0.24.1',
                      'pandas>=0.20', 'xarray>=0.10.0'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
    ],
)
