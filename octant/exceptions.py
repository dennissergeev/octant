# -*- coding: utf-8 -*-
"""
Exceptions specific to octant package.
"""


class OctantError(Exception):
    """Base class for errors in octant package."""
    pass


class NotYetImplementedError(OctantError):
    """
    Raised by missing functionality.

    Different meaning to NotImplementedError, which is for abstract methods.
    """
    pass


class ArgumentError(OctantError):
    """Raised when argument type is not recognized"""
    pass


class LoadError(OctantError):
    """Raised when input files or directories are not found"""
    pass
