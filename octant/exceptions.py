# -*- coding: utf-8 -*-
"""Exceptions specific to octant package."""


class OctantWarning(UserWarning):
    """Base class for warnings in octant package."""

    pass


class MissingConfWarning(OctantWarning):
    """Tracking settings file (.conf) is missing in the tracks directory."""

    pass


class DeprecatedWarning(OctantWarning):
    """Warning for a deprecated feature."""

    pass


class InconsistencyWarning(OctantWarning):
    """Something is inconsistent, e.g. TrackRun metadata."""

    pass


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
    """Raised when argument type is not recognized."""

    pass


class LoadError(OctantError):
    """Raised when input files or directories are not found."""

    pass


class GridError(OctantError):
    """Raised when lon/lat grids are not correct."""

    pass


class ConcatenationError(OctantError):
    """Raised when there is something wrong with extend() method."""

    pass


class SelectError(OctantError):
    """Raised when wrong subset category is selected."""

    pass


class NotCategorisedError(OctantError):
    """Raised when operation on categories cannot proceed because TrackRun is not categorised."""

    pass
