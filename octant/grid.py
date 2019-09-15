# -*- coding: utf-8 -*-
"""Operations on geographical grid."""
import numpy as np

from .params import EARTH_RADIUS


def cell_centres(bounds, bound_position=0.5):
    """
    Calculate coordinate cell centres.

    Inspired by SciTools iris package.

    Parameters
    ----------
    bounds: numpy.array
        One-dimensional array of cell boundaries of shape (M,)
    bound_position: bool, optional
        The desired position of the bounds relative to the position
        of the points.

    Returns
    -------
    centres: numpy.array
        Array of shape (M+1,)

    Examples
    --------
    >>> a = np.arange(-1, 3., 1.)
    >>> a
    array([-1,  0,  1,  2])
    >>> cell_centres(a)
    array([-0.5,  0.5,  1.5])

    See Also
    --------
    octant.grid.cell_bounds
    """
    assert bounds.ndim == 1, "Only 1D points are allowed"
    deltas = np.diff(bounds) * bound_position
    centres = bounds[:-1] + deltas
    return centres


def cell_bounds(points, bound_position=0.5):
    """
    Calculate coordinate cell boundaries.

    Inspired by SciTools iris package.

    Parameters
    ----------
    points: numpy.array
        One-dimensional array of uniformy spaced values of shape (M,)
    bound_position: bool, optional
        The desired position of the bounds relative to the position
        of the points.

    Returns
    -------
    bounds: numpy.array
        Array of shape (M+1,)

    Examples
    --------
    >>> a = np.arange(-1, 2.5, 0.5)
    >>> a
    array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    >>> cell_bounds(a)
    array([-1.25, -0.75, -0.25,  0.25,  0.75,  1.25,  1.75,  2.25])

    See Also
    --------
    octant.grid.cell_centres
    """
    assert points.ndim == 1, "Only 1D points are allowed"
    diffs = np.diff(points)
    assert (diffs == diffs[0]).all(), "The function only works for uniformly spaced points"
    delta = diffs[0] * bound_position
    bounds = np.concatenate([[points[0] - delta], points + delta])
    return bounds


def _iris_guess_bounds(points, bound_position=0.5):
    """Simplified function from iris.coord.Coord."""
    diffs = np.diff(points)
    diffs = np.insert(diffs, 0, diffs[0])
    diffs = np.append(diffs, diffs[-1])

    min_bounds = points - diffs[:-1] * bound_position
    max_bounds = points + diffs[1:] * (1 - bound_position)

    return np.array([min_bounds, max_bounds]).transpose()


def _quadrant_area(radian_lat_bounds, radian_lon_bounds, r_planet):
    """
    Calculate spherical segment areas.

    Taken from iris library.

    Area weights are calculated for each lat/lon cell as:
        .. math::
            r^2 (lon_1 - lon_0) ( sin(lat_1) - sin(lat_0))

    The resulting array will have a shape of
    *(radian_lat_bounds.shape[0], radian_lon_bounds.shape[0])*
    The calculations are done at 64 bit precision and the returned array
    will be of type numpy.float64.

    Parameters
    ----------
    radian_lat_bounds: numpy.array
        Array of latitude bounds (radians) of shape (N, 2)
    radian_lon_bounds: numpy.array
        Array of longitude bounds (radians) of shape (N, 2)
    r_planet: float
        Radius of the planet (currently assumed spherical)
    """
    # ensure pairs of bounds
    if (
        radian_lat_bounds.shape[-1] != 2
        or radian_lon_bounds.shape[-1] != 2
        or radian_lat_bounds.ndim != 2
        or radian_lon_bounds.ndim != 2
    ):
        raise ValueError("Bounds must be [n,2] array")

    # fill in a new array of areas
    radius_sqr = r_planet ** 2
    radian_lat_64 = radian_lat_bounds.astype(np.float64)
    radian_lon_64 = radian_lon_bounds.astype(np.float64)

    ylen = np.sin(radian_lat_64[:, 1]) - np.sin(radian_lat_64[:, 0])
    xlen = radian_lon_64[:, 1] - radian_lon_64[:, 0]
    areas = radius_sqr * np.outer(ylen, xlen)

    # we use abs because backwards bounds (min > max) give negative areas.
    return np.abs(areas)


def grid_cell_areas(lon1d, lat1d, r=EARTH_RADIUS):
    """Simplified iris function to calculate grid cell areas."""
    lon_bounds_radian = np.deg2rad(_iris_guess_bounds(lon1d))
    lat_bounds_radian = np.deg2rad(_iris_guess_bounds(lat1d))
    area = _quadrant_area(lat_bounds_radian, lon_bounds_radian, r)
    return area
