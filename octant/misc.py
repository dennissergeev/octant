# -*- coding: utf-8 -*-
"""Miscellanea."""
import operator
from collections.abc import Iterable

import numpy as np

import xarray as xr

from .decor import get_pbar
from .exceptions import ArgumentError
from .params import EARTH_RADIUS, KM2M
from .utils import great_circle, mask_tracks, mean_arr_along_track

DENSITY_TYPES = ["point", "track", "genesis", "lysis"]


def _exclude_by_first_day(df, m, d):
    """Check if OctantTrack starts on certain day and month."""
    return not ((df.time.dt.month[0] == m).any() and (df.time.dt.day[0] == d).any())


def _exclude_by_last_day(df, m, d):
    """Check if OctantTrack ends on certain day and month."""
    return not ((df.time.dt.month[-1] == m).any() and (df.time.dt.day[-1] == d).any())


def calc_all_dens(tr_obj, lon2d, lat2d, subsets=None, density_types=DENSITY_TYPES, **kwargs):
    """
    Calculate all types of cyclone density for subsets of TrackRun.

    Parameters
    ----------
    lon2d: numpy.ndarray
        2D array of longitudes
    lat2d: numpy.ndarray
        2D array of latitudes
    subsets: list, optional
        Subsets of `TrackRun` to process. By default, all subsets are processed.
    density_types: list, optional
        Types of cyclone density
    **kwargs: dict
        Keyword arguments passed to `octant.core.TrackRun.density()`.
        Should not include `subset` and `by` keywords, because they are passed separately.

    Returns
    -------
    da: xarray.DataArray
       4d array with dimensions (subset, dens_type, latitude, longitude)

    See Also
    --------
    octant.core.TrackRun.density
    """
    pbar = get_pbar()

    if subsets is None:
        if tr_obj.is_categorised:
            subsets = tr_obj.cat_labels
        else:
            subsets = None
    else:
        if not isinstance(subsets, Iterable) or isinstance(subsets, str):
            raise ArgumentError("`subsets` should be a sequence of strings")

    subset_dim = xr.DataArray(name="subset", dims=("subset"), data=subsets)
    dens_dim = xr.DataArray(name="dens_type", dims=("dens_type"), data=density_types)
    list1 = []
    for subset in pbar(subsets):  # , desc="subsets"):
        list2 = []
        for by in pbar(density_types):  # , desc="density_types"):
            list2.append(tr_obj.density(lon2d, lat2d, by=by, subset=subset, **kwargs))
        list1.append(xr.concat(list2, dim=dens_dim))
    da = xr.concat(list1, dim=subset_dim)
    return da.rename("density")


def bin_count_tracks(tr_obj, start_year, n_winters, by="M"):
    """
    Take `octant.TrackRun` and count cyclone tracks by month or by winter.

    Parameters
    ----------
    tr_obj: octant.core.TrackRun
        TrackRun object
    start_year: int
        Start year
    n_winters: int
        Number of years

    Returns
    -------
    counter: numpy.ndarray
        Binned counts of shape (N,)

    """
    pbar = get_pbar()

    if by.upper() == "M":
        counter = np.zeros(12, dtype=int)
        for _, df in pbar(tr_obj.gb, leave=False):  # , desc="tracks"):
            track_months = df.time.dt.month.unique()
            for m in track_months:
                counter[m - 1] += 1
    if by.upper() == "W":
        # winter
        counter = np.zeros(n_winters, dtype=int)
        for _, df in pbar(tr_obj.gb, leave=False):  # , desc="tracks"):
            track_months = df.time.dt.month.unique()
            track_years = df.time.dt.year.unique()

            for i in range(n_winters):
                if track_months[-1] <= 6:
                    if track_years[0] == i + start_year + 1:
                        counter[i] += 1
                else:
                    if track_years[-1] == i + start_year:
                        counter[i] += 1
    return counter


def check_by_mask(
    ot,
    trackrun,
    lsm,
    lmask_thresh=1,
    dist=50.0,
    time_frac=0.5,
    check_domain_bounds=True,
    r_planet=EARTH_RADIUS,
):
    """
    Check how close the OctantTrack is to masked points.

    Check if the given track spends less than `time_frac` of its lifetime
    within `dist` away from the land or domain boundaries (if check_domain_bounds is True).

    This function can be passed to `octant.core.TrackRun.classify()` to filter
    through cyclone tracks.

    Parameters
    ----------
    ot: octant.parts.OctantTrack
        Cyclone track to check
    trackrun: octant.core.TrackRun
        (parent) track run instance to get lon/lat boundaries if present
    lsm: xarray.DataArray
        Two-dimensional land-sea mask
    lmask_thresh: float, optional
        Threshold of `lsm` values, for flexible land-mask filtering
    dist: float, optional
        distance in km, passed to mask_tracks() function
    time_frac: float, optional
        Threshold for track's lifetime (0-1)
    check_domain_bounds: bool, optional
        If true, include domain boundary (taken from TrackRun.conf if available) in the mask
    r_planet: float, optional
        Radius of the planet in metres
        Default: EARTH_RADIUS

    Returns
    -------
    flag: bool
        The track is far away the land mask or from the boundaries.

    Examples
    --------
    >>> from octant.core import TrackRun
    >>> import xarray as xr
    >>> land_mask = xr.open_dataarray("path/to/land/mask/file")
    >>> tr = TrackRun("path/to/directory/with/tracks/")
    >>> random_track = tr.data.loc[123]
    >>> check_by_mask(random_track, tr, land_mask, lmask_thresh=0.5)
    True

    See Also
    --------
    octant.core.TrackRun.classify, octant.utils.mask_tracks, octant.misc.check_far_from_boundaries
    """
    assert isinstance(lsm, xr.DataArray), "lsm variable should be an `xarray.DataArray`"
    lon2d, lat2d = np.meshgrid(lsm.longitude, lsm.latitude)
    if check_domain_bounds:
        l_mask = add_domain_bounds_to_mask(lsm, trackrun.conf.extent)
    else:
        l_mask = lsm
    mask_c = ((l_mask.values >= lmask_thresh) * 1.0).astype("double", order="C")
    lon2d_c = lon2d.astype("double", order="C")
    lat2d_c = lat2d.astype("double", order="C")
    flag = (
        mask_tracks(mask_c, lon2d_c, lat2d_c, ot.lonlat_c, dist * KM2M, r_planet=r_planet)
        <= time_frac
    )
    return flag


def check_far_from_boundaries(ot, lonlat_box, dist, r_planet=EARTH_RADIUS):
    """
    Check if track is not too close to boundaries.

    Parameters
    ----------
    ot: octant.parts.OctantTrack
        Individual cyclone-track object
    lonlat_box: list
        Boundaries of longitude-latitude rectangle (lon_min, lon_max, lat_min, lat_max)
        Note that the order matters!
    dist: float
        Minimum distance from a boundary in kilometres
    r_planet: float, optional
        Radius of the planet in metres
        Default: EARTH_RADIUS

    Returns
    -------
    result: bool
        True if track is not too close to boundaries

    Examples
    --------
    >>> from octant.core import TrackRun
    >>> tr = TrackRun("path/to/directory/with/tracks/")
    >>> random_track = tr.data.loc[123]
    >>> check_far_from_boundaries(random_track, lonlat_box=[-10, 20, 60, 80], dist=250)
    True

    >>> from functools import partial
    >>> conds = [
            ('bound', [partial(check_far_from_boundaries, lonlat_box=tr.conf.extent, dist=100)])
        ]  # construct a condition for tracks to be within the boundaries taken from the TrackRun
    >>> tr.classify(conds)
    >>> tr.cat_labels
    ['bound']

    See Also
    --------
    octant.parts.OctantTrack.within_rectangle, octant.utils.check_by_mask
    """
    # Preliminary check: track is within the rectangle
    # (Could be the case for a small rectangle.)
    result = (
        (ot.lon >= lonlat_box[0])
        & (ot.lon <= lonlat_box[1])
        & (ot.lat >= lonlat_box[2])
        & (ot.lat <= lonlat_box[3])
    ).all()
    if not result:
        return False

    # Main check
    for i, ll in enumerate(lonlat_box):

        def _func(row):
            args = [row.lon, row.lon, row.lat, row.lat].copy()
            args[2 * (i // 2)] = ll
            return great_circle(*args, r_planet=r_planet)

        result &= (ot.apply(_func, axis=1) > dist * KM2M).all()

    return result


def check_by_arr_thresh(ot, arr, arr_thresh, oper, dist, reduce="mean", r_planet=EARTH_RADIUS):
    """
    Check if the mean value of `arr` along the track satisfies the threshold.

    This function can be passed to `octant.core.TrackRun.classify()` to filter
    through cyclone tracks.

    Parameters
    ----------
    ot: octant.parts.OctantTrack
        Cyclone track to check
    arr: xarray.DataArray
        Two-dimensional array
    arr_thresh: float
        Threshold used for `arr` values
    oper: str
        Math operator the mean array value to the threshold
        Can be one of (lt|le|gt|ge)
    dist: float
        Distance in km, passed to mask_tracks() function
    reduce: str, optional
        How to select values along the track (mean|any)
    r_planet: float, optional
        Radius of the planet in metres
        Default: EARTH_RADIUS

    Returns
    -------
    flag: bool
        True if the track satisfies conditions above.

    Examples
    --------
    Check that ocean fraction is greater than 75% within 111 km radius

    >>> from octant.core import TrackRun
    >>> import xarray as xr
    >>> land_mask = xr.open_dataarray("path/to/land/mask/file")
    >>> tr = TrackRun("path/to/directory/with/tracks/")
    >>> random_track = tr.data.loc[123]
    >>> check_by_arr_thresh(random_track, land_mask, 0.25, "le", 111.0)
    True

    See Also
    --------
    octant.core.TrackRun.classify, octant.utils.mean_arr_along_track,
    octant.misc.check_by_mask, octant.misc.check_far_from_boundaries
    """
    allowed_ops = ["lt", "le", "gt", "ge"]
    if oper not in allowed_ops:
        # TODO: create chk_var() function
        raise ArgumentError(f"oper={oper} should be one of {allowed_ops}")
    allowed_ops = ["mean", "all", "any"]
    if reduce not in allowed_ops:
        raise ArgumentError(f"reduce={reduce} should be one of {allowed_ops}")
    op = getattr(operator, oper)
    assert isinstance(arr, xr.DataArray), "arr should be an `xarray.DataArray`"
    lon2d, lat2d = np.meshgrid(arr.longitude, arr.latitude)
    arr_c = arr.values.astype("double", order="C")
    lon2d_c = lon2d.astype("double", order="C")
    lat2d_c = lat2d.astype("double", order="C")
    mean_vals = mean_arr_along_track(
        arr_c, lon2d_c, lat2d_c, ot.lonlat_c, dist * KM2M, r_planet=r_planet
    ).base
    if reduce == "mean":
        flag = op(mean_vals.mean(), arr_thresh)
    elif reduce == "all":
        flag = op(mean_vals, arr_thresh).all()
    elif reduce == "any":
        flag = op(mean_vals, arr_thresh).any()
    return flag


def add_domain_bounds_to_mask(mask, lonlat_box):
    """
    Add a frame representing domain boundaries to a mask.

    Parameters
    ----------
    mask: xarray.DataArray
        Two-dimensional mask
    lonlat_box: list
        Boundaries of longitude-latitude rectangle (lon_min, lon_max, lat_min, lat_max)
        Note that the order matters!

    Returns
    -------
    new_mask: xarray.DataArray
        New mask including the old mask and domain boundaries

    Examples
    --------
    >>> from octant.core import TrackRun
    >>> import xarray as xr
    >>> land_mask = xr.open_dataarray("path/to/land/mask/file")
    >>> tr = TrackRun("path/to/directory/with/tracks/")
    >>> new_mask = add_domain_bounds_to_mask(land_mask, lonlat_box=tr.conf.extent)

    See Also
    --------
    octant.utils.mask_tracks, octant.misc.check_far_from_boundaries
    """
    assert isinstance(mask, xr.DataArray), "mask variable should be an `xarray.DataArray`"
    lon2d, lat2d = np.meshgrid(mask.longitude, mask.latitude)

    lon1, lon2, lat1, lat2 = lonlat_box

    inner_idx = np.ones(mask.shape, dtype=bool)
    if lon1 is not None:
        inner_idx &= lon2d >= lon1
    if lon2 is not None:
        inner_idx &= lon2d <= lon2
    if lat1 is not None:
        inner_idx &= lat2d >= lat1
    if lat2 is not None:
        inner_idx &= lat2d <= lat2
    new_mask = mask.where(inner_idx, 1.0)
    return new_mask
