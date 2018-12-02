# -*- coding: utf-8 -*-
"""Miscellanea."""
import numpy as np

import xarray as xr

from .core import CATS
from .decor import pbar


SUBSETS = [i for i in CATS.keys() if i != 'unknown']
DENSITY_TYPES = ['point', 'track', 'genesis', 'lysis']


def calc_all_dens(tr_obj, lon2d, lat2d, method='radius', r=111.3 * 2):
    """
    Calculate all types of cyclone density for all SUBSETS of TrackRun.

    Arguments
    ---------
    lon2d: numpy.ndarray
        2D array of longitudes
    lat2d: numpy.ndarray
        2D array of latitudes
    method: str
        Method used in octant.core.TrackRun.density()

    Returns
    -------
    da: xarray.DataArray
       4d array with dimensions (subset, dens_type, latitude, longitude)

    """
    subset_dim = xr.DataArray(name='subset', dims=('subset'), data=SUBSETS)
    dens_dim = xr.DataArray(name='dens_type', dims=('dens_type'),
                            data=DENSITY_TYPES)
    list1 = []
    for subset in pbar(SUBSETS, leave=False, desc='subsets'):
        list2 = []
        for by in pbar(DENSITY_TYPES, leave=False, desc='density_types'):
            list2.append(tr_obj.density(lon2d, lat2d, by=by,
                         method=method, r=r, subset=subset))
        list1.append(xr.concat(list2, dim=dens_dim))
    da = xr.concat(list1, dim=subset_dim)
    return da.rename('density')


def bin_count_tracks(tr_obj, start_year, n_winters, by='M'):
    """
    Take `octant.TrackRun` and count cyclone tracks by month or by winter.

    Returns
    -------
    counter: numpy array of shape (N,)
        Binned counts

    """
    if by.upper() == 'M':
        counter = np.zeros(12, dtype=int)
        for _, df in pbar(tr_obj.groupby('track_idx'),
                          leave=False, desc='tracks'):
            track_months = df.time.dt.month.unique()
            for m in track_months:
                counter[m - 1] += 1
    if by.upper() == 'W':
        # winter
        counter = np.zeros(n_winters, dtype=int)
        for _, df in pbar(tr_obj.groupby('track_idx'),
                          leave=False, desc='tracks'):
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
