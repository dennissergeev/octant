# -*- coding: utf-8 -*-
"""
Miscellanea
"""
import numpy as np
import xarray as xr
try:
    # Check if it's Jupyter Notebook
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str.lower():
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

from .core import CATS


subsets = [i for i in CATS.keys() if i != 'unknown']
density_types = ['point', 'track', 'genesis', 'lysis']


def calc_all_dens(tr_obj, lon2d, lat2d, method='radius', r=111.3 * 2):
    """
    Calculate all types of cyclone density for all subsets of a TrackRun object

    Returns
    -------
    da: xarray.DataArray
       4d array with dimensions (subset, dens_type, latitude, longitude)
    """
    subset_dim = xr.DataArray(name='subset', dims=('subset'), data=subsets)
    dens_dim = xr.DataArray(name='dens_type', dims=('dens_type'),
                            data=density_types)
    list1 = []
    for subset in tqdm(subsets, leave=False):
        list2 = []
        for by in tqdm(density_types, leave=False):
            list2.append(tr_obj.density(lon2d, lat2d, by=by,
                         method=method, r=r, subset=subset))
        list1.append(xr.concat(list2, dim=dens_dim))
    da = xr.concat(list1, dim=subset_dim)
    return da.rename('density')


def bin_count_tracks(tr_obj, start_year, n_winters, by='M'):
    """
    Take `octant.TrackRun` and count cyclone tracks by month or by winter

    Returns
    -------
    counter: numpy array of shape (N,)
        Binned counts
    """
    if by.upper() == 'M':
        counter = np.zeros(12, dtype=int)
        for _, df in tqdm(tr_obj.groupby('track_idx'),
                          leave=False, desc='tracks'):
            track_months = df.time.dt.month.unique()
            for m in track_months:
                counter[m-1] += 1
    if by.upper() == 'W':
        # winter
        counter = np.zeros(n_winters, dtype=int)
        for _, df in tqdm(tr_obj.groupby('track_idx'),
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
