# -*- coding: utf-8 -*-
"""
Miscellanea
"""
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
