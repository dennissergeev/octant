# -*- coding: utf-8 -*-
"""
Classes and functions for the analysis of PMCTRACK output
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .utils import (great_circle, density_grid_rad, mask_tracks,
                    distance_metric, total_dist)

HOUR = np.timedelta64(1, 'h')
m2km = 1e-3

TrackSubset = namedtuple('TrackSubset', ['good', 'pmc', 'polarlows'])


class OctantSeries(pd.Series):
    @property
    def _constructor(self):
        return OctantSeries


class OctantTrack(pd.DataFrame):
    """
    Instance of cyclone track

    Subclass of `pandas.DataFrame`
    """
    def __init__(self, *args, **kw):
        super(OctantTrack, self).__init__(*args, **kw)

    @property
    def _constructor(self):
        return OctantTrack  # replace with self.__class__?

    _constructor_sliced = OctantSeries

    @classmethod
    def from_df(cls, df):
        return cls.from_records(df.to_records())

    @classmethod
    def from_mux_df(cls, df):
        return cls.from_records(df.to_records(index=True),
                                index=df.index.names)

    @property
    def coord_view(self):
        return (self.lon.values.view('double'),
                self.lat.values.view('double'),
                self.time.values.view('int64'))

    @property
    def lonlat(self):
        return self[['lon', 'lat']].values

    @property
    def lonlat_c(self):
        return self.lonlat.astype('double', order='C')

    @property
    def lifetime_h(self):
        return (self.time.values[-1]
                - self.time.values[0]) / HOUR

    @property
    def gen_lys_dist_km(self):
        return great_circle(self.lonlat[0, 0], self.lonlat[-1, 0],
                            self.lonlat[0, 1], self.lonlat[-1, 1]) * m2km

    @property
    def total_dist_km(self):
        return total_dist(self.lonlat_c) * m2km

    @property
    def average_speed(self):
        return self.total_dist_km / self.lifetime_h

    @property
    def max_vort(self):
        return np.nanmax(self.vo.values)

    @property
    def mean_vort(self):
        return np.nanmean(self.vo.values)


class TrackRun:
    """
    Results of tracking experiment

    Attributes
    ----------
    filelist: list
        list of "vortrack" files
    conf: TrackSettings
        Configuration used for tracking
    """
    # Keywords for `pandas.read_csv()` used in `load_data()` method
    _load_kw = dict(delimiter='\s+',
                    names=['lon', 'lat', 'vo', 'time', 'area', 'vortex_type'],
                    parse_dates=['time'])
    mux_names = ['track_idx', 'row_idx']

    def __init__(self, dirname=None):
        """
        Arguments
        ---------
        dirname: pathlib.Path or path.Path, optional
            Path to the directory with tracking output
            If present, load the data during on init
        """
        self.dirname = dirname
        self.conf = None
        self.data = pd.DataFrame()
        self.filelist = []
        self.sources = []
        # self._density = None
        if isinstance(self.dirname, Path):
            # Read all files and store in self.all
            # as a list of `pandas.DataFrame`s
            self.load_data(self.dirname)
        elif self.dirname is not None:
            raise TypeError('dirname should be Path-like object')

        if not self.data.empty:
            # Define time step
            for (_, ot) in self.data.groupby('track_idx'):
                if ot.shape[0] > 1:
                    self.tstep_h = ot.time.diff().values[-1] / HOUR
                    break

    def __len__(self):
        return self.data.index.get_level_values(0).to_series().nunique()

    def __repr__(self):
        s = '\n'.join(self.sources)
        return f'TrackRun({s}, {(len(self))})'

    def __add__(self, other):
        new = self.__class__()
        new.extend(self)
        new.extend(other)
        return new

    def load_data(self, dirname, primary_only=True, conf_file=None):
        """
        Arguments
        ---------
        dirname: pathlib.Path or path.Path
            Path to the directory with tracking output
        conf_file: pathlib.Path or path.Path, optional
            Path to the configuration file. If omitted, an attempt is
            made to find a .conf file in the `dirname` directory
        primary_only: bool, optional
            Load only "primary" vortices and skip the merged ones
        """

        if primary_only:
            wcard = 'vortrack*0001.txt'
        else:
            wcard = 'vortrack*.txt'
        self.filelist = sorted([*dirname.glob(wcard)])
        self.sources.append(str(dirname))

        # Load configuration
        if conf_file is None:
            conf_file = list(dirname.glob('*.conf'))[0]
        try:
            self.conf = TrackSettings(conf_file)
        except (IndexError, AttributeError):
            pass

        # Load the tracks
        _data = []
        for fname in self.filelist:
            _data.append(OctantTrack.from_df(pd.read_csv(fname,
                                                         **self._load_kw)))
        self.data = pd.concat(_data, keys=range(len(_data)),
                              names=self.mux_names)
        del _data

    def extend(self, other, adapt_conf=True):
        """
        Extend the TrackRun by appending elements from another TrackRun
        Arguments
        ---------
        TODO
        """
        new_data = pd.concat([self.data, other.data])
        new_track_idx = new_data.index.get_level_values(0).to_series()
        new_track_idx = new_track_idx.ne(new_track_idx.shift()).cumsum() - 1

        mux = pd.MultiIndex.from_arrays([new_track_idx,
                                         new_data.index.get_level_values(1)],
                                        names=new_data.index.names)
        new_data.set_index(mux)

        self.data = new_data

        if adapt_conf and other.conf is not None:
            if self.conf is None:
                self.conf = other.conf.copy()
            else:
                for field in self.conf._fields:
                    if getattr(self.conf, field) != getattr(other.conf, field):
                        setattr(self.conf, field, None)
        self.sources.extend(other.sources)
        if getattr(self, 'tstep_h', None) is None:
            self.tstep_h = getattr(other, 'tstep_h', None)
        else:
            if getattr(other, 'tstep_h', None) is not None:
                assert self.tstep_h == other.tstep_h

    def categorise(self, time_thresh=6, filt_by_time=True,
                   dist_thresh=300., filt_by_dist=True,
                   type_thresh=0.2, lsm=None, filt_by_land=True,
                   filt_by_domain_bounds=True, coast_rad=30.):
        """
        Sort the loaded tracks by PMC/PL criteria

        The function populates the following lists:
          - self.Subset.good
          - self.Subset.pmc
          - self.Subset.polarlows

        Arguments
        ---------
        time_thresh: int, optional
            Time threshold (hours) for basic filtering
        filt_by_time: bool, optional
            Filter by the time threshold
        dist_thresh: float, optional
            Distance in km
            Used for classifying vortices depending on the total distance
            travelled or distance between genesis and lysis.
        filt_by_dist: bool, optional
            Filter by the distance threshold
        lsm: xarray.DataArray of shape 2, optional
            Land-sea mask
            If present, tracks that spend > 0.5 of their lifetime
            within `coast_rad` radius from the coastline are discarded
        coast_rad: float, optional
            Radius in km
            Used for discarding vortices stuck near the coastline
        filt_by_land: bool, optional
            Filter by the proximity to coast given by the land mask (`lsm`)
        filt_by_domain_bounds: bool, optional
            Filter by the proximity to domain boundarie, which are taken from
            the `self.conf` instance if present: lon1, lon2, lat2, lat2
        type_thresh: float, optional
            Ratio of time steps when `vortex_type` is not equal to 0 (PMC) to
            the total lifetime of the vortex. Should be within 0-1 range.
            `type_thresh=1./7.` means that if the number of time steps when the
            vortex is considered a synoptic-scale low or a cold front is more
            than one-seventh of the whole lifetime of the PMC, then the PMC is
            excluded as a synoptic-scale disturbance [Watanabe et al., 2016].
        """
        filt_by_mask = False
        if isinstance(lsm, xr.DataArray):
            lon2d, lat2d = np.meshgrid(lsm.longitude, lsm.latitude)
            l_mask = np.zeros_like(lon2d)
            if filt_by_land:
                l_mask = lsm.values
                filt_by_mask = True
            boundary_mask = np.zeros_like(lon2d)
            if filt_by_domain_bounds:
                filt_by_mask = True
                inner_idx = True
                if getattr(self.conf, 'lon1', None):
                    inner_idx &= lon2d >= self.conf.lon1
                if getattr(self.conf, 'lon2', None):
                    inner_idx &= lon2d <= self.conf.lon2
                if getattr(self.conf, 'lat1', None):
                    inner_idx &= lat2d >= self.conf.lat1
                if getattr(self.conf, 'lat2', None):
                    inner_idx &= lat2d <= self.conf.lat2
                boundary_mask = np.zeros_like(lon2d)
                boundary_mask[~inner_idx] = 1.
            self.themask = (((boundary_mask == 1.) | (l_mask == 1.)) * 1.)
            themask_c = self.themask.astype('double', order='C')
            lon2d_c = lon2d.astype('double', order='C')
            lat2d_c = lat2d.astype('double', order='C')

        for ct in self.all:
            good_flag = True
            pmc_flag = False
            polarlow_flag = False
            if (filt_by_time and ct.lifetime_h < time_thresh):
                good_flag = False
            if (good_flag and filt_by_mask
                and mask_tracks(themask_c, lon2d_c, lat2d_c,
                                ct.lonlat_c, coast_rad * 1e3) > 0.5):
                good_flag = False
            if good_flag and filt_by_dist:
                if ct.gen_lys_dist_km < dist_thresh:
                    good_flag = False

            if good_flag:
                self.Subset.good.append(ct)
                if (((ct.vortex_type != 0).sum()
                     / ct.shape[0] < type_thresh)
                   and (ct.total_dist_km > dist_thresh)):
                    pmc_flag = True
                if pmc_flag:
                    self.Subset.pmc.append(ct)
                    if polarlow_flag:
                        self.Subset.polarlows.append(ct)

    def match_tracks(self, others, subset='good', method='simple',
                     interpolate_to='other',
                     thresh_dist=250., time_frac_thresh=0.5, beta=10.):
        """
        Match tracked vortices to a list of vortices from another data source

        Arguments
        ---------
        others: list
            List of `pandas.DataFrame`s
        subset: str, optional
            Subset to match (good|pmc|polarlows)
        method: str, optional
            Method of matching (intersection|simple|bs2000)
        interpolate_to: str, optional
            Interpolate `TrackRun` times to `other` times, or vice versa
        thresh_dist: float, optional
            Radius (km) threshold of distances between vortices.
            Used in 'intersection' and 'simple' methods
        time_frac_thresh: float, optional
            Fraction of a vortex lifetime used as a threshold in 'intersection'
            and 'simple' methods
        beta: float, optional
            $\beta$ parameter used in 'bs2000' method
        Returns
        -------
        match_pairs: list
            Index pairs of `other` vortices matching a vortex in `TrackRun`
            in a form (<index of `TrackRun` subset>, <index of `other`>)
        """
        sub = self.Subset._asdict()[subset]
        other_tracks = [CycloneTrack(df) for df in others]
        match_pairs = []
        if method == 'intersection':
            for idx, ct in enumerate(sub):
                for other_idx, other_ct in enumerate(other_tracks):
                    times = other_ct.df.time.values
                    time_match_thresh = (time_frac_thresh
                                         * (times[-1] - times[0]) / HOUR)

                    intersect = pd.merge(other_ct.df, ct.df, how='inner',
                                         left_on='time', right_on='time')
                    n_match_times = intersect.shape[0]
                    # = len(set(df.kt.values).intersection(set(stars_times)))
                    dist = (intersect[['lon_x', 'lon_y', 'lat_x', 'lat_y']]
                            .apply(lambda x: great_circle(*x.values),
                                   axis=1))

                    prox_time = ((dist < (thresh_dist * 1e3)).sum()
                                 * self.tstep_h)
                    if ((n_match_times * self.tstep_h > time_match_thresh)
                       and prox_time > time_match_thresh):
                        match_pairs.append((idx, other_idx))
                        break
                #     else:
                #         _flag = False
                # if _flag:
                #     hits.append(df)
                # else:
                #     misses.append(df)

        elif method == 'simple':
            ll = ['lon', 'lat']
            match_pairs = []
            for other_idx, other_ct in enumerate(other_tracks):
                candidates = []
                for idx, ct in enumerate(sub):
                    if interpolate_to == 'other':
                        df1, df2 = ct.df, other_ct.df
                    elif interpolate_to == 'self':
                        df1, df2 = other_ct.df, ct.df
                    l_start = max(df1.time.values[0], df2.time.values[0])
                    e_end = min(df1.time.values[-1], df2.time.values[-1])
                    if (e_end - l_start) / HOUR > 0:
                        # df1 = df1.set_index('time')[ll]
                        # ts = pd.Series(index=df2.time)
                        # new_df1 = (pd.concat([df1, ts]).sort_index()
                        #            .interpolate(method='values')
                        #            .loc[ts.index])[ll]
                        tmp_df2 = pd.DataFrame(data=dict(lon=np.nan,
                                                         lat=np.nan,
                                                         time=df2.time),
                                               index=df2.index)
                        new_df1 = (pd.concat([df1[[*ll, 'time']], tmp_df2],
                                             keys='time')
                                   .set_index('time')
                                   .sort_index()
                                   .interpolate(method='values')
                                   .loc[tmp_df2.time])[ll]
                        new_df1 = new_df1[~new_df1.lon.isnull()]

                        # thr = (time_frac_thresh * 0.5
                        #       * (df2.time.values[-1] - df2.time.values[0]
                        #          + df1.time.values[-1] - df2.time.values[0]))
                        thr = time_frac_thresh * df2.shape[0]
                        dist_diff = np.full(new_df1.shape[0], 9e20)
                        for i, ((x1, y1),
                                (x2, y2)) in enumerate(zip(new_df1[ll].values,
                                                           df2[ll].values)):
                            dist_diff[i] = great_circle(x1, x2, y1, y2)
                        within_r_idx = dist_diff < (thresh_dist * 1e3)
                        # if within_r_idx.any():
                        #     if (new_df1[within_r_idx].index[-1]
                        #        - new_df1[within_r_idx].index[0]) > thr:
                        #         candidates.append((idx, within_r_idx.sum()))
                        if within_r_idx.sum() > thr:
                            candidates.append((idx, within_r_idx.sum()))
                if len(candidates) > 0:
                    candidates = sorted(candidates, key=lambda x: x[1])
                    final_idx = candidates[-1][0]
                    match_pairs.append((final_idx, other_idx))

        elif method == 'bs2000':
            dist_matrix = np.empty((len(sub), len(others)))
            for i, ct in enumerate(sub):
                x1, y1, t1 = ct.coord_view
                for j, other_ct in enumerate(other_tracks):
                    x2, y2, t2 = other_ct.coord_view
                    dist_matrix[i, j] = distance_metric(x1, y1, t1,
                                                        x2, y2, t2,
                                                        beta=float(beta))
            for i, idx1 in enumerate(np.nanargmin(dist_matrix, axis=0)):
                for j, idx2 in enumerate(np.nanargmin(dist_matrix, axis=1)):
                    if i == idx2 and j == idx1:
                        match_pairs.append((idx1, idx2))

        else:
            raise ValueError(f'Unknown method: {method}')

        return match_pairs

    # @property
    def density(self, lon2d, lat2d, subset='good', method='radius', r=100.):
        """
        Create a density map

        Arguments
        ---------
        r: float, optional
            Radius in km
            Used when method='radius'
        lon2d, lat2d
        TODO
        """
        # if self._density is None or redo=True:
        sub = self.Subset._asdict()[subset]
        lon2d_c = lon2d.astype('double', order='C')
        lat2d_c = lat2d.astype('double', order='C')
        r_metres = r * 1e3
        if method == 'radius':
            dens = np.zeros_like(lon2d_c)
            for df in sub:
                lonlat = df[['lon', 'lat']].values.astype('double', order='C')
                dens = density_grid_rad(lon2d_c, lat2d_c, lonlat, dens,
                                        r_metres).base
        # else:
        #     return self._density
        return dens


class TrackSettings:
    def __init__(self, fname_path=None):
        self._fields = []
        if isinstance(fname_path, Path):
            with fname_path.open('r') as f:
                conf_list = [line for line in f.read().split('\n')
                             if not line.startswith('#') and len(line) > 0]
            for line in conf_list:
                if not line.startswith('#'):
                    k, v = line.split('=')
                    self._fields.append(k)
                    try:
                        self.__dict__.update({k: int(v)})
                    except ValueError:
                        try:
                            self.__dict__.update({k: float(v)})
                        except ValueError:
                            v = str(v).strip('"').strip("'")
                            self.__dict__.update({k: v})
                # try:
                #    exec(line, None, self.__dict__)
                # except SyntaxError:
                #    k, v = line.split('=')
                #    self.__dict__.update({k: str(v)})
                #    self._fields.append(k)
        self._fields = tuple(self._fields)

    def copy(self):
        new = self.__class__()
        new.__dict__ = self.__dict__.copy()
        return new

    @property
    def extent(self):
        extent_keys = ['lon1', 'lon2', 'lat1', 'lat2']
        extent = []
        for k in extent_keys:
            try:
                extent.append(getattr(self, k, None))
            except AttributeError:
                extent.append(None)
        return extent

    def __len__(self):
        return len(self._fields)

    def __repr__(self):
        return ('Settings used for '
                f'PMC tracking algorithm ({len(self)})')

    def __str__(self):
        summary = '\n'.join([f'{k} = {getattr(self, k, None)}'
                             for k in self._fields])
        return f'Settings used for PMC tracking algorithm:\n\n{summary}'
