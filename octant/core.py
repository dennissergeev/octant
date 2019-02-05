# -*- coding: utf-8 -*-
"""Classes and functions for the analysis of PMCTRACK output."""
import operator
import warnings
from functools import partial
from pathlib import Path

import numpy as np

import pandas as pd

import xarray as xr

from .decor import ReprTrackRun, get_pbar
from .exceptions import (
    ArgumentError,
    ConcatenationError,
    DeprecatedWarning,
    GridError,
    LoadError,
    MissingConfWarning,
)
from .grid import cell_bounds, cell_centres, grid_cell_areas
from .misc import _exclude_by_first_day, _exclude_by_last_day
from .params import ARCH_KEY, ARCH_KEY_CAT, COLUMNS, HOUR, M2KM
from .parts import TrackSettings
from .utils import (
    distance_metric,
    great_circle,
    mask_tracks,
    point_density_cell,
    point_density_rad,
    total_dist,
    track_density_cell,
    track_density_rad,
)


class _OctantSeries(pd.Series):
    """`pandas.Series` subclass used in octant library."""

    @property
    def _constructor(self):
        return _OctantSeries


class OctantTrack(pd.DataFrame):
    """
    Instance of cyclone track.

    DataFrame with a bunch of extra methods and properties.
    """

    def __init__(self, *args, **kw):
        """Initialise octant.core.OctantTrack."""
        super(OctantTrack, self).__init__(*args, **kw)

    @property
    def _constructor(self):
        return OctantTrack  # replace with self.__class__?

    _constructor_sliced = _OctantSeries

    @property
    def gb(self):
        """Shortcut to group by track_idx index."""
        return self.groupby("track_idx")

    @classmethod
    def from_df(cls, df):
        """Create OctantTrack from pandas.DataFrame."""
        return cls.from_records(df.to_records(index=False))

    @classmethod
    def from_mux_df(cls, df):
        """Create OctantTrack from a multi-index pandas.DataFrame."""
        if df.shape[0] > 0:
            return cls.from_records(df.to_records(index=True), index=df.index.names)
        else:
            return cls(columns=df.columns, index=df.index)

    @property
    def coord_view(self):
        """Numpy view of track coordinates: longitude, latitude, time."""
        return (
            self.lon.values.view("double"),
            self.lat.values.view("double"),
            self.time.values.view("int64"),
        )

    @property
    def lonlat(self):
        """Values of longitude and latitude as 2D numpy array."""
        return self[["lon", "lat"]].values

    @property
    def lonlat_c(self):
        """Values of longitude and latitude as C-ordered 2D numpy array."""
        return self.lonlat.astype("double", order="C")

    @property
    def tridlonlat(self):
        """Values of track index, longitude, latitude as 2D numpy array."""
        return self.reset_index("track_idx")[["track_idx", "lon", "lat"]].values

    @property
    def tridlonlat_c(self):
        """Values of track index, longitude, latitude as C-order 2D array."""
        return self.tridlonlat.astype("double", order="C")

    @property
    def lifetime_h(self):
        """Track duration in hours."""
        if self.shape[0] > 0:
            return (self.time.values[-1] - self.time.values[0]) / HOUR
        else:
            return 0

    @property
    def gen_lys_dist_km(self):
        """Distance between genesis and lysis of the cyclone track in km."""
        if self.shape[0] > 0:
            return (
                great_circle(
                    self.lonlat[0, 0], self.lonlat[-1, 0], self.lonlat[0, 1], self.lonlat[-1, 1]
                )
                * M2KM
            )
        else:
            return 0

    @property
    def total_dist_km(self):
        """Total track distance in km."""
        return total_dist(self.lonlat_c) * M2KM

    @property
    def average_speed(self):
        """Average cyclone propagation speed in km per hour."""
        return self.total_dist_km / self.lifetime_h

    @property
    def max_vort(self):
        """Maximum vorticity of the cyclone track."""
        return np.nanmax(self.vo.values)

    @property
    def mean_vort(self):
        """Mean vorticity of the cyclone track."""
        return np.nanmean(self.vo.values)

    def within_rectangle(self, lon0, lon1, lat0, lat1, thresh=1):
        """
        Check that OctantTrack is within a rectangle for a fraction of its lifetime.

        Parameters
        ----------
        lon0, lon1, lat0, lat1: float
            Boundaries of longitude-latitude rectangle (lon_min, lon_max, lat_min, lat_max)
        thresh: float, optional
            Time threshold. By default, set to maximum, i.e. rack should be within the box entirely.

        Returns
        -------
        bool

        Examples
        --------
        Test that cyclone spends no more than a third of its life time outside the box

        >>> box = [-10, 25, 68, 78]
        >>> ot.within_rectangle(*bbox, thresh=0.67)
        True
        """
        time_within = self[
            (self.lon >= lon0) & (self.lon <= lon1) & (self.lat >= lat0) & (self.lat <= lat1)
        ].lifetime_h
        return time_within / self.lifetime_h >= thresh

    def plot_track(self, ax=None, **kwargs):
        """
        Plot cyclone track using as plot function from plotting submodule.

        Closed circle shows the beginning, open circle - the end of the track.

        Parameters
        ----------
        ax: matplotlib axes object, optional
            Axes in which to plot the track
            If not given, a new figure with cartopy geoaxes is created
        transform: matplotlib transform, optional
            Default: cartopy.crs.PlateCarree()
        kwargs: other keyword arguments
            Options to pass to matplotlib plot() function
        Returns
        -------
        ax: matplotlib axes object
            The same ax as the input ax (if given), or a newly created axes
        """
        from .plotting import plot

        return plot(self, ax=ax, **kwargs)


class TrackRun:
    """
    Results of tracking experiment.

    Attributes
    ----------
    data: octant.core.OctantTrack
        DataFrame-like container of tracking locations, times, and other data
    filelist: list
        List of source "vortrack" files; see PMCTRACK docs for more info
    conf: octant.parts.TrackSettings
        Configuration used for tracking
    is_categorised: bool
        Flag if categorisation has been applied to the TrackRun
    cats: None or pandas.DataFrame
        DataFrame with the same index as data and the number of columns equal to
        the number of categories; None if `is_categorised` is False
    """

    _mux_names = ["track_idx", "row_idx"]

    def __init__(self, dirname=None, columns=COLUMNS, **kwargs):
        """
        Initialise octant.core.TrackRun.

        Parameters
        ----------
        dirname: pathlib.Path, optional
            Path to the directory with tracking output
            If present, load the data during on init
        columns: sequence of str, optional
            List of column names. Should contain 'time' to parse datetimes.
        kwargs: other keyword arguments
            Parameters passed to load_data()
        """
        self.dirname = dirname
        self.conf = None
        mux = pd.MultiIndex.from_arrays([[], []], names=self._mux_names)
        self.columns = columns
        self.data = OctantTrack(index=mux, columns=self.columns)
        self.filelist = []
        self.sources = []
        self.cats = None
        # self.cats = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []],
        #                          names=self._mux_names), columns=[])
        self._cats = {"unknown": 0}
        self.is_categorised = False
        self._cat_inclusive = False
        # self._density = None
        if isinstance(self.dirname, Path):
            # Read all files and store in self.all
            # as a list of `pandas.DataFrame`s
            self.load_data(self.dirname, columns=self.columns, **kwargs)
        elif self.dirname is not None:
            raise LoadError("To load data, `dirname` should be Path-like object")

        if not self.data.empty:
            # Define time step
            for (_, ot) in self.gb:
                if ot.shape[0] > 1:
                    self.tstep_h = ot.time.diff().values[-1] / HOUR
                    break

    def __len__(self):
        """Get the number of cyclone tracks within TrackRun."""
        return self.data.index.get_level_values(0).to_series().nunique()

    def __repr__(self):  # noqa
        rtr = ReprTrackRun(self)
        return rtr.str_repr(short=True)

    def __str__(self):  # noqa
        rtr = ReprTrackRun(self)
        return rtr.str_repr(short=False)

    def _repr_html_(self):
        rtr = ReprTrackRun(self)
        return rtr.html_repr()

    def __add__(self, other):
        """Combine two TrackRun objects together."""
        new = self.__class__()
        new.extend(self)
        new.extend(other)
        return new

    def __getitem__(self, subset):  # noqa
        if (subset in [slice(None), None, "all"]) or self.size() == 0:
            return self.data
        else:
            if self._cat_inclusive:
                return self.data[self.data.cat >= self._cats[subset]]
            else:
                return self.data[self.data.cat == self._cats[subset]]

    @property
    def _pbar(self):
        """Get progress bar."""
        return get_pbar()

    @property
    def gb(self):
        """Group by track index."""
        return self.data.gb

    def size(self, subset=None):
        """Size of subset of tracks."""
        return self[subset].index.get_level_values(0).to_series().nunique()

    def rename_cats(self, **mapping):
        """
        Rename categories of the TrackRun.

        Parameters
        ----------
        mapping: dict
            How to rename categories, {old_key: new_key}
        """
        for old_key, new_key in mapping.items():
            try:
                self._cats[new_key] = self._cats.pop(old_key)
            except KeyError:
                pass

    def load_data(
        self, dirname, columns=COLUMNS, wcard="vortrack*0001.txt", conf_file=None, scale_vo=1e-3
    ):
        """
        Read tracking results from a directory into `TrackRun.data` attribute.

        Parameters
        ----------
        dirname: pathlib.Path
            Path to the directory with tracking output
        columns: sequence of str, optional
            List of column names. Should contain 'time' to parse datetimes.
        conf_file: pathlib.Path, optional
            Path to the configuration file. If omitted, an attempt is
            made to find a .conf file in the `dirname` directory
        wcard: str
            Wildcard for files to read in. By default, loads only "primary" vortices
            and skips merged. See PMCTRACK docs for more info.
        scale_vo: float, optional
            Scale vorticity values column to SI units (s-1)
            By default PMCTRACK writes out vorticity in (x10-3 s-1),
            so scale_vo default is 1e-3. To switch off scaling, set it to 1.
        """
        if not dirname.is_dir():
            raise LoadError(f"No such directory: {dirname}")
        # deprecated...
        # if primary_only:
        #     wcard = "vortrack*0001.txt"
        # else:
        #     wcard = "vortrack*.txt"
        self.filelist = sorted([*dirname.glob(wcard)])
        self.sources.append(str(dirname))

        # Load configuration
        if conf_file is None:
            try:
                conf_file = list(dirname.glob("*.conf"))[0]
                self.conf = TrackSettings(conf_file)
            except (IndexError, AttributeError):
                msg = (
                    "Track settings file (.conf) in the `dirname` directory"
                    "is missing or could not be read"
                )
                warnings.warn(msg, MissingConfWarning)

        # Load the tracks
        self.columns = columns
        load_kw = {"delimiter": r"\s+", "names": self.columns, "parse_dates": ["time"]}  # noqa
        _data = []
        for fname in self._pbar(self.filelist):
            _data.append(OctantTrack.from_df(pd.read_csv(fname, **load_kw)))
        if len(_data) > 0:
            self.data = pd.concat(_data, keys=range(len(_data)), names=self._mux_names)
            self.data["cat"] = 0
            # Scale vorticity to (s-1)
            self.data["vo"] *= scale_vo
        del _data

    @classmethod
    def from_archive(cls, filename):
        """
        Construct TrackRun object from HDF5 file.

        Parameters
        ----------
        filename: str
            File path to HDF5 file

        Returns
        -------
        octant.core.TrackRun

        """
        with pd.HDFStore(filename, mode="r") as store:
            df = store[ARCH_KEY]
            metadata = store.get_storer(ARCH_KEY).attrs.metadata
            _is_cat = metadata["is_categorised"]
            if _is_cat:
                df_cat = store.get(ARCH_KEY_CAT)
        tr = cls()
        if df.shape[0] > 0:
            tr.data = OctantTrack.from_mux_df(df.set_index(cls._mux_names))
        else:
            tr.data = OctantTrack.from_mux_df(df)
        metadata["conf"] = TrackSettings.from_dict(metadata["conf"])
        tr.__dict__.update(metadata)
        if _is_cat:
            tr.cats = df_cat.set_index(cls._mux_names)
        return tr

    def to_archive(self, filename):
        """
        Save TrackRun and its metadata to HDF5 file.

        Parameters
        ----------
        filename: str
            File path to HDF5 file
        """
        with pd.HDFStore(filename, mode="w") as store:
            if self.size() > 0:
                df = pd.DataFrame.from_records(self.data.to_records(index=True))
            else:
                df = pd.DataFrame(columns=self.columns, index=self.data.index)
            store.put(ARCH_KEY, df)
            metadata = {
                k: v
                for k, v in self.__dict__.items()
                if k not in ["data", "filelist", "conf", "cats"]
            }
            metadata["conf"] = getattr(self.conf, "to_dict", lambda: {})()
            store.get_storer(ARCH_KEY).attrs.metadata = metadata
            # Store DataFrame with categorisation data
            if self.is_categorised:
                df_cat = pd.DataFrame.from_records(self.cats.to_records(index=True))
                store.put(ARCH_KEY_CAT, df_cat)

    def extend(self, other, adapt_conf=True):
        """
        Extend the TrackRun by appending elements from another TrackRun.

        Parameters
        ---------
        other: octant.core.TrackRun
            Another TrackRun
        adapt_conf: bool
            Merge TrackSettings (.conf attribute) of each of the TrackRuns
            This is done by retaining matching values and setting other to None
        """
        # Check if category metadata match
        if (self.size() > 0) and (other.size() > 0):
            for attr in ["_cats", "_cat_inclusive", "is_categorised"]:
                a, b = getattr(self, attr), getattr(other, attr)
                if a != b:
                    raise ConcatenationError(
                        f"Categorisation metadata is different for '{attr}': {a} != {b}"
                    )
        elif other.size() > 0:
            for attr in ["_cats", "_cat_inclusive", "is_categorised"]:
                setattr(self, attr, getattr(other, attr))
        if getattr(self, "tstep_h", None) is None:
            self.tstep_h = getattr(other, "tstep_h", None)
        else:
            if getattr(other, "tstep_h", None) is not None:
                if self.tstep_h != other.tstep_h:
                    raise ConcatenationError(
                        "Extending by a TrackRun with different timestep is not allowed"
                    )
        if adapt_conf and other.conf is not None:
            if self.conf is None:
                self.conf = other.conf.copy()
            else:
                for field in self.conf._fields:
                    if getattr(self.conf, field) != getattr(other.conf, field):
                        setattr(self.conf, field, None)
        self.sources.extend(other.sources)

        new_data = pd.concat([self.data, other.data], sort=False)
        new_track_idx = new_data.index.get_level_values(0).to_series()
        new_track_idx = new_track_idx.ne(new_track_idx.shift()).cumsum() - 1

        mux = pd.MultiIndex.from_arrays(
            [new_track_idx, new_data.index.get_level_values(1)], names=new_data.index.names
        )
        self.data = new_data.set_index(mux)

    def time_slice(self, start=None, end=None):
        """
        Subset TrackRun by time using pandas boolean indexing.

        Parameters
        ----------
        start: str or datetime.datetime, optional
            Start of the slice (inclusive)
        stop: str or datetime.datetime, optional
            End of the slice (inclusive)

        Returns
        -------
        octant.core.TrackRun

        Examples
        --------
        >>> from octant.core import TrackRun
        >>> tr = TrackRun(path_to_directory_with_tracks)
        >>> sub_tr = tr.time_slice('2018-09-04', '2018-11-25')
        """
        if (start is None) and (end is None):
            return self
        else:
            crit = True
            if start is not None:
                crit &= self.data.time >= start
            if end is not None:
                crit &= self.data.time <= end
            # Create a copy of this TrackRun
            result = self.__class__()
            result.extend(self)
            # Replace data with TrackRun.data sliced by start or end
            result.data = result.data[crit]
            # Clear up sources to avoid confusion
            result.sources = []
            result.dirname = None
            result.filelist = []
            try:
                result.conf.dt_start = None
                result.conf.dt_end = None
            except AttributeError:
                pass
            return result

    def categorise(
        self,
        filt_by_time=True,
        filt_by_dist=True,
        filt_by_vort=False,
        filt_by_domain_bounds=True,
        filt_by_land=True,
        filt_by_percentile=True,
        strong_percentile=95,
        time_thresh0=6,
        time_thresh1=9,
        dist_thresh=300.0,
        type_thresh=0.2,
        lsm=None,
        coast_rad=50.0,
        vort_thresh0=3e-4,
        vort_thresh1=4.5e-4,
    ):
        """
        Classify the loaded tracks.

        .. deprecated:: 0.0.18

        Criteria:
         - lifetime
         - proximity to land or domain boundaries
         - distance
         - type
         - vorticity maximum

        Parameters
        ----------
        filt_by_time: bool, optional
            Filter by the time threshold
        time_thresh0: int, optional
            Time threshold (hours) for basic filtering
        time_thresh1: int, optional
            Time threshold (hours) for strong filtering
        filt_by_dist: bool, optional
            Filter by the distance threshold (dist. between genesis and lysis)
        dist_thresh: float, optional
            Distance in km
            Used for classifying vortices by distance between genesis and lysis
        filt_by_land: bool, optional
            Filter by the proximity to coast given by the land mask (`lsm`)
        filt_by_domain_bounds: bool, optional
            Filter by the proximity to domain boundarie, which are taken from
            the `self.conf` instance if present: lon1, lon2, lat2, lat2
        lsm: xarray.DataArray, optional
            Two-dimensional land-sea mask
            If present, tracks that spend > 0.5 of their lifetime
            within `coast_rad` radius from the coastline are discarded
        coast_rad: float, optional
            Radius in km
            Used for discarding vortices stuck near the coastline
        type_thresh: float, optional
            Ratio of time steps when `vortex_type` is not equal to 0 (PMC) to
            the total lifetime of the vortex. Should be within 0-1 range.
            `type_thresh=1./7.` means that if the number of time steps when the
            vortex is considered a synoptic-scale low or a cold front is more
            than one-seventh of the whole lifetime of the PMC, then the PMC is
            excluded as a synoptic-scale disturbance [Watanabe et al., 2016].
        filt_by_vort: bool, optional
            Filter by vorticity maximum (strong criteria)
            [Watanabe et al., 2016, p.2509]
        vort_thresh0: float, optional
            Vorticity threshold for strong filtering (s-1)
        vort_thresh1: float, optional
            Higher vorticity threshold for strong filtering (s-1)
        filt_by_percentile: bool, optional
            Filter strongest cyclones by the percentile of the maximum vort.
        strong_percentile: float, optional
            Percentile to define strong category of cyclones
            E.g. 95 means the top 5% strongest cyclones.
        """
        warnings.warn("Use the new classify() method", DeprecatedWarning)
        self._cats.update({"basic": 1, "moderate": 2, "strong": 3})
        self._cat_inclusive = True
        self.data.cat = 0  # Reset categories
        if filt_by_percentile and filt_by_vort:
            raise ArgumentError(("Either filt_by_percentile or filt_by_vort" "should be on"))
        # Save filtering params just in case
        self._cat_params = {k: v for k, v in locals().items() if k != "self"}
        # 0. Prepare mask for spatial filtering
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
                if getattr(self.conf, "lon1", None):
                    inner_idx &= lon2d >= self.conf.lon1
                if getattr(self.conf, "lon2", None):
                    inner_idx &= lon2d <= self.conf.lon2
                if getattr(self.conf, "lat1", None):
                    inner_idx &= lat2d >= self.conf.lat1
                if getattr(self.conf, "lat2", None):
                    inner_idx &= lat2d <= self.conf.lat2
                boundary_mask[~inner_idx] = 1.0
            self.themask = ((boundary_mask == 1.0) | (l_mask == 1.0)) * 1.0
            themask_c = self.themask.astype("double", order="C")
            lon2d_c = lon2d.astype("double", order="C")
            lat2d_c = lat2d.astype("double", order="C")

        for i, ot in self._pbar(self.gb):  # , desc="tracks"):
            basic_flag = True
            moderate_flag = True
            # 1. Minimal filter
            # 1.1. Filter by duration threshold
            if filt_by_time and (ot.lifetime_h < time_thresh0):
                basic_flag = False
            # 1.2. Filter by land mask and domain boundaries
            if (
                basic_flag
                and filt_by_mask
                and mask_tracks(themask_c, lon2d_c, lat2d_c, ot.lonlat_c, coast_rad * 1e3) > 0.5
            ):
                basic_flag = False

            if basic_flag:
                self.data.loc[i, "cat"] = self._cats["basic"]
                # 2. moderate filter
                # 2.1. Filter by flag assigned by the tracking algorithm
                if (ot.vortex_type != 0).sum() / ot.shape[0] > type_thresh:
                    moderate_flag = False
                # 2.2. Filter by distance between genesis and lysis
                if filt_by_dist and (ot.gen_lys_dist_km < dist_thresh):
                    moderate_flag = False

                if moderate_flag:
                    self.data.loc[i, "cat"] = self._cats["moderate"]
                    # 3. Strong filter
                    # 3.1. Filter by vorticity [Watanabe et al., 2016, p.2509]
                    if filt_by_vort:
                        if (ot.vo > vort_thresh1).any() or (
                            ((ot.vo > vort_thresh0).sum() > 1) and (ot.lifetime_h > time_thresh1)
                        ):
                            self.data.loc[i, "cat"] = self._cats["strong"]
            else:
                self.data.loc[i, "cat"] = self._cats["unknown"]
        if filt_by_percentile:
            # 3.2 Filter by percentile-defined vorticity threshold
            vo_per_track = self["moderate"].gb.apply(lambda x: x.max_vort)
            if len(vo_per_track) > 0:
                vo_thresh = np.percentile(vo_per_track, strong_percentile)
                strong = vo_per_track[vo_per_track > vo_thresh]
                self.data.loc[strong.index, "cat"] = self._cats["strong"]

        self.is_categorised = True

    def classify(self, conditions, inclusive=True, clear=True):
        """
        Classify the loaded tracks.

        This is a more abstract and flexible method, replacing the hard-coded categorise() method.

        Parameters
        ----------
        conditions: list
            List of tuples. Each tuple is a (label, list) pair containing the category label and
            a list of functions each of which has OctantTrack as its only argument.
            The method assigns numbers to the labels in the same order
            that they are given, starting from number 1 (see examples).
        inclusive: bool, optional
            If true, a higher category is a subset of lower category;
            otherwise categories are independent.
        clear: bool, optional
            If true, existing TrackRun categories are deleted.

        Examples
        --------
        Simple check using OctantTrack properties

        >>> from octant.core import TrackRun
        >>> tr = TrackRun(path_to_directory_with_tracks)
        >>> def myfun(x):
                bool_flag = (x.vortex_type != 0).sum() / x.shape[0] < 0.2
                return bool_flag
        >>> conds = [
            ('category_a', [lambda ot: ot.lifetime_h >= 6]),  # only 1 function checking lifetime
            ('category_b', [myfun,
                            lambda ot: ot.gen_lys_dist_km > 300.0])  # 2 functions
        ]
        >>> tr.classify(conds)
        >>> tr.size('category_a'), tr.size('category_b')
        31, 10

        For more examples, see example notebooks.

        See Also
        --------
        octant.misc.check_by_mask
        """
        if clear:
            self.clear_categories()
        start = max(self._cats.values()) + 1
        cat_dict = {label: num for num, (label, _) in enumerate(conditions, start)}
        self._cat_inclusive = inclusive
        for i, ot in self._pbar(self.gb):
            prev_flag = True
            for label, funcs in conditions:
                if inclusive:
                    _flag = prev_flag
                else:
                    _flag = True

                for func in funcs:
                    _flag &= func(ot)
                if _flag:
                    self.data.loc[i, "cat"] = cat_dict[label]
                if inclusive:
                    prev_flag = _flag
        self._cats.update(cat_dict)
        self.is_categorised = True

    def categorise_by_percentile(self, subset="unknown", perc=95, by="max_vort", oper="ge"):
        """
        Categorise by percentile.

        Parameters
        ----------
        subset: str, optional
            Subset of Trackrun to apply percentile to.
        perc: float, optional
            Percentile to define a category of cyclones.
            E.g. (perc=95, oper='ge') is the top 5% cyclones.
        by: str, optional
            Property of OctantTrack to apply percentile to.
        oper: str, optional
            Math operator to select track above or below the percentile threshold.
            Can be one of (lt|le|gt|ge).

        Examples
        --------
        >>> from octant.core import TrackRun
        >>> tr = TrackRun(path_to_directory_with_tracks)
        >>> tr._cats
        {'unknown': 0}
        >>> tr.categorise_by_percentile(perc=90, oper="gt")
        >>> tr._cats
        {'unknown': 0, 'max_vort__gt__90pc': 1}
        >>> tr.size("unknown")
        71
        >>> tr.size("max_vort__gt__90pc")
        7


        See Also
        --------
        octant.core.TrackRun.classify
        """
        allowed_ops = ["lt", "le", "gt", "ge"]
        if oper not in allowed_ops:
            raise ArgumentError(f"oper={oper} should be one of {allowed_ops}")
        op = getattr(operator, oper)
        if subset == "unknown":
            label = ""
        else:
            label = f"{subset}__with__"
        label += f"{by}__{oper}__{perc}pc"
        assert label not in self._cats, f"New label={label} clashes with existing one"
        v_per_track = self[subset].gb.apply(lambda x: getattr(x, by))
        if len(v_per_track) > 0:
            cat_id = max(self._cats.values()) + 1
            thresh = np.percentile(v_per_track, perc)
            above_thresh = v_per_track[op(v_per_track, thresh)]
            self.data.loc[above_thresh.index, "cat"] = cat_id
            self._cats.update({label: cat_id})
            self.is_categorised = True

    def clear_categories(self, subset=None, inclusive=None):
        """
        Clear TrackRun of its categories.

        If categories are inclusive, it destroys child categories

        Parameters
        ----------
        subset: str, optional
            If None (default), all categories are removed.
        inclusive: bool or None, optional
            If supplied, is used instead of _cat_inclusive attribute.

        Examples
        --------
        Inclusive categories

        >>> tr = TrackRun(path_to_directory_with_tracks)
        >>> tr._cats
        {'unknown': 0, 'pmc': 1, 'max_vort__ge__90pc': 2}
        >>> tr._cat_inclusive
        True
        >>> tr.clear_categories(subset='pmc')
        >>> tr._cats
        {'unknown': 0}

        Non-inclusive

        >>> tr = TrackRun(path_to_directory_with_tracks)
        >>> tr._cats
        {'unknown': 0, 'pmc': 1, 'max_vort__ge__90pc': 2}
        >>> tr.clear_categories(subset='pmc', inclusive=False)
        >>> tr._cats
        {'unknown': 0, 'max_vort__ge__90pc': 2}
        """
        if inclusive is not None:
            inc = inclusive
        else:
            inc = self._cat_inclusive
        if subset is None:
            # clear all categories
            self.data.cat = 0
            self._cats = {"unknown": 0}
        else:
            cat_id = self._cats[subset]
            # Do not use self[subset].cat = 0 ! - SettingWithCopyWarning
            if inc:
                self.data.loc[self.data.cat >= cat_id, "cat"] = cat_id - 1
                self._cats = {k: v for k, v in self._cats.items() if v < cat_id}
            else:
                self.data.loc[self.data.cat == cat_id, "cat"] = 0
                self._cats.pop(subset)
        if self._cats == {"unknown": 0}:
            self.is_categorised = False
            self._cat_inclusive = False

    def match_tracks(
        self,
        others,
        subset=None,
        method="simple",
        interpolate_to="other",
        thresh_dist=250.0,
        time_frac_thresh=0.5,
        return_dist_matrix=False,
        beta=100.0,
    ):
        """
        Match tracked vortices to a list of vortices from another data source.

        Parameters
        ----------
        others: list or octant.core.TrackRun
            List of dataframes or a TrackRun instance
        subset: str, optional
            Subset (category) of TrackRun to match.
            If not given, the matching is done for all categories.
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
        return_dist_matrix: bool, optional
            Used when method='bs2000'. If True, the method returns a tuple
            of matching pairs and distance matrix used to calculate them
        beta: float, optional
            Parameter used in 'bs2000' method
            E.g. beta=100 corresponds to 10 m/s average steering wind
        Returns
        -------
        match_pairs: list
            Index pairs of `other` vortices matching a vortex in `TrackRun`
            in a form (<index of `TrackRun` subset>, <index of `other`>)
        dist_matrix: numpy.ndarray
            2D array, returned if return_dist_matrix=True
        """
        # Recursive call for each of the available categies
        if subset is None:
            result = {}
            for subset_key in self._cats.keys():
                result[subset_key] = self.match_tracks(
                    others,
                    subset=subset_key,
                    method=method,
                    interpolate_to=interpolate_to,
                    thresh_dist=thresh_dist,
                    time_frac_thresh=time_frac_thresh,
                    return_dist_matrix=return_dist_matrix,
                    beta=beta,
                )
            return result

        # Select subset
        sub_gb = self[subset].gb
        if len(sub_gb) == 0 or len(others) == 0:
            return []
        if isinstance(others, list):
            # match against a list of DataFrames of tracks
            other_gb = pd.concat(
                [OctantTrack.from_df(df) for df in others],
                keys=range(len(others)),
                names=self._mux_names,
            ).gb
        elif isinstance(others, self.__class__):
            # match against another TrackRun
            other_gb = others[subset].gb
        else:
            raise ArgumentError('Argument "others" ' f"has a wrong type: {type(others)}")
        match_pairs = []
        if method == "intersection":
            for idx, ot in self._pbar(sub_gb):  # , desc="self tracks"):
                for other_idx, other_ot in self._pbar(other_gb, leave=False):
                    times = other_ot.time.values
                    time_match_thresh = time_frac_thresh * (times[-1] - times[0]) / HOUR

                    intersect = pd.merge(other_ot, ot, how="inner", left_on="time", right_on="time")
                    n_match_times = intersect.shape[0]
                    if n_match_times > 0:
                        _tstep_h = intersect.time.diff().values[-1] / HOUR
                        dist = intersect[["lon_x", "lon_y", "lat_x", "lat_y"]].apply(
                            lambda x: great_circle(*x.values), axis=1
                        )
                        prox_time = (dist < (thresh_dist * 1e3)).sum() * _tstep_h
                        if (
                            n_match_times * _tstep_h > time_match_thresh
                        ) and prox_time > time_match_thresh:
                            match_pairs.append((idx, other_idx))
                            break

        elif method == "simple":
            # TODO: explain
            ll = ["lon", "lat"]
            match_pairs = []
            for other_idx, other_ct in self._pbar(other_gb):  # , desc="other tracks"):
                candidates = []
                for idx, ct in self._pbar(sub_gb, leave=False):  # , desc="self tracks"):
                    if interpolate_to == "other":
                        df1, df2 = ct.copy(), other_ct
                    elif interpolate_to == "self":
                        df1, df2 = other_ct, ct.copy()
                    l_start = max(df1.time.values[0], df2.time.values[0])
                    e_end = min(df1.time.values[-1], df2.time.values[-1])
                    if (e_end - l_start) / HOUR > 0:
                        # df1 = df1.set_index('time')[ll]
                        # ts = pd.Series(index=df2.time)
                        # new_df1 = (pd.concat([df1, ts]).sort_index()
                        #            .interpolate(method='values')
                        #            .loc[ts.index])[ll]
                        tmp_df2 = pd.DataFrame(
                            data={"lon": np.nan, "lat": np.nan, "time": df2.time}, index=df2.index
                        )
                        new_df1 = (
                            pd.concat([df1[[*ll, "time"]], tmp_df2], ignore_index=True, keys="time")
                            .set_index("time")
                            .sort_index()
                            .interpolate(method="values")
                            .loc[tmp_df2.time]
                        )[ll]
                        new_df1 = new_df1[~new_df1.lon.isnull()]

                        # thr = (time_frac_thresh * 0.5
                        #       * (df2.time.values[-1] - df2.time.values[0]
                        #          + df1.time.values[-1] - df2.time.values[0]))
                        thr = time_frac_thresh * df2.shape[0]
                        dist_diff = np.full(new_df1.shape[0], 9e20)
                        for i, ((x1, y1), (x2, y2)) in enumerate(
                            zip(new_df1[ll].values, df2[ll].values)
                        ):
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

        elif method == "bs2000":
            # sub_list = [i[0] for i in list(sub_gb)]
            sub_indices = list(sub_gb.indices.keys())
            other_indices = list(other_gb.indices.keys())
            dist_matrix = np.full((len(sub_gb), len(other_gb)), 9e20)
            for i, (_, ct) in enumerate(self._pbar(sub_gb, leave=False)):  # , desc="self tracks"):
                x1, y1, t1 = ct.coord_view
                for j, (_, other_ct) in enumerate(self._pbar(other_gb, leave=False)):
                    x2, y2, t2 = other_ct.coord_view
                    dist_matrix[i, j] = distance_metric(x1, y1, t1, x2, y2, t2, beta=float(beta))
            for i, idx1 in enumerate(np.nanargmin(dist_matrix, axis=0)):
                for j, idx2 in enumerate(np.nanargmin(dist_matrix, axis=1)):
                    if i == idx2 and j == idx1:
                        match_pairs.append((sub_indices[idx1], other_indices[idx2]))
            if return_dist_matrix:
                return match_pairs, dist_matrix
        else:
            raise ArgumentError(f"Unknown method: {method}")

        return match_pairs

    def density(
        self,
        lon1d,
        lat1d,
        by="point",
        subset=None,
        method="cell",
        r=222.0,
        exclude_first={"m": 10, "d": 1},
        exclude_last={"m": 4, "d": 30},
        grid_centres=True,
        weight_by_area=True,
    ):
        """
        Calculate different types of cyclone density for a given lon-lat grid.

        - `point`: all points of all tracks
        - `track`: each track only once for a given cell or circle
        - `genesis`: starting positions (excluding start date of tracking)
        - `lysis`: ending positions (excluding final date of tracking)

        Parameters
        ----------
        lon1d: numpy.ndarray
            Longitude points array of shape (M,)
        lat1d: numpy.ndarray
            Latitude points array of shape (N,)
        by: str, optional
            Type of cyclone density (point|track|genesis|lysis)
        subset: str, optional
            Subset (category) of TrackRun to calculate density from.
            If not given, the calculation is done for all categories.
        method: str, optional
            Method to calculate density (radius|cell)
        r: float, optional
            Radius in km
            Used when method='radius'
        exclude_first: dict, optional
            Exclude start date (month, day)
        exclude_last: dict, optional
            Exclude end date (month, day)
        grid_centres: bool, optional
            If true, the function assumes that lon (M,) and lat (N,) arrays are grid centres
            and calculates boundaries, arrays of shape (M+1,) and (N+1,) so that the density
            values refer to centre points given.
            If false, the density is calculated between grid points.
        weight_by_area: bool, optional
            Weight result by area of grid cells.
        Returns
        -------
        dens: xarray.DataArray
            Array of track density of shape (M, N) with useful metadata in attrs
        """
        # Recursive call for each of the available categies
        if subset is None:
            result = {}
            for subset_key in self._cats.keys():
                result[subset_key] = self.density(
                    lon1d,
                    lat1d,
                    by=by,
                    subset=subset_key,
                    method=method,
                    r=r,
                    exclude_first=exclude_first,
                    exclude_last=exclude_last,
                    grid_centres=grid_centres,
                    weight_by_area=weight_by_area,
                )
            return result

        # Redefine grid if necessary
        if grid_centres:
            # Input arrays are centres of grid cells, so cell boundaries need to be calculated
            lon = cell_bounds(lon1d)
            lat = cell_bounds(lat1d)
            # Prepare coordinates for output
            xlon = xr.IndexVariable(dims="longitude", data=lon1d, attrs={"units": "degrees_east"})
            xlat = xr.IndexVariable(dims="latitude", data=lat1d, attrs={"units": "degrees_north"})
        else:
            # Input arrays are boundaries of grid cells, so cell centres need to be
            # calculated for the output
            lon, lat = lon1d, lat1d
            # Prepare coordinates for output
            xlon = xr.IndexVariable(
                dims="longitude", data=cell_centres(lon1d), attrs={"units": "degrees_east"}
            )
            xlat = xr.IndexVariable(
                dims="latitude", data=cell_centres(lat1d), attrs={"units": "degrees_north"}
            )

        # Create 2D mesh
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Select subset
        sub_df = self[subset]
        # Prepare coordinates for cython
        lon2d_c = lon2d.astype("double", order="C")
        lat2d_c = lat2d.astype("double", order="C")

        # Select method
        if method == "radius":
            # Convert radius to metres
            r_metres = r * 1e3
            units = f"per {round(np.pi * r**2)} km2"
            if by == "track":
                cy_func = partial(track_density_rad, rad=r_metres)
            else:
                cy_func = partial(point_density_rad, rad=r_metres)
        elif method == "cell":
            # TODO: make this check more flexible
            if (np.diff(lon2d[0, :]) < 0).any() or (np.diff(lat2d[:, 0]) < 0).any():
                raise GridError("Grid values must be in an ascending order")
            units = "1"
            if by == "track":
                cy_func = track_density_cell
            else:
                cy_func = point_density_cell

        # Convert dataframe columns to C-ordered arrays
        if by == "point":
            sub_data = sub_df.lonlat_c
        elif by == "track":
            sub_data = sub_df.tridlonlat_c
        elif by == "genesis":
            sub_data = (
                sub_df.gb.filter(_exclude_by_first_day, **exclude_first).xs(0, level="row_idx")
            ).lonlat_c
        elif by == "lysis":
            sub_data = (
                self[subset].gb.tail(1).gb.filter(_exclude_by_last_day, **exclude_last)
            ).lonlat_c
        else:
            raise ArgumentError("`by` should be one of point|track|genesis|lysis")

        data = cy_func(lon2d_c, lat2d_c, sub_data).base

        if weight_by_area:
            # calculate area in metres
            area = grid_cell_areas(xlon.values, xlat.values)
            data /= area
            data *= 1e6  # convert to km^{-2}
            units = "km-2"

        dens = xr.DataArray(
            data,
            name=f"{by}_density",
            attrs={"units": units, "subset": subset, "method": method},
            dims=("latitude", "longitude"),
            coords={"longitude": xlon, "latitude": xlat},
        )
        return dens
