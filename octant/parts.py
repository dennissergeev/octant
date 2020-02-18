# -*- coding: utf-8 -*-
"""Parts of octant package."""
import numpy as np

import pandas as pd

from .decor import ReprTrackSettings
from .exceptions import LoadError
from .params import HOUR, M2KM
from .utils import great_circle, total_dist


__all__ = ("OctantTrack", "TrackSettings")


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
        # TODO: include planet radius
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
        if self.lifetime_h == 0:
            return np.nan
        else:
            return self.total_dist_km / self.lifetime_h

    @property
    def max_vort(self):
        """Maximum vorticity of the cyclone track."""
        return np.nanmax(self.vo.values)

    def within_rectangle(self, lon0, lon1, lat0, lat1, time_frac=1):
        """
        Check that OctantTrack is within a rectangle for a fraction of its lifetime.

        Parameters
        ----------
        lon0, lon1, lat0, lat1: float
            Boundaries of longitude-latitude rectangle (lon_min, lon_max, lat_min, lat_max)
        time_frac: float, optional
            Time fraction threshold.
            By default, set to maximum, i.e. track should be within the box entirely.

        Returns
        -------
        bool

        Examples
        --------
        Test that cyclone spends no more than a third of its life time outside the box

        >>> bbox = [-10, 25, 68, 78]
        >>> ot.within_rectangle(*bbox, time_frac=0.67)
        True

        See Also
        --------
        octant.misc.check_far_from_boundaries
        """
        _within = self[
            (self.lon >= lon0) & (self.lon <= lon1) & (self.lat >= lat0) & (self.lat <= lat1)
        ]
        if self.lifetime_h == 0:
            return _within.shape[0] == 1
        else:
            return _within.lifetime_h / self.lifetime_h >= time_frac

    def plot_track(self, ax=None, **kwargs):
        """
        Plot cyclone track using as plot function from plotting submodule.

        Filled circle shows the beginning, empty circle - the end of the track.

        Parameters
        ----------
        ax: matplotlib axes object, optional
            Axes in which to plot the track
            If not given, a new figure with cartopy geoaxes is created
        transform: matplotlib transform, optional
            Default: cartopy.crs.PlateCarree()
        kwargs: dict, optional
            Options to pass to matplotlib plot() function
        Returns
        -------
        ax: matplotlib axes object
            The same ax as the input ax (if given), or a newly created axes
        """
        from .plotting import plot

        return plot(self, ax=ax, **kwargs)


class TrackSettings:
    """
    Dictionary-like container of tracking settings.

    TrackSettings object is constructed by reading `.conf` file used by
    the tracking algorithm.

    Note: the `.conf` file can only have lines with key-value pairs, e.g.
    `lon1=20` or comment lines starting with #
    """

    def __init__(self, fname_path=None):
        """
        Initialise TrackSettings.

        Parameters
        ----------
        fname_path: pathlib.Path, optional
            Path to `.conf` file with settings
            (usually is in the same folder as the tracking output)
        """
        self._fields = []
        if fname_path is not None:
            try:
                with fname_path.open("r") as f:
                    conf_list = [
                        line
                        for line in f.read().split("\n")
                        if not line.startswith("#") and len(line) > 0
                    ]
                for line in conf_list:
                    if not line.startswith("#"):
                        k, v = line.split("=")
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
            except (FileNotFoundError, AttributeError):
                raise LoadError("Check that `fname_path` is a correct Path and formatted correctly")
        self._fields = tuple(self._fields)

    def copy(self):
        """Create a copy of TrackSettings."""
        new = self.__class__()
        new.__dict__ = self.__dict__.copy()
        return new

    @property
    def extent(self):
        """List of lon1, lon2, lat1, lat2 showing the region used for tracking."""
        extent_keys = ["lon1", "lon2", "lat1", "lat2"]
        extent = []
        for k in extent_keys:
            try:
                extent.append(getattr(self, k, None))
            except AttributeError:
                extent.append(None)
        return extent

    def __len__(self):  # noqa
        return len(self._fields)

    def __repr__(self):  # noqa
        rtr = ReprTrackSettings(self)
        return rtr.str_repr(short=True)

    def __str__(self):  # noqa
        rtr = ReprTrackSettings(self)
        return rtr.str_repr(short=False)

    def _repr_html_(self):
        rtr = ReprTrackSettings(self)
        return rtr.html_repr()

    def to_dict(self):
        """Convert TrackSettings to a dictionary."""
        return {k: self.__dict__[k] for k in self._fields}

    @classmethod
    def from_dict(cls, data):
        """
        Construct TrackSettings from a dictionary.

        Parameters
        ----------
        data: dict
            Dictionary with appropriate keys

        Returns
        -------
        octant.parts.TrackSettings

        """
        ts = cls()
        ts.__dict__.update(data)
        ts._fields = list(data.keys())
        return ts
