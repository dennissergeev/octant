# -*- coding: utf-8 -*-
"""
Plotting functions for octant package
"""
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxesSubplot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt

DFLT_TRANSFORM = ccrs.PlateCarree()
DFLT_COLOR = 'C0'


def plot(df, ax=None, transform=DFLT_TRANSFORM, **kwargs):
    """
    Plot cyclone track using as a line plot (using matplotlib.pyplot.plot()).

    Closed circle shows the beginning, open circle - the end of the track.

    Arguments
    ---------
    df: DataFrame-like
        Instance of cyclone track to plot. Longitudes and latitudes are taken
        from 'lon' and 'lat' columns.
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
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=DFLT_TRANSFORM)
        extent = [df.lon.min()-5, df.lon.max()+5,
                  df.lat.min()-2, df.lat.max()+2]
        ax.set_extent(extent, crs=transform)
        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        ax.coastlines()
    if isinstance(ax, GeoAxesSubplot):
        mapkw = dict(transform=transform)
    else:
        mapkw = {}
    color = kwargs.pop('color', DFLT_COLOR)
    l, = ax.plot(df.lon, df.lat, color=color, **kwargs, **mapkw)
    color = l.get_color()  # hold color value if None is given
    ax.plot(df.lon.values[0], df.lat.values[0],
            marker='o', mfc=color, mec=color, **mapkw)
    ax.plot(df.lon.values[-1], df.lat.values[-1],
            marker='o', mfc='w', mec=color, **mapkw)
    return ax
