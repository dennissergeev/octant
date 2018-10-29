# -*- coding: utf-8 -*-
"""Auxiliary classes for octant package."""
from pathlib import Path


class TrackSettings:
    """
    Dictionary-like container of tracking settings.

    TrackSettings object is constructed by reading .conf file used by
    the tracking algorithm.

    Note: the .conf file can only have lines with
    key-value pairs, e.g.
    lon1=20
    or comment lines starting with #
    """

    def __init__(self, fname_path=None):
        """
        Initialise TrackSettings.

        Arguments
        ---------
        fname_path: pathlib.Path
            Path to .conf file with settings
            (usually is in the same folder as the tracking output)
        """
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
        """Create a copy of TrackSettings."""
        new = self.__class__()
        new.__dict__ = self.__dict__.copy()
        return new

    @property
    def extent(self):
        """List of lon1, lon2, lat1, lat2 showing the region used for tracking."""
        extent_keys = ['lon1', 'lon2', 'lat1', 'lat2']
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
        return ('Settings used for '
                f'PMC tracking algorithm ({len(self)})')

    def __str__(self):  # noqa
        summary = '\n'.join([f'{k} = {getattr(self, k, None)}'
                             for k in self._fields])
        return f'Settings used for PMC tracking algorithm:\n\n{summary}'

    def to_dict(self):
        """Convert TrackSettings to a dictionary."""
        return {k: self.__dict__[k] for k in self._fields}

    @classmethod
    def from_dict(cls, data):
        """
        Construct TrackSettings from a dictionary.

        Arguments
        ---------
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
