# -*- coding: utf-8 -*-
"""Input-output functions."""
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd

from .core import OctantTrack
from .decor import get_pbar
from .exceptions import LoadError
from .params import MUX_NAMES, SCALE_VO

ARCH_KEY = "trackrun"
ARCH_KEY_CAT = ARCH_KEY + "_categories"


class CSVLoader(ABC):
    """Abstract base class for reading CSV files with tracking output."""

    def __init__(self, dirname=Path.cwd(), files_wildcard="*"):
        """
        Instantiate base loader.

        Parameters
        ----------
        dirname: pathlib.Path, optional
            Path to the directory with tracking output.
        files_wildcard: str, optional
            Wildcard to find files in the directory.
        """
        if not dirname.is_dir():
            raise LoadError(f"No such directory: {dirname}")
        self.filelist = sorted(dirname.glob(files_wildcard))

    @property
    @abstractmethod
    def read_csv_kw(self):
        """Reserve property for read_csv() keywords."""
        return None

    @abstractmethod
    def __call__(self):
        """Reserve method for reading data."""
        pass

    @property
    def _pbar(self):
        """Get progress bar."""
        return get_pbar()


class PMCTRACKLoader(CSVLoader):
    """Loader of PMCTRACK output."""

    def __init__(self, dirname=Path.cwd(), files_wildcard="vortrack*0001.txt"):
        """
        Instantiate PMCTRACK loader with a different default wildcard.

        Parameters
        ----------
        dirname: pathlib.Path, optional
            Path to the directory with tracking output.
        files_wildcard: str, optional
            Wildcard to find files in the directory.
            By default, loads only "primary" vortices and skips merged.
            See PMCTRACK docs for more info.
        """
        super(PMCTRACKLoader, self).__init__(dirname=dirname, files_wildcard=files_wildcard)

    @property
    def read_csv_kw(self):
        """Keyword arguments for `pandas.read_csv()`."""
        return {
            "delimiter": r"\s+",
            "names": ["lon", "lat", "vo", "time", "area", "vortex_type"],
            "parse_dates": ["time"],
        }

    def __call__(self):
        """Read CSV files and collate them into a `TrackRun.data`-like dataframe."""
        _data = []
        for fname in self._pbar(self.filelist):
            _data.append(OctantTrack.from_df(pd.read_csv(fname, **self.read_csv_kw)))
        if len(_data) > 0:
            result = pd.concat(_data, keys=range(len(_data)), names=MUX_NAMES)
            if "vo" in result.columns:
                # Scale vorticity to (s-1)
                result["vo"] *= SCALE_VO
        del _data
        return result


class PMCTRACKLoaderWithSLP(PMCTRACKLoader):
    """Same as PMCTRACKLoader but with SLP column."""

    @property
    def read_csv_kw(self):
        """Keyword arguments for `pandas.read_csv()`."""
        return {
            "delimiter": r"\s+",
            "names": ["lon", "lat", "vo", "time", "area", "vortex_type", "slp"],
            "parse_dates": ["time"],
        }


class STARSLoader(CSVLoader):
    """Loader of STARS database."""

    @property
    def read_csv_kw(self):
        """Keyword arguments for `pandas.read_csv()`."""
        return {
            "delimiter": r"\s+",
            "parse_dates": {"time": [1, 2, 3, 4, 5]},
            "date_parser": lambda *x: datetime.strptime(" ".join(x), "%Y %m %d %H %M"),
            "dtype": {k: v for k, v in enumerate((int,) + 5 * (str,) + 4 * (float,))},
            "skiprows": 5,
        }

    def __call__(self):
        """Read CSV files and collate them into a `TrackRun.data`-like dataframe."""
        _data = []
        for fname in self._pbar(self.filelist):
            df = pd.read_csv(fname, **self.read_csv_kw)
            for _, track in df.groupby("N"):
                _data.append(OctantTrack.from_df(track))
        if len(_data) > 0:
            result = pd.concat(_data, keys=range(len(_data)), names=MUX_NAMES)
        del _data
        return result
