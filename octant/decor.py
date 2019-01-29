# -*- coding: utf-8 -*-
"""Bells and whistles."""
from . import RUNTIME


_template = """
<style>
  table.octant {{
      white-space: pre;
      border: 1px solid;
      border-color: #f9f9ef;
      font-family: monaco, monospace;
  }}
  th.octant {{
      background: #084469;
      color: #fefefe;
      border-left: 1px solid;
      border-color: #0a507a;
      font-size: 1.05em;
      min-width: 50px;
      max-width: 125px;
  }}
  .octant-la {{
      text-align: left !important;
      white-space: pre;
  }}
  .octant-ra {{
      text-align: right !important;
      white-space: pre;
  }}
</style>
<table class="octant" id="{id}">
    {header}
    {content}
</table>
        """


def get_pbar(use="fastprogress"):
    """
    Get progress bar if the run-time option is enabled and modules are installed.

    Parameters
    ----------
    use: str, optional
        What library to use: fastprogress (default) or tqdm.

    Returns
    -------
    _pbar: fastprogress.progress_bar or tqdm.tqdm or tqdm.tqdm_notebook
         or an empty wrapper
    """
    # noqa
    def _pbar(obj, **pbar_kw):
        """Empty progress bar."""
        return obj

    if RUNTIME.enable_progress_bar:
        from functools import partial

        try:
            if use == "fastprogress":
                # if fastprogress is installed
                from fastprogress import progress_bar

                return partial(progress_bar)
            elif use == "tqdm":
                # if tqdm is installed
                try:
                    # Check if it's Jupyter Notebook
                    ipy_str = str(type(get_ipython()))
                    if "zmqshell" in ipy_str.lower():
                        from tqdm import tqdm_notebook as tqdm
                    else:
                        from tqdm import tqdm
                except NameError:
                    from tqdm import tqdm
                return partial(tqdm)
        except ImportError:
            return _pbar
    else:
        return _pbar


class ReprTrackRun:
    """
    Produce representations of a TrackRun instance.

    This includes:
    * ``str_repr``: provides __repr__ or __str__ view as a string
    * ``html_repr``: represents TrackRun as an HTML object, available in Jupyter notebooks.
        Specifically, this is presented as an HTML table.
    """

    _template = _template

    def __init__(self, trackrun):
        """
        Initialise ReprTrackRun.

        Parameters
        ----------
        trackrun: octant.core.TrackRun
            TrackRun instance
        """
        self.tr_id = id(trackrun)
        self.trackrun = trackrun
        self.name = f"{self.trackrun.__module__}.{self.trackrun.__class__.__name__}"
        self.n_tracks = len(trackrun)
        self.data_cols = tuple(trackrun.data.columns)
        self.ncol = len(self.data_cols)
        self.longname = "Cyclone tracking results"

    def _make_header(self):
        top_cell = f'<th class="octant octant-la" colspan="{self.ncol+1}">{self.longname}</th>'
        cells = ['<tr class="octant">', top_cell, "</tr>"]
        return "\n".join(cells)

    def _make_content(self):
        cells = []
        if self.trackrun.is_categorised:
            cells.append('<tr class="octant octant-ra">')
            cells.append(f'<td rowspan="{len(self.trackrun._cats)+1}">Categories</td>')
            cells.append('<tr class="octant">')
            cells.append(
                '<td class="octant octant-la"></td>'
                f'<td colspan="{self.ncol-2}">{self.n_tracks}</td>'
                '<td class="octant octant-la">in total</td>'
            )
            cells.append("</tr>")
            cells.append("</tr>")
            for cat_label in self.trackrun._cats.keys():
                if cat_label != "unknown":
                    cells.append('<tr class="octant">')
                    cells.append(
                        '<td class="octant octant-ra">of which</td>'
                        '<td class="octant octant-ra"'
                        f'colspan="{self.ncol-2}">{self.trackrun.size(cat_label)}</td>'
                        f'<td class="octant octant-la">{cat_label}</td>'
                    )
                    cells.append("</tr>")
        else:
            # Total number of tracks
            cells.append('<tr class="octant">')
            cells.append('<td class="octant octant-la">Number of tracks</td>')
            cells.append(f'<td class="octant octant-la" colspan="{self.ncol}">{self.n_tracks}</td>')
            cells.append("</tr>")

        # List of columns of the .data container
        cells.append('<tr class="octant">')
        cells.append('<td class="octant octant-ra">Data columns</td>')
        cells.append(
            f'<td class="octant octant-la" colspan="{self.ncol}">{", ".join(self.data_cols)}</td>'
        )
        # for col in self.data_cols:
        #     cells.append(f'<td class="octant octant-la">{col}</td>')
        cells.append("</tr>")

        if self.trackrun.sources:
            # List source directories
            cells.append('<tr class="octant">')
            cells.append(
                f'<td class="octant octant-ra"\
                rowspan="{len(self.trackrun.sources)+1}">Sources</td>'
            )
            for src in self.trackrun.sources:
                cells.append('<tr class="octant">')
                cells.append(
                    f'<td class="octant octant-la"\
                             colspan="{self.ncol}">{src}</td>'
                )
                cells.append("</tr>")
            cells.append("</tr>")
        return "\n".join(cells)

    def html_repr(self):
        """HTML representation of TrackRun used in Jupyter Notebooks."""
        header = self._make_header()
        content = self._make_content()
        return self._template.format(id=self.tr_id, header=header, content=content)

    def str_repr(self, short=False):
        """Represent TrackRun as string."""
        summary = ["<octant.core.{}>".format(type(self.trackrun).__name__)]
        summary.append("[{} tracks]".format(len(self.trackrun)))
        if short:
            return " ".join(summary)

        if len(self.trackrun) > 0:
            summary.append("\nData columns:")
            summary.append(" | ".join(self.trackrun.data.columns))

        if self.trackrun.is_categorised:
            if self.trackrun._cat_inclusive:
                _pre = "of which "
            else:
                _pre = ""
            summary.append("\nCategories:")
            summary.append("         {:>8d} in total".format(self.trackrun.size()))
            for cat_label in self.trackrun._cats.keys():
                if cat_label != "unknown":
                    summary.append(
                        "{}{:>8d} are {}".format(_pre, self.trackrun.size(cat_label), cat_label)
                    )

        if self.trackrun.sources:
            summary.append("\nSources:")
            summary.append("\n".join(self.trackrun.sources))

        # if self.trackrun.conf is not None:
        #     summary.append(u'\nTracking settings:')
        return "\n".join(summary)


class ReprTrackSettings:
    """
    Representations of TrackSettings.

    This includes:
    * ``str_repr``: provides __repr__ or __str__ view as a string
    * ``html_repr``: represents TrackSettings as an HTML object, available in Jupyter notebooks.
        Specifically, this is presented as an HTML table.
    """

    _template = _template

    def __init__(self, ts):
        """
        Initialise ReprTrackSettings.

        Parameters
        ----------
        ts: octant.parts.TrackSettings
            TrackSettings instance
        """
        self.ts_id = id(ts)
        self.ts = ts
        self.name = f"{self.ts.__module__}.{self.ts.__class__.__name__}"
        self.n_set = len(ts)
        self.ncol = 2
        self.longname = f"Tracking algorithm settings ({self.n_set})"

    def _make_header(self):
        top_cell = f'<th class="octant" colspan="{self.ncol}">{self.longname}</th>'
        cells = ['<tr class="octant octant-la">', top_cell, "</tr>"]
        return "\n".join(cells)

    def _make_content(self):
        cells = []
        if self.n_set > 0:
            # List all settings
            for s in self.ts._fields:
                cells.append('<tr class="octant">')
                cells.append(
                    (
                        f'<td class="octant octant-ra">{s} =</td>'
                        f'<td class="octant octant-la">{getattr(self.ts, s, None)}</td>'
                    )
                )
                cells.append("</tr>")
        return "\n".join(cells)

    def html_repr(self):
        """HTML representation of TrackSettings used in Jupyter Notebooks."""
        header = self._make_header()
        content = self._make_content()
        return self._template.format(id=self.ts_id, header=header, content=content)

    def str_repr(self, short=False):
        """Represent TrackSettings as string."""
        summary = [self.longname]
        if self.n_set > 0 and not short:
            summary.append("")
            summary += [f"{k} = {getattr(self.ts, k, None)}" for k in self.ts._fields]

        return "\n".join(summary)
