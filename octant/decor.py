# -*- coding: utf-8 -*-
"""Bells and whistles."""
# This module is temporarily disabled
from . import RUNTIME


def get_pbar():
    """Get progress bar if the run-time option is enabled."""
    # noqa
    def _pbar(obj, **tqdm_kw):
        """Empty progress bar."""
        return obj

    if RUNTIME.enable_progress_bar:
        from functools import partial

        try:
            # if fastprogress is installed
            from fastprogress import progress_bar

            return partial(progress_bar)
        except ImportError:
            try:
                # If tqdm is installed
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

    _template = """
<style>
  table.octant {{
      white-space: pre;
      border: 1px solid;
      border-color: #9c9c9c;
      font-family: monaco, monospace;
  }}
  th.octant {{
      background: #333333;
      color: #e0e0e0;
      border-left: 1px solid;
      border-color: #9c9c9c;
      font-size: 1.05em;
      min-width: 50px;
      max-width: 125px;
  }}
</style>
<table class="octant" id="{id}">
    {header}
    {content}
</table>
        """

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

    def _make_header(self):
        top_cell = f'<th class="octant octant-word-cell" colspan="{self.ncol+1}">{self.name}</th>'
        cells = ['<tr class="octant">', top_cell, "</tr>"]
        return "\n".join(cells)

    def _make_content(self):
        cells = []
        if self.trackrun.is_categorised:
            cells.append('<tr class="octant">')
            cells.append(f'<td rowspan="{len(self.trackrun._cats)+1}">Categories</td>')
            cells.append('<tr class="octant">')
            cells.append(
                '<td class="octant octant-word-cell"></td>'
                f'<td colspan="{self.ncol-2}">{self.n_tracks}</td><td>in total</td>'
            )
            cells.append("</tr>")
            cells.append("</tr>")
            for cat_label in self.trackrun._cats.keys():
                if cat_label != "unknown":
                    cells.append('<tr class="octant">')
                    cells.append(
                        '<td class="octant octant-word-cell">of which</td>'
                        '<td class="octant octant-word-cell"'
                        f'colspan="{self.ncol-2}">{self.trackrun.size(cat_label)}</td>'
                        f'<td class="octant octant-word-cell">{cat_label}</td>'
                    )
                    cells.append("</tr>")
        else:
            # Total number of tracks
            cells.append('<tr class="octant">')
            cells.append('<td class="octant octant-word-cell">Number of tracks</td>')
            cells.append(
                f'<td class="octant octant-word-cell" colspan="{self.ncol}">{self.n_tracks}</td>'
            )
            cells.append("</tr>")

        # List of columns of the .data container
        cells.append('<tr class="octant">')
        cells.append('<td class="octant octant-word-cell">Data columns</td>')
        for col in self.data_cols:
            cells.append(f'<td class="octant octant-word-cell">{col}</td>')
        cells.append("</tr>")

        if self.trackrun.sources:
            # List source directories
            cells.append('<tr class="octant">')
            cells.append(
                f'<td class="octant octant-word-cell"\
                rowspan="{len(self.trackrun.sources)+1}">Sources</td>'
            )
            for src in self.trackrun.sources:
                cells.append('<tr class="octant">')
                cells.append(
                    f'<td class="octant octant-word-cell"\
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
            if self.trackrun._cats_inclusive:
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
