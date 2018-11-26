# -*- coding: utf-8 -*-
"""Bells and whistles."""
# This module is temporarily disabled

# global DISABLE_TQDM
# TODO: global runtime switch
DISABLE_TQDM = True


def pbar(obj, **tqdm_kw):
    """Empty progress bar."""
    return obj


if not DISABLE_TQDM:
    try:
        # If tqdm is installed
        try:
            # Check if it's Jupyter Notebook
            ipy_str = str(type(get_ipython()))
            if 'zmqshell' in ipy_str.lower() and False:
                from tqdm import tqdm_notebook as tqdm
            else:
                from tqdm import tqdm
        except NameError:
            from tqdm import tqdm
        from functools import partial
        pbar = partial(tqdm, leave=False, disable=DISABLE_TQDM)  # noqa
    except ImportError:
        pass


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
        self.tr_id = id(trackrun)
        self.trackrun = trackrun
        self.n_tracks = len(self.trackrun)
        self.name = self.trackrun.__class__.__name__

    def str_repr(self, short=False):
        summary = [u'<octant.core.{}>'.format(type(self.trackrun).__name__)]
        summary.append(u'[{} tracks]'.format(len(self.trackrun)))
        if short:
            return ' '.join(summary)

        if len(self.trackrun) > 0:
            summary.append(u'\nData columns:')
            summary.append(u' | '.join(self.trackrun.data.columns))

        if self.trackrun.is_categorised:
            summary.append(u'\nCategories:')
            summary.append(u'         {:>8d} in total'.format(self.trackrun.size()))
            for cat_label in self.trackrun.cats.keys():
                if cat_label != 'unknown':
                    summary.append(u'of which {:>8d} are {}'.format(self.trackrun.size(cat_label),
                                                                    cat_label))

        if self.trackrun.sources:
            summary.append(u'\nSources:')
            summary.append(u'\n'.join(self.trackrun.sources))

        # if self.trackrun.conf is not None:
        #     summary.append(u'\nTracking settings:')
        return '\n'.join(summary)

    def _make_header(self):
        # TODO
        tlc_template = \
            '<th class="octant octant-word-cell">{self.name} ({self.n_tracks})</th>'
        top_left_cell = tlc_template.format(self=self)
        cells = ['<tr class="octant">', top_left_cell]
        cells.append(
            '<th class="octant octant-word-cell">{}</th>'.format('Data columns'))
        return '\n'.join(cells)

    def _make_content(self):
        cells = ['<tr class="octant">']
        # TODO
        return '\n'.join(cells)

    def html_repr(self):
        header = self._make_header()
        content = self._make_content()
        return self._template.format(id=self.tr_id,
                                     header=header,
                                     content=content)
