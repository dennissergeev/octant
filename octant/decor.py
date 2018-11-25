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


def trackrun_repr(trackrun, short=False):
    summary = [u'<octant.core.{}>'.format(type(trackrun).__name__)]
    summary.append(u'[{} tracks]'.format(len(trackrun)))
    if short:
        return ' '.join(summary)

    if len(trackrun) > 0:
        summary.append(u'\nData columns:')
        summary.append(u' | '.join(trackrun.columns))

    if trackrun.is_categorised:
        summary.append(u'\nCategories:')
        summary.append(u'         {:>8d} in total'.format(trackrun.size()))
        for cat_label in trackrun.cats.keys():
            if cat_label != 'unknown':
                summary.append(u'of which {:>8d} are {}'.format(trackrun.size(cat_label),
                                                                cat_label))

    if trackrun.sources:
        summary.append(u'\nSources:')
        summary.append(u'\n'.join(trackrun.sources))

    # if trackrun.conf is not None:
    #     summary.append(u'\nTracking settings:')
    return '\n'.join(summary)
