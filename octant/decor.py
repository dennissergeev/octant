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
