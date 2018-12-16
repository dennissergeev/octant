.. _api:

#############
API reference
#############

Here the reference documentation is provided for octant's public API.
If you are new to the package or just trying to get a feel for the
overall workflow, you are better off starting in the :ref:`overview`,
or :ref:`examples` sections of the documentation.

.. warning::

   octant is under active development. It is likely that some breaking changes to
   the codebase will occur in the future as octant is improved.

Core hierarchy
==============
The core API is built on top of `pandas` package,
and the main unit is :py:class:`OctantTrack`, which is a subclass of :py:class:`pandas.DataFrame`
with extra methods and properties relevant to analysing cyclone tracking output.


.. autoclass:: octant.core.TrackRun
    :members:
    :undoc-members:

    .. automethod:: octant.core.TrackRun.__init__

Auxiliary parts
===============
.. autoclass:: octant.parts.TrackSettings
    :members:
    :undoc-members:

Miscellanea
===========
Various helper functions

.. autofunction:: octant.misc.calc_all_dens

.. autofunction:: octant.misc.bin_count_tracks


Utilities
=========
Cythonised functions used by methods of :py:class:`octant.core.TrackRun`.

.. autofunction:: octant.utils.great_circle
