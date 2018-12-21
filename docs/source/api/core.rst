Core hierarchy
==============
The core API is built on top of `pandas` package,
and the main unit is :py:class:`OctantTrack`, which is a subclass of :py:class:`pandas.DataFrame`
with extra methods and properties relevant to analysing cyclone tracking output.

.. autoclass:: octant.core.OctantTrack

.. autoclass:: octant.core.TrackRun
    :show-inheritance: False
