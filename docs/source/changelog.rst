Changelog
=========

.. default-role:: py:obj

v0.0.19
-------

:Release: unreleased
:Date: XXX

* Add `classify_by_percentile()` and `clear_categories()` methods
* Improve `density()` method with cell-based calculation, add weighting by grid cell area
* Add helper functions to deal with lon-lat grids to the `grid` submodule
* Add short-cut property `gb` to `OctantTrack` to group by track indices
* Optimise keyword arguments in `TrackRun.load_data` and `TrackRun.__init__`
* Minor bug fixes, more tests


v0.0.18
-------

:Release: v0.0.18
:Date: 1 January 2019

* New method `classify()` - replacing the deprecated `categorise()` in `TrackRun`
* Remove hard-coded categories (`_cats` attribute) - now created dynamically by `classify()`
* New example notebook showing the flexibility of `classify()`
* Part of `categorise()` dealing with land mask is moved to a new function in `misc` module
* Style fixes, integration with black using pre-commit hook


v0.0.17
-------

:Release: v0.0.17
:Date: 15 December 2018

* First release with documentation
