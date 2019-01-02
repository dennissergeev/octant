Changelog
=========

.. default-role:: py:obj


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
