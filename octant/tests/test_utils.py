"""Test the utils submodule."""
# import numpy as np
import numpy.testing as npt

from octant.utils import great_circle


def test_great_circle():
    """Test great circle calculation."""
    dist = great_circle(10., 20., 30., 40.)
    true_dist = 1435334.9068947

    npt.assert_almost_equal(dist, true_dist)
