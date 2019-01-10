"""Test grid submodule."""
import numpy as np
import numpy.testing as npt

from octant import grid


def test_cell_centres():
    """Test cell_centres."""
    arr = np.array([-1, 0, 1, 2])
    des = np.array([-0.5, 0.5, 1.5])
    act = grid.cell_centres(arr)
    npt.assert_allclose(act, des)
    des = np.array([-0.8, 0.2, 1.2])
    act = grid.cell_centres(arr, bound_position=0.2)
    npt.assert_allclose(act, des)


def test_cell_bounds():
    """Test cell_bounds."""
    arr = np.array([26.0, 27.0, 28.0, 29.0, 30.0])
    des = np.array([25.5, 26.5, 27.5, 28.5, 29.5, 30.5])
    act = grid.cell_bounds(arr)
    npt.assert_allclose(act, des)
    des = np.array([25.0, 27.0, 28.0, 29.0, 30.0, 31.0])
    act = grid.cell_bounds(arr, bound_position=1)
    npt.assert_allclose(act, des)


def test__iris_guess_bounds():
    """Test _iris_guess_bounds."""
    arr = np.array([26.0, 27.0, 28.0, 29.0, 30.0])
    des = np.array([[25.5, 26.5], [26.5, 27.5], [27.5, 28.5], [28.5, 29.5], [29.5, 30.5]])
    act = grid._iris_guess_bounds(arr)
    npt.assert_allclose(act, des)


def test_grid_cell_areas():
    """Test area calculation."""
    des = np.array(
        [
            [1.236_230_658_6e10, 1.236_230_658_6e10, 1.236_230_658_6e10],
            [1.236_418_971_2e10, 1.236_418_971_2e10, 1.236_418_971_2e10],
        ]
    )
    lon = np.array([0, 1, 2])
    lat = np.array([-1, 0])
    act = grid.grid_cell_areas(lon, lat)
    npt.assert_allclose(act, des)
