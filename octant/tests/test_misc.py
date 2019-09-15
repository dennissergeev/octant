"""Test the misc submodule."""
from pathlib import Path

from octant import core, misc

import pytest


TEST_DATA = Path(__file__).parent / "test_data"
TEST_DIR = TEST_DATA / "era5_run000"


@pytest.fixture(scope="module")
def trackrun():
    """Load and cache data."""
    return core.TrackRun(TEST_DIR)


def test_check_far_from_boundaries(trackrun):
    """Test check_far_from_boundaries() on a cached TrackRun."""
    a_track = trackrun.data.loc[9]
    assert misc.check_far_from_boundaries(a_track, [-20, 30, 65, 80], dist=200)
    assert not misc.check_far_from_boundaries(a_track, [-10, 30, 73, 80], dist=200)
    assert not misc.check_far_from_boundaries(a_track, [-20, 30, 70, 80], dist=1e3)
