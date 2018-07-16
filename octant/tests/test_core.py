"""Test the core submodule."""
from pathlib import Path

from octant import core

import pytest

TEST_DIR = Path(__file__).parent / 'test_data'


@pytest.fixture(scope='module')
def trackrun():
    """Load and cache data."""
    return core.TrackRun(TEST_DIR)


def test_load_data():
    """Create an empty TrackRun instance and load data afterwards."""
    tr = core.TrackRun()
    tr.load_data(TEST_DIR)
    assert len(tr) == 76
    assert tr.size() == 76
    assert not tr.is_categorised


def test_categorise(trackrun):
    """Use cached TrackRun instance and test categorise() method."""
    trackrun.categorise()
    assert trackrun.is_categorised
    assert trackrun.size('basic') == 31
    assert trackrun.size('moderate') == 10
    assert trackrun.size('strong') == 1


def test_conf(trackrun):
    """Use cached TrackRun instance and check the configuration attribute."""
    assert hasattr(trackrun, 'conf')
    assert len(trackrun.conf) == 43
    assert len(trackrun.conf.extent) == 4
    assert trackrun.conf.extent == [-10, 40, 67, 78]
