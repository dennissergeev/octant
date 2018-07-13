from octant import core
from pathlib import Path
import pytest

TEST_DIR = Path('.') / 'test_data'


@pytest.fixture(scope='module')
def trackrun():
    return core.TrackRun(TEST_DIR)


def test_load_data():
    tr = core.TrackRun()
    tr.load_data(TEST_DIR)
    assert len(tr) == 58
    assert tr.size() == 58
    assert not tr.is_categorised


def test_categorise(trackrun):
    trackrun.categorise()
    assert trackrun.is_categorised
    assert trackrun.size('basic') == 21
    assert trackrun.size('moderate') == 5
    assert trackrun.size('strong') == 1


def test_conf(trackrun):
    assert hasattr(trackrun, 'conf')
    assert len(trackrun.conf) == 43
    assert len(trackrun.conf.extent) == 4
    assert trackrun.conf.extent == [5, 55, 67, 79]
