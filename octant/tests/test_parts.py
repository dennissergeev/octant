"""Test parts submodule."""
from pathlib import Path

from octant import parts
from octant.exceptions import LoadError

import pytest

TEST_FNAME = (
    Path(__file__).parent / "test_data" / "era5_run000" / "era5_run000_accacia_settings.conf"
)


def test_empty_tracksettings():
    """Test empty TrackSettings."""
    ts = parts.TrackSettings()
    assert len(ts) == 0
    assert ts.extent == [None, None, None, None]
    assert ts.to_dict() == {}


def test_tracksettings():
    """Test TrackSettings."""
    ts = parts.TrackSettings(TEST_FNAME)
    assert ts.__repr__() == "Tracking algorithm settings (43)"
    assert len(ts.__str__()) == 754
    assert ts.__str__().startswith("Tracking algorithm settings")
    assert len(ts) == 43
    assert ts.extent == [-10, 40, 67, 78]
    assert isinstance(ts.to_dict(), dict)
    new = parts.TrackSettings.from_dict(ts.to_dict())
    assert isinstance(new, parts.TrackSettings)
    assert len(new) == 43


def test_copy():
    """Test TrackSettings.copy."""
    ts = parts.TrackSettings(TEST_FNAME)
    new = ts.copy()
    assert isinstance(new, parts.TrackSettings)
    assert len(new) == 43
    assert new.__repr__() == "Tracking algorithm settings (43)"
    assert new.extent == [-10, 40, 67, 78]


def test_loaderror():
    """Test raising LoadError."""
    with pytest.raises(LoadError):
        parts.TrackSettings(str(TEST_FNAME))
