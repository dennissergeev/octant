"""Test the core submodule."""
import contextlib
import itertools
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from octant import core, parts

import pandas as pd

import pytest

TEST_DIR = Path(__file__).parent / "test_data" / "era5_run000"
REF_DM = Path(__file__).parent / "test_data" / "distance_matrix.npy"
REF_FILE = Path(__file__).parent / "test_data" / "pmc_loc_time_ch4_20Mar-02Apr.txt"

_counter = itertools.count()


@contextlib.contextmanager
def create_tmp_file(suffix=".h5", allow_cleanup_failure=False):
    """
    Temporary file context.

    Code taken from xarray test module
    """
    temp_dir = Path(tempfile.mkdtemp())
    path = temp_dir / "temp-{}{}".format(next(_counter), suffix)
    try:
        yield path
    finally:
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            if not allow_cleanup_failure:
                raise


@pytest.fixture(scope="module")
def trackrun():
    """Load and cache data."""
    return core.TrackRun(TEST_DIR)


@pytest.fixture(scope="module")
def ref_set():
    """Load reference dataset."""
    ref = pd.read_csv(
        REF_FILE,
        delimiter="\t",
        names=["N", "time", "lon", "lat"],
        parse_dates=["time"],
        date_parser=lambda x: datetime.strptime(x, "%Y%m%d%H%M"),
    )
    ref_tracks = []
    for i, df in ref.groupby("N"):
        ot = core.OctantTrack.from_df(df)
        if ot.lifetime_h >= 6:
            ref_tracks.append(ot)
    assert len(ref_tracks) == 27
    return ref_tracks


def test_load_data():
    """Create an empty TrackRun instance and load data afterwards."""
    tr = core.TrackRun()
    tr.load_data(TEST_DIR)
    assert len(tr) == 76
    assert tr.size() == 76
    assert not tr.is_categorised


def test_archive(trackrun):
    """Test to_archive() and from_archive() methods."""
    with create_tmp_file() as f:
        trackrun.to_archive(f)
        another = core.TrackRun.from_archive(f)
        assert another.data.equals(trackrun.data)
        assert another.sources == trackrun.sources
        assert hasattr(another, "conf")
        assert isinstance(another.conf, parts.TrackSettings)


def test_categorise(trackrun):
    """Use cached TrackRun instance and test categorise() method."""
    trackrun.categorise()
    assert trackrun.is_categorised
    assert trackrun.size("basic") == 31
    assert trackrun.size("moderate") == 10
    assert trackrun.size("strong") == 1


def test_classify(trackrun):
    """Use cached TrackRun instance and test classify() method."""
    conds = [
        ("a", [lambda ot: ot.lifetime_h >= 6]),
        (
            "b",
            [
                lambda ot: (ot.vortex_type != 0).sum() / ot.shape[0] < 0.2,
                lambda ot: ot.gen_lys_dist_km > 300.0,
            ],
        ),
    ]
    trackrun.classify(conds, inclusive=False)
    assert trackrun.is_categorised
    assert trackrun.size("a") == 21
    assert trackrun.size("b") == 11


def test_classify_incl(trackrun):
    """Use cached TrackRun instance and test classify() method with inclusive=True."""
    conds = [
        ("a", [lambda ot: ot.lifetime_h >= 6]),
        (
            "b",
            [
                lambda ot: (ot.vortex_type != 0).sum() / ot.shape[0] < 0.2,
                lambda ot: ot.gen_lys_dist_km > 300.0,
            ],
        ),
    ]
    trackrun.classify(conds, inclusive=True)
    assert trackrun.is_categorised
    assert trackrun.size("a") == 31
    assert trackrun.size("b") == 10


def test_conf(trackrun):
    """Use cached TrackRun instance and check the configuration attribute."""
    assert hasattr(trackrun, "conf")
    assert len(trackrun.conf) == 43
    assert len(trackrun.conf.extent) == 4
    assert trackrun.conf.extent == [-10, 40, 67, 78]


def test_match_bs2000(trackrun, ref_set):
    """Use cached TrackRun and tracks from ref_set to test match_tracks() method."""
    subset = "moderate"
    match_pairs, dm = trackrun.match_tracks(
        ref_set, subset=subset, method="bs2000", beta=50.0, return_dist_matrix=True
    )
    assert len(match_pairs) == 5
    assert dm.shape == (trackrun.size(subset), len(ref_set))
    actual_dm = np.load(REF_DM)
    np.testing.assert_almost_equal(actual_dm, dm)
