"""The macaque viral-load loader.

These tests need the upstream repository, which is gitignored and absent from
a fresh checkout, so they skip rather than fail when it is not there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip.data import ASSAY_FLOOR, DATA_DIR, load_viral_loads  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="upstream repo not cloned into external/",
)


@pytest.fixture(scope="module")
def data():
    return load_viral_loads()


def test_arms_and_animal_counts(data):
    assert set(data) == {"control", "treated"}
    assert len(data["control"]) == 3
    assert len(data["treated"]) == 4


def test_no_missing_or_floor_values_survive(data):
    for arm in data.values():
        for s in arm:
            assert np.all(np.isfinite(s.vl))
            assert np.all(s.vl > ASSAY_FLOOR)
            assert np.all(s.t_days >= 0.0)
            assert len(s.t_days) == len(s.vl)


def test_times_are_sorted_and_in_days(data):
    for arm in data.values():
        for s in arm:
            assert np.all(np.diff(s.t_days) > 0)
            assert s.t_weeks[-1] == pytest.approx(s.t_days[-1] / 7.0)


def test_treated_followed_longer_than_controls(data):
    """The asymmetry S2's check 1 exists to correct for."""
    last_c = max(s.t_days[-1] for s in data["control"])
    last_t = max(s.t_days[-1] for s in data["treated"])
    assert last_t > last_c


def test_truncate_is_a_prefix(data):
    s = data["treated"][0]
    cut = s.truncate(17 * 7.0)
    assert len(cut.vl) <= len(s.vl)
    assert np.all(cut.t_days <= 17 * 7.0)
    assert np.allclose(cut.vl, s.vl[: len(cut.vl)])


def test_known_values_spotcheck(data):
    """Guard against a silent column shift in the loader."""
    ctrl = {s.animal: s for s in data["control"]}
    assert "40412" in ctrl
    # week 2 of animal 40412 is 1.44e8 copies/mL in the source table
    s = ctrl["40412"]
    i = int(np.argmin(np.abs(s.t_days - 14.0)))
    assert s.vl[i] == pytest.approx(1.44e8, rel=1e-6)
