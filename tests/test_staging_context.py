import pytest

import prism_vm as pv

pytestmark = pytest.mark.m3


def test_hyperstrata_precedes_order():
    assert pv.hyperstrata_precedes(0, 0, 0, 1)
    assert pv.hyperstrata_precedes(0, 2, 1, 0)
    assert not pv.hyperstrata_precedes(1, 0, 0, 5)
    assert not pv.hyperstrata_precedes(0, 0, 0, 0)


def test_staging_context_forgets_detail():
    fine = pv.StagingContext(n=3, s=2, t=5, tile=7)
    coarse = pv.StagingContext(n=2, s=1, t=3, tile=None)
    assert pv.staging_context_forgets_detail(fine, coarse)
    assert not pv.staging_context_forgets_detail(coarse, fine)

    same_tile = pv.StagingContext(n=1, s=1, t=1, tile=5)
    assert pv.staging_context_forgets_detail(same_tile, same_tile)

    other_tile = pv.StagingContext(n=1, s=1, t=1, tile=4)
    assert not pv.staging_context_forgets_detail(same_tile, other_tile)
