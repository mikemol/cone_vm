import pytest

from tests import harness

pytestmark = pytest.mark.m2


def test_bsp_candidates_matches_baseline_add_zero():
    expr = "(add zero (suc zero))"
    assert harness.pretty_bsp_candidates(expr, max_steps=32) == harness.pretty_baseline(expr)


def test_bsp_candidates_matches_baseline_mul_zero():
    expr = "(mul zero (suc zero))"
    assert harness.pretty_bsp_candidates(expr, max_steps=32) == harness.pretty_baseline(expr)


@pytest.mark.m3
def test_bsp_candidates_matches_baseline_add_suc():
    expr = "(add (suc zero) (suc zero))"
    assert harness.pretty_bsp_candidates(expr, max_steps=32) == harness.pretty_baseline(expr)


@pytest.mark.m3
def test_bsp_candidates_matches_baseline_mul_suc():
    expr = "(mul (suc zero) (suc zero))"
    assert harness.pretty_bsp_candidates(expr, max_steps=32) == harness.pretty_baseline(expr)
