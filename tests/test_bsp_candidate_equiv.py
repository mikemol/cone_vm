import pytest

from tests import harness

pytestmark = pytest.mark.m2


def test_bsp_candidates_matches_baseline_add_zero():
    expr = "(add zero (suc zero))"
    harness.assert_baseline_equals_bsp_candidates(expr, max_steps=32)


def test_bsp_candidates_matches_baseline_mul_zero():
    expr = "(mul zero (suc zero))"
    harness.assert_baseline_equals_bsp_candidates(expr, max_steps=32)


@pytest.mark.m3
def test_bsp_candidates_matches_baseline_add_suc():
    expr = "(add (suc zero) (suc zero))"
    harness.assert_baseline_equals_bsp_candidates(expr, max_steps=32)


@pytest.mark.m3
def test_bsp_candidates_matches_baseline_mul_suc():
    expr = "(mul (suc zero) (suc zero))"
    harness.assert_baseline_equals_bsp_candidates(expr, max_steps=32)


@pytest.mark.m3
@pytest.mark.parametrize(
    "expr",
    [
        "(add zero (suc zero))",
        "(add (suc zero) (suc zero))",
        "(mul zero (suc zero))",
        "(mul (suc zero) (suc zero))",
    ],
)
def test_bsp_candidates_matches_intrinsic(expr):
    assert harness.denote_pretty_bsp_candidates(
        expr, max_steps=64
    ) == harness.denote_pretty_bsp_intrinsic(expr, max_steps=64)
