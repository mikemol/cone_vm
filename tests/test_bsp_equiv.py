import pytest

from tests import harness

pytestmark = pytest.mark.m1


def test_bsp_matches_baseline_add():
    expr = "(add (suc zero) (suc zero))"
    harness.assert_baseline_equals_bsp_intrinsic(expr)


def test_bsp_matches_baseline_add_zero():
    expr = "(add zero (suc (suc zero)))"
    harness.assert_baseline_equals_bsp_intrinsic(expr)
