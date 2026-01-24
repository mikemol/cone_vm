import pytest

from tests import harness

pytestmark = pytest.mark.m3


@pytest.mark.parametrize(
    "expr",
    [
        "zero",
        "(suc zero)",
        "(add (suc zero) (suc zero))",
        "(mul (suc (suc zero)) (suc zero))",
        "(add (suc (suc zero)) (mul (suc zero) zero))",
    ],
)
def test_scheduler_commutation_small_suite(expr):
    harness.assert_arena_schedule_invariance(expr, steps=4)
