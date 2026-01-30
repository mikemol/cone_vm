import random

import pytest

import prism_vm as pv
from tests import harness

pytestmark = pytest.mark.m2


def _rand_expr(rng, depth):
    if depth <= 0:
        return "zero"
    roll = rng.random()
    if roll < 0.35:
        return "zero"
    if roll < 0.7:
        return f"(suc {_rand_expr(rng, depth - 1)})"
    return f"(add {_rand_expr(rng, depth - 1)} {_rand_expr(rng, depth - 1)})"


def test_strata_validator_random_programs():
    rng = random.Random(0)
    for _ in range(12):
        expr = _rand_expr(rng, 4)
        harness.denote_pretty_bsp_candidates(
            expr, max_steps=64, validate_mode=pv.ValidateMode.STRICT
        )
