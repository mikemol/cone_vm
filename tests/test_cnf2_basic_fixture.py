from pathlib import Path

import pytest

from tests import harness

pytestmark = pytest.mark.m3


def _load_expressions():
    path = Path(__file__).with_name("cnf2_basic.txt")
    lines = path.read_text().splitlines()
    exprs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        exprs.append(line)
    return exprs


@pytest.mark.parametrize("expr", _load_expressions())
def test_cnf2_basic_fixture_equivalence(expr):
    assert harness.pretty_bsp_candidates(expr, max_steps=64) == harness.pretty_baseline(
        expr
    )
