from pathlib import Path
import re

import pytest

import prism_vm as pv
from tests import harness

intern1 = harness.intern1

pytestmark = pytest.mark.m4


TOKEN_RE = re.compile(r"[A-Za-z_]+|==|[(),]")


def _coord_leaf(ledger, op):
    node_id, ledger = intern1(ledger, op, 0, 0)
    return int(node_id), ledger


def _parse_coord_expr(tokens, ledger):
    if not tokens:
        raise ValueError("unexpected end of coord expression")
    tok = tokens.pop(0)
    if tok == "zero":
        return _coord_leaf(ledger, pv.OP_COORD_ZERO)
    if tok == "one":
        return _coord_leaf(ledger, pv.OP_COORD_ONE)
    if tok == "pair":
        if not tokens or tokens.pop(0) != "(":
            raise ValueError("expected '(' after pair")
        left, ledger = _parse_coord_expr(tokens, ledger)
        if not tokens or tokens.pop(0) != ",":
            raise ValueError("expected ',' in pair")
        right, ledger = _parse_coord_expr(tokens, ledger)
        if not tokens or tokens.pop(0) != ")":
            raise ValueError("expected ')' after pair")
        node_id, ledger = intern1(ledger, pv.OP_COORD_PAIR, left, right)
        return int(node_id), ledger
    if tok == "xor":
        if not tokens or tokens.pop(0) != "(":
            raise ValueError("expected '(' after xor")
        left, ledger = _parse_coord_expr(tokens, ledger)
        if not tokens or tokens.pop(0) != ",":
            raise ValueError("expected ',' in xor")
        right, ledger = _parse_coord_expr(tokens, ledger)
        if not tokens or tokens.pop(0) != ")":
            raise ValueError("expected ')' after xor")
        coord_id, ledger = pv.coord_xor(ledger, left, right)
        return int(coord_id), ledger
    raise ValueError(f"unknown coord token: {tok}")


def _load_fixture_lines(name):
    path = Path(__file__).with_name(name)
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def test_coord_basic_fixture():
    ledger = pv.init_ledger()
    for line in _load_fixture_lines("coord_basic.txt"):
        if "==" in line:
            left_str, right_str = [part.strip() for part in line.split("==", 1)]
            left_tokens = TOKEN_RE.findall(left_str)
            right_tokens = TOKEN_RE.findall(right_str)
            left_id, ledger = _parse_coord_expr(left_tokens, ledger)
            right_id, ledger = _parse_coord_expr(right_tokens, ledger)
            left_norm, ledger = pv.coord_norm(ledger, left_id)
            right_norm, ledger = pv.coord_norm(ledger, right_id)
            assert int(left_norm) == int(right_norm)
        else:
            tokens = TOKEN_RE.findall(line)
            coord_id, ledger = _parse_coord_expr(tokens, ledger)
            norm1, ledger = pv.coord_norm(ledger, coord_id)
            norm2, ledger = pv.coord_norm(ledger, norm1)
            assert int(norm1) == int(norm2)


def test_coord_noop_fixture():
    ops = {
        "add": (pv.OP_ADD, pv.ZERO_PTR, pv.ZERO_PTR),
        "mul": (pv.OP_MUL, pv.ZERO_PTR, pv.ZERO_PTR),
        "suc": (pv.OP_SUC, pv.ZERO_PTR, 0),
        "zero": (pv.OP_ZERO, 0, 0),
    }
    for line in _load_fixture_lines("coord_noop.txt"):
        ledger = pv.init_ledger()
        op, a1, a2 = ops[line]
        node_id, ledger = intern1(ledger, op, a1, a2)
        count_before = int(ledger.count)
        norm_id, ledger = pv.coord_norm(ledger, node_id)
        assert int(norm_id) == node_id
        assert int(ledger.count) == count_before
