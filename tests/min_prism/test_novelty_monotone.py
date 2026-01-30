import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3


def _build_simple_ledger():
    ledger = pv.init_ledger()
    suc_ids, ledger = mph.intern_nodes(
        ledger, [pv.OP_SUC], [pv.ZERO_PTR], [0]
    )
    suc_id = int(suc_ids[0])
    add_ids, ledger = mph.intern_nodes(
        ledger, [pv.OP_ADD], [pv.ZERO_PTR], [suc_id]
    )
    return ledger, int(add_ids[0])


def _assert_monotone(sets):
    for prev, curr in zip(sets, sets[1:]):
        assert prev.issubset(curr)


def test_novelty_monotone_intrinsic():
    ledger, root = _build_simple_ledger()
    frontier = jnp.array([root], dtype=jnp.int32)
    novelty = []
    for _ in range(3):
        novelty.append(mph.novelty_set(ledger))
        ledger, frontier = pv.cycle_intrinsic(ledger, frontier)
    novelty.append(mph.novelty_set(ledger))
    _assert_monotone(novelty)


def test_novelty_monotone_cnf2():
    ledger, root = _build_simple_ledger()
    frontier = pv._committed_ids(jnp.array([root], dtype=jnp.int32))
    novelty = []
    for _ in range(3):
        novelty.append(mph.novelty_set(ledger))
        ledger, frontier_prov, _, q_map = pv.cycle_candidates(ledger, frontier)
        frontier = pv.apply_q(q_map, frontier_prov)
    novelty.append(mph.novelty_set(ledger))
    _assert_monotone(novelty)


def test_novelty_fixed_point_cnf2_simple():
    ledger, root = _build_simple_ledger()
    frontier = pv._committed_ids(jnp.array([root], dtype=jnp.int32))
    prev = mph.novelty_set(ledger)
    stable = False
    for _ in range(6):
        ledger, frontier_prov, _, q_map = pv.cycle_candidates(ledger, frontier)
        frontier = pv.apply_q(q_map, frontier_prov)
        curr = mph.novelty_set(ledger)
        if curr == prev:
            stable = True
            break
        prev = curr
    assert stable
