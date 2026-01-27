import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3


def _assert_rebuild_matches(ledger):
    keys = set(mph.canon_state_ledger(ledger)[0])
    rebuilt, _ = mph.rebuild_ledger_from_keys(keys)
    assert mph.canon_state_ledger(rebuilt) == mph.canon_state_ledger(ledger)


def test_cqrs_replay_matches_intrinsic_cycles():
    vm = pv.PrismVM_BSP()
    root_ptr = harness.parse_expr(vm, "(add (suc zero) (suc zero))")
    frontier = jnp.array([int(root_ptr)], dtype=jnp.int32)
    for _ in range(3):
        vm.ledger, frontier = pv.cycle_intrinsic(vm.ledger, frontier)
    _assert_rebuild_matches(vm.ledger)


def test_cqrs_replay_matches_cnf2_cycles():
    if not pv._cnf2_enabled():
        pytest.skip("CNF-2 disabled")
    vm = pv.PrismVM_BSP()
    root_ptr = harness.parse_expr(vm, "(mul (suc zero) (suc zero))")
    frontier = pv._committed_ids(jnp.array([int(root_ptr)], dtype=jnp.int32))
    for _ in range(3):
        vm.ledger, frontier_prov, _, q_map = pv.cycle_candidates(vm.ledger, frontier)
        frontier = pv.apply_q(q_map, frontier_prov)
    _assert_rebuild_matches(vm.ledger)
