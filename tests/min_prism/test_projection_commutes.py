import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3


def _build_ledger_with_tail():
    ledger = pv.init_ledger()
    suc0_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc0 = int(suc0_ids[0])
    suc1_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([suc0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc1 = int(suc1_ids[0])
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc0], dtype=jnp.int32),
        jnp.array([suc1], dtype=jnp.int32),
    )
    add_id = int(add_ids[0])
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([suc1], dtype=jnp.int32),
        jnp.array([suc0], dtype=jnp.int32),
    )
    mul_id = int(mul_ids[0])
    # Tail nodes that should be irrelevant to the frontier.
    tail_pairs = [
        (suc0, suc1),
        (suc1, suc0),
        (add_id, mul_id),
        (mul_id, add_id),
    ]
    for a1, a2 in tail_pairs:
        _, ledger = pv.intern_nodes(
            ledger,
            jnp.array([pv.OP_SORT], dtype=jnp.int32),
            jnp.array([a1], dtype=jnp.int32),
            jnp.array([a2], dtype=jnp.int32),
        )
    frontier = jnp.array([add_id], dtype=jnp.int32)
    keys = list(mph.canon_state_ledger(ledger)[0])
    frontier_key = mph.structural_key_for_id(ledger, add_id)
    frontier_idx = keys.index(frontier_key)
    max_id = max(frontier_idx, len(keys) - 3)
    return ledger, frontier, max_id


def _build_ledger_with_tail_multi_frontier():
    ledger = pv.init_ledger()
    suc0_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc0 = int(suc0_ids[0])
    suc1_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([suc0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc1 = int(suc1_ids[0])
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc0], dtype=jnp.int32),
        jnp.array([suc1], dtype=jnp.int32),
    )
    add_id = int(add_ids[0])
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([suc0], dtype=jnp.int32),
        jnp.array([suc1], dtype=jnp.int32),
    )
    mul_id = int(mul_ids[0])
    tail_pairs = [
        (suc0, suc1),
        (suc1, suc0),
        (add_id, mul_id),
        (mul_id, add_id),
    ]
    for a1, a2 in tail_pairs:
        _, ledger = pv.intern_nodes(
            ledger,
            jnp.array([pv.OP_SORT], dtype=jnp.int32),
            jnp.array([a1], dtype=jnp.int32),
            jnp.array([a2], dtype=jnp.int32),
        )
    frontier = jnp.array([add_id, mul_id], dtype=jnp.int32)
    keys = list(mph.canon_state_ledger(ledger)[0])
    frontier_keys = [
        mph.structural_key_for_id(ledger, add_id),
        mph.structural_key_for_id(ledger, mul_id),
    ]
    frontier_idx = max(keys.index(k) for k in frontier_keys)
    max_id = max(frontier_idx, len(keys) - 3)
    return ledger, frontier, max_id


def test_projection_commutes_cycle_intrinsic():
    ledger, frontier, max_id = _build_ledger_with_tail()
    full_ledger, full_frontier = pv.cycle_intrinsic(ledger, frontier)
    lhs_ledger, lhs_map = mph.project_ledger(full_ledger, max_id)
    lhs_state = mph.canon_state_ledger(lhs_ledger)
    lhs_frontier = mph.map_ids(full_frontier, lhs_map)

    proj_ledger, mapping = mph.project_ledger(ledger, max_id)
    proj_frontier = mph.map_ids(frontier, mapping)
    proj_ledger, proj_frontier = pv.cycle_intrinsic(
        proj_ledger, proj_frontier
    )
    proj_ledger, rhs_map = mph.project_ledger(proj_ledger, max_id)
    rhs_state = mph.canon_state_ledger(proj_ledger)
    proj_frontier = mph.map_ids(proj_frontier, rhs_map)
    assert lhs_state == rhs_state

    lhs_key = mph.structural_key_for_id(
        lhs_ledger, int(lhs_frontier[0])
    )
    rhs_key = mph.structural_key_for_id(
        proj_ledger, int(proj_frontier[0])
    )
    assert lhs_key == rhs_key


def test_projection_commutes_cycle_intrinsic_two_steps():
    ledger, frontier, max_id = _build_ledger_with_tail()
    full_ledger, full_frontier = pv.cycle_intrinsic(ledger, frontier)
    full_ledger, full_frontier = pv.cycle_intrinsic(full_ledger, full_frontier)
    lhs_ledger, lhs_map = mph.project_ledger(full_ledger, max_id)
    lhs_state = mph.canon_state_ledger(lhs_ledger)
    lhs_frontier = mph.map_ids(full_frontier, lhs_map)

    proj_ledger, mapping = mph.project_ledger(ledger, max_id)
    proj_frontier = mph.map_ids(frontier, mapping)
    proj_ledger, proj_frontier = pv.cycle_intrinsic(
        proj_ledger, proj_frontier
    )
    proj_ledger, proj_frontier = pv.cycle_intrinsic(
        proj_ledger, proj_frontier
    )
    proj_ledger, rhs_map = mph.project_ledger(proj_ledger, max_id)
    rhs_state = mph.canon_state_ledger(proj_ledger)
    proj_frontier = mph.map_ids(proj_frontier, rhs_map)
    assert lhs_state == rhs_state

    lhs_key = mph.structural_key_for_id(
        lhs_ledger, int(lhs_frontier[0])
    )
    rhs_key = mph.structural_key_for_id(
        proj_ledger, int(proj_frontier[0])
    )
    assert lhs_key == rhs_key


def test_projection_commutes_cycle_candidates():
    ledger, frontier, max_id = _build_ledger_with_tail()
    frontier = pv._committed_ids(frontier)
    full_ledger, full_frontier_prov, _, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    full_frontier = pv.apply_q(q_map, full_frontier_prov).a
    lhs_ledger, lhs_map = mph.project_ledger(full_ledger, max_id)
    lhs_state = mph.canon_state_ledger(lhs_ledger)
    lhs_frontier = mph.map_ids(full_frontier, lhs_map)

    proj_ledger, mapping = mph.project_ledger(ledger, max_id)
    proj_frontier = pv._committed_ids(mph.map_ids(frontier.a, mapping))
    proj_ledger, proj_frontier_prov, _, proj_q = pv.cycle_candidates(
        proj_ledger, proj_frontier
    )
    proj_frontier = pv.apply_q(proj_q, proj_frontier_prov).a
    proj_ledger, rhs_map = mph.project_ledger(proj_ledger, max_id)
    rhs_state = mph.canon_state_ledger(proj_ledger)
    proj_frontier = mph.map_ids(proj_frontier, rhs_map)
    assert lhs_state == rhs_state

    lhs_key = mph.structural_key_for_id(
        lhs_ledger, int(lhs_frontier[0])
    )
    rhs_key = mph.structural_key_for_id(
        proj_ledger, int(proj_frontier[0])
    )
    assert lhs_key == rhs_key


def test_projection_commutes_cycle_candidates_two_steps():
    ledger, frontier, max_id = _build_ledger_with_tail()
    frontier_ids = pv._committed_ids(frontier)
    full_ledger, full_frontier_prov, _, q_map = pv.cycle_candidates(
        ledger, frontier_ids
    )
    full_frontier = pv.apply_q(q_map, full_frontier_prov).a
    full_frontier = pv._committed_ids(full_frontier)
    full_ledger, full_frontier_prov, _, q_map = pv.cycle_candidates(
        full_ledger, full_frontier
    )
    full_frontier = pv.apply_q(q_map, full_frontier_prov).a
    lhs_ledger, lhs_map = mph.project_ledger(full_ledger, max_id)
    lhs_state = mph.canon_state_ledger(lhs_ledger)
    lhs_frontier = mph.map_ids(full_frontier, lhs_map)

    proj_ledger, mapping = mph.project_ledger(ledger, max_id)
    proj_frontier = pv._committed_ids(mph.map_ids(frontier, mapping))
    proj_ledger, proj_frontier_prov, _, proj_q = pv.cycle_candidates(
        proj_ledger, proj_frontier
    )
    proj_frontier = pv._committed_ids(pv.apply_q(proj_q, proj_frontier_prov).a)
    proj_ledger, proj_frontier_prov, _, proj_q = pv.cycle_candidates(
        proj_ledger, proj_frontier
    )
    proj_frontier = pv.apply_q(proj_q, proj_frontier_prov).a
    proj_ledger, rhs_map = mph.project_ledger(proj_ledger, max_id)
    rhs_state = mph.canon_state_ledger(proj_ledger)
    proj_frontier = mph.map_ids(proj_frontier, rhs_map)
    assert lhs_state == rhs_state

    lhs_key = mph.structural_key_for_id(
        lhs_ledger, int(lhs_frontier[0])
    )
    rhs_key = mph.structural_key_for_id(
        proj_ledger, int(proj_frontier[0])
    )
    assert lhs_key == rhs_key


def test_projection_commutes_cycle_candidates_frontier_permutation():
    states = []
    frontier_keys = []
    for permute in (False, True):
        ledger, frontier, max_id = _build_ledger_with_tail_multi_frontier()
        if permute:
            frontier = frontier[::-1]
        frontier_ids = pv._committed_ids(frontier)
        full_ledger, full_frontier_prov, _, q_map = pv.cycle_candidates(
            ledger, frontier_ids
        )
        full_frontier = pv.apply_q(q_map, full_frontier_prov).a
        lhs_ledger, lhs_map = mph.project_ledger(full_ledger, max_id)
        lhs_state = mph.canon_state_ledger(lhs_ledger)
        lhs_frontier = mph.map_ids(full_frontier, lhs_map)
        states.append(lhs_state)
        frontier_keys.append(
            sorted(
                mph.structural_key_for_id(lhs_ledger, int(fid))
                for fid in jax.device_get(lhs_frontier)
            )
        )

        proj_ledger, mapping = mph.project_ledger(ledger, max_id)
        proj_frontier = pv._committed_ids(mph.map_ids(frontier, mapping))
        proj_ledger, proj_frontier_prov, _, proj_q = pv.cycle_candidates(
            proj_ledger, proj_frontier
        )
        proj_frontier = pv.apply_q(proj_q, proj_frontier_prov).a
        proj_ledger, rhs_map = mph.project_ledger(proj_ledger, max_id)
        rhs_state = mph.canon_state_ledger(proj_ledger)
        proj_frontier = mph.map_ids(proj_frontier, rhs_map)
        assert lhs_state == rhs_state

        rhs_keys = sorted(
            mph.structural_key_for_id(proj_ledger, int(fid))
            for fid in jax.device_get(proj_frontier)
        )
        assert frontier_keys[-1] == rhs_keys

    assert states[0] == states[1]
    assert frontier_keys[0] == frontier_keys[1]
