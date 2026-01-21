import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv


def _require_cycle_candidates():
    if not hasattr(pv, "cycle_candidates"):
        pytest.xfail("cycle_candidates not implemented")


def test_cycle_candidates_add_zero():
    _require_cycle_candidates()
    ledger = pv.init_ledger()

    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    y_id = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    frontier = jnp.array([add_ids[0]], dtype=jnp.int32)
    start_count = int(ledger.count)
    ledger, next_frontier, strata = pv.cycle_candidates(ledger, frontier)
    stratum0, stratum1 = strata
    assert int(next_frontier.shape[0]) == 1
    assert int(next_frontier[0]) == int(y_id)
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 0
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 0


def test_cycle_candidates_mul_zero():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    )
    frontier = jnp.array([mul_ids[0]], dtype=jnp.int32)
    start_count = int(ledger.count)
    ledger, next_frontier, strata = pv.cycle_candidates(ledger, frontier)
    stratum0, stratum1 = strata
    assert int(next_frontier.shape[0]) == 1
    assert int(next_frontier[0]) == pv.ZERO_PTR
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 0
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 0


def test_cycle_candidates_add_suc():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_x_id = suc_ids[0]
    y_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    y_id = y_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_x_id], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    frontier = jnp.array([add_ids[0]], dtype=jnp.int32)
    start_count = int(ledger.count)
    ledger, next_frontier, strata = pv.cycle_candidates(ledger, frontier)
    stratum0, stratum1 = strata
    assert int(next_frontier.shape[0]) == 1
    next_id = next_frontier[0]
    assert int(ledger.opcode[next_id]) == pv.OP_ADD
    assert int(ledger.arg1[next_id]) == pv.ZERO_PTR
    assert int(ledger.arg2[next_id]) == int(y_id)
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 1
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 1
    l2_id = int(stratum1.start)
    assert int(ledger.opcode[l2_id]) == pv.OP_SUC
    assert int(ledger.arg1[l2_id]) == int(next_id)


def test_cycle_candidates_mul_suc():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_x_id = suc_ids[0]
    y_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    y_id = y_ids[0]
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([suc_x_id], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    frontier = jnp.array([mul_ids[0]], dtype=jnp.int32)
    start_count = int(ledger.count)
    ledger, next_frontier, strata = pv.cycle_candidates(ledger, frontier)
    stratum0, stratum1 = strata
    assert int(next_frontier.shape[0]) == 1
    next_id = int(next_frontier[0])
    assert next_id == int(stratum1.start)
    assert int(ledger.opcode[next_id]) == pv.OP_ADD
    assert int(ledger.arg1[next_id]) == int(y_id)
    assert int(ledger.arg2[next_id]) == int(stratum0.start)
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 1
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 1


def test_cycle_candidates_noop_on_suc():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    frontier = jnp.array([suc_ids[0]], dtype=jnp.int32)
    start_count = int(ledger.count)
    ledger, next_frontier, strata = pv.cycle_candidates(ledger, frontier)
    stratum0, stratum1 = strata
    assert int(next_frontier.shape[0]) == 1
    assert int(next_frontier[0]) == int(suc_ids[0])
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 0
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 0


def test_cycle_candidates_validate_stratum_random_frontier():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC, pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1, 1], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
    )
    suc_x_id = suc_ids[0]
    y_id = suc_ids[1]
    add_zero_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    add_suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_x_id], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    mul_zero_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    mul_suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([suc_x_id], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    pool = jnp.array(
        [
            add_zero_ids[0],
            add_suc_ids[0],
            mul_zero_ids[0],
            mul_suc_ids[0],
            suc_x_id,
            y_id,
        ],
        dtype=jnp.int32,
    )
    key = jax.random.PRNGKey(0)
    for _ in range(4):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (5,), 0, pool.shape[0])
        frontier = pool[idx]
        ledger, next_frontier, strata = pv.cycle_candidates(
            ledger, frontier, validate_stratum=True
        )
        stratum0, stratum1 = strata
        assert int(next_frontier.shape[0]) == int(frontier.shape[0])
        assert int(stratum0.start) <= int(ledger.count)
        assert int(stratum1.start) <= int(ledger.count)
