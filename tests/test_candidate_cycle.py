import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv


pytestmark = [
    pytest.mark.m2,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


def _require_cycle_candidates():
    assert hasattr(pv, "cycle_candidates")


def _commit_frontier(frontier_prov, q_map):
    return pv.apply_q(q_map, frontier_prov)


def _build_suc_add_suc_frontier():
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_zero = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_zero], dtype=jnp.int32),
        jnp.array([suc_zero], dtype=jnp.int32),
    )
    add_id = add_ids[0]
    root_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([add_id], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([root_ids[0]], dtype=jnp.int32))
    return ledger, frontier


def _ledger_snapshot(ledger):
    return (
        jax.device_get(ledger.opcode),
        jax.device_get(ledger.arg1),
        jax.device_get(ledger.arg2),
        jax.device_get(ledger.keys_b0_sorted),
        jax.device_get(ledger.keys_b1_sorted),
        jax.device_get(ledger.keys_b2_sorted),
        jax.device_get(ledger.keys_b3_sorted),
        jax.device_get(ledger.keys_b4_sorted),
        jax.device_get(ledger.ids_sorted),
    )


def _assert_ledger_snapshot(ledger, snapshot):
    fields = _ledger_snapshot(ledger)
    for field, expected in zip(fields, snapshot):
        assert (field == expected).all()


def test_cycle_candidates_rejects_when_cnf2_disabled(monkeypatch):
    _require_cycle_candidates()
    monkeypatch.setattr(pv, "_cnf2_enabled", lambda: False)
    ledger = pv.init_ledger()
    frontier = pv._committed_ids(jnp.array([pv.ZERO_PTR], dtype=jnp.int32))
    with pytest.raises(RuntimeError, match="cycle_candidates disabled until m2"):
        pv.cycle_candidates(ledger, frontier)


def test_cycle_candidates_empty_frontier_no_mutation():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    snapshot = _ledger_snapshot(ledger)
    frontier = pv._committed_ids(jnp.zeros((0,), dtype=jnp.int32))
    out_ledger, frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    mapped = pv.apply_q(q_map, frontier_prov).a
    stratum0, stratum1, stratum2 = strata
    assert int(out_ledger.count) == int(ledger.count)
    assert int(stratum0.count) == 0
    assert int(stratum1.count) == 0
    assert int(stratum2.count) == 0
    assert bool(jnp.array_equal(frontier_prov.a, frontier.a))
    assert bool(jnp.array_equal(mapped, frontier.a))
    _assert_ledger_snapshot(out_ledger, snapshot)


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
    frontier = pv._committed_ids(jnp.array([add_ids[0]], dtype=jnp.int32))
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    stratum0, stratum1, stratum2 = strata
    assert int(next_frontier.a.shape[0]) == 1
    assert int(next_frontier.a[0]) == int(y_id)
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 0
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 0
    assert int(stratum2.start) == start_count + int(stratum0.count) + int(stratum1.count)
    assert int(stratum2.count) == 0


def test_cycle_candidates_add_zero_right():
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
        jnp.array([y_id], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([add_ids[0]], dtype=jnp.int32))
    ledger, next_frontier_prov, _, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    assert int(next_frontier.a.shape[0]) == 1
    assert int(next_frontier.a[0]) == int(y_id)


def test_cycle_candidates_mul_zero():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([mul_ids[0]], dtype=jnp.int32))
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    stratum0, stratum1, stratum2 = strata
    assert int(next_frontier.a.shape[0]) == 1
    assert int(next_frontier.a[0]) == pv.ZERO_PTR
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 0
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 0
    assert int(stratum2.start) == start_count + int(stratum0.count) + int(stratum1.count)
    assert int(stratum2.count) == 0


@pytest.mark.m3
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
    frontier = pv._committed_ids(jnp.array([add_ids[0]], dtype=jnp.int32))
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    stratum0, stratum1, stratum2 = strata
    assert int(next_frontier.a.shape[0]) == 1
    next_id = next_frontier.a[0]
    assert int(next_id) == int(stratum1.start)
    assert int(ledger.opcode[next_id]) == pv.OP_SUC
    assert int(ledger.arg1[next_id]) == int(stratum0.start)
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 1
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 1
    assert int(stratum2.start) == start_count + int(stratum0.count) + int(stratum1.count)
    assert int(stratum2.count) == 0


@pytest.mark.m3
def test_cycle_candidates_add_suc_right():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    base_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    base_id = base_ids[0]
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_id = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([base_id], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([add_ids[0]], dtype=jnp.int32))
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    stratum0, stratum1, stratum2 = strata
    assert int(next_frontier.a.shape[0]) == 1
    next_id = next_frontier.a[0]
    assert int(next_id) == int(stratum1.start)
    assert int(ledger.opcode[next_id]) == pv.OP_SUC
    assert int(ledger.arg1[next_id]) == int(stratum0.start)
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 1
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 1
    assert int(stratum2.start) == start_count + int(stratum0.count) + int(stratum1.count)
    assert int(stratum2.count) == 0


@pytest.mark.m3
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
    frontier = pv._committed_ids(jnp.array([mul_ids[0]], dtype=jnp.int32))
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    stratum0, stratum1, stratum2 = strata
    assert int(next_frontier.a.shape[0]) == 1
    next_id = int(next_frontier.a[0])
    assert next_id == int(stratum1.start)
    assert int(ledger.opcode[next_id]) == pv.OP_ADD
    arg1 = int(ledger.arg1[next_id])
    arg2 = int(ledger.arg2[next_id])
    assert {arg1, arg2} == {int(y_id), int(stratum0.start)}
    assert arg1 <= arg2
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 1
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 1
    assert int(stratum2.start) == start_count + int(stratum0.count) + int(stratum1.count)
    assert int(stratum2.count) == 0


@pytest.mark.m3
def test_cycle_candidates_mul_suc_right():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    base_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    base_id = base_ids[0]
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_id = suc_ids[0]
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([base_id], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([mul_ids[0]], dtype=jnp.int32))
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    stratum0, stratum1, stratum2 = strata
    assert int(next_frontier.a.shape[0]) == 1
    next_id = int(next_frontier.a[0])
    assert next_id == int(stratum1.start)
    assert int(ledger.opcode[next_id]) == pv.OP_ADD
    arg1 = int(ledger.arg1[next_id])
    arg2 = int(ledger.arg2[next_id])
    assert {arg1, arg2} == {int(base_id), int(stratum0.start)}
    assert arg1 <= arg2
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 1
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 1
    assert int(stratum2.start) == start_count + int(stratum0.count) + int(stratum1.count)
    assert int(stratum2.count) == 0


def test_cycle_candidates_noop_on_suc():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([suc_ids[0]], dtype=jnp.int32))
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    stratum0, stratum1, stratum2 = strata
    assert int(next_frontier.a.shape[0]) == 1
    assert int(next_frontier.a[0]) == int(suc_ids[0])
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 0
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 0
    assert int(stratum2.start) == start_count + int(stratum0.count) + int(stratum1.count)
    assert int(stratum2.count) == 0


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
        frontier = pv._committed_ids(pool[idx])
        ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
            ledger, frontier, validate_stratum=True
        )
        next_frontier = _commit_frontier(next_frontier_prov, q_map)
        stratum0, stratum1, stratum2 = strata
        assert int(next_frontier.a.shape[0]) == int(frontier.a.shape[0])
        assert int(stratum0.start) <= int(ledger.count)
        assert int(stratum1.start) <= int(ledger.count)
        assert int(stratum2.start) <= int(ledger.count)


def test_cycle_candidates_validate_stratum_trips_on_within_refs(monkeypatch):
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_id = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([add_ids[0]], dtype=jnp.int32))
    real_intern = pv.intern_nodes

    def bad_intern(ledger_in, batch_or_ops, a1=None, a2=None):
        ids, new_ledger = real_intern(ledger_in, batch_or_ops, a1, a2)
        start = int(ledger_in.count)
        end = int(new_ledger.count)
        if end > start:
            idx = jnp.arange(start, end, dtype=jnp.int32)
            new_arg1 = new_ledger.arg1.at[idx].set(idx)
            new_ledger = new_ledger._replace(arg1=new_arg1)
        return ids, new_ledger

    monkeypatch.setattr(pv, "intern_nodes", bad_intern)
    with pytest.raises(ValueError, match="Stratum contains within-tier references"):
        pv.cycle_candidates(ledger, frontier, validate_stratum=True)


def test_cycle_candidates_stop_path_on_corrupt():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([suc_ids[0]], dtype=jnp.int32))
    snapshot = _ledger_snapshot(ledger)
    ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))

    out_ledger, frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    mapped = pv.apply_q(q_map, frontier_prov).a
    stratum0, stratum1, stratum2 = strata
    assert int(out_ledger.count) == int(ledger.count)
    assert int(stratum0.count) == 0
    assert int(stratum1.count) == 0
    assert int(stratum2.count) == 0
    assert bool(jnp.array_equal(frontier_prov.a, frontier.a))
    assert bool(jnp.array_equal(mapped, frontier.a))
    _assert_ledger_snapshot(out_ledger, snapshot)


def test_cycle_candidates_stop_path_on_oom():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([suc_ids[0]], dtype=jnp.int32))
    snapshot = _ledger_snapshot(ledger)
    ledger = ledger._replace(oom=jnp.array(True, dtype=jnp.bool_))

    with pytest.raises(RuntimeError, match="Ledger capacity exceeded"):
        pv.cycle_candidates(ledger, frontier)
    _assert_ledger_snapshot(ledger, snapshot)


def test_cycle_candidates_q_map_composes_across_strata():
    _require_cycle_candidates()
    ledger, frontier = _build_suc_add_suc_frontier()
    ledger, next_frontier_prov, strata, q_map = pv.cycle_candidates(
        ledger, frontier
    )
    stratum0, stratum1, stratum2 = strata
    assert int(stratum0.count) > 0
    assert int(stratum1.count) > 0
    assert int(stratum2.count) > 0

    def _expected_ids(prov_ids):
        ops = ledger.opcode[prov_ids]
        a1 = q_map(pv._provisional_ids(ledger.arg1[prov_ids])).a
        a2 = q_map(pv._provisional_ids(ledger.arg2[prov_ids])).a
        expected, _ = pv.intern_nodes(ledger, ops, a1, a2)
        return expected

    for stratum in strata:
        count_i = int(stratum.count)
        ids = stratum.start + jnp.arange(count_i, dtype=jnp.int32)
        mapped = q_map(pv._provisional_ids(ids)).a
        expected = _expected_ids(ids)
        assert bool(jnp.array_equal(mapped, expected))


def test_cycle_candidates_does_not_mutate_preexisting_rows():
    _require_cycle_candidates()
    ledger, frontier = _build_suc_add_suc_frontier()
    start_count = int(ledger.count)
    pre_ops = jax.device_get(ledger.opcode[:start_count])
    pre_a1 = jax.device_get(ledger.arg1[:start_count])
    pre_a2 = jax.device_get(ledger.arg2[:start_count])

    ledger, _, _, _ = pv.cycle_candidates(ledger, frontier)

    post_ops = jax.device_get(ledger.opcode[:start_count])
    post_a1 = jax.device_get(ledger.arg1[:start_count])
    post_a2 = jax.device_get(ledger.arg2[:start_count])
    assert bool(jnp.array_equal(pre_ops, post_ops))
    assert bool(jnp.array_equal(pre_a1, post_a1))
    assert bool(jnp.array_equal(pre_a2, post_a2))
