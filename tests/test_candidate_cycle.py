import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

i32 = harness.i32
i32v = harness.i32v
intern_nodes = harness.intern_nodes
intern1 = harness.intern1
committed_ids = harness.committed_ids
cycle_candidates = harness.cycle_candidates_static_bound


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
    suc_zero, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, suc_zero, suc_zero)
    root_id, ledger = intern1(ledger, pv.OP_SUC, add_id, 0)
    frontier = committed_ids(root_id)
    return ledger, frontier


def _build_frontier_permutation_sample():
    ledger = pv.init_ledger()
    suc_ids, ledger = intern_nodes(
        ledger, [pv.OP_SUC, pv.OP_SUC], [pv.ZERO_PTR, pv.ZERO_PTR], [0, 0]
    )
    suc_x_id = suc_ids[0]
    y_id = suc_ids[1]
    add_zero_id, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, y_id)
    add_suc_id, ledger = intern1(ledger, pv.OP_ADD, suc_x_id, y_id)
    mul_zero_id, ledger = intern1(ledger, pv.OP_MUL, pv.ZERO_PTR, y_id)
    mul_suc_id, ledger = intern1(ledger, pv.OP_MUL, suc_x_id, y_id)
    frontier = i32(
        [add_zero_id, add_suc_id, mul_zero_id, mul_suc_id, suc_x_id]
    )
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
    ledger = pv.init_ledger()
    frontier = committed_ids(pv.ZERO_PTR)
    with pytest.raises(RuntimeError, match="cycle_candidates disabled until m2"):
        cycle_candidates(ledger, frontier, cnf2_enabled_fn=lambda: False)


def test_cycle_candidates_empty_frontier_no_mutation():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    snapshot = _ledger_snapshot(ledger)
    frontier = committed_ids([])
    out_ledger, frontier_prov, strata, q_map = cycle_candidates(
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

    y_id, ledger = intern1(ledger, pv.OP_SUC, 1, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, 1, y_id)
    frontier = committed_ids(add_id)
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = cycle_candidates(
        ledger, frontier, validate_mode=pv.ValidateMode.STRICT
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

    y_id, ledger = intern1(ledger, pv.OP_SUC, 1, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, y_id, 1)
    frontier = committed_ids(add_id)
    ledger, next_frontier_prov, _, q_map = cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    assert int(next_frontier.a.shape[0]) == 1
    assert int(next_frontier.a[0]) == int(y_id)


def test_cycle_candidates_mul_zero():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    mul_id, ledger = intern1(ledger, pv.OP_MUL, 1, 1)
    frontier = committed_ids(mul_id)
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = cycle_candidates(
        ledger, frontier, validate_mode=pv.ValidateMode.STRICT
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
    suc_x_id, ledger = intern1(ledger, pv.OP_SUC, 1, 0)
    y_id, ledger = intern1(ledger, pv.OP_SUC, 1, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, suc_x_id, y_id)
    frontier = committed_ids(add_id)
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = cycle_candidates(
        ledger, frontier, validate_mode=pv.ValidateMode.STRICT
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
def test_cycle_candidates_slot1_visibility_boundaries():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = intern_nodes(ledger, [pv.OP_SUC, pv.OP_SUC], [1, 1], [0, 0])
    suc_x_id = suc_ids[0]
    y_id = suc_ids[1]
    add_id, ledger = intern1(ledger, pv.OP_ADD, suc_x_id, y_id)
    frontier = committed_ids(add_id)
    ledger, _, strata, _ = cycle_candidates(
        ledger, frontier, validate_mode=pv.ValidateMode.STRICT
    )
    stratum0, stratum1, _ = strata
    assert int(stratum0.count) > 0
    assert int(stratum1.count) > 0
    ids = stratum1.start + jnp.arange(int(stratum1.count), dtype=jnp.int32)
    a1 = jax.device_get(ledger.arg1[ids])
    a2 = jax.device_get(ledger.arg2[ids])
    start1 = int(stratum1.start)
    assert (a1 < start1).all()
    assert (a2 < start1).all()
    slot0_id = int(stratum0.start)
    assert bool(jnp.any(a1 == slot0_id) | jnp.any(a2 == slot0_id))


@pytest.mark.m3
def test_cycle_candidates_slot1_visibility_across_cycles():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = intern_nodes(ledger, [pv.OP_SUC, pv.OP_SUC], [1, 1], [0, 0])
    suc_x_id = suc_ids[0]
    y_id = suc_ids[1]
    add_id, ledger = intern1(ledger, pv.OP_ADD, suc_x_id, y_id)
    mul_id, ledger = intern1(ledger, pv.OP_MUL, suc_x_id, y_id)
    frontier = committed_ids([add_id, mul_id])
    saw_slot1 = False
    for _ in range(3):
        ledger, frontier_prov, strata, q_map = cycle_candidates(
            ledger, frontier, validate_mode=pv.ValidateMode.STRICT
        )
        stratum0, stratum1, _ = strata
        if int(stratum1.count) == 0:
            frontier = _commit_frontier(frontier_prov, q_map)
            continue
        saw_slot1 = True
        start0 = int(stratum0.start)
        start1 = int(stratum1.start)
        end0 = start0 + int(stratum0.count)
        ids = stratum1.start + jnp.arange(int(stratum1.count), dtype=jnp.int32)
        a1 = jax.device_get(ledger.arg1[ids])
        a2 = jax.device_get(ledger.arg2[ids])
        assert (a1 < start1).all()
        assert (a2 < start1).all()
        if int(stratum0.count) > 0:
            slot0_ref = ((a1 >= start0) & (a1 < end0)) | (
                (a2 >= start0) & (a2 < end0)
            )
            assert slot0_ref.all()
        frontier = _commit_frontier(frontier_prov, q_map)
    assert saw_slot1


@pytest.mark.m3
def test_cycle_candidates_add_suc_right():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    base_id, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, pv.ZERO_PTR)
    suc_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, base_id, suc_id)
    frontier = committed_ids(add_id)
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = cycle_candidates(
        ledger, frontier, validate_mode=pv.ValidateMode.STRICT
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
    suc_x_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    y_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    mul_id, ledger = intern1(ledger, pv.OP_MUL, suc_x_id, y_id)
    frontier = committed_ids(mul_id)
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = cycle_candidates(
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
    base_id, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, pv.ZERO_PTR)
    suc_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    mul_id, ledger = intern1(ledger, pv.OP_MUL, base_id, suc_id)
    frontier = committed_ids(mul_id)
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = cycle_candidates(
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
    suc_id, ledger = intern1(ledger, pv.OP_SUC, 1, 0)
    frontier = committed_ids(suc_id)
    start_count = int(ledger.count)
    ledger, next_frontier_prov, strata, q_map = cycle_candidates(
        ledger, frontier
    )
    next_frontier = _commit_frontier(next_frontier_prov, q_map)
    stratum0, stratum1, stratum2 = strata
    assert int(next_frontier.a.shape[0]) == 1
    assert int(next_frontier.a[0]) == int(suc_id)
    assert int(stratum0.start) == start_count
    assert int(stratum0.count) == 0
    assert int(stratum1.start) == start_count + int(stratum0.count)
    assert int(stratum1.count) == 0
    assert int(stratum2.start) == start_count + int(stratum0.count) + int(stratum1.count)
    assert int(stratum2.count) == 0


def test_cycle_candidates_validate_stratum_random_frontier():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_ids, ledger = intern_nodes(ledger, [pv.OP_SUC, pv.OP_SUC], [1, 1], [0, 0])
    suc_x_id = suc_ids[0]
    y_id = suc_ids[1]
    add_zero_id, ledger = intern1(ledger, pv.OP_ADD, 1, y_id)
    add_suc_id, ledger = intern1(ledger, pv.OP_ADD, suc_x_id, y_id)
    mul_zero_id, ledger = intern1(ledger, pv.OP_MUL, 1, y_id)
    mul_suc_id, ledger = intern1(ledger, pv.OP_MUL, suc_x_id, y_id)
    pool = i32([add_zero_id, add_suc_id, mul_zero_id, mul_suc_id, suc_x_id, y_id])
    key = jax.random.PRNGKey(0)
    for _ in range(4):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (5,), 0, pool.shape[0])
        frontier = committed_ids(pool[idx])
        ledger, next_frontier_prov, strata, q_map = cycle_candidates(
            ledger, frontier, validate_mode=pv.ValidateMode.STRICT
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
    suc_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, suc_id, suc_id)
    frontier = committed_ids(add_id)
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

    with pytest.raises(ValueError, match="Stratum contains within-tier references"):
        cycle_candidates(
            ledger, frontier, validate_mode=pv.ValidateMode.STRICT, intern_fn=bad_intern
        )


@pytest.mark.m3
def test_cycle_candidates_wrap_microstrata_validate():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_zero, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    root, ledger = intern1(ledger, pv.OP_ADD, suc_zero, suc_zero)
    for _ in range(2):
        root, ledger = intern1(ledger, pv.OP_SUC, root, 0)
    frontier = committed_ids(root)
    ledger2, _, strata, _ = cycle_candidates(
        ledger, frontier, validate_mode=pv.ValidateMode.STRICT
    )
    stratum2 = strata[2]
    assert int(stratum2.count) > 1
    assert not bool(pv.validate_stratum_no_within_refs(ledger2, stratum2))
    assert bool(pv.validate_stratum_no_future_refs(ledger2, stratum2))


def test_cycle_candidates_frontier_permutation_invariant():
    _require_cycle_candidates()
    ledger_a, frontier_a = _build_frontier_permutation_sample()
    ledger_b, frontier_b = _build_frontier_permutation_sample()
    perm = jnp.array([2, 0, 4, 1, 3], dtype=jnp.int32)
    inv_perm = jnp.argsort(perm)
    frontier_perm = frontier_b[perm]

    ledger_a, next_frontier_prov_a, _, q_map_a = cycle_candidates(
        ledger_a, committed_ids(frontier_a)
    )
    next_frontier_a = pv.apply_q(q_map_a, next_frontier_prov_a).a

    ledger_b, next_frontier_prov_b, _, q_map_b = cycle_candidates(
        ledger_b, committed_ids(frontier_perm)
    )
    next_frontier_b = pv.apply_q(q_map_b, next_frontier_prov_b).a
    assert bool(jnp.array_equal(next_frontier_a, next_frontier_b[inv_perm]))


@pytest.mark.m3
def test_cycle_candidates_emits_from_pre_step_ledger(monkeypatch):
    _require_cycle_candidates()
    ledger, frontier = _build_suc_add_suc_frontier()
    pre_count = int(pv._host_int_value(ledger.count))
    real_emit = pv.emit_candidates

    def _emit_checked(ledger_in, frontier_ids):
        assert int(pv._host_int_value(ledger_in.count)) == pre_count
        return real_emit(ledger_in, frontier_ids)

    cycle_candidates(ledger, frontier, emit_candidates_fn=_emit_checked)


def test_cycle_candidates_stop_path_on_corrupt():
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    frontier = committed_ids(suc_id)
    snapshot = _ledger_snapshot(ledger)
    ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))

    out_ledger, frontier_prov, strata, q_map = cycle_candidates(
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
    suc_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    frontier = committed_ids(suc_id)
    snapshot = _ledger_snapshot(ledger)
    ledger = ledger._replace(oom=jnp.array(True, dtype=jnp.bool_))

    with pytest.raises(RuntimeError, match="Ledger capacity exceeded"):
        cycle_candidates(ledger, frontier)
    _assert_ledger_snapshot(ledger, snapshot)


def test_cycle_candidates_stop_path_on_oom_is_non_mutating(monkeypatch):
    _require_cycle_candidates()
    ledger = pv.init_ledger()
    suc_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    frontier = committed_ids(suc_id)
    snapshot = _ledger_snapshot(ledger)
    ledger = ledger._replace(oom=jnp.array(True, dtype=jnp.bool_))
    out_ledger, frontier_prov, strata, q_map = cycle_candidates(
        ledger, frontier, host_raise_if_bad_fn=lambda *args, **kwargs: None
    )
    mapped = pv.apply_q(q_map, frontier_prov).a
    stratum0, stratum1, stratum2 = strata
    assert bool(out_ledger.oom)
    assert int(out_ledger.count) == int(ledger.count)
    assert int(stratum0.count) == 0
    assert int(stratum1.count) == 0
    assert int(stratum2.count) == 0
    assert bool(jnp.array_equal(mapped, frontier_prov.a))
    _assert_ledger_snapshot(out_ledger, snapshot)


def test_cycle_candidates_q_map_composes_across_strata():
    _require_cycle_candidates()
    ledger, frontier = _build_suc_add_suc_frontier()
    ledger, next_frontier_prov, strata, q_map = cycle_candidates(
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
        expected, _ = intern_nodes(ledger, ops, a1, a2)
        return expected

    for stratum in strata:
        count_i = int(stratum.count)
        ids = stratum.start + jnp.arange(count_i, dtype=jnp.int32)
        mapped = q_map(pv._provisional_ids(ids)).a
        expected = _expected_ids(ids)
        assert bool(jnp.array_equal(mapped, expected))


def test_cycle_candidates_q_map_idempotent_on_frontier():
    _require_cycle_candidates()
    ledger, frontier = _build_suc_add_suc_frontier()
    ledger, next_frontier_prov, _, q_map = cycle_candidates(ledger, frontier)
    mapped = pv.apply_q(q_map, next_frontier_prov).a
    remapped = pv.apply_q(q_map, pv._provisional_ids(mapped)).a
    assert bool(jnp.array_equal(mapped, remapped))


@pytest.mark.m3
def test_cycle_candidates_does_not_mutate_preexisting_rows():
    _require_cycle_candidates()
    ledger, frontier = _build_suc_add_suc_frontier()
    start_count = int(ledger.count)
    pre_ops = jax.device_get(ledger.opcode[:start_count])
    pre_a1 = jax.device_get(ledger.arg1[:start_count])
    pre_a2 = jax.device_get(ledger.arg2[:start_count])

    ledger, _, _, _ = cycle_candidates(ledger, frontier)

    post_ops = jax.device_get(ledger.opcode[:start_count])
    post_a1 = jax.device_get(ledger.arg1[:start_count])
    post_a2 = jax.device_get(ledger.arg2[:start_count])
    assert bool(jnp.array_equal(pre_ops, post_ops))
    assert bool(jnp.array_equal(pre_a1, post_a1))
    assert bool(jnp.array_equal(pre_a2, post_a2))
