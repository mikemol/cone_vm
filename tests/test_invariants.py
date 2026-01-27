import re

import jax.numpy as jnp
import pytest

import prism_vm as pv


def _small_add_manifest(cap):
    ops = jnp.zeros(cap, dtype=jnp.int32)
    a1 = jnp.zeros(cap, dtype=jnp.int32)
    a2 = jnp.zeros(cap, dtype=jnp.int32)
    ops = ops.at[1].set(pv.OP_ZERO)
    ops = ops.at[2].set(pv.OP_SUC)
    a1 = a1.at[2].set(1)
    ops = ops.at[3].set(pv.OP_ADD)
    a1 = a1.at[3].set(2)
    a2 = a2.at[3].set(2)
    return pv.Manifest(
        ops,
        a1,
        a2,
        jnp.array(cap, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
    )


def _add_manifest_two_suc(cap):
    ops = jnp.zeros(cap, dtype=jnp.int32)
    a1 = jnp.zeros(cap, dtype=jnp.int32)
    a2 = jnp.zeros(cap, dtype=jnp.int32)
    ops = ops.at[1].set(pv.OP_ZERO)
    ops = ops.at[2].set(pv.OP_SUC)
    a1 = a1.at[2].set(1)
    ops = ops.at[3].set(pv.OP_SUC)
    a1 = a1.at[3].set(2)
    ops = ops.at[4].set(pv.OP_ADD)
    a1 = a1.at[4].set(3)
    a2 = a2.at[4].set(1)
    return pv.Manifest(
        ops,
        a1,
        a2,
        jnp.array(5, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
    )


def _small_mul_manifest(cap):
    ops = jnp.zeros(cap, dtype=jnp.int32)
    a1 = jnp.zeros(cap, dtype=jnp.int32)
    a2 = jnp.zeros(cap, dtype=jnp.int32)
    ops = ops.at[1].set(pv.OP_ZERO)
    ops = ops.at[2].set(pv.OP_SUC)
    a1 = a1.at[2].set(1)
    ops = ops.at[3].set(pv.OP_MUL)
    a1 = a1.at[3].set(2)
    a2 = a2.at[3].set(1)
    return pv.Manifest(
        ops,
        a1,
        a2,
        jnp.array(cap, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
    )


def _mul_manifest_commutative(cap):
    ops = jnp.zeros(cap, dtype=jnp.int32)
    a1 = jnp.zeros(cap, dtype=jnp.int32)
    a2 = jnp.zeros(cap, dtype=jnp.int32)
    ops = ops.at[1].set(pv.OP_ZERO)
    ops = ops.at[2].set(pv.OP_SUC)
    a1 = a1.at[2].set(1)
    ops = ops.at[3].set(pv.OP_SUC)
    a1 = a1.at[3].set(2)
    ops = ops.at[4].set(pv.OP_MUL)
    a1 = a1.at[4].set(2)
    a2 = a2.at[4].set(3)
    return pv.Manifest(
        ops,
        a1,
        a2,
        jnp.array(5, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
    )


@pytest.mark.m1
def test_manifest_capacity_guard():
    vm = pv.PrismVM()
    cap = int(vm.manifest.opcode.shape[0])
    vm.active_count_host = cap
    with pytest.raises(ValueError):
        vm._cons_raw(
            pv.OP_SUC, pv._manifest_ptr(pv.ZERO_PTR), pv._manifest_ptr(0)
        )
    assert pv._host_bool_value(vm.manifest.oom)


@pytest.mark.m3
def test_arena_capacity_guard():
    vm = pv.PrismVM_BSP_Legacy()
    cap = int(vm.arena.opcode.shape[0])
    vm.arena = vm.arena._replace(count=jnp.array(cap, dtype=jnp.int32))
    with pytest.raises(ValueError):
        vm._alloc(pv.OP_SUC, pv._arena_ptr(pv.ZERO_PTR), pv._arena_ptr(0))
    assert pv._host_bool_value(vm.arena.oom)


@pytest.mark.m1
def test_ledger_capacity_guard():
    ledger = pv.init_ledger()
    ledger = ledger._replace(count=jnp.array(pv.MAX_ID, dtype=jnp.int32))
    ids, new_ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC, pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR, pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0, pv.ZERO_PTR], dtype=jnp.int32),
    )
    assert pv._host_bool_value(new_ledger.corrupt)
    assert not pv._host_bool_value(new_ledger.oom)
    assert pv._host_int_value(new_ledger.count) == pv._host_int_value(ledger.count)
    assert int(jnp.sum(ids)) == 0


@pytest.mark.m1
def test_intern_nodes_early_out_on_oom_returns_zero_ids():
    ledger = pv.init_ledger()
    ledger = ledger._replace(oom=jnp.array(True, dtype=jnp.bool_))
    ops = jnp.array([pv.OP_ZERO, pv.OP_SUC], dtype=jnp.int32)
    a1 = jnp.array([0, pv.ZERO_PTR], dtype=jnp.int32)
    a2 = jnp.array([0, 0], dtype=jnp.int32)
    ids, new_ledger = pv.intern_nodes(ledger, ops, a1, a2)
    new_ledger.count.block_until_ready()
    assert pv._host_bool_value(new_ledger.oom) == pv._host_bool_value(ledger.oom)
    assert pv._host_bool_value(new_ledger.corrupt) == pv._host_bool_value(ledger.corrupt)
    assert pv._host_int_value(new_ledger.count) == pv._host_int_value(ledger.count)
    assert int(ids[0]) == pv.ZERO_PTR
    assert int(ids[1]) == 0


@pytest.mark.m1
def test_intern_nodes_early_out_on_corrupt_returns_zero_ids():
    ledger = pv.init_ledger()
    ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))
    ops = jnp.array([pv.OP_ZERO, pv.OP_SUC], dtype=jnp.int32)
    a1 = jnp.array([0, pv.ZERO_PTR], dtype=jnp.int32)
    a2 = jnp.array([0, 0], dtype=jnp.int32)
    ids, new_ledger = pv.intern_nodes(ledger, ops, a1, a2)
    new_ledger.count.block_until_ready()
    assert pv._host_bool_value(new_ledger.oom) == pv._host_bool_value(ledger.oom)
    assert pv._host_bool_value(new_ledger.corrupt) == pv._host_bool_value(ledger.corrupt)
    assert pv._host_int_value(new_ledger.count) == pv._host_int_value(ledger.count)
    assert int(ids[0]) == pv.ZERO_PTR
    assert int(ids[1]) == 0


@pytest.mark.m1
def test_kernel_add_oom():
    cap = 4
    manifest = _small_add_manifest(cap)
    new_manifest, _ = pv.kernel_add(manifest, jnp.int32(3))
    assert bool(new_manifest.oom)
    assert int(new_manifest.active_count) == cap


@pytest.mark.m1
def test_kernel_add_partial_allocation_sets_oom():
    cap = 6
    manifest = _add_manifest_two_suc(cap)
    new_manifest, _ = pv.kernel_add(manifest, jnp.int32(4))
    assert bool(new_manifest.oom)
    assert int(new_manifest.active_count) == cap


@pytest.mark.m1
def test_kernel_add_oom_preserves_null_slot():
    cap = 4
    manifest = _small_add_manifest(cap)
    new_manifest, _ = pv.kernel_add(manifest, jnp.int32(3))
    assert int(new_manifest.opcode[0]) == pv.OP_NULL
    assert int(new_manifest.arg1[0]) == 0
    assert int(new_manifest.arg2[0]) == 0


@pytest.mark.m1
def test_kernel_mul_oom():
    cap = 4
    manifest = _small_mul_manifest(cap)
    new_manifest, _ = pv.kernel_mul(manifest, jnp.int32(3))
    assert bool(new_manifest.oom)
    assert int(new_manifest.active_count) == cap


@pytest.mark.m1
def test_kernel_mul_canonicalizes_add_args():
    manifest = _mul_manifest_commutative(8)
    pre_count = int(manifest.active_count)
    new_manifest, _ = pv.kernel_mul(manifest, jnp.int32(4))
    found = False
    for idx in range(pre_count, int(new_manifest.active_count)):
        if int(new_manifest.opcode[idx]) != pv.OP_ADD:
            continue
        found = True
        assert int(new_manifest.arg1[idx]) <= int(new_manifest.arg2[idx])
    assert found


@pytest.mark.m3
def test_op_interact_oom():
    cap = 4
    ops = jnp.zeros(cap, dtype=jnp.int32)
    a1 = jnp.zeros(cap, dtype=jnp.int32)
    a2 = jnp.zeros(cap, dtype=jnp.int32)
    rank = jnp.full(cap, pv.RANK_COLD, dtype=jnp.int8)
    ops = ops.at[1].set(pv.OP_ZERO)
    ops = ops.at[2].set(pv.OP_SUC)
    a1 = a1.at[2].set(1)
    ops = ops.at[3].set(pv.OP_ADD)
    a1 = a1.at[3].set(2)
    a2 = a2.at[3].set(2)
    rank = rank.at[3].set(pv.RANK_HOT)
    arena = pv.Arena(
        ops,
        a1,
        a2,
        rank,
        jnp.array(cap, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
        jnp.zeros(3, dtype=jnp.uint32),
    )
    new_arena = pv.op_interact(arena)
    assert bool(new_arena.oom)
    assert int(new_arena.count) == cap


@pytest.mark.m3
def test_op_interact_partial_allocation_sets_oom():
    cap = 7
    ops = jnp.zeros(cap, dtype=jnp.int32)
    a1 = jnp.zeros(cap, dtype=jnp.int32)
    a2 = jnp.zeros(cap, dtype=jnp.int32)
    rank = jnp.full(cap, pv.RANK_COLD, dtype=jnp.int8)
    ops = ops.at[1].set(pv.OP_ZERO)
    ops = ops.at[2].set(pv.OP_SUC)
    a1 = a1.at[2].set(1)
    ops = ops.at[3].set(pv.OP_SUC)
    a1 = a1.at[3].set(1)
    ops = ops.at[4].set(pv.OP_ADD)
    a1 = a1.at[4].set(2)
    a2 = a2.at[4].set(3)
    ops = ops.at[5].set(pv.OP_ADD)
    a1 = a1.at[5].set(3)
    a2 = a2.at[5].set(2)
    rank = rank.at[4].set(pv.RANK_HOT).at[5].set(pv.RANK_HOT)
    arena = pv.Arena(
        ops,
        a1,
        a2,
        rank,
        jnp.array(6, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
        jnp.zeros(3, dtype=jnp.uint32),
    )
    new_arena = pv.op_interact(arena)
    assert bool(new_arena.oom)
    assert int(new_arena.count) == cap


@pytest.mark.m3
def test_op_interact_canonicalizes_spawned_add():
    cap = 8
    ops = jnp.zeros(cap, dtype=jnp.int32)
    a1 = jnp.zeros(cap, dtype=jnp.int32)
    a2 = jnp.zeros(cap, dtype=jnp.int32)
    rank = jnp.full(cap, pv.RANK_COLD, dtype=jnp.int8)
    ops = ops.at[1].set(pv.OP_ZERO)
    ops = ops.at[4].set(pv.OP_SUC)
    a1 = a1.at[4].set(1)
    ops = ops.at[2].set(pv.OP_SUC)
    a1 = a1.at[2].set(4)
    ops = ops.at[3].set(pv.OP_ADD)
    a1 = a1.at[3].set(2)
    a2 = a2.at[3].set(2)
    rank = rank.at[3].set(pv.RANK_HOT)
    arena = pv.Arena(
        ops,
        a1,
        a2,
        rank,
        jnp.array(5, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
        jnp.zeros(3, dtype=jnp.uint32),
    )
    new_arena = pv.op_interact(arena)
    add_idx = int(arena.count)
    assert int(new_arena.opcode[add_idx]) == pv.OP_ADD
    assert int(new_arena.arg1[add_idx]) == 2
    assert int(new_arena.arg2[add_idx]) == 4


@pytest.mark.m2
def test_validate_stratum_no_within_refs_jax_ok():
    ledger = pv.init_ledger()
    ledger = ledger._replace(
        arg1=ledger.arg1.at[2].set(1),
        arg2=ledger.arg2.at[2].set(0),
        count=jnp.array(3, dtype=jnp.int32),
    )
    stratum = pv.Stratum(
        start=jnp.array(2, dtype=jnp.int32),
        count=jnp.array(1, dtype=jnp.int32),
    )
    assert bool(pv.validate_stratum_no_within_refs_jax(ledger, stratum))


@pytest.mark.m2
def test_validate_stratum_no_within_refs_jax_bad():
    ledger = pv.init_ledger()
    ledger = ledger._replace(
        arg1=ledger.arg1.at[2].set(2),
        arg2=ledger.arg2.at[2].set(0),
        count=jnp.array(3, dtype=jnp.int32),
    )
    stratum = pv.Stratum(
        start=jnp.array(2, dtype=jnp.int32),
        count=jnp.array(1, dtype=jnp.int32),
    )
    assert not bool(pv.validate_stratum_no_within_refs_jax(ledger, stratum))


@pytest.mark.m3
def test_compact_candidates_preserves_order():
    enabled = jnp.array([0, 1, 0, 1, 1, 0], dtype=jnp.int32)
    opcode = jnp.arange(enabled.shape[0], dtype=jnp.int32)
    candidates = pv.CandidateBuffer(
        enabled=enabled,
        opcode=opcode,
        arg1=opcode + 10,
        arg2=opcode + 20,
    )
    compacted, count, idx = pv.compact_candidates_with_index(candidates)
    expected = jnp.array([1, 3, 4], dtype=jnp.int32)
    count_int = int(count)
    assert count_int == 3
    assert bool(jnp.all(idx[:count_int] == expected))
    assert bool(jnp.all(compacted.opcode[:count_int] == opcode[expected]))


@pytest.mark.m1
def test_intern_nodes_opcode_bucket():
    ledger = pv.init_ledger()
    ops = jnp.array([pv.OP_ADD, pv.OP_MUL], dtype=jnp.int32)
    a1 = jnp.array([pv.ZERO_PTR, pv.ZERO_PTR], dtype=jnp.int32)
    a2 = jnp.array([pv.ZERO_PTR, pv.ZERO_PTR], dtype=jnp.int32)
    ids, ledger = pv.intern_nodes(ledger, ops, a1, a2)
    ids2, _ = pv.intern_nodes(ledger, ops, a1, a2)
    assert int(ids[0]) != int(ids[1])
    assert int(ids2[0]) == int(ids[0])
    assert int(ids2[1]) == int(ids[1])


@pytest.mark.m1
def test_optimize_ptr_zero_rules():
    vm = pv.PrismVM()
    zero = vm.cons(pv.OP_ZERO, pv._manifest_ptr(0), pv._manifest_ptr(0))
    one = vm.cons(pv.OP_SUC, zero, pv._manifest_ptr(0))

    add_left = vm.cons(pv.OP_ADD, zero, one)
    ptr, reason = pv.optimize_ptr(vm.manifest, jnp.int32(add_left))
    assert pv._host_int_value(reason) == 1
    assert pv._host_int_value(ptr) == int(one)

    add_right = vm.cons(pv.OP_ADD, one, zero)
    ptr, reason = pv.optimize_ptr(vm.manifest, jnp.int32(add_right))
    assert pv._host_int_value(reason) == 1
    assert pv._host_int_value(ptr) == int(one)

    mul_left = vm.cons(pv.OP_MUL, zero, one)
    ptr, reason = pv.optimize_ptr(vm.manifest, jnp.int32(mul_left))
    assert pv._host_int_value(reason) == 2
    assert pv._host_int_value(ptr) == pv.ZERO_PTR

    mul_right = vm.cons(pv.OP_MUL, one, zero)
    ptr, reason = pv.optimize_ptr(vm.manifest, jnp.int32(mul_right))
    assert pv._host_int_value(reason) == 2
    assert pv._host_int_value(ptr) == pv.ZERO_PTR


@pytest.mark.m1
def test_trace_cache_refresh_after_eval():
    vm = pv.PrismVM()
    tokens = re.findall(r"\(|\)|[a-z]+", "(add (suc zero) (suc zero))")
    ptr = vm.parse(tokens)
    pre_count = pv._host_int_value(vm.manifest.active_count)
    vm.eval(ptr)
    post_count = pv._host_int_value(vm.manifest.active_count)
    assert post_count > pre_count
    for idx in range(pre_count, post_count):
        op = pv._host_int_value(vm.manifest.opcode[idx])
        a1 = pv._host_int_value(vm.manifest.arg1[idx])
        a2 = pv._host_int_value(vm.manifest.arg2[idx])
        a1, a2 = pv._key_order_commutative_host(op, a1, a2)
        sig = (op, pv._manifest_ptr(a1), pv._manifest_ptr(a2))
        cached = vm.trace_cache.get(sig)
        assert cached is not None
        cached_op = pv._host_int_value(vm.manifest.opcode[int(cached)])
        cached_a1 = pv._host_int_value(vm.manifest.arg1[int(cached)])
        cached_a2 = pv._host_int_value(vm.manifest.arg2[int(cached)])
        cached_a1, cached_a2 = pv._key_order_commutative_host(
            cached_op, cached_a1, cached_a2
        )
        assert (cached_op, pv._manifest_ptr(cached_a1), pv._manifest_ptr(cached_a2)) == sig


@pytest.mark.m1
def test_baseline_eval_commutative_nodes_are_canonicalized():
    vm = pv.PrismVM()
    tokens = re.findall(r"\(|\)|[a-z]+", "(mul (suc (suc zero)) (suc zero))")
    ptr = vm.parse(tokens)
    pre_count = pv._host_int_value(vm.manifest.active_count)
    vm.eval(ptr)
    post_count = pv._host_int_value(vm.manifest.active_count)
    assert post_count > pre_count
    found = False
    for idx in range(pre_count, post_count):
        op = pv._host_int_value(vm.manifest.opcode[idx])
        if op not in (pv.OP_ADD, pv.OP_MUL):
            continue
        found = True
        before = pv._host_int_value(vm.manifest.active_count)
        a1 = pv._host_int_value(vm.manifest.arg1[idx])
        a2 = pv._host_int_value(vm.manifest.arg2[idx])
        ptr_rev = vm.cons(op, pv._manifest_ptr(a2), pv._manifest_ptr(a1))
        after = pv._host_int_value(vm.manifest.active_count)
        assert after == before
        c1, c2 = pv._key_order_commutative_host(op, a1, a2)
        assert vm.trace_cache.get((op, pv._manifest_ptr(c1), pv._manifest_ptr(c2))) == ptr_rev
    assert found


@pytest.mark.m1
def test_mul_commutative_interning():
    ledger = pv.init_ledger()
    ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_id = int(ids[0])
    ids1, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    ids2, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    assert int(ids1[0]) == int(ids2[0])


@pytest.mark.m1
def test_add_commutative_interning():
    ledger = pv.init_ledger()
    ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_id = int(ids[0])
    ids1, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    ids2, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    assert int(ids1[0]) == int(ids2[0])


@pytest.mark.m1
def test_mul_commutative_baseline_cons():
    vm = pv.PrismVM()
    zero = pv._manifest_ptr(pv.ZERO_PTR)
    one = vm.cons(pv.OP_SUC, zero, pv._manifest_ptr(0))
    mul1 = vm.cons(pv.OP_MUL, zero, one)
    mul2 = vm.cons(pv.OP_MUL, one, zero)
    assert mul1 == mul2


@pytest.mark.m1
def test_add_commutative_baseline_cons():
    vm = pv.PrismVM()
    zero = pv._manifest_ptr(pv.ZERO_PTR)
    one = vm.cons(pv.OP_SUC, zero, pv._manifest_ptr(0))
    add1 = vm.cons(pv.OP_ADD, zero, one)
    add2 = vm.cons(pv.OP_ADD, one, zero)
    assert add1 == add2
