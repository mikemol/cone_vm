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
    a2 = a2.at[3].set(1)
    return pv.Manifest(
        ops,
        a1,
        a2,
        jnp.array(cap, dtype=jnp.int32),
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
        vm._cons_raw(pv.OP_SUC, pv.ZERO_PTR, 0)
    assert bool(vm.manifest.oom)


@pytest.mark.m3
def test_arena_capacity_guard():
    vm = pv.PrismVM_BSP_Legacy()
    cap = int(vm.arena.opcode.shape[0])
    vm.arena = vm.arena._replace(count=jnp.array(cap, dtype=jnp.int32))
    with pytest.raises(ValueError):
        vm._alloc(pv.OP_SUC, pv.ZERO_PTR, 0)
    assert bool(vm.arena.oom)


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
    assert bool(new_ledger.corrupt)
    assert not bool(new_ledger.oom)
    assert int(new_ledger.count) == int(ledger.count)
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
    assert bool(new_ledger.oom) == bool(ledger.oom)
    assert bool(new_ledger.corrupt) == bool(ledger.corrupt)
    assert int(new_ledger.count) == int(ledger.count)
    assert int(jnp.sum(ids)) == 0


@pytest.mark.m1
def test_intern_nodes_early_out_on_corrupt_returns_zero_ids():
    ledger = pv.init_ledger()
    ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))
    ops = jnp.array([pv.OP_ZERO, pv.OP_SUC], dtype=jnp.int32)
    a1 = jnp.array([0, pv.ZERO_PTR], dtype=jnp.int32)
    a2 = jnp.array([0, 0], dtype=jnp.int32)
    ids, new_ledger = pv.intern_nodes(ledger, ops, a1, a2)
    new_ledger.count.block_until_ready()
    assert bool(new_ledger.oom) == bool(ledger.oom)
    assert bool(new_ledger.corrupt) == bool(ledger.corrupt)
    assert int(new_ledger.count) == int(ledger.count)
    assert int(jnp.sum(ids)) == 0


@pytest.mark.m1
def test_kernel_add_oom():
    cap = 4
    manifest = _small_add_manifest(cap)
    new_manifest, _ = pv.kernel_add(manifest, jnp.int32(3))
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
    a2 = a2.at[3].set(1)
    rank = rank.at[3].set(pv.RANK_HOT)
    arena = pv.Arena(
        ops,
        a1,
        a2,
        rank,
        jnp.array(cap, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
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
    a2 = a2.at[3].set(1)
    rank = rank.at[3].set(pv.RANK_HOT)
    arena = pv.Arena(
        ops,
        a1,
        a2,
        rank,
        jnp.array(5, dtype=jnp.int32),
        jnp.array(False, dtype=jnp.bool_),
    )
    new_arena = pv.op_interact(arena)
    add_idx = int(arena.count)
    assert int(new_arena.opcode[add_idx]) == pv.OP_ADD
    assert int(new_arena.arg1[add_idx]) == 1
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
    zero = vm.cons(pv.OP_ZERO, 0, 0)
    one = vm.cons(pv.OP_SUC, zero, 0)

    add_left = vm.cons(pv.OP_ADD, zero, one)
    ptr, optimized = pv.optimize_ptr(vm.manifest, jnp.int32(add_left))
    assert bool(optimized)
    assert int(ptr) == one

    add_right = vm.cons(pv.OP_ADD, one, zero)
    ptr, optimized = pv.optimize_ptr(vm.manifest, jnp.int32(add_right))
    assert bool(optimized)
    assert int(ptr) == one

    mul_left = vm.cons(pv.OP_MUL, zero, one)
    ptr, optimized = pv.optimize_ptr(vm.manifest, jnp.int32(mul_left))
    assert bool(optimized)
    assert int(ptr) == pv.ZERO_PTR

    mul_right = vm.cons(pv.OP_MUL, one, zero)
    ptr, optimized = pv.optimize_ptr(vm.manifest, jnp.int32(mul_right))
    assert bool(optimized)
    assert int(ptr) == pv.ZERO_PTR


@pytest.mark.m1
def test_trace_cache_refresh_after_eval():
    vm = pv.PrismVM()
    tokens = re.findall(r"\(|\)|[a-z]+", "(add (suc zero) (suc zero))")
    ptr = vm.parse(tokens)
    pre_count = int(vm.manifest.active_count)
    vm.eval(ptr)
    post_count = int(vm.manifest.active_count)
    assert post_count > pre_count
    for idx in range(pre_count, post_count):
        op = int(vm.manifest.opcode[idx])
        a1 = int(vm.manifest.arg1[idx])
        a2 = int(vm.manifest.arg2[idx])
        a1, a2 = pv._canonicalize_commutative_host(op, a1, a2)
        sig = (op, a1, a2)
        cached = vm.trace_cache.get(sig)
        assert cached is not None
        cached_op = int(vm.manifest.opcode[cached])
        cached_a1 = int(vm.manifest.arg1[cached])
        cached_a2 = int(vm.manifest.arg2[cached])
        cached_a1, cached_a2 = pv._canonicalize_commutative_host(
            cached_op, cached_a1, cached_a2
        )
        assert (cached_op, cached_a1, cached_a2) == sig


@pytest.mark.m1
def test_baseline_eval_commutative_nodes_are_canonicalized():
    vm = pv.PrismVM()
    tokens = re.findall(r"\(|\)|[a-z]+", "(mul (suc (suc zero)) (suc zero))")
    ptr = vm.parse(tokens)
    pre_count = int(vm.manifest.active_count)
    vm.eval(ptr)
    post_count = int(vm.manifest.active_count)
    assert post_count > pre_count
    found = False
    for idx in range(pre_count, post_count):
        op = int(vm.manifest.opcode[idx])
        if op not in (pv.OP_ADD, pv.OP_MUL):
            continue
        found = True
        before = int(vm.manifest.active_count)
        a1 = int(vm.manifest.arg1[idx])
        a2 = int(vm.manifest.arg2[idx])
        ptr_rev = vm.cons(op, a2, a1)
        after = int(vm.manifest.active_count)
        assert after == before
        c1, c2 = pv._canonicalize_commutative_host(op, a1, a2)
        assert vm.trace_cache.get((op, c1, c2)) == ptr_rev
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
    zero = pv.ZERO_PTR
    one = vm.cons(pv.OP_SUC, zero, 0)
    mul1 = vm.cons(pv.OP_MUL, zero, one)
    mul2 = vm.cons(pv.OP_MUL, one, zero)
    assert mul1 == mul2


@pytest.mark.m1
def test_add_commutative_baseline_cons():
    vm = pv.PrismVM()
    zero = pv.ZERO_PTR
    one = vm.cons(pv.OP_SUC, zero, 0)
    add1 = vm.cons(pv.OP_ADD, zero, one)
    add2 = vm.cons(pv.OP_ADD, one, zero)
    assert add1 == add2
