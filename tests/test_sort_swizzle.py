import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = [
    pytest.mark.m3,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


def _arena_with_edges():
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_ADD).at[3].set(pv.OP_SUC),
        arg1=arena.arg1.at[2].set(3).at[3].set(1),
        arg2=arena.arg2.at[2].set(1).at[3].set(0),
        rank=arena.rank.at[1].set(pv.RANK_COLD)
        .at[2].set(pv.RANK_HOT)
        .at[3].set(pv.RANK_COLD),
        count=jnp.array(4, dtype=jnp.int32),
    )
    return arena


def test_swizzle_preserves_edges():
    assert hasattr(pv, "op_sort_and_swizzle"), "op_sort_and_swizzle missing"
    arena = _arena_with_edges()
    sorted_arena, inv_perm = pv.op_sort_and_swizzle_with_perm(arena)
    add_new = int(inv_perm[2])
    assert int(sorted_arena.opcode[0]) == pv.OP_NULL
    assert int(sorted_arena.opcode[add_new]) == pv.OP_ADD
    assert int(sorted_arena.arg1[add_new]) == int(inv_perm[3])
    assert int(sorted_arena.arg2[add_new]) == int(inv_perm[1])
    assert int(sorted_arena.count) == int(arena.count)


def test_swizzle_null_pointer_stays_zero():
    assert hasattr(pv, "op_sort_and_swizzle"), "op_sort_and_swizzle missing"
    arena = _arena_with_edges()
    sorted_arena, inv_perm = pv.op_sort_and_swizzle_with_perm(arena)
    suc_new = int(inv_perm[3])
    assert int(sorted_arena.opcode[0]) == pv.OP_NULL
    assert int(sorted_arena.opcode[suc_new]) == pv.OP_SUC
    assert int(sorted_arena.arg2[suc_new]) == 0


def test_sort_swizzle_root_remap():
    assert hasattr(pv, "op_sort_and_swizzle_with_perm"), "op_sort_and_swizzle_with_perm missing"
    arena = _arena_with_edges()
    sorted_arena, inv_perm = pv.op_sort_and_swizzle_with_perm(arena)
    root_old = jnp.array(3, dtype=jnp.int32)
    root_new = jnp.where(root_old != 0, inv_perm[root_old], 0)
    assert int(root_new) != 0
    assert int(sorted_arena.opcode[int(root_new)]) == pv.OP_SUC
    assert int(sorted_arena.arg1[root_new]) == int(inv_perm[1])


def test_sort_swizzle_preserves_count_and_oom():
    assert hasattr(pv, "op_sort_and_swizzle_with_perm"), "op_sort_and_swizzle_with_perm missing"
    arena = _arena_with_edges()
    arena = arena._replace(oom=jnp.array(True, dtype=jnp.bool_))
    sorted_arena, _ = pv.op_sort_and_swizzle_with_perm(arena)
    assert int(sorted_arena.count) == int(arena.count)
    assert bool(jax.device_get(sorted_arena.oom)) == bool(jax.device_get(arena.oom))


def test_swizzle_does_not_create_new_edges():
    assert hasattr(pv, "op_sort_and_swizzle_with_perm"), "op_sort_and_swizzle_with_perm missing"
    arena = _arena_with_edges()
    sorted_arena, inv_perm = pv.op_sort_and_swizzle_with_perm(arena)
    perm = pv._invert_perm(inv_perm)
    count = int(arena.count)
    old_ops = jax.device_get(arena.opcode)
    old_a1 = jax.device_get(arena.arg1)
    old_a2 = jax.device_get(arena.arg2)
    new_ops = jax.device_get(sorted_arena.opcode)
    new_a1 = jax.device_get(sorted_arena.arg1)
    new_a2 = jax.device_get(sorted_arena.arg2)
    perm_h = jax.device_get(perm)
    inv_h = jax.device_get(inv_perm)
    for new_idx in range(count):
        old_idx = int(perm_h[new_idx])
        exp_a1 = 0 if int(old_a1[old_idx]) == 0 else int(inv_h[old_a1[old_idx]])
        exp_a2 = 0 if int(old_a2[old_idx]) == 0 else int(inv_h[old_a2[old_idx]])
        assert int(new_ops[new_idx]) == int(old_ops[old_idx])
        assert int(new_a1[new_idx]) == exp_a1
        assert int(new_a2[new_idx]) == exp_a2
