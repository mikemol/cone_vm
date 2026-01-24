import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m3


def test_interact_add_zero():
    assert hasattr(pv, "op_interact"), "op_interact missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_SUC).at[3].set(pv.OP_ADD),
        arg1=arena.arg1.at[2].set(1).at[3].set(1),
        arg2=arena.arg2.at[2].set(0).at[3].set(2),
        rank=arena.rank.at[3].set(pv.RANK_HOT),
        count=jnp.array(4, dtype=jnp.int32),
    )
    updated = pv.op_interact(arena)
    assert int(updated.opcode[3]) == pv.OP_SUC
    assert int(updated.arg1[3]) == 1
    assert int(updated.arg2[3]) == 0
    assert int(updated.count) == 4


def test_interact_add_suc():
    assert hasattr(pv, "op_interact"), "op_interact missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_SUC).at[4].set(pv.OP_SUC).at[3].set(pv.OP_ADD),
        arg1=arena.arg1.at[2].set(1).at[4].set(1).at[3].set(4),
        arg2=arena.arg2.at[2].set(0).at[4].set(0).at[3].set(2),
        rank=arena.rank.at[3].set(pv.RANK_HOT),
        count=jnp.array(5, dtype=jnp.int32),
    )
    updated = pv.op_interact(arena)
    assert int(updated.opcode[3]) == pv.OP_SUC
    assert int(updated.arg1[3]) == 5
    assert int(updated.arg2[3]) == 0
    assert int(updated.opcode[5]) == pv.OP_ADD
    assert int(updated.arg1[5]) == 1
    assert int(updated.arg2[5]) == 4
    assert int(updated.count) == 6


def test_interact_non_hot_noop():
    assert hasattr(pv, "op_interact"), "op_interact missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_SUC).at[3].set(pv.OP_ADD),
        arg1=arena.arg1.at[2].set(1).at[3].set(2),
        arg2=arena.arg2.at[2].set(0).at[3].set(1),
        rank=arena.rank.at[3].set(pv.RANK_COLD),
        count=jnp.array(4, dtype=jnp.int32),
    )
    updated = pv.op_interact(arena)
    assert bool(jnp.array_equal(updated.opcode, arena.opcode))
    assert bool(jnp.array_equal(updated.arg1, arena.arg1))
    assert bool(jnp.array_equal(updated.arg2, arena.arg2))
    assert int(updated.count) == 4
