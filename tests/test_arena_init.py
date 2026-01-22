import jax.numpy as jnp

import pytest

import prism_vm as pv

pytestmark = pytest.mark.m4


def test_arena_init_zero_seed():
    assert hasattr(pv, "init_arena"), "init_arena missing"
    assert hasattr(pv, "OP_ZERO"), "OP_ZERO missing"
    arena = pv.init_arena()
    assert int(arena.opcode[1]) == pv.OP_ZERO
    assert int(arena.arg1[1]) == 0
    assert int(arena.arg2[1]) == 0
    assert int(arena.count) == 2


def test_arena_init_null_free():
    assert hasattr(pv, "init_arena"), "init_arena missing"
    assert hasattr(pv, "RANK_FREE"), "RANK_FREE missing"
    arena = pv.init_arena()
    ranks = arena.rank
    all_free = jnp.all(ranks == pv.RANK_FREE)
    assert bool(all_free)
