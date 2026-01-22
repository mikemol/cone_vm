import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m4


def test_rank_classification():
    assert hasattr(pv, "op_rank"), "op_rank missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_ADD).at[3].set(pv.OP_ZERO)
    )
    ranked = pv.op_rank(arena)
    assert int(ranked.rank[2]) == pv.RANK_HOT
    assert int(ranked.rank[3]) == pv.RANK_COLD


def test_rank_null_is_free():
    assert hasattr(pv, "op_rank"), "op_rank missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[4].set(pv.OP_NULL),
        rank=arena.rank.at[4].set(pv.RANK_HOT),
    )
    ranked = pv.op_rank(arena)
    assert int(ranked.rank[4]) == pv.RANK_FREE
