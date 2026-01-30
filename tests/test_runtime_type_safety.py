import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

intern_nodes = harness.intern_nodes

pytestmark = pytest.mark.m3


def _small_arena(size: int = 16) -> pv.Arena:
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg1 = jnp.zeros(size, dtype=jnp.int32)
    arg2 = jnp.zeros(size, dtype=jnp.int32)
    rank = jnp.zeros(size, dtype=jnp.int8)
    count = jnp.array(2, dtype=jnp.int32)
    oom = jnp.array(False, dtype=jnp.bool_)
    servo = jnp.zeros(3, dtype=jnp.uint32)
    return pv.Arena(
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
        rank=rank,
        count=count,
        oom=oom,
        servo=servo,
    )


def test_intern_nodes_prefix_compare_handles_scalar_available():
    # Exercises prefix <= available; if available is not scalar-like, this fails.
    ledger = pv.init_ledger()
    ops = jnp.array([pv.OP_SUC], dtype=jnp.int32)
    a1 = jnp.array([pv.ZERO_PTR], dtype=jnp.int32)
    a2 = jnp.array([0], dtype=jnp.int32)
    ids, new_ledger = intern_nodes(ledger, ops, a1, a2)
    new_ledger.count.block_until_ready()
    assert ids.shape == (1,)
    assert int(new_ledger.count) >= int(ledger.count)


def test_blocked_perm_shift_handles_arrays():
    # Exercises bit-shift on array scalars; should not error on ndarray inputs.
    arena = _small_arena()
    morton = jnp.arange(arena.rank.shape[0], dtype=jnp.uint32)
    perm = pv._blocked_perm(arena, block_size=4, morton=morton, active_count=8)
    assert perm.shape == (arena.rank.shape[0],)
    assert perm.dtype == jnp.int32
