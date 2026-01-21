import jax.numpy as jnp

import prism_vm as pv


def test_cycle_root_remap():
    assert hasattr(pv, "cycle"), "cycle missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_ADD).at[3].set(pv.OP_SUC),
        arg1=arena.arg1.at[2].set(1).at[3].set(1),
        arg2=arena.arg2.at[2].set(1).at[3].set(0),
        count=jnp.array(4, dtype=jnp.int32),
    )
    updated, new_root = pv.cycle(arena, 3)
    assert int(new_root) == 2
    assert int(updated.opcode[new_root]) == pv.OP_SUC
    assert int(updated.arg1[new_root]) == 1


def test_cycle_without_sort_keeps_root():
    assert hasattr(pv, "cycle"), "cycle missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[3].set(pv.OP_SUC),
        arg1=arena.arg1.at[3].set(1),
        arg2=arena.arg2.at[3].set(0),
        count=jnp.array(4, dtype=jnp.int32),
    )
    updated, new_root = pv.cycle(arena, 3, do_sort=False)
    assert int(new_root) == 3
    assert int(updated.opcode[3]) == pv.OP_SUC
