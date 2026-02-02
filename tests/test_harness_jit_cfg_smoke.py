import jax.numpy as jnp

import prism_vm as pv
from tests import harness


def test_harness_jit_cfg_smoke():
    arena = pv.init_arena()
    ledger = pv.init_ledger()
    root = jnp.int32(0)
    frontier = jnp.array([0], dtype=jnp.int32)

    op_interact = harness.make_op_interact_jit_cfg()
    arena_out = op_interact(arena)
    assert arena_out is not None

    cycle = harness.make_cycle_jit_cfg(
        sort_cfg=pv.ArenaSortConfig(do_sort=False)
    )
    arena_out, root_out = cycle(arena, root)
    assert arena_out is not None
    assert root_out is not None

    cycle_intrinsic = harness.make_cycle_intrinsic_jit_cfg()
    ledger_out, frontier_out = cycle_intrinsic(ledger, frontier)
    assert ledger_out is not None
    assert frontier_out is not None
