import jax.numpy as jnp

import ic_vm as ic
from tests import harness


def test_ic_compact_active_pairs_result_smoke():
    state = ic.ic_init(4)
    node_type = state.node_type
    node_type = node_type.at[1].set(ic.TYPE_CON)
    node_type = node_type.at[2].set(ic.TYPE_CON)
    state = state._replace(node_type=node_type)
    state = ic.ic_wire(state, 1, int(ic.PORT_PRINCIPAL), 2, int(ic.PORT_PRINCIPAL))

    compact_jit = harness.make_ic_compact_active_pairs_result_jit()
    result, active = compact_jit(state)

    assert int(result.count) == 1
    assert int(result.idx[0]) == 1
    assert bool(active[1])
