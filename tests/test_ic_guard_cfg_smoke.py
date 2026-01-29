import jax
import jax.numpy as jnp

import ic_vm as ic


def test_graph_config_with_guard_smoke():
    state = ic.ic_init(4)
    guard_cfg = ic.ICGuardConfig()
    graph_cfg = ic.graph_config_with_guard(guard_cfg=guard_cfg)
    state = ic.ic_wire_jax_cfg(
        state,
        jnp.uint32(1),
        jnp.uint32(ic.PORT_PRINCIPAL),
        jnp.uint32(2),
        jnp.uint32(ic.PORT_PRINCIPAL),
        cfg=graph_cfg,
    )
    state.ports.block_until_ready()
    ptr = state.ports[1, ic.PORT_PRINCIPAL]
    assert int(ptr) != 0

