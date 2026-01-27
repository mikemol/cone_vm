import jax.numpy as jnp
import pytest

import ic_vm as ic

pytestmark = pytest.mark.m6


def test_ic_port_encode_decode_roundtrip():
    node = jnp.uint32(7)
    port = jnp.uint32(2)
    ptr = ic.encode_port(node, port)
    out_node, out_port = ic.decode_port(ptr)
    assert int(out_node) == int(node)
    assert int(out_port) == int(port)


def test_ic_init_free_stack():
    state = ic.ic_init(4)
    assert state.free_top == 4
    assert list(map(int, state.free_stack)) == [3, 2, 1, 0]
    assert state.ports.shape == (4, 3)


def test_ic_find_active_pairs_principal_link():
    state = ic.ic_init(2)
    state = ic.ic_wire(state, 0, 0, 1, 0)
    pairs, active = ic.ic_find_active_pairs(state)
    assert bool(active[0])
    assert int(pairs[0]) == 0
