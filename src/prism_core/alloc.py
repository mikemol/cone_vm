"""Shared allocator backends (device + host).

These helpers assume a state object with fields:
  free_stack, free_top, oom, corrupt
and a `_replace` method (e.g. NamedTuple).
"""

from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class AllocConfig:
    """Allocator DI bundle (shared)."""

    set_oom_on_fail: bool = False


DEFAULT_ALLOC_CONFIG = AllocConfig()


def alloc_raw(state, count: int) -> Tuple[object, jnp.ndarray, jnp.ndarray]:
    if count > state.free_stack.shape[0]:
        return state, jnp.zeros((count,), dtype=jnp.uint32), jnp.bool_(False)
    top = state.free_top.astype(jnp.int32)
    ok = (top >= count) & (~state.oom) & (~state.corrupt)

    def _do(s):
        start = top - count
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (count,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32), jnp.bool_(True)

    def _fail(s):
        return s, jnp.zeros((count,), dtype=jnp.uint32), jnp.bool_(False)

    return jax.lax.cond(ok, _do, _fail, state)


def alloc_pad(state, count: int) -> Tuple[object, jnp.ndarray, jnp.ndarray]:
    state2, ids, ok = alloc_raw(state, count)
    if count == 0:
        ids4 = jnp.zeros((4,), dtype=jnp.uint32)
    elif count == 1:
        ids4 = jnp.concatenate([ids, jnp.zeros((3,), dtype=jnp.uint32)], axis=0)
    elif count == 2:
        ids4 = jnp.concatenate([ids, jnp.zeros((2,), dtype=jnp.uint32)], axis=0)
    elif count == 4:
        ids4 = ids
    else:
        ids4 = jnp.zeros((4,), dtype=jnp.uint32)
        ok = jnp.bool_(False)
    return state2, ids4, ok


def init_nodes_jax(state, ids4: jnp.ndarray, count: jnp.ndarray, node_type: jnp.uint8):
    mask = jnp.arange(4, dtype=jnp.int32) < count.astype(jnp.int32)
    node_type_curr = state.node_type[ids4]
    node_type_update = jnp.where(mask, node_type, node_type_curr)
    node_type_arr = state.node_type.at[ids4].set(node_type_update)
    ports_curr = state.ports[ids4]
    ports_update = jnp.where(mask[:, None], jnp.uint32(0), ports_curr)
    ports_arr = state.ports.at[ids4].set(ports_update)
    return state._replace(node_type=node_type_arr, ports=ports_arr)


@partial(jax.jit, static_argnames=("set_oom_on_fail",))
def alloc_jax(state, count: jnp.ndarray, node_type: jnp.uint8, set_oom_on_fail: bool = False):
    """Device-only allocator for construction (no host sync)."""
    c = jnp.asarray(count, dtype=jnp.int32)
    idx = jnp.where(
        c == 0,
        0,
        jnp.where(c == 1, 1, jnp.where(c == 2, 2, jnp.where(c == 4, 3, 4))),
    ).astype(jnp.int32)

    def _case0(s):
        return alloc_pad(s, 0)

    def _case1(s):
        return alloc_pad(s, 1)

    def _case2(s):
        return alloc_pad(s, 2)

    def _case4(s):
        return alloc_pad(s, 4)

    def _bad(s):
        return s, jnp.zeros((4,), dtype=jnp.uint32), jnp.bool_(False)

    def _run(s):
        return jax.lax.switch(idx, (_case0, _case1, _case2, _case4, _bad), s)

    def _halt(s):
        return s, jnp.zeros((4,), dtype=jnp.uint32), jnp.bool_(False)

    state2, ids4, ok = jax.lax.cond(state.corrupt, _halt, _run, state)
    if set_oom_on_fail:
        state2 = state2._replace(oom=state2.oom | ((~ok) & (~state.corrupt)))

    def _do_init(s):
        return init_nodes_jax(s, ids4, c, node_type)

    do_init = ok & (c > 0)
    state2 = jax.lax.cond(do_init, _do_init, lambda s: s, state2)
    return state2, ids4, ok


def alloc_jax_cfg(state, count: jnp.ndarray, node_type: jnp.uint8, *, cfg: AllocConfig = DEFAULT_ALLOC_CONFIG):
    """alloc_jax wrapper for a fixed AllocConfig."""
    return alloc_jax(
        state,
        count,
        node_type,
        set_oom_on_fail=cfg.set_oom_on_fail,
    )


def alloc2(state):
    top = state.free_top.astype(jnp.int32)
    ok = (top >= 2) & (~state.oom) & (~state.corrupt)

    def _do(s):
        start = top - 2
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (2,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32)

    def _fail(s):
        def _keep(s_in):
            return s_in

        def _oom(s_in):
            return s_in._replace(oom=jnp.bool_(True))

        s2 = jax.lax.cond(s.corrupt, _keep, _oom, s)
        return s2, jnp.zeros((2,), dtype=jnp.uint32)

    return jax.lax.cond(ok, _do, _fail, state)


def alloc2_cfg(state, *, cfg: AllocConfig = DEFAULT_ALLOC_CONFIG):
    """alloc2 wrapper for a fixed AllocConfig."""
    return alloc2(state)


def alloc4(state):
    top = state.free_top.astype(jnp.int32)
    ok = (top >= 4) & (~state.oom) & (~state.corrupt)

    def _do(s):
        start = top - 4
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (4,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32)

    def _fail(s):
        def _keep(s_in):
            return s_in

        def _oom(s_in):
            return s_in._replace(oom=jnp.bool_(True))

        s2 = jax.lax.cond(s.corrupt, _keep, _oom, s)
        return s2, jnp.zeros((4,), dtype=jnp.uint32)

    return jax.lax.cond(ok, _do, _fail, state)


def alloc4_cfg(state, *, cfg: AllocConfig = DEFAULT_ALLOC_CONFIG):
    """alloc4 wrapper for a fixed AllocConfig."""
    return alloc4(state)


def free2(state, nodes: jnp.ndarray):
    top = state.free_top.astype(jnp.int32)
    cap = state.free_stack.shape[0]
    ok = (top + 2) <= cap

    def _do(s):
        fs = s.free_stack
        fs = fs.at[top + 0].set(nodes[0])
        fs = fs.at[top + 1].set(nodes[1])
        return s._replace(free_stack=fs, free_top=jnp.uint32(top + 2))

    def _fail(s):
        def _keep(s_in):
            return s_in

        def _corrupt(s_in):
            return s_in._replace(corrupt=jnp.bool_(True))

        return jax.lax.cond(s.corrupt, _keep, _corrupt, s)

    return jax.lax.cond(ok & (~state.oom) & (~state.corrupt), _do, _fail, state)


def free2_cfg(state, nodes: jnp.ndarray, *, cfg: AllocConfig = DEFAULT_ALLOC_CONFIG):
    """free2 wrapper for a fixed AllocConfig."""
    return free2(state, nodes)


def host_flag(value: jnp.ndarray) -> bool:
    return bool(jax.device_get(value))


def alloc_nodes(state, count: int) -> Tuple[object, jnp.ndarray]:
    n = int(count)
    if n == 0:
        return state, jnp.zeros((0,), dtype=jnp.uint32)
    if host_flag(state.corrupt):
        return state, jnp.zeros((n,), dtype=jnp.uint32)
    free_top = int(state.free_top)
    if free_top < n or host_flag(state.oom):
        return state._replace(oom=jnp.bool_(True)), jnp.zeros((n,), dtype=jnp.uint32)
    idx = state.free_stack[free_top - n:free_top]
    free_top = free_top - n
    return state._replace(free_top=jnp.uint32(free_top)), idx


def alloc_host(state, count: int, node_type: jnp.uint8) -> Tuple[object, jnp.ndarray]:
    state, nodes = alloc_nodes(state, count)
    if nodes.size == 0 or host_flag(state.oom) or host_flag(state.corrupt):
        return state, nodes
    node_type_arr = state.node_type.at[nodes].set(node_type)
    ports = state.ports.at[nodes].set(jnp.uint32(0))
    return state._replace(node_type=node_type_arr, ports=ports), nodes


def free_nodes(state, nodes: jnp.ndarray):
    if nodes.size == 0:
        return state
    if host_flag(state.corrupt):
        return state
    count = int(nodes.shape[0])
    free_top = int(state.free_top)
    cap = int(state.free_stack.shape[0])
    if free_top + count > cap:
        return state._replace(corrupt=jnp.bool_(True))
    free_stack = state.free_stack
    free_stack = free_stack.at[free_top:free_top + count].set(nodes)
    return state._replace(free_stack=free_stack, free_top=jnp.uint32(free_top + count))


__all__ = [
    "AllocConfig",
    "DEFAULT_ALLOC_CONFIG",
    "alloc_raw",
    "alloc_pad",
    "init_nodes_jax",
    "alloc_jax",
    "alloc_jax_cfg",
    "alloc2",
    "alloc2_cfg",
    "alloc4",
    "alloc4_cfg",
    "free2",
    "free2_cfg",
    "host_flag",
    "alloc_nodes",
    "alloc_host",
    "free_nodes",
]
