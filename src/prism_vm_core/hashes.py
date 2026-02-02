from dataclasses import dataclass
import jax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_vm_core.domains import _host_int_value
from prism_vm_core.ontology import OP_ADD, OP_MUL

# dataflow-bundle: a1, a2, ops


_TEST_GUARDS = _jax_safe.TEST_GUARDS


@dataclass(frozen=True)
class _RootStructArgs:
    ops: object
    a1: object
    a2: object


def _root_struct_hash_host(ops, a1, a2, root_i, count, limit):
    if root_i <= 0 or root_i >= count:
        return 0
    bundle = _RootStructArgs(ops=ops, a1=a1, a2=a2)
    cache = {}
    visiting = set()

    def _hash(idx):
        if idx <= 0 or idx >= count:
            return 0
        if idx in cache:
            return cache[idx]
        if idx in visiting:
            return 0x9E3779B9
        if len(cache) >= int(limit):
            return 0
        visiting.add(idx)
        op = int(bundle.ops[idx])
        h1 = _hash(int(bundle.a1[idx]))
        h2 = _hash(int(bundle.a2[idx]))
        if op in (OP_ADD, OP_MUL) and h2 < h1:
            h1, h2 = h2, h1
        h = (op * 1315423911) ^ (h1 + 0x9E3779B9) ^ ((h2 << 1) & 0xFFFFFFFF)
        visiting.remove(idx)
        cache[idx] = h & 0xFFFFFFFF
        return cache[idx]

    return int(_hash(int(root_i))) & 0xFFFFFFFF


def _arena_root_hash_host(arena, root_ptr, limit=64):
    if not _TEST_GUARDS:
        return 0
    root_i = _host_int_value(root_ptr)
    if root_i == 0:
        return 0
    count = _host_int_value(arena.count)
    if count <= 0 or root_i >= count:
        return 0
    ops = jax.device_get(arena.opcode[:count])
    a1 = jax.device_get(arena.arg1[:count])
    a2 = jax.device_get(arena.arg2[:count])
    return _root_struct_hash_host(ops, a1, a2, root_i, count, limit)


def _ledger_root_hash_host(ledger, root_id, limit=64):
    if not _TEST_GUARDS:
        return 0
    root_i = _host_int_value(root_id)
    if root_i == 0:
        return 0
    count = _host_int_value(ledger.count)
    if count <= 0 or root_i >= count:
        return 0
    ops = jax.device_get(ledger.opcode[:count])
    a1 = jax.device_get(ledger.arg1[:count])
    a2 = jax.device_get(ledger.arg2[:count])
    return _root_struct_hash_host(ops, a1, a2, root_i, count, limit)


def _ledger_roots_hash_host(ledger, root_ids, limit=64):
    roots = jax.device_get(root_ids)
    return tuple(_ledger_root_hash_host(ledger, int(r), limit) for r in roots)


__all__ = [
    "_arena_root_hash_host",
    "_ledger_root_hash_host",
    "_ledger_roots_hash_host",
    "_root_struct_hash_host",
]
