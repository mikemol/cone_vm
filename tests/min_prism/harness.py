import jax
import jax.numpy as jnp

import prism_vm as pv

_COMMUTATIVE_OPS = {pv.OP_ADD, pv.OP_MUL}


def _structural_keyer(ledger):
    count = int(pv._host_int_value(ledger.count))
    ops = jax.device_get(ledger.opcode[:count])
    a1 = jax.device_get(ledger.arg1[:count])
    a2 = jax.device_get(ledger.arg2[:count])
    cache = {}
    visiting = set()

    def key(idx):
        if idx in cache:
            return cache[idx]
        if idx in visiting:
            key_val = (-1, idx, ())
            cache[idx] = key_val
            return key_val
        if idx < 0 or idx >= count:
            return (pv.OP_NULL, (), ())
        visiting.add(idx)
        op = int(ops[idx])
        if op == pv.OP_NULL:
            key_val = (pv.OP_NULL, (), ())
        else:
            k1 = key(int(a1[idx]))
            k2 = key(int(a2[idx]))
            if op in _COMMUTATIVE_OPS and k2 < k1:
                k1, k2 = k2, k1
            key_val = (op, k1, k2)
        visiting.remove(idx)
        cache[idx] = key_val
        return key_val

    return key


def canon_state_ledger(ledger):
    keys = _ordered_keys(ledger)
    oom = bool(jax.device_get(ledger.oom))
    corrupt = bool(jax.device_get(ledger.corrupt))
    return tuple(keys), oom, corrupt


def project_ledger(ledger, max_id):
    key = _structural_keyer(ledger)
    ordered = _ordered_keys(ledger)
    keep = set(ordered[: int(max_id) + 1])
    keep = _closure_keys(keep, ordered)
    new_ledger, key_map = _rebuild_ledger_from_keys(keep)
    count = int(pv._host_int_value(ledger.count))
    mapping = {}
    for idx in range(count):
        mapping[idx] = key_map.get(key(idx), 0)
    new_ledger = new_ledger._replace(oom=ledger.oom, corrupt=ledger.corrupt)
    return new_ledger, mapping


def map_ids(ids, mapping):
    host_ids = jax.device_get(ids)
    mapped = [mapping.get(int(i), 0) for i in host_ids]
    return jnp.array(mapped, dtype=jnp.int32)


def structural_key_for_id(ledger, idx):
    key = _structural_keyer(ledger)
    return key(int(idx))


def _ordered_keys(ledger):
    count = int(pv._host_int_value(ledger.count))
    key = _structural_keyer(ledger)
    keys = {key(i) for i in range(count)}
    key_null = key(0)
    key_zero = key(1)
    keys.discard(key_null)
    keys.discard(key_zero)
    return [key_null, key_zero] + sorted(keys)


def _closure_keys(keep, ordered):
    ordered_set = set(ordered)
    changed = True
    while changed:
        changed = False
        for key in list(keep):
            op, k1, k2 = key
            for child in (k1, k2):
                if child in ordered_set and child not in keep:
                    keep.add(child)
                    changed = True
    return keep


def _rebuild_ledger_from_keys(keys):
    new_ledger = pv.init_ledger()
    key_map = {}
    key_null = (pv.OP_NULL, (), ())
    key_zero = (pv.OP_ZERO, key_null, key_null)
    key_map[key_null] = 0
    key_map[key_zero] = 1

    def emit(key):
        nonlocal new_ledger
        if key in key_map:
            return key_map[key]
        op, k1, k2 = key
        if op == pv.OP_NULL:
            return 0
        if op == pv.OP_ZERO:
            return 1
        a1 = emit(k1) if k1 in keys else 0
        a2 = emit(k2) if k2 in keys else 0
        ids_out, new_ledger = pv.intern_nodes(
            new_ledger,
            jnp.array([op], dtype=jnp.int32),
            jnp.array([a1], dtype=jnp.int32),
            jnp.array([a2], dtype=jnp.int32),
        )
        key_map[key] = int(ids_out[0])
        return key_map[key]

    for key in sorted(keys):
        emit(key)
    return new_ledger, key_map
