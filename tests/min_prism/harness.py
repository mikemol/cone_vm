import jax
import jax.numpy as jnp

import prism_vm as pv

_COMMUTATIVE_OPS = {pv.OP_ADD, pv.OP_MUL}


def i32(values):
    return jnp.asarray(values, dtype=jnp.int32)


def i32v(*values):
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = tuple(values[0])
    return i32(list(values))


def intern_nodes(ledger, ops, a1s, a2s, **kwargs):
    return pv.intern_nodes(ledger, i32(ops), i32(a1s), i32(a2s), **kwargs)


def intern1(ledger, op, a1, a2, **kwargs):
    ids, ledger = intern_nodes(ledger, [op], [a1], [a2], **kwargs)
    return int(ids[0]), ledger


def committed_ids(values):
    return pv._committed_ids(i32(values))


def make_cycle_intrinsic_jit(**kwargs):
    """Build a jitted intrinsic cycle entrypoint with fixed DI."""
    return pv.cycle_intrinsic_jit(**kwargs)


def make_cycle_candidates_jit(**kwargs):
    """Build a jitted CNF-2 cycle entrypoint with fixed DI."""
    return pv.cycle_candidates_jit(**kwargs)


def make_cycle_candidates_static_jit(**kwargs):
    """Build a jitted CNF-2 cycle entrypoint (static policy)."""
    return pv.cycle_candidates_static_jit(**kwargs)


def make_cycle_candidates_value_jit(**kwargs):
    """Build a jitted CNF-2 cycle entrypoint (policy as JAX value)."""
    return pv.cycle_candidates_value_jit(**kwargs)


def make_cnf2_bound_static_cfg(
    *,
    safety_policy=None,
    guard_cfg=None,
    cfg=None,
):
    if cfg is None:
        cfg = pv.DEFAULT_CNF2_CONFIG
    if guard_cfg is not None:
        cfg = pv.cnf2_config_with_guard(guard_cfg, cfg=cfg)
    if safety_policy is None:
        safety_policy = pv.DEFAULT_SAFETY_POLICY
    binding = pv.resolve_policy_binding(
        policy=safety_policy,
        policy_value=None,
        context="tests.min_prism.harness.make_cnf2_bound_static_cfg",
    )
    bound = pv.cnf2_config_bound(binding, cfg=cfg)
    return bound.bind_cfg()


_DEFAULT_CNF2_BOUND_CFG, _DEFAULT_CNF2_BOUND_POLICY = make_cnf2_bound_static_cfg()


def cycle_candidates_static_bound(
    ledger,
    frontier_ids,
    *,
    cfg=None,
    safety_policy=None,
    guard_cfg=None,
    **kwargs,
):
    if cfg is None and safety_policy is None and guard_cfg is None:
        cfg = _DEFAULT_CNF2_BOUND_CFG
        safety_policy = _DEFAULT_CNF2_BOUND_POLICY
    elif cfg is None or safety_policy is None:
        cfg, safety_policy = make_cnf2_bound_static_cfg(
            safety_policy=safety_policy,
            guard_cfg=guard_cfg,
            cfg=cfg,
        )
    return pv.cycle_candidates_static(
        ledger,
        frontier_ids,
        cnf2_cfg=cfg,
        safe_gather_policy=safety_policy,
        **kwargs,
    )


def make_cycle_jit(**kwargs):
    """Build a jitted BSP arena cycle entrypoint with fixed DI."""
    return pv.cycle_jit(**kwargs)


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


def novelty_set(ledger):
    return set(canon_state_ledger(ledger)[0])


def fixed_point_steps_intrinsic(
    ledger,
    root_id,
    max_steps=8,
    *,
    cycle_intrinsic_fn=None,
    cycle_intrinsic_kwargs=None,
):
    """Run intrinsic cycles until novelty stabilizes or max_steps reached."""
    frontier = jnp.array([int(root_id)], dtype=jnp.int32)
    if cycle_intrinsic_fn is None:
        cycle_intrinsic_fn = pv.cycle_intrinsic
    if cycle_intrinsic_kwargs is None:
        cycle_intrinsic_kwargs = {}
    prev = novelty_set(ledger)
    for step in range(max_steps):
        ledger, frontier = cycle_intrinsic_fn(
            ledger, frontier, **cycle_intrinsic_kwargs
        )
        curr = novelty_set(ledger)
        if curr == prev:
            return step + 1, ledger, frontier, True
        prev = curr
    return max_steps, ledger, frontier, False


def fixed_point_steps_cnf2(
    ledger,
    root_id,
    max_steps=8,
    *,
    cycle_candidates_fn=None,
    cycle_candidates_kwargs=None,
):
    """Run CNF-2 cycles until novelty stabilizes or max_steps reached."""
    frontier = pv._committed_ids(jnp.array([int(root_id)], dtype=jnp.int32))
    if cycle_candidates_fn is None:
        cycle_candidates_fn = cycle_candidates_static_bound
        cycle_candidates_kwargs = {}
    if cycle_candidates_kwargs is None:
        cycle_candidates_kwargs = {}
    prev = novelty_set(ledger)
    for step in range(max_steps):
        ledger, frontier_prov, _, q_map = cycle_candidates_fn(
            ledger, frontier, **cycle_candidates_kwargs
        )
        frontier = pv.apply_q(q_map, frontier_prov)
        curr = novelty_set(ledger)
        if curr == prev:
            return step + 1, ledger, frontier, True
        prev = curr
    return max_steps, ledger, frontier, False


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
        ids_out, new_ledger = intern_nodes(new_ledger, [op], [a1], [a2])
        key_map[key] = int(ids_out[0])
        return key_map[key]

    for key in sorted(keys):
        emit(key)
    return new_ledger, key_map


def rebuild_ledger_from_keys(keys):
    return _rebuild_ledger_from_keys(keys)


def enumerate_keys(max_depth=2, ops=None):
    if ops is None:
        ops = [pv.OP_SUC, pv.OP_ADD, pv.OP_MUL]
    key_null = (pv.OP_NULL, (), ())
    key_zero = (pv.OP_ZERO, key_null, key_null)
    unary_ops = {pv.OP_SUC}
    binary_ops = {pv.OP_ADD, pv.OP_MUL, pv.OP_SORT}

    levels = {0: {key_null, key_zero}}
    for depth in range(1, max_depth + 1):
        prior = set().union(*levels.values())
        new = set()
        for op in ops:
            if op in unary_ops:
                for k1 in prior:
                    new.add((op, k1, key_null))
            elif op in binary_ops:
                for k1 in prior:
                    for k2 in prior:
                        a1, a2 = k1, k2
                        if op in _COMMUTATIVE_OPS and a2 < a1:
                            a1, a2 = a2, a1
                        new.add((op, a1, a2))
        levels[depth] = new
    return set().union(*levels.values())
