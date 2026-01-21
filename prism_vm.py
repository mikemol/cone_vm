from jax import jit, lax
import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Callable, Tuple
import os
import re
import time

# --- 1. Ontology (Opcodes) ---
OP_NULL = 0
OP_ZERO = 1
OP_SUC  = 2
OP_ADD  = 10
OP_MUL  = 11
OP_SORT = 99

OP_NAMES = {
    0: "NULL", 1: "zero", 2: "suc",
    10: "add", 11: "mul", 99: "sort"
}

MAX_ROWS = 1024 * 32
MAX_NODES = 1024 * 64

# --- Rank (2-bit Scheduler) ---
RANK_HOT = 0
RANK_WARM = 1  # Reserved for future policies.
RANK_COLD = 2
RANK_FREE = 3

# --- 2. Manifest (Heap) ---
class Manifest(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    active_count: jnp.ndarray

class Arena(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    rank:   jnp.ndarray
    count:  jnp.ndarray

class Ledger(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    keys_hi_sorted: jnp.ndarray
    keys_lo_sorted: jnp.ndarray
    ids_sorted: jnp.ndarray
    count:  jnp.ndarray

def init_manifest():
    return Manifest(
        opcode=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg1=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg2=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        active_count=jnp.array(1, dtype=jnp.int32)
    )

def init_arena():
    arena = Arena(
        opcode=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        arg1=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        arg2=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        rank=jnp.full(MAX_NODES, RANK_FREE, dtype=jnp.int8),
        count=jnp.array(1, dtype=jnp.int32),
    )
    arena = arena._replace(
        opcode=arena.opcode.at[1].set(OP_ZERO),
        arg1=arena.arg1.at[1].set(0),
        arg2=arena.arg2.at[1].set(0),
        count=jnp.array(2, dtype=jnp.int32),
    )
    return arena

def init_ledger():
    max_key = jnp.iinfo(jnp.uint32).max

    opcode = jnp.zeros(MAX_NODES, dtype=jnp.int32)
    arg1 = jnp.zeros(MAX_NODES, dtype=jnp.int32)
    arg2 = jnp.zeros(MAX_NODES, dtype=jnp.int32)

    opcode = opcode.at[1].set(OP_ZERO)

    keys_hi_sorted = jnp.full(MAX_NODES, max_key, dtype=jnp.uint32)
    keys_lo_sorted = jnp.full(MAX_NODES, max_key, dtype=jnp.uint32)
    ids_sorted = jnp.zeros(MAX_NODES, dtype=jnp.int32)

    k0_hi, k0_lo = _pack_key(jnp.uint32(OP_NULL), jnp.uint32(0), jnp.uint32(0))
    k1_hi, k1_lo = _pack_key(jnp.uint32(OP_ZERO), jnp.uint32(0), jnp.uint32(0))
    keys_hi_sorted = keys_hi_sorted.at[0].set(k0_hi).at[1].set(k1_hi)
    keys_lo_sorted = keys_lo_sorted.at[0].set(k0_lo).at[1].set(k1_lo)
    ids_sorted = ids_sorted.at[0].set(0).at[1].set(1)

    return Ledger(
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
        keys_hi_sorted=keys_hi_sorted,
        keys_lo_sorted=keys_lo_sorted,
        ids_sorted=ids_sorted,
        count=jnp.array(2, dtype=jnp.int32),
    )

@jit
def op_rank(arena):
    ops = arena.opcode
    is_free = ops == OP_NULL
    is_inst = ops >= 10
    new_rank = jnp.where(is_free, RANK_FREE, jnp.where(is_inst, RANK_HOT, RANK_COLD))
    return arena._replace(rank=new_rank.astype(jnp.int8))

def _invert_perm(perm):
    inv = jnp.empty_like(perm)
    return inv.at[perm].set(jnp.arange(perm.shape[0], dtype=perm.dtype))

def _pack_key(op, a1, a2):
    op_u = op.astype(jnp.uint32)
    a1_u = a1.astype(jnp.uint32) & jnp.uint32(0xFFFF)
    a2_u = a2.astype(jnp.uint32) & jnp.uint32(0xFFFF)
    key_hi = (op_u << 16) | a1_u
    key_lo = a2_u
    return key_hi, key_lo

@jit
def intern_nodes(ledger, proposed_ops, proposed_a1, proposed_a2):
    """
    Batch-intern a list of proposed (op,a1,a2) nodes into the canonical Ledger.

    Args:
      ledger: Ledger
      proposed_ops/a1/a2: int32 arrays, shape [N]

    Returns:
      final_ids: int32 array, shape [N], canonical ids for each proposal
      new_ledger: Ledger, updated
    """
    max_key = jnp.iinfo(jnp.uint32).max

    P_hi, P_lo = _pack_key(proposed_ops, proposed_a1, proposed_a2)
    perm = jnp.lexsort((P_lo, P_hi)).astype(jnp.int32)

    s_hi = P_hi[perm]
    s_lo = P_lo[perm]
    s_ops = proposed_ops[perm]
    s_a1 = proposed_a1[perm]
    s_a2 = proposed_a2[perm]

    is_diff = jnp.concatenate([
        jnp.array([True]),
        (s_hi[1:] != s_hi[:-1]) | (s_lo[1:] != s_lo[:-1]),
    ])

    idx = jnp.arange(s_hi.shape[0], dtype=jnp.int32)

    def scan_fn(carry, x):
        is_leader, i = x
        new_carry = jnp.where(is_leader, i, carry)
        return new_carry, new_carry

    _, leader_idx = lax.scan(scan_fn, jnp.int32(0), (is_diff, idx))

    L_hi = ledger.keys_hi_sorted
    L_lo = ledger.keys_lo_sorted
    L_ids = ledger.ids_sorted

    count = ledger.count.astype(jnp.int32)

    def _lex_less(a_hi, a_lo, b_hi, b_lo):
        return jnp.logical_or(a_hi < b_hi, jnp.logical_and(a_hi == b_hi, a_lo < b_lo))

    def _search_one(t_hi, t_lo):
        lo = jnp.int32(0)
        hi = count

        def cond(state):
            lo_i, hi_i = state
            return lo_i < hi_i

        def body(state):
            lo_i, hi_i = state
            mid = (lo_i + hi_i) // 2
            mid_hi = L_hi[mid]
            mid_lo = L_lo[mid]
            go_right = _lex_less(mid_hi, mid_lo, t_hi, t_lo)
            lo_i = jnp.where(go_right, mid + 1, lo_i)
            hi_i = jnp.where(go_right, hi_i, mid)
            return (lo_i, hi_i)

        lo, _ = lax.while_loop(cond, body, (lo, hi))
        return lo

    insert_pos = jax.vmap(_search_one)(s_hi, s_lo)
    safe_pos = jnp.minimum(insert_pos, count - 1)

    found_match = (
        (insert_pos < count)
        & (L_hi[safe_pos] == s_hi)
        & (L_lo[safe_pos] == s_lo)
    )
    matched_ids = L_ids[safe_pos].astype(jnp.int32)

    is_new = is_diff & (~found_match)

    spawn = is_new.astype(jnp.int32)
    offsets = jnp.cumsum(spawn) - spawn
    num_new = jnp.sum(spawn).astype(jnp.int32)

    write_start = ledger.count.astype(jnp.int32)
    new_ids_for_sorted = jnp.where(found_match, matched_ids, write_start + offsets)

    leader_ids = jnp.where(is_diff, new_ids_for_sorted, jnp.int32(0))
    ids_sorted_order = leader_ids[leader_idx]

    inv_perm = _invert_perm(perm)
    final_ids = ids_sorted_order[inv_perm]

    new_opcode = ledger.opcode
    new_arg1 = ledger.arg1
    new_arg2 = ledger.arg2

    write_idx = jnp.where(is_new, write_start + offsets, jnp.int32(-1))
    valid_w = write_idx >= 0
    safe_w = jnp.where(valid_w, write_idx, 0)

    new_opcode = new_opcode.at[safe_w].set(
        jnp.where(valid_w, s_ops, new_opcode[0]), mode="drop"
    )
    new_arg1 = new_arg1.at[safe_w].set(
        jnp.where(valid_w, s_a1, new_arg1[0]), mode="drop"
    )
    new_arg2 = new_arg2.at[safe_w].set(
        jnp.where(valid_w, s_a2, new_arg2[0]), mode="drop"
    )

    new_count = ledger.count + num_new

    all_hi, all_lo = _pack_key(new_opcode, new_arg1, new_arg2)
    valid_all = jnp.arange(all_hi.shape[0], dtype=jnp.int32) < new_count
    sortable_hi = jnp.where(valid_all, all_hi, max_key)
    sortable_lo = jnp.where(valid_all, all_lo, max_key)

    order = jnp.lexsort((sortable_lo, sortable_hi)).astype(jnp.int32)
    new_keys_hi_sorted = sortable_hi[order]
    new_keys_lo_sorted = sortable_lo[order]
    new_ids_sorted = order

    new_ledger = Ledger(
        opcode=new_opcode,
        arg1=new_arg1,
        arg2=new_arg2,
        keys_hi_sorted=new_keys_hi_sorted,
        keys_lo_sorted=new_keys_lo_sorted,
        ids_sorted=new_ids_sorted,
        count=new_count,
    )
    return final_ids, new_ledger

def _active_prefix_count(arena):
    size = arena.rank.shape[0]
    # Avoid host-syncing on arena.count; sort full size for consistent behavior.
    return size

def _apply_perm_and_swizzle(arena, perm):
    inv_perm = _invert_perm(perm)
    new_ops = arena.opcode[perm]
    new_arg1 = arena.arg1[perm]
    new_arg2 = arena.arg2[perm]
    new_rank = arena.rank[perm]
    swizzled_arg1 = jnp.where(new_arg1 != 0, inv_perm[new_arg1], 0)
    swizzled_arg2 = jnp.where(new_arg2 != 0, inv_perm[new_arg2], 0)
    active_count = jnp.sum(new_rank != RANK_FREE).astype(jnp.int32)
    return Arena(new_ops, swizzled_arg1, swizzled_arg2, new_rank, active_count), inv_perm

@jit
def _op_sort_and_swizzle_with_perm_full(arena):
    size = arena.rank.shape[0]
    idx = jnp.arange(size, dtype=jnp.int32)
    sort_key = arena.rank.astype(jnp.int32) * (size + 1) + idx
    perm = jnp.argsort(sort_key)
    return _apply_perm_and_swizzle(arena, perm)

def _op_sort_and_swizzle_with_perm_prefix(arena, active_count):
    size = arena.rank.shape[0]
    if active_count <= 1:
        perm = jnp.arange(size, dtype=jnp.int32)
        return _apply_perm_and_swizzle(arena, perm)
    idx = jnp.arange(active_count, dtype=jnp.int32)
    sort_key = arena.rank[:active_count].astype(jnp.int32) * (active_count + 1) + idx
    perm_active = jnp.argsort(sort_key)
    tail = jnp.arange(active_count, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return _apply_perm_and_swizzle(arena, perm)

def op_sort_and_swizzle_with_perm(arena):
    active_count = _active_prefix_count(arena)
    size = arena.rank.shape[0]
    if active_count >= size:
        return _op_sort_and_swizzle_with_perm_full(arena)
    return _op_sort_and_swizzle_with_perm_prefix(arena, active_count)

def op_sort_and_swizzle(arena):
    sorted_arena, _ = op_sort_and_swizzle_with_perm(arena)
    return sorted_arena

def _blocked_perm(arena, block_size, morton=None, active_count=None):
    size = int(arena.rank.shape[0])
    if block_size <= 0 or size % block_size != 0:
        raise ValueError("block_size must evenly divide arena size")
    num_blocks = size // block_size
    if active_count is None or active_count >= size:
        active_blocks = num_blocks
    else:
        active_blocks = (active_count + block_size - 1) // block_size
        if active_blocks < 0:
            active_blocks = 0
        if active_blocks > num_blocks:
            active_blocks = num_blocks

    ranks = arena.rank.reshape((num_blocks, block_size)).astype(jnp.uint32)
    idx = jnp.arange(block_size, dtype=jnp.uint32)
    idx_u = idx & jnp.uint32(0xFFFF)
    if morton is None:
        morton_u = jnp.zeros_like(ranks, dtype=jnp.uint32)
    else:
        morton_u = morton.reshape((num_blocks, block_size)).astype(jnp.uint32) & jnp.uint32(0x3FFF)

    if active_blocks <= 0:
        return jnp.arange(size, dtype=jnp.int32)

    if active_blocks == num_blocks:
        sort_key = (ranks << 30) | (morton_u << 16) | idx_u
        perm_local = jnp.argsort(sort_key, axis=1)
        base = (jnp.arange(num_blocks, dtype=jnp.uint32) * block_size)[:, None]
        perm = (base + perm_local).reshape((size,)).astype(jnp.int32)
        return perm

    ranks_active = ranks[:active_blocks]
    morton_active = morton_u[:active_blocks]
    if active_count is not None and active_count < active_blocks * block_size:
        base = (jnp.arange(active_blocks, dtype=jnp.uint32) * block_size)[:, None]
        block_idx = base + idx_u[None, :]
        tail_mask = block_idx >= active_count
        ranks_active = jnp.where(tail_mask, jnp.uint32(RANK_FREE), ranks_active)
        morton_active = jnp.where(tail_mask, jnp.uint32(0), morton_active)

    sort_key = (ranks_active << 30) | (morton_active << 16) | idx_u
    perm_local = jnp.argsort(sort_key, axis=1)
    base = (jnp.arange(active_blocks, dtype=jnp.uint32) * block_size)[:, None]
    perm_active = (base + perm_local).reshape((active_blocks * block_size,)).astype(jnp.int32)
    tail = jnp.arange(active_blocks * block_size, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return perm

def op_sort_and_swizzle_blocked_with_perm(arena, block_size, morton=None):
    active_count = _active_prefix_count(arena)
    perm = _blocked_perm(arena, block_size, morton=morton, active_count=active_count)
    return _apply_perm_and_swizzle(arena, perm)

def op_sort_and_swizzle_blocked(arena, block_size, morton=None):
    sorted_arena, _ = op_sort_and_swizzle_blocked_with_perm(
        arena, block_size, morton=morton
    )
    return sorted_arena

def _apply_perm_to_morton(morton, inv_perm):
    if morton is None:
        return None
    perm = _invert_perm(inv_perm)
    return morton[perm]

def _walk_block_sizes(start_block_size, size):
    sizes = []
    if start_block_size <= 0 or start_block_size >= size:
        return sizes
    block_size = start_block_size
    while block_size < size:
        next_block = block_size * 2
        if next_block >= size:
            sizes.append(size)
            break
        if size % next_block != 0:
            sizes.append(size)
            break
        sizes.append(next_block)
        block_size = next_block
    return sizes

def op_sort_and_swizzle_hierarchical_with_perm(
    arena, l2_block_size, l1_block_size, morton=None, do_global=False
):
    size = int(arena.rank.shape[0])
    if l2_block_size <= 0 or l1_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if size % l2_block_size != 0 or size % l1_block_size != 0:
        raise ValueError("block sizes must evenly divide arena size")
    if l1_block_size % l2_block_size != 0:
        raise ValueError("l1_block_size must be a multiple of l2_block_size")

    arena, inv_perm = op_sort_and_swizzle_blocked_with_perm(
        arena, l2_block_size, morton=morton
    )
    morton = _apply_perm_to_morton(morton, inv_perm)
    inv_perm_total = inv_perm

    if l1_block_size > l2_block_size:
        arena, inv_perm_l1 = op_sort_and_swizzle_blocked_with_perm(
            arena, l1_block_size, morton=morton
        )
        morton = _apply_perm_to_morton(morton, inv_perm_l1)
        inv_perm_total = inv_perm_l1[inv_perm_total]

    if do_global and l1_block_size < size:
        for block_size in _walk_block_sizes(l1_block_size, size):
            arena, inv_perm_global = op_sort_and_swizzle_blocked_with_perm(
                arena, block_size, morton=morton
            )
            morton = _apply_perm_to_morton(morton, inv_perm_global)
            inv_perm_total = inv_perm_global[inv_perm_total]

    return arena, inv_perm_total

def op_sort_and_swizzle_hierarchical(
    arena, l2_block_size, l1_block_size, morton=None, do_global=False
):
    sorted_arena, _ = op_sort_and_swizzle_hierarchical_with_perm(
        arena,
        l2_block_size,
        l1_block_size,
        morton=morton,
        do_global=do_global,
    )
    return sorted_arena

def swizzle_2to1_host(x, y):
    z = 0
    for i in range(10):
        x0 = (x >> (2 * i)) & 1
        x1 = (x >> (2 * i + 1)) & 1
        y0 = (y >> i) & 1
        z |= (x0 << (3 * i))
        z |= (x1 << (3 * i + 1))
        z |= (y0 << (3 * i + 2))
    return z

@jit
def swizzle_2to1_dev(x, y):
    x = x.astype(jnp.uint32)
    y = y.astype(jnp.uint32)
    z = jnp.zeros_like(x, dtype=jnp.uint32)

    def body(i, val):
        z_acc, x_in, y_in = val
        x_bits = x_in & jnp.uint32(3)
        y_bit = y_in & jnp.uint32(1)
        chunk = (y_bit << 2) | x_bits
        z_acc = z_acc | (chunk << (3 * i))
        return (z_acc, x_in >> 2, y_in >> 1)

    res, _, _ = lax.fori_loop(0, 10, body, (z, x, y))
    return res

def _build_pallas_swizzle(backend):
    try:
        import jax as jax_module
        import jax.experimental.pallas as pl
        if backend == "triton":
            import jax.experimental.pallas.triton  # noqa: F401
    except Exception:
        return None

    if jax_module.default_backend() == "cpu":
        return None
    if backend == "triton" and jax_module.default_backend() != "gpu":
        return None

    def kernel(x_ref, y_ref, out_ref):
        x_val = x_ref[0].astype(jnp.uint32)
        y_val = y_ref[0].astype(jnp.uint32)
        z = jnp.uint32(0)
        for i in range(10):
            x_bits = x_val & jnp.uint32(3)
            y_bit = y_val & jnp.uint32(1)
            chunk = (y_bit << 2) | x_bits
            z = z | (chunk << (3 * i))
            x_val = x_val >> 2
            y_val = y_val >> 1
        out_ref[0] = z

    def swizzle(x, y):
        out_shape = jax_module.ShapeDtypeStruct(x.shape, jnp.uint32)
        in_specs = [
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
        ]
        out_specs = pl.BlockSpec((1,), lambda i: (i,))
        grid = (x.shape[0],)
        return pl.pallas_call(
            kernel,
            out_shape=out_shape,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            backend="triton" if backend == "triton" else None,
        )(x, y)

    return swizzle

_SWIZZLE_BACKEND = os.environ.get("PRISM_SWIZZLE_BACKEND", "jax").strip().lower()
_SWIZZLE_ACCEL = None
if _SWIZZLE_BACKEND in ("pallas", "triton"):
    _SWIZZLE_ACCEL = _build_pallas_swizzle(_SWIZZLE_BACKEND)
    if _SWIZZLE_ACCEL is None:
        _SWIZZLE_BACKEND = "jax"

def swizzle_2to1(x, y):
    if _SWIZZLE_ACCEL is not None:
        return _SWIZZLE_ACCEL(x, y)
    return swizzle_2to1_dev(x, y)

@jit
def op_morton(arena):
    size = arena.opcode.shape[0]
    idx = jnp.arange(size, dtype=jnp.uint32)
    x = idx
    y = jnp.zeros_like(idx)
    return swizzle_2to1(x, y)

@jit
def _op_sort_and_swizzle_morton_with_perm_full(arena, morton):
    size = arena.rank.shape[0]
    idx = jnp.arange(size, dtype=jnp.uint32)
    rank_u = arena.rank.astype(jnp.uint32)
    morton_u = morton.astype(jnp.uint32) & jnp.uint32(0x3FFF)
    idx_u = idx & jnp.uint32(0xFFFF)
    sort_key = (rank_u << 30) | (morton_u << 16) | idx_u
    perm = jnp.argsort(sort_key).astype(jnp.int32)
    return _apply_perm_and_swizzle(arena, perm)

def _op_sort_and_swizzle_morton_with_perm_prefix(arena, morton, active_count):
    size = arena.rank.shape[0]
    if active_count <= 1:
        perm = jnp.arange(size, dtype=jnp.int32)
        return _apply_perm_and_swizzle(arena, perm)
    idx = jnp.arange(active_count, dtype=jnp.uint32)
    rank_u = arena.rank[:active_count].astype(jnp.uint32)
    morton_u = morton[:active_count].astype(jnp.uint32) & jnp.uint32(0x3FFF)
    idx_u = idx & jnp.uint32(0xFFFF)
    sort_key = (rank_u << 30) | (morton_u << 16) | idx_u
    perm_active = jnp.argsort(sort_key).astype(jnp.int32)
    tail = jnp.arange(active_count, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return _apply_perm_and_swizzle(arena, perm)

def op_sort_and_swizzle_morton_with_perm(arena, morton):
    active_count = _active_prefix_count(arena)
    size = arena.rank.shape[0]
    if active_count >= size:
        return _op_sort_and_swizzle_morton_with_perm_full(arena, morton)
    return _op_sort_and_swizzle_morton_with_perm_prefix(arena, morton, active_count)

def op_sort_and_swizzle_morton(arena, morton):
    sorted_arena, _ = op_sort_and_swizzle_morton_with_perm(arena, morton)
    return sorted_arena

@jit
def op_interact(arena):
    ops = arena.opcode
    a1 = arena.arg1
    a2 = arena.arg2
    is_hot = arena.rank == RANK_HOT
    is_add = ops == OP_ADD
    op_x = ops[a1]
    mask_zero = is_hot & is_add & (op_x == OP_ZERO)
    mask_suc = is_hot & is_add & (op_x == OP_SUC)

    # First: local rewrites that don't allocate.
    y_op = ops[a2]
    y_a1 = a1[a2]
    y_a2 = a2[a2]
    new_ops = jnp.where(mask_zero, y_op, ops)
    new_a1 = jnp.where(mask_zero, y_a1, a1)
    new_a2 = jnp.where(mask_zero, y_a2, a2)

    # Second: allocation for suc-case.
    spawn = mask_suc.astype(jnp.int32)
    offsets = jnp.cumsum(spawn) - spawn
    total_spawn = jnp.sum(spawn).astype(jnp.int32)
    base_free = arena.count
    new_add_idx = base_free + offsets

    new_ops = jnp.where(mask_suc, OP_SUC, new_ops)
    new_a1 = jnp.where(mask_suc, new_add_idx, new_a1)
    new_a2 = jnp.where(mask_suc, 0, new_a2)

    # Scatter-create the spawned add nodes only where mask_suc is true.
    idxs = jnp.where(mask_suc, new_add_idx, jnp.int32(-1))
    grandchild_x = a1[a1]
    payload_op = jnp.full_like(idxs, OP_ADD)
    payload_a1 = grandchild_x
    payload_a2 = a2

    valid = idxs >= 0
    idxs2 = jnp.where(valid, idxs, 0)

    final_ops = new_ops.at[idxs2].set(
        jnp.where(valid, payload_op, new_ops[0]), mode="drop"
    )
    final_a1 = new_a1.at[idxs2].set(
        jnp.where(valid, payload_a1, new_a1[0]), mode="drop"
    )
    final_a2 = new_a2.at[idxs2].set(
        jnp.where(valid, payload_a2, new_a2[0]), mode="drop"
    )

    return Arena(final_ops, final_a1, final_a2, arena.rank, arena.count + total_spawn)

@jit
def cycle_intrinsic(ledger, frontier_ids):
    f_ops = ledger.opcode[frontier_ids]
    f_a1 = ledger.arg1[frontier_ids]
    f_a2 = ledger.arg2[frontier_ids]

    op_a1 = ledger.opcode[f_a1]
    is_add_suc = (f_ops == OP_ADD) & (op_a1 == OP_SUC)
    is_mul_suc = (f_ops == OP_MUL) & (op_a1 == OP_SUC)
    is_add_zero = (f_ops == OP_ADD) & (op_a1 == OP_ZERO)
    is_mul_zero = (f_ops == OP_MUL) & (op_a1 == OP_ZERO)

    val_x = ledger.arg1[f_a1]
    val_y = f_a2

    l1_ops = jnp.zeros_like(f_ops)
    l1_a1 = jnp.zeros_like(f_a1)
    l1_a2 = jnp.zeros_like(f_a2)
    l1_ops = jnp.where(is_add_suc, OP_ADD, l1_ops)
    l1_a1 = jnp.where(is_add_suc, val_x, l1_a1)
    l1_a2 = jnp.where(is_add_suc, val_y, l1_a2)
    l1_ops = jnp.where(is_mul_suc, OP_MUL, l1_ops)
    l1_a1 = jnp.where(is_mul_suc, val_x, l1_a1)
    l1_a2 = jnp.where(is_mul_suc, val_y, l1_a2)

    l1_ids, ledger = intern_nodes(ledger, l1_ops, l1_a1, l1_a2)

    l2_ops = jnp.zeros_like(f_ops)
    l2_a1 = jnp.zeros_like(f_a1)
    l2_a2 = jnp.zeros_like(f_a2)
    l2_ops = jnp.where(is_add_suc, OP_SUC, l2_ops)
    l2_a1 = jnp.where(is_add_suc, l1_ids, l2_a1)
    l2_ops = jnp.where(is_mul_suc, OP_ADD, l2_ops)
    l2_a1 = jnp.where(is_mul_suc, val_y, l2_a1)
    l2_a2 = jnp.where(is_mul_suc, l1_ids, l2_a2)

    l2_ids, ledger = intern_nodes(ledger, l2_ops, l2_a1, l2_a2)

    next_frontier = frontier_ids
    next_frontier = jnp.where(is_add_zero, f_a2, next_frontier)
    next_frontier = jnp.where(is_mul_zero, jnp.int32(1), next_frontier)
    next_frontier = jnp.where(is_add_suc, l1_ids, next_frontier)
    next_frontier = jnp.where(is_mul_suc, l2_ids, next_frontier)
    return ledger, next_frontier

def cycle(
    arena,
    root_ptr,
    do_sort=True,
    use_morton=False,
    block_size=None,
    morton=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
):
    """Run one BSP cycle; keep root_ptr as a JAX scalar to avoid host sync."""
    arena = op_rank(arena)
    root_arr = jnp.asarray(root_ptr, dtype=jnp.int32)
    if do_sort:
        morton_arr = None
        if use_morton or morton is not None:
            morton_arr = morton if morton is not None else op_morton(arena)
        if l2_block_size is not None or l1_block_size is not None:
            if l2_block_size is None:
                l2_block_size = l1_block_size
            if l1_block_size is None:
                l1_block_size = l2_block_size
            arena, inv_perm = op_sort_and_swizzle_hierarchical_with_perm(
                arena,
                l2_block_size,
                l1_block_size,
                morton=morton_arr,
                do_global=do_global,
            )
        elif block_size is not None:
            arena, inv_perm = op_sort_and_swizzle_blocked_with_perm(
                arena, block_size, morton=morton_arr
            )
        elif morton_arr is not None:
            arena, inv_perm = op_sort_and_swizzle_morton_with_perm(arena, morton_arr)
        else:
            arena, inv_perm = op_sort_and_swizzle_with_perm(arena)
        root_arr = jnp.where(root_arr != 0, inv_perm[root_arr], 0)
    arena = op_interact(arena)
    return arena, root_arr

# --- 3. JAX Kernels (Static) ---
# --- 3. JAX Kernels (Static) ---
@jit
def kernel_add(manifest, ptr):
    ops, a1, a2, count = manifest.opcode, manifest.arg1, manifest.arg2, manifest.active_count
    init_x = a1[ptr]
    init_y = a2[ptr]
    init_val = (init_x, init_y, True, ops, a1, count)

    def cond(v): return v[2]

    def body(v):
        curr_x, curr_y, active, b_ops, b_a1, b_count = v
        op_x = b_ops[curr_x]
        is_suc = (op_x == OP_SUC)
        next_x = jnp.where(is_suc, b_a1[curr_x], curr_x)
        w_idx = b_count
        next_ops = b_ops.at[w_idx].set(OP_SUC)
        next_a1  = b_a1.at[w_idx].set(curr_y)
        next_y = jnp.where(is_suc, w_idx, curr_y)
        next_active = is_suc
        next_count = jnp.where(is_suc, b_count + 1, b_count)
        return (next_x, next_y, next_active, next_ops, next_a1, next_count)

    _, final_y, _, f_ops, f_a1, f_count = lax.while_loop(cond, body, init_val)
    return manifest._replace(opcode=f_ops, arg1=f_a1, active_count=f_count), final_y

# Kernel stub for MUL
@jit
def kernel_mul(manifest, ptr):
    ops, a1, a2, count = manifest.opcode, manifest.arg1, manifest.arg2, manifest.active_count
    init_x = a1[ptr]
    y = a2[ptr]
    init_acc = jnp.array(1, dtype=jnp.int32)
    init_val = (init_x, init_acc, ops, a1, a2, count)

    def cond(v):
        curr_x, _, b_ops, _, _, _ = v
        return b_ops[curr_x] == OP_SUC

    def body(v):
        curr_x, acc, b_ops, b_a1, b_a2, b_count = v
        next_x = b_a1[curr_x]
        add_idx = b_count
        next_ops = b_ops.at[add_idx].set(OP_ADD)
        next_a1 = b_a1.at[add_idx].set(y)
        next_a2 = b_a2.at[add_idx].set(acc)
        next_count = b_count + 1
        add_manifest = Manifest(next_ops, next_a1, next_a2, next_count)
        updated_manifest, next_acc = kernel_add(add_manifest, add_idx)
        return (
            next_x,
            next_acc,
            updated_manifest.opcode,
            updated_manifest.arg1,
            updated_manifest.arg2,
            updated_manifest.active_count,
        )

    _, final_acc, f_ops, f_a1, f_a2, f_count = lax.while_loop(cond, body, init_val)
    return manifest._replace(opcode=f_ops, arg1=f_a1, arg2=f_a2, active_count=f_count), final_acc

def _dispatch_identity(args):
    manifest, ptr = args
    return manifest, ptr

def _dispatch_add(args):
    manifest, ptr = args
    return kernel_add(manifest, ptr)

def _dispatch_mul(args):
    manifest, ptr = args
    return kernel_mul(manifest, ptr)

@jit
def optimize_ptr(manifest, ptr):
    ops, a1s, a2s = manifest.opcode, manifest.arg1, manifest.arg2
    op = ops[ptr]
    a1 = a1s[ptr]
    a2 = a2s[ptr]
    op_a1 = ops[a1]
    is_add = op == OP_ADD
    is_zero = op_a1 == OP_ZERO
    optimized = jnp.logical_and(is_add, is_zero)
    out_ptr = jnp.where(optimized, a2, ptr)
    return out_ptr, optimized

@jit
def dispatch_kernel(manifest, ptr):
    opt_ptr, opt_applied = optimize_ptr(manifest, ptr)
    op = manifest.opcode[opt_ptr]
    case_index = jnp.where(op == OP_ADD, 1, jnp.where(op == OP_MUL, 2, 0))
    new_manifest, res_ptr = lax.switch(
        case_index,
        (_dispatch_identity, _dispatch_add, _dispatch_mul),
        (manifest, opt_ptr),
    )
    return new_manifest, res_ptr, opt_applied


# --- 4. Prism VM (Host Logic) ---
class PrismVM:
    def __init__(self):
        print("⚡ Prism IR: Initializing Host Context...")
        self.manifest = init_manifest()
        # Trace Cache: (opcode, arg1, arg2) -> ptr
        self.trace_cache: Dict[Tuple[int, int, int], int] = {}
        # Initialize Universe (Seed with ZERO)
        self._cons_raw(OP_ZERO, 0, 0)
        self.trace_cache[(OP_ZERO, 0, 0)] = 1

        self.kernels = {OP_ADD: kernel_add, OP_MUL: kernel_mul}

    def _cons_raw(self, op, a1, a2):
        """Physical allocation (Device Write)"""
        idx = int(self.manifest.active_count)
        self.manifest = self.manifest._replace(
            opcode=self.manifest.opcode.at[idx].set(op),
            arg1=self.manifest.arg1.at[idx].set(a1),
            arg2=self.manifest.arg2.at[idx].set(a2),
            active_count=jnp.array(idx + 1, dtype=jnp.int32)
        )
        return idx

    def cons(self, op, a1=0, a2=0):
        """
        The Smart Allocator.
        1. Checks Cache (Deduplication).
        2. Allocates if new.
        """
        signature = (op, a1, a2)
        if signature in self.trace_cache:
            return self.trace_cache[signature]
        ptr = self._cons_raw(op, a1, a2)
        self.trace_cache[signature] = ptr
        return ptr

    # --- STATIC ANALYSIS ENGINE ---
    def analyze_and_optimize(self, ptr):
        """
        Examines the IR at `ptr` BEFORE execution.
        Performs trivial reductions (Constant Folding / Identity Elimination).
        """
        ptr_arr = jnp.array(ptr, dtype=jnp.int32)
        opt_ptr, opt_applied = optimize_ptr(self.manifest, ptr_arr)
        if bool(opt_applied):
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        return int(opt_ptr)

    def eval(self, ptr):
        """
        The Hybrid Interpreter.
        1. Static Analysis (Host)
        2. Dispatch (Device)
        """
        ptr_arr = jnp.array(ptr, dtype=jnp.int32)
        new_manifest, res_ptr, opt_applied = dispatch_kernel(self.manifest, ptr_arr)
        res_ptr.block_until_ready()
        self.manifest = new_manifest
        if bool(opt_applied):
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        return int(res_ptr)

    # --- PARSING & DISPLAY ---
    def parse(self, tokens):
        token = tokens.pop(0)
        if token == 'zero': return self.cons(OP_ZERO)
        if token == 'suc':  return self.cons(OP_SUC, self.parse(tokens))
        if token in ['add', 'mul']:
            op = OP_ADD if token == 'add' else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self.cons(op, a1, a2)
        if token == '(': 
            val = self.parse(tokens)
            tokens.pop(0)
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr):
        op = int(self.manifest.opcode[ptr])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:  return f"(suc {self.decode(int(self.manifest.arg1[ptr]))})"
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"


class PrismVM_BSP_Legacy:
    def __init__(self):
        print("⚡ Prism IR: Initializing BSP Arena (Legacy)...")
        self.arena = init_arena()

    def _alloc(self, op, a1=0, a2=0):
        idx = int(self.arena.count)
        self.arena = self.arena._replace(
            opcode=self.arena.opcode.at[idx].set(op),
            arg1=self.arena.arg1.at[idx].set(a1),
            arg2=self.arena.arg2.at[idx].set(a2),
            count=jnp.array(idx + 1, dtype=jnp.int32),
        )
        return idx

    def parse(self, tokens):
        token = tokens.pop(0)
        if token == "zero": return self._alloc(OP_ZERO, 0, 0)
        if token == "suc":  return self._alloc(OP_SUC, self.parse(tokens), 0)
        if token in ["add", "mul"]:
            op = OP_ADD if token == "add" else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self._alloc(op, a1, a2)
        if token == "(":
            val = self.parse(tokens)
            tokens.pop(0)
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr):
        op = int(self.arena.opcode[ptr])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:  return f"(suc {self.decode(int(self.arena.arg1[ptr]))})"
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"

class PrismVM_BSP:
    def __init__(self):
        print("⚡ Prism IR: Initializing BSP Ledger...")
        self.ledger = init_ledger()

    def _intern(self, op, a1=0, a2=0):
        ids, self.ledger = intern_nodes(
            self.ledger,
            jnp.array([op], dtype=jnp.int32),
            jnp.array([a1], dtype=jnp.int32),
            jnp.array([a2], dtype=jnp.int32),
        )
        return int(ids[0])

    def parse(self, tokens):
        token = tokens.pop(0)
        if token == "zero": return self._intern(OP_ZERO, 0, 0)
        if token == "suc":  return self._intern(OP_SUC, self.parse(tokens), 0)
        if token in ["add", "mul"]:
            op = OP_ADD if token == "add" else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self._intern(op, a1, a2)
        if token == "(":
            val = self.parse(tokens)
            tokens.pop(0)
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr):
        op = int(self.ledger.opcode[ptr])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:  return f"(suc {self.decode(int(self.ledger.arg1[ptr]))})"
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"


def make_vm(mode="baseline"):
    if mode == "bsp":
        return PrismVM_BSP()
    return PrismVM()

def _rank_counts(arena):
    hot = int(jnp.sum(arena.rank == RANK_HOT))
    warm = int(jnp.sum(arena.rank == RANK_WARM))
    cold = int(jnp.sum(arena.rank == RANK_COLD))
    free = int(jnp.sum(arena.rank == RANK_FREE))
    return hot, warm, cold, free

# --- 5. Telemetric REPL ---
def run_program_lines(lines, vm=None):
    if vm is None:
        vm = PrismVM()
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith('#'):
            continue
        start_rows = int(vm.manifest.active_count)
        t0 = time.perf_counter()
        tokens = re.findall(r'\(|\)|[a-z]+', inp)
        ir_ptr = vm.parse(tokens)
        parse_ms = (time.perf_counter() - t0) * 1000
        mid_rows = int(vm.manifest.active_count)
        ir_allocs = mid_rows - start_rows
        ir_op = OP_NAMES.get(int(vm.manifest.opcode[ir_ptr]), "?")
        print(f"   ├─ IR Build: {ir_op} @ {ir_ptr}")
        if ir_allocs == 0:
            print(f"   ├─ Cache   : \033[96mHIT (No new IR rows)\033[0m")
        else:
            print(f"   ├─ Cache   : MISS (+{ir_allocs} IR rows)")
        t1 = time.perf_counter()
        res_ptr = vm.eval(ir_ptr)
        eval_ms = (time.perf_counter() - t1) * 1000
        end_rows = int(vm.manifest.active_count)
        exec_allocs = end_rows - mid_rows
        print(f"   ├─ Execute : {eval_ms:.2f}ms")
        if exec_allocs > 0:
            print(f"   ├─ Kernel  : +{exec_allocs} rows allocated")
        else:
            print(f"   ├─ Kernel  : \033[96mSKIPPED (Static Optimization)\033[0m")
        print(f"   └─ Result  : \033[92m{vm.decode(res_ptr)}\033[0m")
    return vm

def run_program_lines_bsp(
    lines,
    vm=None,
    cycles=1,
    do_sort=True,
    use_morton=False,
    block_size=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
):
    if vm is None:
        vm = PrismVM_BSP()
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith("#"):
            continue
        tokens = re.findall(r"\(|\)|[a-z]+", inp)
        root_ptr = vm.parse(tokens)
        frontier = jnp.array([root_ptr], dtype=jnp.int32)
        for _ in range(max(1, cycles)):
            vm.ledger, frontier = cycle_intrinsic(vm.ledger, frontier)
        root_ptr = frontier[0]
        root_ptr_int = int(root_ptr)
        print(f"   ├─ Ledger   : {int(vm.ledger.count)} nodes")
        print(f"   └─ Result  : \033[92m{vm.decode(root_ptr_int)}\033[0m")
    return vm

def repl(mode="baseline", use_morton=False, block_size=None):
    if mode == "bsp":
        vm = PrismVM_BSP()
        print("\n🔮 Prism IR Shell (BSP Ledger)")
        print("   Try: (add (suc zero) (suc zero))")
    else:
        vm = PrismVM()
        print("\n🔮 Prism IR Shell (Static Analysis + Deduplication)")
        print("   Try: (add (suc zero) (suc zero))")
        print("   Try: (add zero (suc (suc zero))) <- Triggers Optimizer")
    while True:
        try:
            inp = input("\nλ> ").strip()
            if inp == "exit": break
            if not inp: continue
            if mode == "bsp":
                run_program_lines_bsp(
                    [inp], vm, use_morton=use_morton, block_size=block_size
                )
            else:
                run_program_lines([inp], vm)
        except Exception as e:
            print(f"   ERROR: {e}")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    mode = "baseline"
    cycles = 1
    do_sort = True
    use_morton = False
    block_size = None
    path = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--mode", "-m") and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
            continue
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--cycles" and i + 1 < len(args):
            cycles = int(args[i + 1])
            i += 2
            continue
        if arg.startswith("--cycles="):
            cycles = int(arg.split("=", 1)[1])
            i += 1
            continue
        if arg == "--no-sort":
            do_sort = False
            i += 1
            continue
        if arg == "--morton":
            use_morton = True
            i += 1
            continue
        if arg == "--block-size" and i + 1 < len(args):
            block_size = int(args[i + 1])
            i += 2
            continue
        if arg.startswith("--block-size="):
            block_size = int(arg.split("=", 1)[1])
            i += 1
            continue
        if path is None:
            path = arg
            i += 1
            continue
        i += 1
    if path:
        with open(path) as f:
            lines = f.readlines()
        if mode == "bsp":
            run_program_lines_bsp(
                lines,
                cycles=cycles,
                do_sort=do_sort,
                use_morton=use_morton,
                block_size=block_size,
            )
        else:
            run_program_lines(lines)
    else:
        repl(mode=mode, use_morton=use_morton, block_size=block_size)
