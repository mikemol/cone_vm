import functools
import inspect
import os

import jax
import jax.numpy as jnp
from jax import jit, lax

from prism_core import jax_safe as _jax_safe
from prism_core.safety import SafetyPolicy
from prism_vm_core.domains import _host_int, _host_int_value
from prism_vm_core.guards import (
    GuardConfig,
    DEFAULT_GUARD_CONFIG,
    resolve_safe_gather_fn,
    guard_null_row_cfg,
    guard_slot0_perm_cfg,
    guard_swizzle_args_cfg,
)
from prism_vm_core.ontology import OP_NULL
from prism_vm_core.structures import Arena


_TEST_GUARDS = _jax_safe.TEST_GUARDS


_BINCOUNT_HAS_LENGTH = "length" in inspect.signature(jnp.bincount).parameters


def _bincount_256(x, weights):
    # Fixed-size bincount keeps JIT shapes static across JAX versions.
    if _BINCOUNT_HAS_LENGTH:
        return jnp.bincount(x, weights=weights, minlength=256, length=256)
    out = jnp.zeros(256, dtype=weights.dtype)
    return out.at[x].add(weights)


def _bincount_32(x, weights):
    # Fixed-size bincount keeps JIT shapes static across JAX versions.
    if _BINCOUNT_HAS_LENGTH:
        return jnp.bincount(x, weights=weights, minlength=32, length=32)
    out = jnp.zeros(32, dtype=weights.dtype)
    return out.at[x].add(weights)


# --- Rank (2-bit Scheduler) ---
RANK_HOT = 0
RANK_WARM = 1  # Reserved for future policies.
RANK_COLD = 2
RANK_FREE = 3


@jit
def _blind_spectral_probe(arena):
    # Entropyₐ: BSPˢ observer that measures spatial XOR magnitudes.
    size = arena.opcode.shape[0]
    idx = jnp.arange(size, dtype=jnp.uint32)
    count = arena.count.astype(jnp.uint32)
    live = idx < count
    hot = (arena.rank == RANK_HOT) & live
    hot_w = hot.astype(jnp.float32)
    arg1 = arena.arg1.astype(jnp.uint32)
    arg2 = arena.arg2.astype(jnp.uint32)
    xor1 = jnp.bitwise_xor(idx, arg1)
    xor2 = jnp.bitwise_xor(idx, arg2)

    def _msb_index(x):
        x_f = jnp.where(x > 0, jnp.log2(x.astype(jnp.float32)), 0.0)
        return jnp.floor(x_f).astype(jnp.int32)

    m1 = jnp.clip(_msb_index(xor1), 0, 31)
    m2 = jnp.clip(_msb_index(xor2), 0, 31)
    hist = _bincount_32(m1, hot_w) + _bincount_32(m2, hot_w)
    spectrum = hist / (jnp.sum(hist) + jnp.float32(1e-6))
    return lax.stop_gradient(spectrum)


_SERVO_MASK_DEFAULT = jnp.uint32(0xFFFFFFFF)


def _servo_mask_from_k(k):
    k = jnp.clip(k, 0, 31).astype(jnp.uint32)
    low_bits = (jnp.uint32(1) << k) - jnp.uint32(1)
    return jnp.where(k == 0, _SERVO_MASK_DEFAULT, jnp.bitwise_not(low_bits))


def _servo_mask_to_k(mask):
    mask = jnp.where(mask == 0, _SERVO_MASK_DEFAULT, mask)
    low_bit = mask & (jnp.uint32(0) - mask)
    low_bit = jnp.where(low_bit == 0, jnp.uint32(1), low_bit)
    k = jnp.floor(jnp.log2(low_bit.astype(jnp.float32))).astype(jnp.int32)
    return jnp.clip(k, 0, 31)


@jit
def _servo_update(arena):
    mask = arena.servo[0].astype(jnp.uint32)
    mask = jnp.where(mask == 0, _SERVO_MASK_DEFAULT, mask)
    k = _servo_mask_to_k(mask)
    spectrum = _blind_spectral_probe(arena)
    idx = jnp.arange(spectrum.shape[0], dtype=jnp.int32)
    start = jnp.maximum(k - 1, 0)
    p_buffer = jnp.sum(jnp.where(idx >= start, spectrum, 0.0))

    size = arena.rank.shape[0]
    ids = jnp.arange(size, dtype=jnp.int32)
    live = ids < arena.count.astype(jnp.int32)
    hot = (arena.rank == RANK_HOT) & live
    hot_count = jnp.sum(hot).astype(jnp.float32)
    denom = jnp.exp2(jnp.maximum(k - 1, 0).astype(jnp.float32))
    d_active = hot_count / denom

    spill = p_buffer > 0.25
    vacuum = (p_buffer < 0.10) & (d_active < 0.4)
    k_up = jnp.minimum(k + 1, 31)
    k_down = jnp.maximum(k - 1, 0)
    k_next = jnp.where(spill, k_up, jnp.where(vacuum, k_down, k))
    mask_next = _servo_mask_from_k(k_next)
    new_servo = arena.servo.at[0].set(mask_next)
    return arena._replace(servo=new_servo)


@jit
def op_rank(arena):
    ops = arena.opcode
    is_free = ops == OP_NULL
    is_inst = ops >= 10
    new_rank = jnp.where(is_free, RANK_FREE, jnp.where(is_inst, RANK_HOT, RANK_COLD))
    return arena._replace(rank=new_rank.astype(jnp.int8))


def _active_prefix_count(arena) -> jnp.ndarray:
    size = arena.rank.shape[0]
    # SYNC: host reads device scalar for active count (m1).
    count = _host_int(arena.count)
    count_i = int(count)
    if _TEST_GUARDS and count_i > size:
        raise RuntimeError(
            f"arena.count overflow: {count_i} exceeds arena size {size}"
        )
    # NOTE: clamp to size hides overflow outside test mode; guard above in tests.
    return _host_int(size) if count_i > size else count


def _invert_perm(perm):
    inv = jnp.empty_like(perm)
    return inv.at[perm].set(jnp.arange(perm.shape[0], dtype=perm.dtype))

def _apply_perm_and_swizzle(
    arena,
    perm,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSP_s renorm: layout-only; must commute with q/denote.
    if guard_cfg is None:
        guard_cfg = DEFAULT_GUARD_CONFIG
    safe_gather_fn = resolve_safe_gather_fn(
        safe_gather_fn=safe_gather_fn,
        policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    inv_perm = _invert_perm(perm)
    new_ops = arena.opcode[perm]
    new_arg1 = arena.arg1[perm]
    new_arg2 = arena.arg2[perm]
    new_rank = arena.rank[perm]
    # Guard pointer swizzles in test mode; mask to live region; NULL stays pinned at 0.
    ids = jnp.arange(new_arg1.shape[0], dtype=jnp.int32)
    live = ids < arena.count.astype(jnp.int32)
    idx1 = jnp.where(live, new_arg1, jnp.int32(0))
    idx2 = jnp.where(live, new_arg2, jnp.int32(0))
    g1 = safe_gather_fn(inv_perm, idx1, "swizzle.arg1")
    g2 = safe_gather_fn(inv_perm, idx2, "swizzle.arg2")
    swizzled_arg1 = jnp.where(live & (new_arg1 != 0), g1, 0)
    swizzled_arg2 = jnp.where(live & (new_arg2 != 0), g2, 0)
    # NOTE: value-bound guards for swizzled args in test mode are deferred to
    # IMPLEMENTATION_PLAN.md.
    # Swizzle is renormalization only; denotation must not change (plan).
    # See IMPLEMENTATION_PLAN.md (m3 denotation invariance).
    guard_slot0_perm_cfg(perm, inv_perm, "swizzle.perm", cfg=guard_cfg)
    guard_null_row_cfg(
        new_ops, swizzled_arg1, swizzled_arg2, "swizzle.row0", cfg=guard_cfg
    )
    guard_swizzle_args_cfg(
        swizzled_arg1,
        swizzled_arg2,
        live,
        arena.count,
        "swizzle.args",
        cfg=guard_cfg,
    )
    return (
        Arena(
            new_ops,
            swizzled_arg1,
            swizzled_arg2,
            new_rank,
            arena.count,
            arena.oom,
            arena.servo,
        ),
        inv_perm,
    )


@functools.partial(
    jax.jit, static_argnames=("safe_gather_fn", "safe_gather_policy", "guard_cfg")
)
def _op_sort_and_swizzle_with_perm_full(
    arena,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    size = arena.rank.shape[0]
    idx = jnp.arange(size, dtype=jnp.int32)
    sort_key = arena.rank.astype(jnp.int32) * (size + 1) + idx
    sort_key = sort_key.at[0].set(jnp.int32(-1))
    perm = jnp.argsort(sort_key)
    return _apply_perm_and_swizzle(
        arena,
        perm,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def _op_sort_and_swizzle_with_perm_prefix(
    arena,
    active_count,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    size = arena.rank.shape[0]
    if active_count <= 1:
        perm = jnp.arange(size, dtype=jnp.int32)
        return _apply_perm_and_swizzle(
            arena,
            perm,
            safe_gather_fn=safe_gather_fn,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
        )
    idx = jnp.arange(active_count, dtype=jnp.int32)
    sort_key = arena.rank[:active_count].astype(jnp.int32) * (active_count + 1) + idx
    sort_key = sort_key.at[0].set(jnp.int32(-1))
    perm_active = jnp.argsort(sort_key)
    tail = jnp.arange(active_count, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return _apply_perm_and_swizzle(
        arena,
        perm,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def op_sort_and_swizzle_with_perm(
    arena,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only.
    active_count = _active_prefix_count(arena)
    active_count_i = int(active_count)
    size = arena.rank.shape[0]
    if active_count_i >= size:
        return _op_sort_and_swizzle_with_perm_full(
            arena,
            safe_gather_fn=safe_gather_fn,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
        )
    return _op_sort_and_swizzle_with_perm_prefix(
        arena,
        active_count_i,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def op_sort_and_swizzle(
    arena,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only.
    sorted_arena, _ = op_sort_and_swizzle_with_perm(
        arena,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    return sorted_arena


def _blocked_perm(arena, block_size, morton=None, active_count=None):
    # BSPˢ: layout/space only.
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
        morton_u = (
            morton.reshape((num_blocks, block_size)).astype(jnp.uint32)
            & jnp.uint32(0x3FFF)
        )
    # Keep NULL row pinned to preserve slot-0 invariants across permutations.
    ranks = ranks.at[0, 0].set(jnp.uint32(0))
    morton_u = morton_u.at[0, 0].set(jnp.uint32(0))

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
    perm_active = (base + perm_local).reshape((active_blocks * block_size,)).astype(
        jnp.int32
    )
    tail = jnp.arange(active_blocks * block_size, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return perm


def op_sort_and_swizzle_blocked_with_perm(
    arena,
    block_size,
    morton=None,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only.
    active_count = _active_prefix_count(arena)
    perm = _blocked_perm(
        arena, block_size, morton=morton, active_count=int(active_count)
    )
    return _apply_perm_and_swizzle(
        arena,
        perm,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def op_sort_and_swizzle_blocked(
    arena,
    block_size,
    morton=None,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only.
    sorted_arena, _ = op_sort_and_swizzle_blocked_with_perm(
        arena,
        block_size,
        morton=morton,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
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
    arena,
    l2_block_size,
    l1_block_size,
    morton=None,
    do_global=False,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only.
    size = int(arena.rank.shape[0])
    if l2_block_size <= 0 or l1_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if size % l2_block_size != 0 or size % l1_block_size != 0:
        raise ValueError("block sizes must evenly divide arena size")
    if l1_block_size % l2_block_size != 0:
        raise ValueError("l1_block_size must be a multiple of l2_block_size")

    arena, inv_perm = op_sort_and_swizzle_blocked_with_perm(
        arena,
        l2_block_size,
        morton=morton,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    morton = _apply_perm_to_morton(morton, inv_perm)
    inv_perm_total = inv_perm

    if l1_block_size > l2_block_size:
        arena, inv_perm_l1 = op_sort_and_swizzle_blocked_with_perm(
            arena,
            l1_block_size,
            morton=morton,
            safe_gather_fn=safe_gather_fn,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
        )
        morton = _apply_perm_to_morton(morton, inv_perm_l1)
        inv_perm_total = inv_perm_l1[inv_perm_total]

    if do_global and l1_block_size < size:
        for block_size in _walk_block_sizes(l1_block_size, size):
            arena, inv_perm_global = op_sort_and_swizzle_blocked_with_perm(
                arena,
                block_size,
                morton=morton,
                safe_gather_fn=safe_gather_fn,
                safe_gather_policy=safe_gather_policy,
                guard_cfg=guard_cfg,
            )
            morton = _apply_perm_to_morton(morton, inv_perm_global)
            inv_perm_total = inv_perm_global[inv_perm_total]

    return arena, inv_perm_total


def op_sort_and_swizzle_hierarchical(
    arena,
    l2_block_size,
    l1_block_size,
    morton=None,
    do_global=False,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only.
    sorted_arena, _ = op_sort_and_swizzle_hierarchical_with_perm(
        arena,
        l2_block_size,
        l1_block_size,
        morton=morton,
        do_global=do_global,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    return sorted_arena


def swizzle_2to1_host(x, y):
    z = 0
    for i in range(10):
        x0 = (x >> (2 * i)) & 1
        x1 = (x >> (2 * i + 1)) & 1
        y0 = (y >> i) & 1
        z |= x0 << (3 * i)
        z |= x1 << (3 * i + 1)
        z |= y0 << (3 * i + 2)
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


_ARENA_COORD_SCHEME = os.environ.get(
    "PRISM_ARENA_COORD_SCHEME", "index"
).strip().lower()
_ARENA_COORD_GRID_LOG2 = None
_ARENA_COORD_GRID_MASK = None
if _ARENA_COORD_SCHEME not in ("index", "grid"):
    raise ValueError(
        f"PRISM_ARENA_COORD_SCHEME must be 'index' or 'grid' (got {_ARENA_COORD_SCHEME!r})"
    )
if _ARENA_COORD_SCHEME == "grid":
    value = os.environ.get("PRISM_ARENA_COORD_GRID_LOG2", "").strip()
    if not value or not value.isdigit():
        raise ValueError("PRISM_ARENA_COORD_GRID_LOG2 must be a non-negative integer")
    _ARENA_COORD_GRID_LOG2 = int(value)
    if _ARENA_COORD_GRID_LOG2 > 31:
        raise ValueError("PRISM_ARENA_COORD_GRID_LOG2 must be <= 31")
    _ARENA_COORD_GRID_MASK = (1 << _ARENA_COORD_GRID_LOG2) - 1


def swizzle_2to1(x, y):
    if _SWIZZLE_ACCEL is not None:
        return _SWIZZLE_ACCEL(x, y)
    return swizzle_2to1_dev(x, y)


def _arena_coords(arena):
    size = arena.opcode.shape[0]
    idx = jnp.arange(size, dtype=jnp.uint32)
    if _ARENA_COORD_SCHEME == "grid":
        x = idx & jnp.uint32(_ARENA_COORD_GRID_MASK)
        y = idx >> _ARENA_COORD_GRID_LOG2
        return x, y
    return idx, jnp.zeros_like(idx)


@jit
def op_morton(arena):
    # BSPˢ: layout/space only.
    x, y = _arena_coords(arena)
    return swizzle_2to1(x, y)


@functools.partial(
    jax.jit, static_argnames=("safe_gather_fn", "safe_gather_policy", "guard_cfg")
)
def _op_sort_and_swizzle_morton_with_perm_full(
    arena,
    morton,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    size = arena.rank.shape[0]
    idx = jnp.arange(size, dtype=jnp.uint32)
    rank_u = arena.rank.astype(jnp.uint32)
    morton_u = morton.astype(jnp.uint32) & jnp.uint32(0x3FFF)
    # Keep row 0 uniquely minimal even if other rows alias the 16-bit idx lane.
    idx_u = idx & jnp.uint32(0xFFFF)
    idx_u = jnp.where(idx_u == 0, jnp.uint32(1), idx_u)
    sort_key = (rank_u << 30) | (morton_u << 16) | idx_u
    sort_key = sort_key.at[0].set(jnp.uint32(0))
    perm = jnp.argsort(sort_key).astype(jnp.int32)
    return _apply_perm_and_swizzle(
        arena,
        perm,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def _op_sort_and_swizzle_morton_with_perm_prefix(
    arena,
    morton,
    active_count,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    size = arena.rank.shape[0]
    if active_count <= 1:
        perm = jnp.arange(size, dtype=jnp.int32)
        return _apply_perm_and_swizzle(
            arena,
            perm,
            safe_gather_fn=safe_gather_fn,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
        )
    idx = jnp.arange(active_count, dtype=jnp.uint32)
    rank_u = arena.rank[:active_count].astype(jnp.uint32)
    morton_u = morton[:active_count].astype(jnp.uint32) & jnp.uint32(0x3FFF)
    # Keep row 0 uniquely minimal even if other rows alias the 16-bit idx lane.
    idx_u = idx & jnp.uint32(0xFFFF)
    idx_u = jnp.where(idx_u == 0, jnp.uint32(1), idx_u)
    sort_key = (rank_u << 30) | (morton_u << 16) | idx_u
    sort_key = sort_key.at[0].set(jnp.uint32(0))
    perm_active = jnp.argsort(sort_key).astype(jnp.int32)
    tail = jnp.arange(active_count, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return _apply_perm_and_swizzle(
        arena,
        perm,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def op_sort_and_swizzle_morton_with_perm(
    arena,
    morton,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only.
    active_count = _active_prefix_count(arena)
    active_count_i = int(active_count)
    size = arena.rank.shape[0]
    if active_count_i >= size:
        return _op_sort_and_swizzle_morton_with_perm_full(
            arena,
            morton,
            safe_gather_fn=safe_gather_fn,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
        )
    return _op_sort_and_swizzle_morton_with_perm_prefix(
        arena,
        morton,
        active_count_i,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def op_sort_and_swizzle_morton(
    arena,
    morton,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only.
    sorted_arena, _ = op_sort_and_swizzle_morton_with_perm(
        arena,
        morton,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    return sorted_arena


@functools.partial(
    jax.jit, static_argnames=("safe_gather_fn", "safe_gather_policy", "guard_cfg")
)
def _op_sort_and_swizzle_servo_with_perm_full(
    arena,
    morton,
    servo_mask,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    size = arena.rank.shape[0]
    idx = jnp.arange(size, dtype=jnp.uint32)
    mask = jnp.where(servo_mask == 0, _SERVO_MASK_DEFAULT, servo_mask).astype(
        jnp.uint32
    )
    masked = morton.astype(jnp.uint32) & mask
    rank_u = arena.rank.astype(jnp.uint32)
    rank_u = rank_u.at[0].set(jnp.uint32(0))
    masked = masked.at[0].set(jnp.uint32(0))
    perm = jnp.lexsort((idx, masked, rank_u)).astype(jnp.int32)
    return _apply_perm_and_swizzle(
        arena,
        perm,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


@functools.partial(
    jax.jit,
    static_argnums=(2,),
    static_argnames=("safe_gather_fn", "safe_gather_policy", "guard_cfg"),
)
def _op_sort_and_swizzle_servo_with_perm_prefix(
    arena,
    morton,
    active_count,
    servo_mask,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    size = arena.rank.shape[0]
    if active_count <= 1:
        perm = jnp.arange(size, dtype=jnp.int32)
        return _apply_perm_and_swizzle(
            arena,
            perm,
            safe_gather_fn=safe_gather_fn,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
        )
    idx = jnp.arange(active_count, dtype=jnp.uint32)
    mask = jnp.where(servo_mask == 0, _SERVO_MASK_DEFAULT, servo_mask).astype(
        jnp.uint32
    )
    masked = morton[:active_count].astype(jnp.uint32) & mask
    rank_u = arena.rank[:active_count].astype(jnp.uint32)
    rank_u = rank_u.at[0].set(jnp.uint32(0))
    masked = masked.at[0].set(jnp.uint32(0))
    perm_active = jnp.lexsort((idx, masked, rank_u)).astype(jnp.int32)
    tail = jnp.arange(active_count, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return _apply_perm_and_swizzle(
        arena,
        perm,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def op_sort_and_swizzle_servo_with_perm(
    arena,
    morton,
    servo_mask,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only (servo-masked Morton).
    active_count = _active_prefix_count(arena)
    active_count_i = int(active_count)
    size = arena.rank.shape[0]
    if active_count_i >= size:
        return _op_sort_and_swizzle_servo_with_perm_full(
            arena,
            morton,
            servo_mask,
            safe_gather_fn=safe_gather_fn,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
        )
    return _op_sort_and_swizzle_servo_with_perm_prefix(
        arena,
        morton,
        active_count_i,
        servo_mask,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )


def op_sort_and_swizzle_servo(
    arena,
    morton,
    servo_mask,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # BSPˢ: layout/space only (servo-masked Morton).
    sorted_arena, _ = op_sort_and_swizzle_servo_with_perm(
        arena,
        morton,
        servo_mask,
        safe_gather_fn=safe_gather_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    return sorted_arena


__all__ = [
    "RANK_HOT",
    "RANK_WARM",
    "RANK_COLD",
    "RANK_FREE",
    "op_rank",
    "op_sort_and_swizzle",
    "op_sort_and_swizzle_blocked",
    "op_sort_and_swizzle_blocked_with_perm",
    "op_sort_and_swizzle_hierarchical",
    "op_sort_and_swizzle_hierarchical_with_perm",
    "op_sort_and_swizzle_morton",
    "op_sort_and_swizzle_morton_with_perm",
    "op_sort_and_swizzle_servo",
    "op_sort_and_swizzle_servo_with_perm",
    "op_sort_and_swizzle_with_perm",
    "op_morton",
    "swizzle_2to1",
    "swizzle_2to1_dev",
    "swizzle_2to1_host",
    "_servo_update",
]
