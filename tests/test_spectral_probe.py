import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m5


def _make_arena(opcode, arg1, arg2, rank, count):
    return pv.Arena(
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
        rank=rank,
        count=count,
        oom=jnp.array(False, dtype=jnp.bool_),
        servo=jnp.zeros(3, dtype=jnp.uint32),
    )


def test_spectral_probe_tree_peak():
    size = 2048
    idx = jnp.arange(size, dtype=jnp.int32)
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg_base = idx - 1024
    arg1 = jnp.where(idx >= 1024, arg_base, jnp.int32(0))
    arg2 = arg1
    rank = jnp.where(idx >= 1024, pv.RANK_HOT, pv.RANK_FREE).astype(jnp.int8)
    count = jnp.array(size, dtype=jnp.int32)
    arena = _make_arena(opcode, arg1, arg2, rank, count)

    spectrum = pv._blind_spectral_probe(arena)
    assert float(spectrum[10]) > 0.8


def test_spectral_probe_noise_spread():
    size = 8192
    idx = jnp.arange(size, dtype=jnp.int32)
    opcode = jnp.zeros(size, dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    k = jax.random.randint(key, (size,), 0, 13, dtype=jnp.int32)
    mask = jnp.left_shift(jnp.int32(1), k)
    arg1 = jnp.bitwise_xor(idx, mask)
    arg2 = arg1
    rank = jnp.full(size, pv.RANK_HOT, dtype=jnp.int8)
    count = jnp.array(size, dtype=jnp.int32)
    arena = _make_arena(opcode, arg1, arg2, rank, count)

    spectrum = pv._blind_spectral_probe(arena)
    entropy = -jnp.sum(spectrum * jnp.log2(spectrum + 1e-6))
    assert float(entropy) > 3.0
