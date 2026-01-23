import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv


pytestmark = pytest.mark.m1


@jax.jit
def _gather_bad_neg(x):
    return pv.safe_gather_1d(x, jnp.int32(-1), "test.neg")


@jax.jit
def _gather_bad_oob(x):
    return pv.safe_gather_1d(x, jnp.int32(x.shape[0]), "test.oob")


@jax.jit
def _gather_ok(x):
    return pv.safe_gather_1d(x, jnp.int32(2), "test.ok")


def _skip_if_no_debug_callback():
    if not pv._HAS_DEBUG_CALLBACK:
        pytest.skip("jax.debug.callback not available")


def test_gather_guard_negative_index_raises():
    _skip_if_no_debug_callback()
    x = jnp.arange(5, dtype=jnp.int32)
    with pytest.raises(RuntimeError, match=r"gather index out of bounds"):
        _gather_bad_neg(x).block_until_ready()


def test_gather_guard_oob_raises():
    _skip_if_no_debug_callback()
    x = jnp.arange(5, dtype=jnp.int32)
    with pytest.raises(RuntimeError, match=r"gather index out of bounds"):
        _gather_bad_oob(x).block_until_ready()


def test_gather_guard_valid_indices_noop():
    _skip_if_no_debug_callback()
    x = jnp.arange(5, dtype=jnp.int32)
    y = _gather_ok(x).block_until_ready()
    assert int(y) == 2
