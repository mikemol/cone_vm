import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv


pytestmark = [
    pytest.mark.m1,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


@jax.jit
def _scatter_bad_neg(x):
    idx = jnp.array([-1], dtype=jnp.int32)
    vals = jnp.array([7], dtype=x.dtype)
    return pv._scatter_drop(x, idx, vals, "test.neg")


@jax.jit
def _scatter_bad_oob(x):
    idx = jnp.array([x.shape[0] + 1], dtype=jnp.int32)
    vals = jnp.array([7], dtype=x.dtype)
    return pv._scatter_drop(x, idx, vals, "test.oob")


@jax.jit
def _scatter_sentinel(x):
    idx = jnp.array([x.shape[0]], dtype=jnp.int32)
    vals = jnp.array([7], dtype=x.dtype)
    return pv._scatter_drop(x, idx, vals, "test.sentinel")


@jax.jit
def _scatter_strict_sentinel(x):
    idx = jnp.array([x.shape[0]], dtype=jnp.int32)
    vals = jnp.array([7], dtype=x.dtype)
    return pv._scatter_strict(x, idx, vals, "test.strict_sentinel")


@jax.jit
def _scatter_ok(x):
    idx = jnp.array([2], dtype=jnp.int32)
    vals = jnp.array([9], dtype=x.dtype)
    return pv._scatter_drop(x, idx, vals, "test.ok")


def _skip_if_no_debug_callback():
    if not pv._HAS_DEBUG_CALLBACK:
        pytest.skip("jax.debug.callback not available")


def test_scatter_guard_sanity_requires_debug_callback():
    assert pv._TEST_GUARDS, "PRISM_TEST_GUARDS must be enabled for guard tests"
    _skip_if_no_debug_callback()
    x = jnp.zeros(4, dtype=jnp.int32)
    with pytest.raises(RuntimeError, match=r"scatter index out of bounds"):
        _scatter_bad_oob(x).block_until_ready()


def test_scatter_guard_negative_index_raises():
    _skip_if_no_debug_callback()
    x = jnp.zeros(4, dtype=jnp.int32)
    with pytest.raises(RuntimeError, match=r"scatter index out of bounds"):
        _scatter_bad_neg(x).block_until_ready()


def test_scatter_guard_oob_raises():
    _skip_if_no_debug_callback()
    x = jnp.zeros(4, dtype=jnp.int32)
    with pytest.raises(RuntimeError, match=r"scatter index out of bounds"):
        _scatter_bad_oob(x).block_until_ready()


def test_scatter_guard_allows_sentinel_drop():
    _skip_if_no_debug_callback()
    x = jnp.arange(4, dtype=jnp.int32)
    y = _scatter_sentinel(x).block_until_ready()
    assert jnp.array_equal(y, x)


def test_scatter_guard_strict_rejects_sentinel():
    _skip_if_no_debug_callback()
    x = jnp.arange(4, dtype=jnp.int32)
    with pytest.raises(RuntimeError, match=r"scatter index out of bounds"):
        _scatter_strict_sentinel(x).block_until_ready()


def test_scatter_guard_valid_index_writes():
    _skip_if_no_debug_callback()
    x = jnp.zeros(4, dtype=jnp.int32)
    y = _scatter_ok(x).block_until_ready()
    assert int(y[2]) == 9
