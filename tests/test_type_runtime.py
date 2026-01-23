import pytest

import jax.numpy as jnp

import prism_vm as pv

pytest.importorskip("jaxtyping")
pytest.importorskip("beartype")

from beartype import beartype
from jaxtyping import Array, Bool, Int32, jaxtyped


@jaxtyped(typechecker=beartype)
def _checked_candidate_indices(
    enabled: Int32[Array, "n"],
) -> tuple[Int32[Array, "n"], Bool[Array, "n"], Int32[Array, ""]]:
    return pv._candidate_indices(enabled)


def test_candidate_indices_runtime_typecheck_accepts_int32():
    enabled = jnp.array([0, 1, 0], dtype=jnp.int32)
    idx, valid, count = _checked_candidate_indices(enabled)
    assert idx.dtype == jnp.int32
    assert valid.dtype == jnp.bool_
    assert count.shape == ()


def test_candidate_indices_runtime_typecheck_rejects_float():
    enabled = jnp.array([0.0, 1.0], dtype=jnp.float32)
    with pytest.raises(TypeError):
        _checked_candidate_indices(enabled)


def test_candidate_indices_runtime_typecheck_rejects_2d():
    enabled = jnp.array([[0, 1]], dtype=jnp.int32)
    with pytest.raises(TypeError):
        _checked_candidate_indices(enabled)
