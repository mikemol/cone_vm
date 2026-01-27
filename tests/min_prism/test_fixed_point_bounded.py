import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3


def _build_roots():
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_id = int(suc_ids[0])
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    return ledger, [int(add_ids[0]), int(mul_ids[0])]


@pytest.mark.parametrize("max_steps", [6])
def test_novelty_fixed_point_bounded_intrinsic(max_steps):
    ledger, roots = _build_roots()
    for root in roots:
        steps, ledger, _, stable = mph.fixed_point_steps_intrinsic(
            ledger, root, max_steps=max_steps
        )
        assert stable
        assert steps <= max_steps


@pytest.mark.parametrize("max_steps", [8])
def test_novelty_fixed_point_bounded_cnf2(max_steps):
    ledger, roots = _build_roots()
    for root in roots:
        steps, ledger, _, stable = mph.fixed_point_steps_cnf2(
            ledger, root, max_steps=max_steps
        )
        assert stable
        assert steps <= max_steps
