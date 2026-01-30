import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

intern_nodes = harness.intern_nodes


pytestmark = pytest.mark.m4


@pytest.fixture(autouse=True)
def _enable_probe(monkeypatch):
    monkeypatch.setenv("PRISM_COORD_NORM_PROBE", "1")


def test_coord_norm_probe_only_runs_for_pairs():
    if not pv._HAS_DEBUG_CALLBACK:
        pytest.skip("jax.debug.callback not available")
    pv.coord_norm_probe_reset()
    ledger = pv.init_ledger()
    ops = jnp.array(
        [pv.OP_ADD, pv.OP_COORD_PAIR, pv.OP_MUL, pv.OP_COORD_PAIR],
        dtype=jnp.int32,
    )
    a1 = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    a2 = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    _, ledger2 = intern_nodes(ledger, ops, a1, a2)
    ledger2.count.block_until_ready()
    assert pv.coord_norm_probe_get() == 2


def test_coord_norm_probe_skips_non_coord_batch():
    if not pv._HAS_DEBUG_CALLBACK:
        pytest.skip("jax.debug.callback not available")
    pv.coord_norm_probe_reset()
    ledger = pv.init_ledger()
    ops = jnp.array([pv.OP_ADD, pv.OP_MUL, pv.OP_SUC, pv.OP_ZERO], dtype=jnp.int32)
    a1 = jnp.array([1, 1, 1, 0], dtype=jnp.int32)
    a2 = jnp.array([1, 1, 0, 0], dtype=jnp.int32)
    _, ledger2 = intern_nodes(ledger, ops, a1, a2)
    ledger2.count.block_until_ready()
    assert pv.coord_norm_probe_get() == 0
