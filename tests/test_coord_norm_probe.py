import jax.numpy as jnp
import pytest

import prism_vm as pv


pytestmark = pytest.mark.m4


@pytest.fixture(autouse=True)
def _enable_probe(monkeypatch):
    monkeypatch.setenv("PRISM_COORD_NORM_PROBE", "1")


@pytest.mark.xfail(
    reason="m4: intern_nodes still vmaps coord_norm over all proposals",
    strict=True,
)
def test_coord_norm_probe_only_runs_for_pairs():
    pv.coord_norm_probe_reset()
    ledger = pv.init_ledger()
    ops = jnp.array(
        [pv.OP_ADD, pv.OP_COORD_PAIR, pv.OP_MUL, pv.OP_COORD_PAIR],
        dtype=jnp.int32,
    )
    a1 = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    a2 = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    _, ledger2 = pv.intern_nodes(ledger, ops, a1, a2)
    ledger2.count.block_until_ready()
    assert pv.coord_norm_probe_get() == 2


@pytest.mark.xfail(
    reason="m4: intern_nodes still vmaps coord_norm even when no coord ops exist",
    strict=True,
)
def test_coord_norm_probe_skips_non_coord_batch():
    pv.coord_norm_probe_reset()
    ledger = pv.init_ledger()
    ops = jnp.array([pv.OP_ADD, pv.OP_MUL, pv.OP_SUC, pv.OP_ZERO], dtype=jnp.int32)
    a1 = jnp.array([1, 1, 1, 0], dtype=jnp.int32)
    a2 = jnp.array([1, 1, 0, 0], dtype=jnp.int32)
    _, ledger2 = pv.intern_nodes(ledger, ops, a1, a2)
    ledger2.count.block_until_ready()
    assert pv.coord_norm_probe_get() == 0
