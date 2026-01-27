import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv


pytestmark = [
    pytest.mark.m2,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


def _build_add_suc_frontier():
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_zero = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_zero], dtype=jnp.int32),
        jnp.array([suc_zero], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([add_ids[0]], dtype=jnp.int32))
    return ledger, frontier


def test_cnf2_metrics_disabled_noop(monkeypatch):
    monkeypatch.delenv("PRISM_CNF2_METRICS", raising=False)
    monkeypatch.setenv("PRISM_ENABLE_CNF2", "1")
    pv.cnf2_metrics_reset()
    ledger, frontier = _build_add_suc_frontier()
    pv.cycle_candidates(ledger, frontier)
    metrics = pv.cnf2_metrics_get()
    assert metrics["cycles"] == 0
    assert metrics["rewrite_child"] == 0
    assert metrics["changed"] == 0
    assert metrics["wrap_emit"] == 0


def test_cnf2_metrics_counts(monkeypatch):
    monkeypatch.setenv("PRISM_CNF2_METRICS", "1")
    monkeypatch.setenv("PRISM_ENABLE_CNF2", "1")
    monkeypatch.setenv("PRISM_ENABLE_CNF2_SLOT1", "1")
    pv.cnf2_metrics_reset()
    ledger, frontier = _build_add_suc_frontier()
    pv.cycle_candidates(ledger, frontier)
    metrics = pv.cnf2_metrics_get()
    assert metrics["cycles"] == 1
    assert metrics["rewrite_child"] >= 1
    assert metrics["changed"] >= 1
    assert metrics["wrap_emit"] == 0
