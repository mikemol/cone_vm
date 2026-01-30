import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

intern1 = harness.intern1
committed_ids = harness.committed_ids


pytestmark = [
    pytest.mark.m2,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


def _build_add_suc_frontier():
    ledger = pv.init_ledger()
    suc_zero, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, suc_zero, suc_zero)
    frontier = committed_ids(add_id)
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
