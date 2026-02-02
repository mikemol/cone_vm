import pytest

import prism_vm as pv
from tests import harness

pytestmark = pytest.mark.m4


def test_damage_metrics_disabled_noop(monkeypatch):
    monkeypatch.delenv("PRISM_DAMAGE_METRICS", raising=False)
    monkeypatch.delenv("PRISM_DAMAGE_TILE_SIZE", raising=False)
    pv.damage_metrics_reset()
    _ = harness.denote_pretty_arena(
        "(add (suc zero) (suc zero))",
        steps=2,
        sort_cfg=pv.ArenaSortConfig(do_sort=True, use_morton=True),
    )
    metrics = pv.damage_metrics_get()
    assert metrics["cycles"] == 0
    assert metrics["edge_total"] == 0
    assert metrics["edge_cross"] == 0


def test_damage_metrics_no_semantic_effect(monkeypatch):
    expr = "(add (suc zero) (suc zero))"
    monkeypatch.delenv("PRISM_DAMAGE_METRICS", raising=False)
    monkeypatch.delenv("PRISM_DAMAGE_TILE_SIZE", raising=False)
    pv.damage_metrics_reset()
    baseline = harness.denote_pretty_arena(
        expr, steps=2, sort_cfg=pv.ArenaSortConfig(do_sort=True, use_morton=True)
    )
    monkeypatch.setenv("PRISM_DAMAGE_METRICS", "1")
    monkeypatch.setenv("PRISM_DAMAGE_TILE_SIZE", "2")
    pv.damage_metrics_reset()
    with_metrics = harness.denote_pretty_arena(
        expr, steps=2, sort_cfg=pv.ArenaSortConfig(do_sort=True, use_morton=True)
    )
    assert baseline == with_metrics
    metrics = pv.damage_metrics_get()
    assert metrics["cycles"] > 0
