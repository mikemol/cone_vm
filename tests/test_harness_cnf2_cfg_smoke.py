import jax.numpy as jnp

import prism_vm as pv
from tests import harness


def test_harness_cnf2_cfg_smoke():
    ledger = pv.init_ledger()
    ledger_state = harness.init_ledger_state()
    cfg = pv.Cnf2Config()
    frontier = jnp.zeros((0,), dtype=jnp.int32)

    candidates = pv.emit_candidates_cfg(ledger, frontier, cfg=cfg)
    compacted, count = pv.compact_candidates_cfg(candidates, cfg=cfg)
    compacted2, count2, idx = pv.compact_candidates_with_index_cfg(
        candidates, cfg=cfg
    )
    ids, ledger2, count3 = pv.intern_candidates_cfg(
        ledger, candidates, cfg=cfg
    )
    ids_state, ledger_state2, count_state = pv.intern_candidates_cfg(
        ledger_state, candidates, cfg=cfg
    )
    _ = pv.scatter_compacted_ids_cfg(
        idx, jnp.zeros_like(idx), jnp.int32(0), candidates.enabled.shape[0], cfg=cfg
    )

    assert compacted is not None
    assert compacted2 is not None
    assert ids is not None
    assert ledger2 is not None
    assert ids_state is not None
    assert ledger_state2 is not None
    assert isinstance(ledger_state2, pv.LedgerState)
    assert int(count) == int(count2)
    assert int(count3) == int(count)
    assert int(count_state) == int(count)

    emit_jit = harness.make_emit_candidates_jit_cfg()
    compact_jit = harness.make_compact_candidates_jit_cfg()
    compact_idx_jit = harness.make_compact_candidates_with_index_jit_cfg()
    intern_jit = harness.make_intern_candidates_jit_cfg()

    candidates_jit = emit_jit(ledger, frontier)
    compacted_jit, count_jit = compact_jit(candidates_jit)
    compacted_idx_jit, count_idx_jit, _ = compact_idx_jit(candidates_jit)
    ids_jit, ledger_jit, count_ids_jit = intern_jit(ledger, candidates_jit)

    assert candidates_jit is not None
    assert compacted_jit is not None
    assert compacted_idx_jit is not None
    assert ids_jit is not None
    assert ledger_jit is not None
    assert int(count_jit) == int(count_idx_jit)
    assert int(count_ids_jit) == int(count_jit)
