import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3


def _build_suc_over_mul():
    ledger = pv.init_ledger()
    mul_ids, ledger = mph.intern_nodes(
        ledger, [pv.OP_MUL], [pv.ZERO_PTR], [pv.ZERO_PTR]
    )
    mul_id = int(mul_ids[0])
    suc_ids, ledger = mph.intern_nodes(ledger, [pv.OP_SUC], [mul_id], [0])
    suc_id = int(suc_ids[0])
    return ledger, suc_id, mul_id


def test_project_ledger_closure_keeps_dependencies():
    ledger, suc_id, mul_id = _build_suc_over_mul()
    keys = list(mph.canon_state_ledger(ledger)[0])
    suc_key = mph.structural_key_for_id(ledger, suc_id)
    mul_key = mph.structural_key_for_id(ledger, mul_id)
    suc_idx = keys.index(suc_key)
    mul_idx = keys.index(mul_key)
    assert suc_idx < mul_idx

    proj_ledger, mapping = mph.project_ledger(ledger, suc_idx)
    assert mapping[mul_id] != 0
    proj_suc_key = mph.structural_key_for_id(
        proj_ledger, mapping.get(suc_id, 0)
    )
    assert proj_suc_key == suc_key


def test_project_ledger_idempotent():
    ledger, suc_id, _ = _build_suc_over_mul()
    keys = list(mph.canon_state_ledger(ledger)[0])
    suc_key = mph.structural_key_for_id(ledger, suc_id)
    max_id = keys.index(suc_key)

    proj_ledger, _ = mph.project_ledger(ledger, max_id)
    proj_again, _ = mph.project_ledger(proj_ledger, max_id)
    assert mph.canon_state_ledger(proj_ledger) == mph.canon_state_ledger(
        proj_again
    )
