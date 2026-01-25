import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

pytestmark = pytest.mark.m3


def test_denotation_not_structural_representation():
    expr = "(add zero (suc zero))"
    norm = "(suc zero)"
    pretty_expr = harness.denote_pretty_bsp_intrinsic(expr, max_steps=64)
    pretty_norm = harness.denote_pretty_bsp_intrinsic(norm, max_steps=64)
    assert pretty_expr == pretty_norm

    vm = pv.PrismVM_BSP()
    root_expr = harness.parse_expr(vm, expr)
    root_norm = harness.parse_expr(vm, norm)
    assert vm.decode(root_expr) != vm.decode(root_norm)

    if not pv._TEST_GUARDS:
        pytest.skip("structural hash only available with test guards")
    hash_expr = pv._ledger_root_hash_host(
        vm.ledger, jnp.int32(int(root_expr))
    )
    hash_norm = pv._ledger_root_hash_host(
        vm.ledger, jnp.int32(int(root_norm))
    )
    assert hash_expr != hash_norm
