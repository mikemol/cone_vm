import pytest

import prism_vm as pv
from tests import harness


@pytest.mark.m1
def test_manifest_q_projection_matches_baseline():
    expr = "(add (suc zero) (suc zero))"
    status, vm, ptr = harness.normalize_baseline(expr, max_steps=16)
    assert status == harness.STATUS_CONVERGED
    expected = vm.decode(ptr)
    ledger, root_id = pv.project_manifest_to_ledger(vm.manifest, ptr)
    vm_bsp = pv.PrismVM_BSP()
    vm_bsp.ledger = ledger
    assert vm_bsp.decode(root_id) == expected


@pytest.mark.m3
def test_arena_q_projection_invariant_across_schedule():
    expr = "(add (suc zero) (suc zero))"

    def _project(sort_cfg):
        vm = pv.PrismVM_BSP_Legacy()
        root_ptr = vm.parse(harness.tokenize(expr))
        arena = vm.arena
        for _ in range(4):
            arena, root_ptr = pv.cycle(
                arena,
                root_ptr,
                sort_cfg=sort_cfg,
            )
            root_ptr = pv._arena_ptr(pv._host_int_value(root_ptr))
        ledger, root_id = pv.project_arena_to_ledger(arena, root_ptr)
        vm_bsp = pv.PrismVM_BSP()
        vm_bsp.ledger = ledger
        return vm_bsp.decode(root_id)

    no_sort = _project(pv.ArenaSortConfig(do_sort=False, use_morton=False))
    morton = _project(pv.ArenaSortConfig(do_sort=True, use_morton=True))
    assert no_sort == morton
