import prism_vm as pv


def test_repl_mode_switch():
    assert hasattr(pv, "make_vm"), "make_vm missing"
    assert hasattr(pv, "PrismVM_BSP"), "PrismVM_BSP missing"
    vm = pv.make_vm("bsp")
    assert isinstance(vm, pv.PrismVM_BSP)
    assert hasattr(vm, "ledger")


def test_repl_default_is_baseline():
    assert hasattr(pv, "make_vm"), "make_vm missing"
    vm = pv.make_vm()
    assert isinstance(vm, pv.PrismVM)
