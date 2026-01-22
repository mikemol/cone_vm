import prism_vm as pv
from tests import harness


def _eval(expr):
    return harness.denote_baseline(expr)


def _count_suc(vm, ptr):
    count = 0
    while True:
        op = int(vm.manifest.opcode[ptr])
        if op == pv.OP_ZERO:
            return count
        assert op == pv.OP_SUC
        ptr = int(vm.manifest.arg1[ptr])
        count += 1


def test_mul_by_zero():
    vm, res = _eval("(mul zero (suc (suc zero)))")
    assert _count_suc(vm, res) == 0


def test_mul_by_one():
    vm, res = _eval("(mul (suc zero) (suc (suc zero)))")
    assert _count_suc(vm, res) == 2


def test_mul_small():
    vm, res = _eval("(mul (suc (suc zero)) (suc (suc (suc zero))))")
    assert _count_suc(vm, res) == 6
