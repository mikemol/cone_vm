import sys
import types

import pytest

import prism_vm as pv

pytestmark = pytest.mark.m4


def test_gpu_watchdog_disabled(monkeypatch):
    monkeypatch.delenv("PRISM_GPU_METRICS", raising=False)
    watchdog = pv._gpu_watchdog_create()
    assert watchdog is None


def test_gpu_watchdog_missing_module(monkeypatch):
    monkeypatch.setenv("PRISM_GPU_METRICS", "1")
    monkeypatch.setitem(sys.modules, "pynvml", None)
    watchdog = pv._gpu_watchdog_create()
    assert watchdog is None


def test_gpu_watchdog_stubbed_metrics(monkeypatch):
    monkeypatch.setenv("PRISM_GPU_METRICS", "1")
    monkeypatch.setenv("PRISM_GPU_INDEX", "0")

    class _Util:
        def __init__(self, gpu, memory):
            self.gpu = gpu
            self.memory = memory

    class _Mem:
        def __init__(self, used, total):
            self.used = used
            self.total = total

    stub = types.SimpleNamespace()
    stub.NVML_CLOCK_SM = 0
    stub.NVMLError = RuntimeError
    stub.nvmlInit = lambda: None
    stub.nvmlShutdown = lambda: None
    stub.nvmlDeviceGetHandleByIndex = lambda idx: f"handle-{idx}"
    stub.nvmlDeviceGetUtilizationRates = lambda handle: _Util(42, 7)
    stub.nvmlDeviceGetMemoryInfo = (
        lambda handle: _Mem(3 * 1024**2, 8 * 1024**2)
    )
    stub.nvmlDeviceGetClockInfo = lambda handle, clock: 1234
    stub.nvmlDeviceGetPowerUsage = lambda handle: 50000

    monkeypatch.setitem(sys.modules, "pynvml", stub)
    watchdog = pv._gpu_watchdog_create()
    assert watchdog is not None
    stats = watchdog.poll()
    assert stats is not None
    assert stats["gpu_util"] == 42
    assert stats["mem_io"] == 7
    assert stats["vram_used_mb"] == 3
    assert stats["vram_total_mb"] == 8
    assert stats["power_w"] == 50.0
    assert stats["sm_clock"] == 1234
    watchdog.close()
