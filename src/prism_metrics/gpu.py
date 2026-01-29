from prism_vm_core.gating import _gpu_metrics_device_index, _gpu_metrics_enabled


class GPUWatchdog:
    def __init__(self, device_index=0):
        self.enabled = False
        self.handle = None
        self._nvml = None
        try:
            import importlib

            nvml = importlib.import_module("pynvml")
            nvml.nvmlInit()
            self.handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml = nvml
            self.enabled = True
        except Exception:
            self.enabled = False
            self._nvml = None
            self.handle = None

    def poll(self):
        if not self.enabled or self._nvml is None:
            return None
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self.handle)
            try:
                power_mw = self._nvml.nvmlDeviceGetPowerUsage(self.handle)
            except Exception:
                power_mw = None
            try:
                clock = self._nvml.nvmlDeviceGetClockInfo(
                    self.handle, self._nvml.NVML_CLOCK_SM
                )
            except Exception:
                clock = None
        except Exception:
            return None

        return {
            "gpu_util": int(getattr(util, "gpu", 0)),
            "mem_io": int(getattr(util, "memory", 0)),
            "vram_used_mb": int(getattr(mem, "used", 0) // (1024**2)),
            "vram_total_mb": int(getattr(mem, "total", 0) // (1024**2)),
            "power_w": (
                float(power_mw) / 1000.0 if power_mw is not None else None
            ),
            "sm_clock": int(clock) if clock is not None else None,
        }

    def close(self):
        if self.enabled and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self.enabled = False
        self._nvml = None
        self.handle = None


def _gpu_watchdog_create():
    if not _gpu_metrics_enabled():
        return None
    watchdog = GPUWatchdog(_gpu_metrics_device_index())
    if not watchdog.enabled:
        return None
    return watchdog


__all__ = ["GPUWatchdog", "_gpu_watchdog_create"]
