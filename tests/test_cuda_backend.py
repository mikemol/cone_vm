import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import pytest


def _gpu_present() -> bool:
    proc_gpus = Path("/proc/driver/nvidia/gpus")
    if proc_gpus.exists():
        try:
            return any(proc_gpus.iterdir())
        except Exception:
            return True
    return os.path.exists("/dev/nvidia0")


@pytest.mark.m1
def test_cuda_backend_available_when_gpu_present():
    if not _gpu_present():
        pytest.skip("no NVIDIA GPU device nodes present")
    cuda_version = getattr(jaxlib, "cuda_version", None)
    try:
        devices = jax.devices("gpu")
    except Exception as exc:
        pytest.fail(f"GPU present but JAX GPU backend failed to initialize: {exc}")
    if not devices:
        pytest.fail("GPU present but JAX reported no GPU devices")
    # JAX 0.8+ may use the CUDA plugin path (cpu jaxlib + cuda plugin). In that
    # case cuda_version can be None while GPU devices are still usable.
    if cuda_version is None:
        x = jax.device_put(jnp.array([1.0], dtype=jnp.float32), device=devices[0])
        if x.device.platform != "gpu":
            pytest.fail(
                "GPU device reported but device_put landed on non-GPU platform"
            )
