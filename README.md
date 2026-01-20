# Prism VM

Prism VM is a small JAX-backed interpreter for a tiny IR (zero/suc/add/mul) with
deduplication, basic static optimization, and kernel dispatch.

## Requirements
- Python via mise (`mise.toml`)
- JAX (CPU or CUDA)

## Setup
CPU-only:
```
mise exec -- python -m pip install jax jaxlib
```

CUDA 12:
```
mise exec -- python -m pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Run
REPL:
```
mise exec -- python prism_vm.py
```

Run a program file:
```
mise exec -- python prism_vm.py tests/test_add_cache.txt
```

## Repo layout
- `prism_vm.py` - VM, kernels, and REPL
- `tests/` - sample program fixtures
- `mise.toml` - Python toolchain config
