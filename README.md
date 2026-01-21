# Prism VM

Prism VM is a small JAX-backed interpreter for a tiny IR (zero/suc/add/mul) with
deduplication, basic static optimization, and kernel dispatch, plus an
experimental BSP arena pipeline.

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
Baseline REPL:
```
mise exec -- python prism_vm.py
```

Baseline program file:
```
mise exec -- python prism_vm.py tests/test_add_cache.txt
```

BSP REPL:
```
mise exec -- python prism_vm.py --mode bsp
```

BSP program file (with multiple cycles):
```
mise exec -- python prism_vm.py --mode bsp --cycles 3 tests/test_add_cache.txt
```

Disable sort in BSP mode:
```
mise exec -- python prism_vm.py --mode bsp --no-sort tests/test_add_cache.txt
```

Enable Morton ordering in BSP mode:
```
mise exec -- python prism_vm.py --mode bsp --morton tests/test_add_cache.txt
```

Enable block-local (hierarchical) sorting in BSP mode:
```
mise exec -- python prism_vm.py --mode bsp --block-size 256 tests/test_add_cache.txt
```

Benchmark compare matrix (baseline + BSP variants, CSV output):
```
mise exec -- python bench_compare.py --runs 3 --cycles 3 --out bench_results.csv
```

## Testing
Install pytest (once):
```
mise exec -- python -m pip install pytest
```

Run the suite:
```
mise exec -- pytest
```

## Repo layout
- `prism_vm.py` - VM, kernels, and REPL
- `tests/` - pytest suite and sample program fixtures
- `mise.toml` - Python toolchain config
- `in/` - design notes and evolution documents
