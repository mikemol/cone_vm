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

Swizzle backend (optional GPU acceleration, falls back to JAX on CPU; set `pallas` or `triton`):
```
PRISM_SWIZZLE_BACKEND=triton mise exec -- python prism_vm.py --mode bsp --morton tests/test_add_cache.txt
```

Benchmark compare matrix (baseline + BSP variants, CSV output):
```
mise exec -- python bench_compare.py --runs 3 --cycles 3 --out bench_results.csv
```

Benchmark with swizzle backend sweep:
```
mise exec -- python bench_compare.py --swizzle-backends jax,pallas,triton --runs 3 --cycles 3 --out bench_results.csv
```

Benchmark with hierarchical modes (L2/L1/global) enabled:
```
mise exec -- python bench_compare.py --hierarchy-l1-mult 4 --runs 3 --cycles 3 --out bench_results.csv
```

Note: hierarchy modes are included by default; use `--hierarchy-no-global` to drop the global stage, or `--hierarchy-morton` to include Morton variants.

Target hierarchy stress workloads only:
```
mise exec -- python bench_compare.py --workloads arena_hierarchy_l2256_l11024 --runs 3 --cycles 3 --out bench_results.csv
```

Override hierarchy workload sizes explicitly:
```
mise exec -- python bench_compare.py --hierarchy-workload-l2 128 --hierarchy-workload-l1 512 --runs 3 --cycles 3 --out bench_results.csv
```

Sweep arena sizes and block sizes to find inflection points:
```
mise exec -- python bench_compare.py --block-sizes 64,128,256,512 --arena-counts 8000,16000,32000 --runs 3 --cycles 3 --out bench_results.csv
```

Phoronix-style suite (CSV + Markdown + SVG/PNG plots, CPU/GPU):
```
mise exec -- python bench_phoronix.py --block-sizes 64,128,256,512 --arena-counts 8000,16000,32000 --runs 3 --cycles 3 --out-dir bench_phoronix
```

Note: `bench_phoronix.py` uses matplotlib if available; otherwise it falls back to a minimal built-in plotter.

## Testing
Install pytest (once):
```
mise exec -- python -m pip install pytest
```

Run the suite:
```
mise exec -- pytest
```

## Policy
This repo uses a self-hosted runner. Read `POLICY_SEED.md` before changing any
workflow or CI behavior. Install advisory hooks with:
```
scripts/install_policy_hooks.sh
```

See `CONTRIBUTING.md` for the guardrails and required checks.

## Milestones
m1 semantic commitments: Ledger interning uses full key-byte equality, univalence
hard-cap is enforced (overflow => corrupt), corrupt/oom are sticky stop-paths
(no further mutation), and baseline vs ledger equivalence holds on the m1 suite.
Changes to these commitments require a milestone bump and updates in
`MILESTONES.md`.

## Repo layout
- `prism_vm.py` - VM, kernels, and REPL
- `tests/` - pytest suite and sample program fixtures
- `mise.toml` - Python toolchain config
- `in/` - design notes and evolution documents
