#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prism_vm as pv


def _build_suc_add_suc_frontier():
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_zero = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_zero], dtype=jnp.int32),
        jnp.array([suc_zero], dtype=jnp.int32),
    )
    add_id = add_ids[0]
    root_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([add_id], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    frontier = pv._committed_ids(jnp.array([root_ids[0]], dtype=jnp.int32))
    return ledger, frontier


def _run_intrinsic(ledger, frontier, iterations: int):
    frontier_ids = frontier.a
    for _ in range(iterations):
        ledger, frontier_ids = pv.cycle_intrinsic(ledger, frontier_ids)
    jax.block_until_ready(ledger.count)
    return ledger, pv._committed_ids(frontier_ids)


def _validate_mode_from_flag(validate: bool) -> pv.ValidateMode:
    return pv.ValidateMode.STRICT if validate else pv.ValidateMode.NONE


def _run_cnf2(ledger, frontier, iterations: int, validate: bool):
    validate_mode = _validate_mode_from_flag(validate)
    for _ in range(iterations):
        ledger, frontier_prov, _, q_map = pv.cycle_candidates(
            ledger, frontier, validate_mode=validate_mode
        )
        frontier = pv.apply_q(q_map, frontier_prov)
    jax.block_until_ready(ledger.count)
    return ledger, frontier


def _find_latest_trace(root: Path) -> Path | None:
    candidates = []
    for pattern in ("*.json", "*.json.gz"):
        candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a JAX profiler trace for a small Prism VM workload."
    )
    parser.add_argument(
        "--engine",
        choices=("intrinsic", "cnf2"),
        default="intrinsic",
        help="Execution engine to trace (default: intrinsic).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Iterations to trace (default: 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations before tracing (default: 1).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/jax-trace"),
        help="Profiler output directory (default: /tmp/jax-trace).",
    )
    parser.add_argument(
        "--validate-stratum",
        action="store_true",
        help="Validate strata when tracing CNF2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip tracing and only validate arguments.",
    )
    args = parser.parse_args()

    if args.iterations < 0 or args.warmup < 0:
        print("iterations and warmup must be non-negative", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"capture_trace: dry-run (out_dir={args.out_dir})")
        return 0

    ledger, frontier = _build_suc_add_suc_frontier()
    if args.engine == "cnf2":
        os.environ.setdefault("PRISM_ENABLE_CNF2", "1")
        if args.warmup:
            ledger, frontier = _run_cnf2(ledger, frontier, args.warmup, args.validate_stratum)
        with jax.profiler.trace(str(args.out_dir), create_perfetto_link=False):
            ledger, frontier = _run_cnf2(
                ledger, frontier, args.iterations, args.validate_stratum
            )
    else:
        if args.warmup:
            ledger, frontier = _run_intrinsic(ledger, frontier, args.warmup)
        with jax.profiler.trace(str(args.out_dir), create_perfetto_link=False):
            ledger, frontier = _run_intrinsic(ledger, frontier, args.iterations)

    trace_path = _find_latest_trace(args.out_dir)
    if trace_path is None:
        print("capture_trace: trace capture complete (no trace file found)")
        return 1
    print(f"capture_trace: trace captured at {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
