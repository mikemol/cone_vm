#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tracemalloc
from typing import Any

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


def _serialize_stat(stat: tracemalloc.StatisticDiff) -> dict[str, Any]:
    return {
        "traceback": [str(line) for line in stat.traceback.format()],
        "size_diff_bytes": int(stat.size_diff),
        "count_diff": int(stat.count_diff),
    }


def _run_memory_audit(
    engine: str, iterations: int, warmup: int, validate: bool, top: int
):
    ledger, frontier = _build_suc_add_suc_frontier()
    tracemalloc.start()

    if engine == "cnf2":
        os.environ.setdefault("PRISM_ENABLE_CNF2", "1")
        if warmup:
            ledger, frontier = _run_cnf2(ledger, frontier, warmup, validate)
        snapshot_start = tracemalloc.take_snapshot()
        ledger, frontier = _run_cnf2(ledger, frontier, iterations, validate)
    else:
        if warmup:
            ledger, frontier = _run_intrinsic(ledger, frontier, warmup)
        snapshot_start = tracemalloc.take_snapshot()
        ledger, frontier = _run_intrinsic(ledger, frontier, iterations)

    jax.block_until_ready(ledger.count)
    snapshot_end = tracemalloc.take_snapshot()

    stats = snapshot_end.compare_to(snapshot_start, "lineno")
    total_growth = sum(stat.size_diff for stat in stats)
    report = {
        "engine": engine,
        "iterations": iterations,
        "warmup": warmup,
        "validate_stratum": bool(validate),
        "jax_backend": jax.default_backend(),
        "growth_bytes": int(total_growth),
        "growth_mb": float(total_growth) / (1024 * 1024),
        "top_deltas": [_serialize_stat(stat) for stat in stats[:top]],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture tracemalloc growth for Prism VM workloads."
    )
    parser.add_argument(
        "--engine",
        choices=("intrinsic", "cnf2"),
        default="intrinsic",
        help="Execution engine to measure (default: intrinsic).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iterations to measure (default: 10).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations before sampling (default: 1).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top allocation deltas to report (default: 10).",
    )
    parser.add_argument(
        "--validate-stratum",
        action="store_true",
        help="Validate strata when measuring CNF2.",
    )
    parser.add_argument(
        "--max-growth-mb",
        type=float,
        default=None,
        help="Fail when growth exceeds this MB (default: no threshold).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write JSON report to this path (default: stdout).",
    )
    args = parser.parse_args()

    if args.iterations < 0 or args.warmup < 0:
        print("iterations and warmup must be non-negative", file=sys.stderr)
        return 2
    if args.max_growth_mb is not None and args.max_growth_mb < 0:
        print("max-growth-mb must be non-negative", file=sys.stderr)
        return 2

    report = _run_memory_audit(
        args.engine, args.iterations, args.warmup, args.validate_stratum, args.top
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is None:
        print(payload)
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
        print(f"audit_memory_stability: wrote {args.json_out}")

    if args.max_growth_mb is not None:
        if report["growth_mb"] > args.max_growth_mb:
            print("audit_memory_stability: FAIL (growth exceeded)")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
