#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import sys
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


def _run_cnf2(ledger, frontier, iterations: int, validate: bool):
    for _ in range(iterations):
        ledger, frontier_prov, _, q_map = pv.cycle_candidates(
            ledger, frontier, validate_stratum=validate
        )
        frontier = pv.apply_q(q_map, frontier_prov)
    jax.block_until_ready(ledger.count)
    return ledger, frontier


def _collect_stats(profile: cProfile.Profile, top: int) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    entries = []
    for func, stat in stats.stats.items():
        cc, nc, tt, ct, _callers = stat
        filename, line, name = func
        entries.append(
            {
                "function": f"{filename}:{line}:{name}",
                "calls": int(nc),
                "self_sec": float(tt),
                "cumulative_sec": float(ct),
                "primitive_calls": int(cc),
            }
        )
    entries.sort(key=lambda item: item["cumulative_sec"], reverse=True)
    return entries[:top]


def _run_profile(engine: str, iterations: int, warmup: int, validate: bool, top: int):
    ledger, frontier = _build_suc_add_suc_frontier()
    if engine == "cnf2":
        os.environ.setdefault("PRISM_ENABLE_CNF2", "1")
        if warmup:
            ledger, frontier = _run_cnf2(ledger, frontier, warmup, validate)
        profile = cProfile.Profile()
        profile.enable()
        ledger, frontier = _run_cnf2(ledger, frontier, iterations, validate)
        profile.disable()
    else:
        if warmup:
            ledger, frontier = _run_intrinsic(ledger, frontier, warmup)
        profile = cProfile.Profile()
        profile.enable()
        ledger, frontier = _run_intrinsic(ledger, frontier, iterations)
        profile.disable()

    stats = pstats.Stats(profile)
    report = {
        "engine": engine,
        "iterations": iterations,
        "warmup": warmup,
        "validate_stratum": bool(validate),
        "jax_backend": jax.default_backend(),
        "total_time_sec": float(stats.total_tt),
        "top_functions": _collect_stats(profile, top),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile host dispatch overhead for Prism VM workloads."
    )
    parser.add_argument(
        "--engine",
        choices=("intrinsic", "cnf2"),
        default="intrinsic",
        help="Execution engine to profile (default: intrinsic).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iterations to profile (default: 10).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations before profiling (default: 1).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Top functions to report (default: 20).",
    )
    parser.add_argument(
        "--validate-stratum",
        action="store_true",
        help="Validate strata when profiling CNF2.",
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

    report = _run_profile(
        args.engine, args.iterations, args.warmup, args.validate_stratum, args.top
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is None:
        print(payload)
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
        print(f"audit_host_performance: wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
