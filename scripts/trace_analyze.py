#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Iterable


KERNEL_HINTS = ("thunk", "fusion", "kernel", "xla")
MEMORY_HINTS = ("scatter", "gather", "copy", "slice", "concat")
COMPUTE_HINTS = ("gemm", "conv", "dot", "fusion")


def _load_trace(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_latest_trace(root: Path) -> Path | None:
    candidates = []
    for pattern in ("*.json", "*.json.gz"):
        candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _iter_events(data: dict) -> Iterable[dict]:
    events = data.get("traceEvents")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                yield event


def analyze_trace(trace_path: Path, min_duty: float, max_scatter: float) -> bool:
    data = _load_trace(trace_path)
    events = list(_iter_events(data))
    if not events:
        print("trace_analyze: empty or invalid traceEvents")
        return False

    total_gpu_time = 0.0
    compute_kernels = 0.0
    memory_kernels = 0.0
    other_kernels = 0.0
    start_time = None
    end_time = None

    for event in events:
        name = event.get("name")
        dur = event.get("dur")
        ts = event.get("ts")
        if name is None or dur is None or ts is None:
            continue
        try:
            dur_val = float(dur)
            ts_val = float(ts)
        except (TypeError, ValueError):
            continue
        if start_time is None or ts_val < start_time:
            start_time = ts_val
        if end_time is None or (ts_val + dur_val) > end_time:
            end_time = ts_val + dur_val

        name_l = str(name).lower()
        if any(hint in name_l for hint in KERNEL_HINTS):
            total_gpu_time += dur_val
            if any(hint in name_l for hint in MEMORY_HINTS):
                memory_kernels += dur_val
            elif any(hint in name_l for hint in COMPUTE_HINTS):
                compute_kernels += dur_val
            else:
                other_kernels += dur_val

    if total_gpu_time == 0.0:
        print("trace_analyze: no GPU kernels detected (CPU trace?)")
        return True

    if start_time is None or end_time is None or end_time <= start_time:
        print("trace_analyze: invalid timeline bounds")
        return False

    wall_time = end_time - start_time
    duty_cycle = (total_gpu_time / wall_time) * 100.0
    scatter_ratio = (memory_kernels / total_gpu_time) * 100.0

    print("trace_analyze: report")
    print(f"  wall_time_ms={wall_time/1000:.2f}")
    print(f"  gpu_active_ms={total_gpu_time/1000:.2f}")
    print(f"  duty_cycle_pct={duty_cycle:.2f}")
    print(f"  scatter_ratio_pct={scatter_ratio:.2f}")
    print(f"  thresholds: duty>={min_duty:.1f} scatter<={max_scatter:.1f}")
    print("trace_analyze: kernel_breakdown_ms")
    print(f"  compute={compute_kernels/1000:.2f}")
    print(f"  memory={memory_kernels/1000:.2f}")
    print(f"  other={other_kernels/1000:.2f}")

    if duty_cycle < min_duty:
        print("trace_analyze: FAIL (low duty cycle)")
        return False
    if scatter_ratio > max_scatter:
        print("trace_analyze: FAIL (high scatter ratio)")
        return False
    print("trace_analyze: PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze JAX Perfetto trace for duty cycle and scatter ratio."
    )
    parser.add_argument(
        "trace",
        nargs="?",
        help="Trace file (.json or .json.gz). When omitted, search /tmp/jax-trace.",
    )
    parser.add_argument(
        "--min-duty",
        type=float,
        default=50.0,
        help="Minimum duty cycle percent to pass (default: 50.0).",
    )
    parser.add_argument(
        "--max-scatter",
        type=float,
        default=30.0,
        help="Maximum scatter ratio percent to pass (default: 30.0).",
    )
    args = parser.parse_args()

    if args.trace:
        trace_path = Path(args.trace)
    else:
        trace_path = _find_latest_trace(Path("/tmp/jax-trace"))
        if trace_path is None:
            print("trace_analyze: no trace files found in /tmp/jax-trace", file=sys.stderr)
            return 1

    if not trace_path.exists():
        print(f"trace_analyze: trace not found: {trace_path}", file=sys.stderr)
        return 1

    try:
        ok = analyze_trace(trace_path, args.min_duty, args.max_scatter)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"trace_analyze: failed to load trace: {exc}", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
