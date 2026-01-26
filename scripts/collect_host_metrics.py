#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _first_top_function(data: dict) -> tuple[str, str]:
    top = data.get("top_functions") or []
    if not top:
        return "", ""
    entry = top[0]
    func = str(entry.get("function", ""))
    cum = entry.get("cumulative_sec", "")
    return func, f"{cum:.6f}" if isinstance(cum, (float, int)) else str(cum)


def _first_top_delta(data: dict) -> tuple[str, str]:
    top = data.get("top_deltas") or []
    if not top:
        return "", ""
    entry = top[0]
    size = entry.get("size_diff_bytes", "")
    trace = entry.get("traceback") or []
    first_line = trace[0] if trace else ""
    return str(size), str(first_line)


def collect_host_metrics(base: Path):
    perf_rows = []
    mem_rows = []
    for path in sorted(base.rglob("host_perf_*.json")):
        data = _load_json(path)
        if not data:
            continue
        func, cum = _first_top_function(data)
        perf_rows.append(
            (
                str(path.relative_to(base)),
                str(data.get("engine", "")),
                str(data.get("iterations", "")),
                f"{data.get('total_time_sec', '')}",
                func,
                cum,
            )
        )
    for path in sorted(base.rglob("host_memory_*.json")):
        data = _load_json(path)
        if not data:
            continue
        size, trace = _first_top_delta(data)
        mem_rows.append(
            (
                str(path.relative_to(base)),
                str(data.get("engine", "")),
                str(data.get("iterations", "")),
                f"{data.get('growth_mb', '')}",
                size,
                trace,
            )
        )
    return perf_rows, mem_rows


def write_summary(out_path: Path, perf_rows, mem_rows) -> None:
    lines = [
        "# Host Telemetry Summary",
        "",
        "## Host Performance Baselines",
        "",
        "| file | engine | iterations | total_time_sec | top_function | top_cumulative_sec |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if perf_rows:
        for row in perf_rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| (none) | 0 | 0 | 0 |  |  |")

    lines.extend(
        [
            "",
            "## Host Memory Baselines",
            "",
            "| file | engine | iterations | growth_mb | top_delta_bytes | top_delta_trace |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if mem_rows:
        for row in mem_rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| (none) | 0 | 0 | 0 | 0 |  |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect host performance/memory telemetry into a summary table."
    )
    parser.add_argument(
        "--base",
        default="collected_report/raw",
        help="Base directory to scan for host_perf_*.json and host_memory_*.json",
    )
    parser.add_argument(
        "--out",
        default="collected_report/host_metrics_summary.md",
        help="Output markdown summary path",
    )
    args = parser.parse_args()
    base = Path(args.base)
    out_path = Path(args.out)
    if not base.exists():
        print(f"Base directory not found: {base}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    perf_rows, mem_rows = collect_host_metrics(base)
    write_summary(out_path, perf_rows, mem_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
