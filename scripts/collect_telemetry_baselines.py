#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _prefer_non_raw(path: Path, base: Path) -> bool:
    rel = path.relative_to(base)
    parts = list(rel.parts)
    if "raw" not in parts:
        return True
    idx = parts.index("raw")
    candidate = base.joinpath(*parts[:idx], *parts[idx + 1 :])
    return not candidate.exists()


def _collect_trace_reports(base: Path):
    rows = []
    for path in sorted(base.rglob("*trace*_report.json")):
        if not _prefer_non_raw(path, base):
            continue
        data = _load_json(path)
        if not data:
            continue
        rows.append(
            (
                str(path.relative_to(base)),
                str(data.get("mode", "")),
                f"{data.get('duty_cycle_pct', '')}",
                f"{data.get('scatter_ratio_pct', '')}",
                f"{data.get('kernel_active_ms', '')}",
                f"{data.get('wall_time_ms', '')}",
            )
        )
    return rows


def _collect_metadata_rows(base: Path):
    rows = []
    for path in sorted(base.rglob("telemetry_metadata*.json")):
        if not _prefer_non_raw(path, base):
            continue
        data = _load_json(path)
        if not data:
            continue
        env = data.get("env") or {}
        jax = data.get("jax") or {}
        py = data.get("python") or {}
        rows.append(
            (
                str(path.relative_to(base)),
                str(data.get("label", "")),
                str(data.get("milestone", "")),
                str(data.get("engine", "")),
                str(data.get("backend", "")),
                str(py.get("version", "")),
                str(jax.get("version", "")),
                str(jax.get("jaxlib_version", "")),
                str(jax.get("device_summary", "")),
                str(env.get("PRISM_ENABLE_SERVO", "")),
                str(env.get("PRISM_DAMAGE_TILE_SIZE", "")),
                str(env.get("PRISM_DAMAGE_METRICS", "")),
                str(env.get("PRISM_SWIZZLE_BACKEND", "")),
            )
        )
    return rows


def write_baselines(
    out_path: Path,
    damage_summary: Path | None,
    host_summary: Path | None,
    trace_rows,
    metadata_rows,
) -> None:
    lines = [
        "# Telemetry Baselines",
        "",
        "This report aggregates telemetry artifacts captured during CI runs.",
        "",
    ]
    if damage_summary and damage_summary.exists():
        lines.append("## Damage Metrics Summary")
        lines.append("")
        lines.append(_read_text(damage_summary) or "(missing)")
        lines.append("")
    if host_summary and host_summary.exists():
        lines.append("## Host Telemetry Summary")
        lines.append("")
        lines.append(_read_text(host_summary) or "(missing)")
        lines.append("")

    lines.append("## Run Metadata Summary")
    lines.append("")
    lines.append(
        "| file | label | milestone | engine | backend | python | jax | jaxlib | devices | servo | damage_tile | damage_metrics | swizzle_backend |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    if metadata_rows:
        for row in metadata_rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| (none) |  |  |  |  |  |  |  |  |  |  |  |  |")

    lines.append("## Trace Analysis Summary")
    lines.append("")
    lines.append(
        "| file | mode | duty_cycle_pct | scatter_ratio_pct | kernel_active_ms | wall_time_ms |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if trace_rows:
        for row in trace_rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| (none) |  |  |  |  |  |")

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect telemetry baselines into a single registry report."
    )
    parser.add_argument(
        "--base",
        default="collected_report/raw",
        help="Base directory to scan for trace reports",
    )
    parser.add_argument(
        "--damage-summary",
        default="collected_report/damage_metrics_summary.md",
        help="Damage metrics summary path",
    )
    parser.add_argument(
        "--host-summary",
        default="collected_report/host_metrics_summary.md",
        help="Host telemetry summary path",
    )
    parser.add_argument(
        "--out",
        default="collected_report/telemetry_baselines.md",
        help="Output registry path",
    )
    args = parser.parse_args()

    base = Path(args.base)
    out_path = Path(args.out)
    if not base.exists():
        print(f"Base directory not found: {base}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_rows = _collect_trace_reports(base)
    damage_summary = Path(args.damage_summary) if args.damage_summary else None
    host_summary = Path(args.host_summary) if args.host_summary else None
    metadata_rows = _collect_metadata_rows(base)
    write_baselines(out_path, damage_summary, host_summary, trace_rows, metadata_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
