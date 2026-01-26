#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


DAMAGE_PATTERN = re.compile(
    r"Damage\s*:\s*cycles=(\d+)\s+hot=(\d+)\s+"
    r"edges=(\d+)/(\d+)\s+rate=([0-9.]+)\s+tile=(\d+)"
)


def collect_damage_metrics(base: Path) -> list[tuple[str, str, str, str, str, str, str]]:
    rows = []
    for path in sorted(base.rglob("damage_metrics*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            match = DAMAGE_PATTERN.search(line)
            if match:
                cycles, hot, edge_cross, edge_total, rate, tile = match.groups()
                rows.append(
                    (
                        str(path.relative_to(base)),
                        cycles,
                        hot,
                        edge_cross,
                        edge_total,
                        rate,
                        tile,
                    )
                )
    return rows


def write_summary(out_path: Path, rows: list[tuple[str, ...]]) -> None:
    lines = [
        "# Damage Metrics Summary",
        "",
        "| file | cycles | hot | edge_cross | edge_total | rate | tile |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if rows:
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| (none) | 0 | 0 | 0 | 0 | 0.0 | 0 |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect damage metrics telemetry into a summary table."
    )
    parser.add_argument(
        "--base",
        default="collected_report/raw",
        help="Base directory to scan for damage_metrics*.txt",
    )
    parser.add_argument(
        "--out",
        default="collected_report/damage_metrics_summary.md",
        help="Output markdown summary path",
    )
    args = parser.parse_args()
    base = Path(args.base)
    out_path = Path(args.out)
    if not base.exists():
        print(f"Base directory not found: {base}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_damage_metrics(base)
    write_summary(out_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
