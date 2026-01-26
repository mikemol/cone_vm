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


def _iter_input_files(items: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob("damage_metrics*.txt")))
        else:
            files.append(path)
    return files


def _collect_by_tile(files: list[Path]) -> dict[int, tuple[int, int]]:
    totals: dict[int, tuple[int, int]] = {}
    for path in files:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            match = DAMAGE_PATTERN.search(line)
            if not match:
                continue
            edge_cross = int(match.group(3))
            edge_total = int(match.group(4))
            tile = int(match.group(6))
            prev_cross, prev_total = totals.get(tile, (0, 0))
            totals[tile] = (prev_cross + edge_cross, prev_total + edge_total)
    return totals


def _rate(edge_cross: int, edge_total: int) -> float:
    return (edge_cross / edge_total) if edge_total else 0.0


def _write_report(out_path: Path, totals: dict[int, tuple[int, int]]) -> None:
    tiles = sorted(totals.keys())
    if len(tiles) < 2:
        raise ValueError("need at least two tile sizes to compute delta")
    lines = ["Damage metrics delta report"]
    for tile in tiles:
        edge_cross, edge_total = totals[tile]
        rate = _rate(edge_cross, edge_total)
        lines.append(
            f"tile={tile} edges={edge_cross}/{edge_total} rate={rate:.4f}"
        )
    smallest = tiles[0]
    largest = tiles[-1]
    rate_small = _rate(*totals[smallest])
    rate_large = _rate(*totals[largest])
    delta = rate_small - rate_large
    lines.append(
        f"delta(rate@{smallest} - rate@{largest})={delta:.4f}"
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute damage metric rate delta between tile sizes."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Damage metrics logs or directories to scan",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output report path",
    )
    args = parser.parse_args()
    files = _iter_input_files(args.inputs)
    totals = _collect_by_tile(files)
    if not totals:
        print("no damage metrics found in inputs", file=sys.stderr)
        return 2
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_report(out_path, totals)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
