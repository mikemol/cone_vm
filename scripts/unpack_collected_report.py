#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import re
from pathlib import Path
import sys
import zipfile


def _sanitize(name: str) -> str:
    name = re.sub(r"[()]+", "", name)
    name = name.replace(" ", "-")
    return name


def _find_latest_zip(root: Path) -> Path | None:
    candidates = sorted(root.glob("collected-report*.zip"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base.with_name(f"{base.name}-{stamp}")
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = base.with_name(f"{base.name}-{stamp}-{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unpack collected-report artifacts for inspection."
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        help="Path to collected-report zip (defaults to latest in artifacts/).",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts",
        help="Directory to unpack into (default: artifacts).",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.zip_path:
        zip_path = Path(args.zip_path)
    else:
        zip_path = _find_latest_zip(out_root)
        if zip_path is None:
            print("No collected-report zip found in artifacts/", file=sys.stderr)
            return 2

    if not zip_path.exists():
        print(f"Zip not found: {zip_path}", file=sys.stderr)
        return 2

    stem = _sanitize(zip_path.stem)
    dest = _unique_dir(out_root / stem)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest)
    except zipfile.BadZipFile as exc:
        print(f"Invalid zip file: {exc}", file=sys.stderr)
        return 2

    print(f"Unpacked to {dest}")
    summary = dest / "telemetry_baselines.md"
    if summary.exists():
        print(f"Telemetry summary: {summary}")
    else:
        print("Telemetry summary not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
