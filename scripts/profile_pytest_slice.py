#!/usr/bin/env python
"""Profile a pytest slice for a bounded duration.

Example:
  mise exec -- python scripts/profile_pytest_slice.py \
    --seconds 60 \
    --out artifacts/profiles/strata_random_programs.prof \
    -- -q tests/test_strata_random_programs.py
"""

from __future__ import annotations

import argparse
import cProfile
import os
import signal
import sys
import time
from pathlib import Path

import pytest


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run pytest with cProfile for a bounded duration.",
        epilog="Pass pytest args after '--'.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=60,
        help="Max profiling duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output .prof path (default: artifacts/profiles/pytest_slice_<stamp>.prof).",
    )
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


def _default_out_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "profiles" / f"pytest_slice_{stamp}.prof"


def main() -> int:
    args, pytest_args = _parse_args()
    out_path = Path(args.out) if args.out else _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = cProfile.Profile()

    def _alarm_handler(_sig, _frame):
        profile.disable()
        profile.dump_stats(out_path)
        print(f"[profile] dumped partial stats to {out_path}", file=sys.stderr)
        os._exit(0)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max(1, int(args.seconds)))

    profile.enable()
    exit_code = pytest.main(pytest_args)
    profile.disable()
    profile.dump_stats(out_path)
    print(f"[profile] dumped stats to {out_path}", file=sys.stderr)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
