#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Sequence
from datetime import datetime, timezone


def _run(cmd: Sequence[str], env: dict, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, check=check)


def _run_capture(cmd: Sequence[str], env: dict) -> str:
    result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def _run_capture_json(cmd: Sequence[str], env: dict) -> list[dict]:
    result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"command failed: {' '.join(cmd)}")
    output = result.stdout.strip()
    if not output:
        return []
    import json  # local import to avoid overhead when unused

    return json.loads(output)


def _unbuffered_env() -> dict:
    env = os.environ.copy()
    env["PAGER"] = "cat"
    env["GH_PAGER"] = "cat"
    env["GIT_PAGER"] = "cat"
    env["LESS"] = "FRX"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _get_branch() -> str:
    return _run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], os.environ)


def _find_run_id(workflow: str, branch: str, env: dict) -> str:
    return _run_capture(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--branch",
            branch,
            "--event",
            "push",
            "--limit",
            "1",
            "--json",
            "databaseId",
            "--jq",
            ".[0].databaseId | tostring",
        ],
        env,
    )


def _parse_iso8601(value: str) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _estimate_watch_duration(
    workflow: str, branch: str, env: dict, history_limit: int
) -> int | None:
    runs = _run_capture_json(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--branch",
            branch,
            "--event",
            "push",
            "--limit",
            str(history_limit),
            "--json",
            "status,createdAt,updatedAt,startedAt",
        ],
        env,
    )
    durations = []
    for run in runs:
        if run.get("status") != "completed":
            continue
        start = _parse_iso8601(run.get("startedAt") or run.get("createdAt"))
        end = _parse_iso8601(run.get("updatedAt"))
        if start and end and end >= start:
            durations.append((end - start).total_seconds())
    if not durations:
        return None
    return int(max(durations))


def _fetch_run_status(run_id: str, env: dict) -> dict:
    data = _run_capture_json(
        [
            "gh",
            "run",
            "view",
            run_id,
            "--json",
            "status,conclusion,createdAt,startedAt,updatedAt",
        ],
        env,
    )
    if not data:
        return {}
    if isinstance(data, dict):
        return data
    return data[0]


def _compute_deadline(
    status: dict,
    expected_duration_s: int | None,
) -> float | None:
    if expected_duration_s is None:
        return None
    start_at = _parse_iso8601(status.get("startedAt") or status.get("createdAt"))
    now = datetime.now(timezone.utc)
    if start_at is None:
        return (now.timestamp() + expected_duration_s * 1.3 + 60)
    elapsed = max(0.0, (now - start_at).total_seconds())
    projected = max(float(expected_duration_s), elapsed) * 1.3 + 60
    return (start_at.timestamp() + projected)


def _watch_poll(
    run_id: str,
    env: dict,
    expected_duration_s: int | None,
    interval_s: int,
) -> int:
    while True:
        status = _fetch_run_status(run_id, env)
        status_str = status.get("status")
        conclusion = status.get("conclusion")
        deadline = _compute_deadline(status, expected_duration_s)
        stamp = datetime.now(timezone.utc).isoformat()
        if deadline is not None:
            remaining = max(0, int(deadline - datetime.now(timezone.utc).timestamp()))
            print(
                f"ci_watch: {stamp} status={status_str} conclusion={conclusion} "
                f"deadline_in={remaining}s"
            )
        else:
            print(f"ci_watch: {stamp} status={status_str} conclusion={conclusion}")
        if status_str == "completed":
            if conclusion == "success":
                return 0
            return 1
        if deadline is not None and datetime.now(timezone.utc).timestamp() > deadline:
            print("ci_watch: watch timeout exceeded", file=sys.stderr)
            return 2
        time.sleep(interval_s)


def _download_artifacts(run_id: str, dest: Path, env: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _run(["gh", "run", "download", run_id, "--dir", str(dest)], env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push, locate latest workflow run, watch it, and download artifacts."
    )
    parser.add_argument(
        "--workflow",
        default=".github/workflows/ci-milestones.yml",
        help="Workflow name or path (default: .github/workflows/ci-milestones.yml).",
    )
    parser.add_argument("--branch", default="", help="Branch to filter runs.")
    parser.add_argument("--remote", default="origin", help="Git remote (default: origin).")
    parser.add_argument("--run-id", default="", help="Use specific run id (skip lookup).")
    parser.add_argument("--no-push", action="store_true", help="Skip git push.")
    parser.add_argument("--no-watch", action="store_true", help="Skip gh run watch.")
    parser.add_argument(
        "--watch-mode",
        choices=("poll", "gh"),
        default="poll",
        help="Watch mode: poll (default) or gh.",
    )
    parser.add_argument(
        "--watch-timeout",
        default="auto",
        help="Max watch seconds, 0 for no timeout, or 'auto' (default).",
    )
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=15,
        help="Polling interval in seconds (default: 15).",
    )
    parser.add_argument(
        "--watch-history",
        type=int,
        default=5,
        help="Number of recent runs to estimate timeout for 'auto' (default: 5).",
    )
    parser.add_argument("--no-logs", action="store_true", help="Skip gh run view --log-failed.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip gh run download artifacts.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for downloaded artifacts (default: artifacts).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("ci_watch: dry-run enabled")
    if not args.dry_run and shutil.which("gh") is None:
        print("gh CLI is required", file=sys.stderr)
        return 1

    branch = args.branch or _get_branch()
    env = _unbuffered_env()

    if not args.no_push:
        cmd = ["git", "push", args.remote, branch]
        print("ci_watch: push", " ".join(cmd))
        if not args.dry_run:
            _run(cmd, env)

    run_id = args.run_id
    if not run_id:
        print(f"ci_watch: lookup run (workflow={args.workflow}, branch={branch})")
        if args.dry_run:
            run_id = "DRY_RUN"
        else:
            run_id = _find_run_id(args.workflow, branch, env)
            if not run_id:
                print("No runs found.", file=sys.stderr)
                return 1

    print(f"ci_watch: run_id={run_id}")

    if not args.no_watch:
        if args.watch_mode == "gh":
            cmd = ["gh", "run", "watch", run_id, "--exit-status"]
            print("ci_watch: watch", " ".join(cmd))
            if not args.dry_run:
                _run(cmd, env, check=True)
        else:
            timeout_raw = args.watch_timeout.strip().lower()
            if timeout_raw == "auto":
                try:
                    expected_duration = _estimate_watch_duration(
                        args.workflow, branch, env, args.watch_history
                    )
                except RuntimeError as exc:
                    print(f"ci_watch: duration estimate failed: {exc}")
                    expected_duration = None
            else:
                expected_duration = int(timeout_raw) if timeout_raw != "0" else None
            print(
                "ci_watch: poll watch "
                f"interval={args.watch_interval}s expected_duration={expected_duration}"
            )
            if not args.dry_run:
                watch_code = _watch_poll(
                    run_id, env, expected_duration, args.watch_interval
                )
                if watch_code != 0:
                    return watch_code

    if not args.no_logs:
        cmd = ["gh", "run", "view", run_id, "--log-failed"]
        print("ci_watch: logs", " ".join(cmd))
        if not args.dry_run:
            _run(cmd, env, check=False)

    if not args.no_download:
        dest = Path(args.artifacts_dir) / f"run-{run_id}"
        print(f"ci_watch: download artifacts -> {dest}")
        if not args.dry_run:
            _download_artifacts(run_id, dest, env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
