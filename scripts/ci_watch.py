#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


def _run(cmd: Sequence[str], env: dict, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, check=check)


def _run_capture(cmd: Sequence[str], env: dict) -> str:
    result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"command failed: {' '.join(cmd)}")
    return result.stdout.strip()


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
        cmd = ["gh", "run", "watch", run_id, "--exit-status"]
        print("ci_watch: watch", " ".join(cmd))
        if not args.dry_run:
            _run(cmd, env, check=True)

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
