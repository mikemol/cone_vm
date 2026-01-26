import subprocess
import sys

import pytest


pytestmark = pytest.mark.m4


def test_ci_watch_dry_run():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ci_watch.py",
            "--dry-run",
            "--no-push",
            "--no-watch",
            "--no-logs",
            "--no-download",
            "--run-id",
            "123",
            "--branch",
            "stage",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ci_watch: dry-run enabled" in result.stdout
