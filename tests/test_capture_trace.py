import subprocess
import sys

import pytest


pytestmark = pytest.mark.m4


def test_capture_trace_dry_run(tmp_path):
    out_dir = tmp_path / "trace"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_trace.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dry-run" in result.stdout
