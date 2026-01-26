import json
import subprocess
import sys

import pytest


pytestmark = pytest.mark.m4


def test_audit_memory_stability_intrinsic(tmp_path):
    out_path = tmp_path / "memory.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_memory_stability.py",
            "--engine",
            "intrinsic",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--top",
            "3",
            "--json-out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["engine"] == "intrinsic"
    assert data["iterations"] == 1
    assert "growth_bytes" in data
    assert "top_deltas" in data
