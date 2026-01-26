import subprocess
import sys

import pytest


pytestmark = pytest.mark.m4


def test_collect_host_metrics_summary(tmp_path):
    base = tmp_path / "raw"
    base.mkdir()
    (base / "host_perf_intrinsic.json").write_text(
        "{\n"
        '  "engine": "intrinsic",\n'
        '  "iterations": 2,\n'
        '  "total_time_sec": 0.25,\n'
        '  "top_functions": [\n'
        '    {"function": "foo.py:1:bar", "cumulative_sec": 0.1}\n'
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )
    (base / "host_memory_intrinsic.json").write_text(
        "{\n"
        '  "engine": "intrinsic",\n'
        '  "iterations": 2,\n'
        '  "growth_mb": 0.01,\n'
        '  "top_deltas": [\n'
        '    {"size_diff_bytes": 128, "traceback": ["file.py:1"]}\n'
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "summary.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_host_metrics.py",
            "--base",
            str(base),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    contents = out_path.read_text(encoding="utf-8")
    assert "host_perf_intrinsic.json" in contents
    assert "host_memory_intrinsic.json" in contents
