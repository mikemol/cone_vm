import json
import subprocess
import sys

import pytest


pytestmark = pytest.mark.m4


def test_collect_telemetry_baselines(tmp_path):
    base = tmp_path / "raw"
    base.mkdir()
    trace = base / "trace_cpu_report.json"
    trace.write_text(
        json.dumps(
            {
                "mode": "cpu",
                "duty_cycle_pct": 99.0,
                "scatter_ratio_pct": 5.0,
                "kernel_active_ms": 12.0,
                "wall_time_ms": 12.1,
            }
        ),
        encoding="utf-8",
    )
    damage_summary = tmp_path / "damage.md"
    damage_summary.write_text("# damage summary\n", encoding="utf-8")
    host_summary = tmp_path / "host.md"
    host_summary.write_text("# host summary\n", encoding="utf-8")
    out_path = tmp_path / "telemetry.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_telemetry_baselines.py",
            "--base",
            str(base),
            "--damage-summary",
            str(damage_summary),
            "--host-summary",
            str(host_summary),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    contents = out_path.read_text(encoding="utf-8")
    assert "trace_cpu_report.json" in contents
    assert "damage summary" in contents
    assert "host summary" in contents
