import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.m1


def _run_trace_analyze(args):
    return subprocess.run(
        [sys.executable, "scripts/trace_analyze.py", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_trace(path, events):
    data = {"traceEvents": events}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_trace_analyze_pass(tmp_path):
    trace = tmp_path / "trace.json"
    events = [
        {"name": "xla_gpu_kernel", "ts": 0, "dur": 100},
        {"name": "fusion_compute_gpu", "ts": 110, "dur": 100},
        {"name": "scatter_gpu_kernel", "ts": 220, "dur": 10},
    ]
    _write_trace(trace, events)
    result = _run_trace_analyze(
        [str(trace), "--min-duty", "10", "--max-scatter", "80"]
    )
    assert result.returncode == 0
    assert "trace_analyze: PASS" in result.stdout


def test_trace_analyze_fail_scatter(tmp_path):
    trace = tmp_path / "trace.json"
    events = [
        {"name": "scatter_gpu_kernel", "ts": 0, "dur": 100},
        {"name": "gather_gpu_kernel", "ts": 110, "dur": 100},
    ]
    _write_trace(trace, events)
    result = _run_trace_analyze(
        [str(trace), "--min-duty", "0", "--max-scatter", "20"]
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_trace_analyze_no_gpu_kernels(tmp_path):
    trace = tmp_path / "trace.json"
    events = [
        {"name": "python_host", "ts": 0, "dur": 100},
    ]
    _write_trace(trace, events)
    result = _run_trace_analyze([str(trace)])
    assert result.returncode == 0
    assert "no kernels detected" in result.stdout


def test_trace_analyze_cpu_mode(tmp_path):
    trace = tmp_path / "trace.json"
    events = [
        {"name": "xla_kernel", "ts": 0, "dur": 100},
        {"name": "scatter_kernel", "ts": 110, "dur": 100},
    ]
    _write_trace(trace, events)
    result = _run_trace_analyze(
        [str(trace), "--min-duty", "99", "--max-scatter", "1"]
    )
    assert result.returncode == 0
    assert "mode=cpu" in result.stdout
