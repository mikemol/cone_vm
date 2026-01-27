import subprocess
import sys

import pytest

pytestmark = pytest.mark.m1


def _run_delta(args):
    result = subprocess.run(
        [sys.executable, "scripts/damage_metrics_delta.py", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result


def test_damage_metrics_delta_report(tmp_path):
    log_small = tmp_path / "damage_metrics.txt"
    log_large = tmp_path / "damage_metrics_tile32.txt"
    log_small.write_text(
        "   ├─ Damage   : cycles=1 hot=2 edges=4/4 rate=1.0000 tile=2\n",
        encoding="utf-8",
    )
    log_large.write_text(
        "   ├─ Damage   : cycles=1 hot=2 edges=1/4 rate=0.2500 tile=32\n",
        encoding="utf-8",
    )
    out = tmp_path / "delta.txt"
    result = _run_delta(
        ["--inputs", str(log_small), str(log_large), "--out", str(out)]
    )
    assert result.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "tile=2 edges=4/4 rate=1.0000" in text
    assert "tile=32 edges=1/4 rate=0.2500" in text
    assert "delta(rate@2 - rate@32)=0.7500" in text


def test_damage_metrics_delta_requires_two_tiles(tmp_path):
    log_small = tmp_path / "damage_metrics.txt"
    log_small.write_text(
        "   ├─ Damage   : cycles=1 hot=2 edges=4/4 rate=1.0000 tile=2\n",
        encoding="utf-8",
    )
    out = tmp_path / "delta.txt"
    result = _run_delta(["--inputs", str(log_small), "--out", str(out)])
    assert result.returncode == 2
