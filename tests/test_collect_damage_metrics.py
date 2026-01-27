import subprocess
import sys

import pytest

pytestmark = pytest.mark.m1


def _run_script(args):
    result = subprocess.run(
        [sys.executable, "scripts/collect_damage_metrics.py", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result


def test_collect_damage_metrics_summary(tmp_path):
    base = tmp_path / "raw"
    base.mkdir()
    (base / "damage_metrics.txt").write_text(
        "   ├─ Damage   : cycles=2 hot=5 edges=3/7 rate=0.4286 tile=4\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.md"
    result = _run_script(["--base", str(base), "--out", str(out)])
    assert result.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "| damage_metrics.txt | 2 | 5 | 3 | 7 | 0.4286 | 4 |" in text


def test_collect_damage_metrics_empty(tmp_path):
    base = tmp_path / "raw"
    base.mkdir()
    (base / "damage_metrics_off.txt").write_text(
        "   ├─ Result   : (suc zero)\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.md"
    result = _run_script(["--base", str(base), "--out", str(out)])
    assert result.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "| (none) | 0 | 0 | 0 | 0 | 0.0 | 0 |" in text


def test_collect_damage_metrics_missing_base(tmp_path):
    missing = tmp_path / "missing"
    out = tmp_path / "summary.md"
    result = _run_script(["--base", str(missing), "--out", str(out)])
    assert result.returncode == 2
