import subprocess
import sys
import zipfile

import pytest


pytestmark = pytest.mark.m4


def test_unpack_collected_report(tmp_path):
    zip_path = tmp_path / "collected-report-test.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("telemetry_baselines.md", "# baselines\n")
    out_dir = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/unpack_collected_report.py",
            "--zip",
            str(zip_path),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Unpacked to" in result.stdout
    assert "telemetry_baselines.md" in result.stdout
