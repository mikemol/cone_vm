import os
import sys

import pytest

# Ensure repo root is importable when pytest uses importlib mode.
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _parse_milestone(value):
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("m"):
        value = value[1:]
    if not value.isdigit():
        raise ValueError(f"invalid milestone: {value!r}")
    return int(value)


def pytest_addoption(parser):
    parser.addoption(
        "--milestone",
        action="store",
        default=os.environ.get("PRISM_MILESTONE", ""),
        help="run tests up to a milestone (m1-m5)",
    )


def pytest_collection_modifyitems(config, items):
    milestone = _parse_milestone(config.getoption("--milestone"))
    if milestone is None:
        return
    for item in items:
        markers = [
            m.name
            for m in item.iter_markers()
            if m.name in {"m1", "m2", "m3", "m4", "m5"}
        ]
        if not markers:
            continue
        required = max(int(m[1:]) for m in markers)
        if milestone < required:
            item.add_marker(
                pytest.mark.skip(reason=f"requires m{required} (running m{milestone})")
            )
