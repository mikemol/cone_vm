import os
import sys
from pathlib import Path

import jax
import pytest

# Enable strict scatter guard in tests unless explicitly overridden.
os.environ.setdefault("PRISM_SCATTER_GUARD", "1")
os.environ.setdefault("PRISM_TEST_GUARDS", "1")

# Ensure repo root is importable when pytest uses importlib mode.
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MILESTONE_MARKERS = {"m1", "m2", "m3", "m4", "m5", "m6"}
_MARKER_DESCRIPTIONS = {
    "m1": "expected to pass from M1 onward",
    "m2": "expected to pass from M2 onward",
    "m3": "expected to pass from M3 onward",
    "m4": "expected to pass from M4 onward",
    "m5": "expected to pass from M5 onward",
    "m6": "expected to pass from M6 onward",
    "backend_matrix": "run the test on cpu and gpu (when available) in one session",
}


def _parse_milestone(value):
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("m"):
        value = value[1:]
    if not value.isdigit():
        raise ValueError(f"invalid milestone: {value!r}")
    return int(value)


def _parse_milestone_selector(value):
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("m"):
        value = value[1:]
    if value.endswith("+"):
        return (_parse_milestone(value[:-1]), None)
    if value.endswith("-"):
        return (None, _parse_milestone(value[:-1]))
    if ".." in value:
        lo, hi = value.split("..", 1)
        return (_parse_milestone(lo), _parse_milestone(hi))
    if "-" in value:
        lo, hi = value.split("-", 1)
        return (_parse_milestone(lo), _parse_milestone(hi))
    milestone = _parse_milestone(value)
    return (milestone, milestone)


def _read_milestone_file(path):
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == "PRISM_MILESTONE":
                return value.strip()
        else:
            return line
    return ""


def _milestone_default():
    value = os.environ.get("PRISM_MILESTONE", "").strip()
    if value:
        return value
    value = _read_milestone_file(os.path.join(ROOT, ".pytest-milestone"))
    if value:
        return value
    return _read_milestone_file(os.path.join(ROOT, ".vscode", "pytest.env"))


def pytest_addoption(parser):
    parser.addoption(
        "--milestone",
        action="store",
        default=_milestone_default(),
        help="run tests up to a milestone (m1-m6)",
    )
    parser.addoption(
        "--milestone-band",
        action="store",
        default="",
        help="run tests in a milestone band (m3, m2-4, m3+, m3-)",
    )
    parser.addoption(
        "--include-unmarked",
        action="store_true",
        default=False,
        help="include unmarked tests when using --milestone-band",
    )


def pytest_configure(config):
    for name, desc in _MARKER_DESCRIPTIONS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")
    selector = _parse_milestone_selector(config.getoption("--milestone-band"))
    if selector is not None:
        low, high = selector
        if low == 1:
            # Treat m1-band runs as baseline coverage (avoid m1-only semantics).
            milestone = _parse_milestone(config.getoption("--milestone"))
        else:
            milestone = high if high is not None else low
    else:
        milestone = _parse_milestone(config.getoption("--milestone"))
    if milestone is None:
        return
    os.environ["PRISM_MILESTONE"] = f"m{milestone}"


def _backend_matrix_backends():
    backends = ["cpu"]
    try:
        gpu_devices = jax.devices("gpu")
    except Exception:
        gpu_devices = []
    if gpu_devices:
        backends.append("gpu")
    return backends


def pytest_generate_tests(metafunc):
    marker = metafunc.definition.get_closest_marker("backend_matrix")
    if marker and "backend_device" in metafunc.fixturenames:
        backends = _backend_matrix_backends()
        ids = [f"{backend}-backend" for backend in backends]
        metafunc.parametrize("backend_device", backends, ids=ids, indirect=True)


@pytest.fixture
def backend_device(request):
    backend = getattr(request, "param", None)
    if backend is None:
        return None
    return jax.devices(backend)[0]


@pytest.fixture(autouse=True)
def _set_default_device(request):
    if not request.node.get_closest_marker("backend_matrix"):
        yield
        return
    device = request.getfixturevalue("backend_device")
    with jax.default_device(device):
        yield


def pytest_collection_modifyitems(config, items):
    selector = _parse_milestone_selector(config.getoption("--milestone-band"))
    milestone = _parse_milestone(config.getoption("--milestone"))
    if selector is None and milestone is None:
        return
    include_unmarked = config.getoption("--include-unmarked")
    deselected = []
    for item in items:
        markers = [m.name for m in item.iter_markers() if m.name in MILESTONE_MARKERS]
        if not markers:
            if selector is not None and not include_unmarked:
                deselected.append(item)
            continue
        required = max(int(m[1:]) for m in markers)
        if selector is not None:
            low, high = selector
            if low is not None and required < low:
                deselected.append(item)
            elif high is not None and required > high:
                deselected.append(item)
        elif milestone is not None and milestone < required:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        for item in deselected:
            items.remove(item)
