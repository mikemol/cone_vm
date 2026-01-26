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


def pytest_configure(config):
    for name, desc in _MARKER_DESCRIPTIONS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")
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
    milestone = _parse_milestone(config.getoption("--milestone"))
    if milestone is None:
        return
    deselected = []
    for item in items:
        markers = [m.name for m in item.iter_markers() if m.name in MILESTONE_MARKERS]
        if not markers:
            continue
        required = max(int(m[1:]) for m in markers)
        if milestone < required:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        for item in deselected:
            items.remove(item)
