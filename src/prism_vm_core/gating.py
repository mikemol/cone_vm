import os

from prism_core import jax_safe as _jax_safe
from prism_core.modes import BspMode, coerce_bsp_mode

_TEST_GUARDS = _jax_safe.TEST_GUARDS


def _parse_milestone_value(value):
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("m"):
        value = value[1:]
    if value.isdigit():
        return int(value)
    return None


def _read_pytest_milestone():
    if not _TEST_GUARDS:
        return None
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    path = os.path.join(repo_root, ".pytest-milestone")
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "PRISM_MILESTONE":
                        return _parse_milestone_value(value)
                else:
                    return _parse_milestone_value(line)
    except FileNotFoundError:
        return None
    return None


def _cnf2_enabled():
    # CNF-2 pipeline is staged for m2+; guard uses env/milestone in tests.
    # See IMPLEMENTATION_PLAN.md (m2 CNF-2 enablement).
    value = os.environ.get("PRISM_ENABLE_CNF2", "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    milestone = _parse_milestone_value(os.environ.get("PRISM_MILESTONE", ""))
    if milestone is None:
        milestone = _read_pytest_milestone()
    return milestone is not None and milestone >= 2


def _cnf2_slot1_enabled():
    # Slot1 continuation is staged for m2+; hyperstrata visibility is enforced
    # under test guards (m3 normative) to justify continuation enablement.
    # See IMPLEMENTATION_PLAN.md (CNF-2 continuation slot).
    value = os.environ.get("PRISM_ENABLE_CNF2_SLOT1", "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    milestone = _parse_milestone_value(os.environ.get("PRISM_MILESTONE", ""))
    if milestone is None:
        milestone = _read_pytest_milestone()
    return milestone is not None and milestone >= 2


def _default_bsp_mode() -> BspMode:
    # CNF-2 becomes the default at m2; intrinsic remains the oracle path.
    # See IMPLEMENTATION_PLAN.md (m1/m2 engine staging).
    return BspMode.CNF2 if _cnf2_enabled() else BspMode.INTRINSIC


def _normalize_bsp_mode(bsp_mode):
    return coerce_bsp_mode(
        bsp_mode, default_fn=_default_bsp_mode, context="bsp_mode"
    )


def _servo_enabled():
    value = os.environ.get("PRISM_ENABLE_SERVO", "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    milestone = _parse_milestone_value(os.environ.get("PRISM_MILESTONE", ""))
    if milestone is None:
        milestone = _read_pytest_milestone()
    return milestone is not None and milestone >= 5


def _gpu_metrics_enabled():
    value = os.environ.get("PRISM_GPU_METRICS", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _gpu_metrics_device_index():
    value = os.environ.get("PRISM_GPU_INDEX", "").strip()
    if not value:
        return 0
    if not value.isdigit():
        raise ValueError("PRISM_GPU_INDEX must be an integer")
    return int(value)


__all__ = [
    "_parse_milestone_value",
    "_read_pytest_milestone",
    "_cnf2_enabled",
    "_cnf2_slot1_enabled",
    "_default_bsp_mode",
    "_normalize_bsp_mode",
    "_servo_enabled",
    "_gpu_metrics_enabled",
    "_gpu_metrics_device_index",
]
