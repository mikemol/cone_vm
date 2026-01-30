from __future__ import annotations


def _require_ptr_domain(ptr, label: str, expected_type):
    if not isinstance(ptr, expected_type):
        raise TypeError(f"{label} expected {expected_type.__name__}")
    return ptr


__all__ = ["_require_ptr_domain"]
