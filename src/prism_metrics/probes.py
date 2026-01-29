import os


_coord_norm_probe_count = 0


def coord_norm_probe_reset():
    # Debug-only probe used by m4 tests to detect coord normalization scope.
    # See IMPLEMENTATION_PLAN.md (m4 coord probes).
    global _coord_norm_probe_count
    _coord_norm_probe_count = 0


def coord_norm_probe_get():
    return int(_coord_norm_probe_count)


def _coord_norm_probe_enabled():
    value = os.environ.get("PRISM_COORD_NORM_PROBE", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _coord_norm_probe_tick(n):
    # Only increments when PRISM_COORD_NORM_PROBE is enabled.
    if not _coord_norm_probe_enabled():
        return
    global _coord_norm_probe_count
    _coord_norm_probe_count += int(n)


def _coord_norm_probe_reset_cb(_):
    coord_norm_probe_reset()


def _coord_norm_probe_assert(has_coord):
    if not _coord_norm_probe_enabled():
        return
    expect = bool(has_coord)
    count = coord_norm_probe_get()
    if expect and count <= 0:
        raise RuntimeError("coord_norm probe missing for coord pair batch")
    if not expect and count != 0:
        raise RuntimeError("coord_norm probe fired for non-coord batch")


__all__ = [
    "coord_norm_probe_reset",
    "coord_norm_probe_get",
    "_coord_norm_probe_enabled",
    "_coord_norm_probe_tick",
    "_coord_norm_probe_reset_cb",
    "_coord_norm_probe_assert",
]
