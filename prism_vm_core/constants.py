from __future__ import annotations

import os

# NOTE: JAX op dtype normalization (int32) is assumed; tighten if drift appears.
MAX_ROWS = 1024 * 32
MAX_KEY_NODES = 1 << 16
LEDGER_CAPACITY = MAX_KEY_NODES - 1
MAX_ID = LEDGER_CAPACITY - 1
MAX_COUNT = MAX_ID + 1  # Next-free upper bound; equals LEDGER_CAPACITY.
# Hard-cap is semantic (univalence), not just capacity.
if LEDGER_CAPACITY >= MAX_KEY_NODES:
    raise ValueError("LEDGER_CAPACITY exceeds 16-bit key packing")

MAX_COORD_STEPS = 8

_PREFIX_SCAN_CHUNK = int(os.environ.get("PRISM_PREFIX_SCAN_CHUNK", "4096") or 4096)
if _PREFIX_SCAN_CHUNK <= 0:
    _PREFIX_SCAN_CHUNK = LEDGER_CAPACITY
if _PREFIX_SCAN_CHUNK > LEDGER_CAPACITY:
    _PREFIX_SCAN_CHUNK = LEDGER_CAPACITY

__all__ = [
    "MAX_ROWS",
    "MAX_KEY_NODES",
    "LEDGER_CAPACITY",
    "MAX_ID",
    "MAX_COUNT",
    "MAX_COORD_STEPS",
    "_PREFIX_SCAN_CHUNK",
]
