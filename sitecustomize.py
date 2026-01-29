"""Auto-add src/ to sys.path for repo-local imports.

This keeps root-level entrypoints (prism_vm.py, ic_vm.py, scripts) working
with the src/ layout without requiring editable installs.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(__file__)
_SRC = os.path.join(_ROOT, "src")

if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
