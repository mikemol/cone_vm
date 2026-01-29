import os
import sys

_ROOT = os.path.dirname(__file__)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ic_core.exports import *  # noqa: F401,F403
from ic_core import exports as _exports

__all__ = _exports.__all__
