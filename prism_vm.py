import os
import sys

_ROOT = os.path.dirname(__file__)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from prism_vm_core.exports import *  # noqa: F401,F403
from prism_vm_core.exports import __all__  # noqa: F401
from prism_cli.repl import main

# NOTE: JAX op dtype normalization (int32) is assumed; tighten if drift appears
# (see IMPLEMENTATION_PLAN.md).

if __name__ == "__main__":
    main()
