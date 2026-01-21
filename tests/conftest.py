import os
import sys

# Ensure repo root is importable when pytest uses importlib mode.
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
