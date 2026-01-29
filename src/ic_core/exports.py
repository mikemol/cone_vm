"""Single control surface for IC exports."""

from ic_core import facade as _facade
from ic_core import types as _types

__all__ = list(_facade.__all__)
for _name in _types.__all__:
    if _name not in __all__:
        __all__.append(_name)

for _name in __all__:
    if hasattr(_facade, _name):
        globals()[_name] = getattr(_facade, _name)
    else:
        globals()[_name] = getattr(_types, _name)
