"""Single control surface for IC exports."""

from ic_core import facade as _facade

__all__ = list(_facade.__all__)

for _name in __all__:
    globals()[_name] = getattr(_facade, _name)

