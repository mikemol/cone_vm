from __future__ import annotations

from enum import Enum

from prism_core.errors import PrismValidateModeError


class ValidateMode(str, Enum):
    STRICT = "strict"
    HYPER = "hyper"


def coerce_validate_mode(
    mode: ValidateMode | str | None, *, context: str | None = None
) -> ValidateMode:
    if mode is None:
        return ValidateMode.STRICT
    if isinstance(mode, ValidateMode):
        return mode
    if isinstance(mode, str):
        if mode == ValidateMode.STRICT.value:
            return ValidateMode.STRICT
        if mode == ValidateMode.HYPER.value:
            return ValidateMode.HYPER
    raise PrismValidateModeError(
        mode=mode,
        allowed=(ValidateMode.STRICT.value, ValidateMode.HYPER.value),
        context=context,
    )


__all__ = [
    "ValidateMode",
    "coerce_validate_mode",
]
