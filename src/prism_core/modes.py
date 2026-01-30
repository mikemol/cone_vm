from __future__ import annotations

from enum import Enum

from prism_core.errors import PrismValidateModeError, PrismBspModeError


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


class BspMode(str, Enum):
    AUTO = "auto"
    INTRINSIC = "intrinsic"
    CNF2 = "cnf2"


def coerce_bsp_mode(
    mode: BspMode | str | None,
    *,
    default_fn=None,
    context: str | None = None,
) -> BspMode:
    if mode is None or mode == "" or mode == BspMode.AUTO:
        return default_fn() if default_fn is not None else BspMode.AUTO
    if isinstance(mode, BspMode):
        return mode
    if isinstance(mode, str):
        if mode == BspMode.AUTO.value:
            return default_fn() if default_fn is not None else BspMode.AUTO
        if mode == BspMode.INTRINSIC.value:
            return BspMode.INTRINSIC
        if mode == BspMode.CNF2.value:
            return BspMode.CNF2
    raise PrismBspModeError(
        mode=mode,
        allowed=(
            BspMode.INTRINSIC.value,
            BspMode.CNF2.value,
            BspMode.AUTO.value,
        ),
        context=context,
    )


__all__ = [
    "BspMode",
    "coerce_bsp_mode",
    "ValidateMode",
    "coerce_validate_mode",
]
