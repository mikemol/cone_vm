from __future__ import annotations

from enum import Enum

from prism_core.errors import (
    PrismValidateModeError,
    PrismBspModeError,
    PrismCnf2ModeError,
)


class ValidateMode(str, Enum):
    NONE = "none"
    STRICT = "strict"
    HYPER = "hyper"


def coerce_validate_mode(
    mode: ValidateMode | str | None, *, context: str | None = None
) -> ValidateMode:
    if mode is None:
        return ValidateMode.NONE
    if isinstance(mode, ValidateMode):
        return mode
    if isinstance(mode, str):
        if mode == ValidateMode.NONE.value:
            return ValidateMode.NONE
        if mode == ValidateMode.STRICT.value:
            return ValidateMode.STRICT
        if mode == ValidateMode.HYPER.value:
            return ValidateMode.HYPER
    raise PrismValidateModeError(
        mode=mode,
        allowed=(
            ValidateMode.NONE.value,
            ValidateMode.STRICT.value,
            ValidateMode.HYPER.value,
        ),
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


class Cnf2Mode(str, Enum):
    AUTO = "auto"
    OFF = "off"
    BASE = "base"
    SLOT1 = "slot1"


def coerce_cnf2_mode(
    mode: Cnf2Mode | str | None, *, context: str | None = None
) -> Cnf2Mode:
    if mode is None or mode == "" or mode == Cnf2Mode.AUTO:
        return Cnf2Mode.AUTO
    if isinstance(mode, Cnf2Mode):
        return mode
    if isinstance(mode, str):
        if mode == Cnf2Mode.AUTO.value:
            return Cnf2Mode.AUTO
        if mode == Cnf2Mode.OFF.value:
            return Cnf2Mode.OFF
        if mode == Cnf2Mode.BASE.value:
            return Cnf2Mode.BASE
        if mode == Cnf2Mode.SLOT1.value:
            return Cnf2Mode.SLOT1
    raise PrismCnf2ModeError(
        mode=mode,
        allowed=(
            Cnf2Mode.OFF.value,
            Cnf2Mode.BASE.value,
            Cnf2Mode.SLOT1.value,
            Cnf2Mode.AUTO.value,
        ),
        context=context,
    )


__all__ = [
    "BspMode",
    "coerce_bsp_mode",
    "Cnf2Mode",
    "coerce_cnf2_mode",
    "ValidateMode",
    "coerce_validate_mode",
]
