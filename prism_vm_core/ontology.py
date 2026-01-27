from __future__ import annotations

from dataclasses import dataclass

# --- 1. Ontology (Opcodes) ---
# Ledger ids 0/1 are semantic reserves (NULL/ZERO); baseline heaps seed ZERO at 1.
OP_NULL = 0
OP_ZERO = 1
OP_SUC = 2
OP_ADD = 10
OP_MUL = 11
OP_SORT = 99
OP_COORD_ZERO = 20
OP_COORD_ONE = 21
OP_COORD_PAIR = 22
ZERO_PTR = 1  # Must stay aligned with OP_ZERO (identity semantics).

OP_NAMES = {
    0: "NULL",
    1: "zero",
    2: "suc",
    10: "add",
    11: "mul",
    99: "sort",
    20: "coord_zero",
    21: "coord_one",
    22: "coord_pair",
}


# Pointer domain wrappers (runtime separation).
@dataclass(frozen=True)
class ManifestPtr:
    i: int

    def __int__(self) -> int:
        return int(self.i)

    def __index__(self) -> int:
        return int(self.i)


@dataclass(frozen=True)
class LedgerId:
    i: int

    def __int__(self) -> int:
        return int(self.i)

    def __index__(self) -> int:
        return int(self.i)


@dataclass(frozen=True)
class ArenaPtr:
    i: int

    def __int__(self) -> int:
        return int(self.i)

    def __index__(self) -> int:
        return int(self.i)


# Host-only scalar markers for sync boundaries.
@dataclass(frozen=True)
class HostInt:
    v: int

    def __int__(self) -> int:
        return int(self.v)

    def __index__(self) -> int:
        return int(self.v)


@dataclass(frozen=True)
class HostBool:
    v: bool

    def __bool__(self) -> bool:
        return bool(self.v)


@dataclass(frozen=True)
class ProvisionalIds:
    a: object


@dataclass(frozen=True)
class CommittedIds:
    a: object


__all__ = [
    "OP_NULL",
    "OP_ZERO",
    "OP_SUC",
    "OP_ADD",
    "OP_MUL",
    "OP_SORT",
    "OP_COORD_ZERO",
    "OP_COORD_ONE",
    "OP_COORD_PAIR",
    "ZERO_PTR",
    "OP_NAMES",
    "ManifestPtr",
    "LedgerId",
    "ArenaPtr",
    "HostInt",
    "HostBool",
    "ProvisionalIds",
    "CommittedIds",
]
