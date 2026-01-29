from __future__ import annotations

from dataclasses import dataclass

from prism_ledger.config import InternConfig


@dataclass(frozen=True, slots=True)
class CoordConfig:
    """Coordination-layer configuration (DI/flags only, no behavior).

    Axis: Interface/Control. Commutes with q. Erased by q.
    """

    intern_cfg: InternConfig | None = None


DEFAULT_COORD_CONFIG = CoordConfig()

__all__ = ["CoordConfig", "DEFAULT_COORD_CONFIG"]
