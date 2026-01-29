from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class InternConfig:
    """Configuration for ledger interning behavior (control-plane).

    These flags must be explicit and are intended to be static args when
    used under JAX JIT.
    """

    op_buckets_full_range: bool = False
    force_spawn_clip: bool = False

    @staticmethod
    def from_env() -> "InternConfig":
        op_buckets_full_range = (
            os.environ.get("PRISM_OP_BUCKETS_FULL_RANGE", "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        force_spawn_clip = (
            os.environ.get("PRISM_FORCE_SPAWN_CLIP", "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        return InternConfig(
            op_buckets_full_range=op_buckets_full_range,
            force_spawn_clip=force_spawn_clip,
        )


DEFAULT_INTERN_CONFIG = InternConfig.from_env()


__all__ = ["InternConfig", "DEFAULT_INTERN_CONFIG"]
