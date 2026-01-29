from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """Safety policy for out-of-bounds handling.

    mode:
      - "corrupt": propagate OOB as semantic corruption
      - "clamp": clamp indices, do not corrupt
      - "drop": ignore OOB (for masked/scatter-drop semantics)
    """

    mode: str = "corrupt"


DEFAULT_SAFETY_POLICY = SafetyPolicy()


def oob_mask(ok, *, policy: SafetyPolicy = DEFAULT_SAFETY_POLICY):
    """Return a corruption mask based on OOB policy."""
    if policy.mode == "corrupt":
        return ~ok
    return jnp.zeros_like(ok, dtype=jnp.bool_)


__all__ = ["SafetyPolicy", "DEFAULT_SAFETY_POLICY", "oob_mask"]
