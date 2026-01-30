from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from typing import TypeAlias


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
PolicyValue: TypeAlias = jnp.ndarray
POLICY_VALUE_CORRUPT = jnp.int32(0)
POLICY_VALUE_CLAMP = jnp.int32(1)
POLICY_VALUE_DROP = jnp.int32(2)


def policy_to_value(policy: SafetyPolicy) -> PolicyValue:
    """Encode SafetyPolicy as a JAX value for JIT-safe policy handling."""
    if policy.mode == "clamp":
        return POLICY_VALUE_CLAMP
    if policy.mode == "drop":
        return POLICY_VALUE_DROP
    return POLICY_VALUE_CORRUPT


def oob_mask(ok, *, policy: SafetyPolicy = DEFAULT_SAFETY_POLICY):
    """Return a corruption mask based on OOB policy."""
    if policy.mode == "corrupt":
        return ~ok
    return jnp.zeros_like(ok, dtype=jnp.bool_)


def oob_any(ok, *, policy: SafetyPolicy = DEFAULT_SAFETY_POLICY):
    """Return True if OOB should be treated as corruption."""
    return jnp.any(oob_mask(ok, policy=policy))


def oob_mask_value(ok, *, policy_value: PolicyValue):
    """Return a corruption mask based on a policy value."""
    policy_val = jnp.asarray(policy_value, dtype=jnp.int32)
    return jnp.where(policy_val == POLICY_VALUE_CORRUPT, ~ok, False)


def oob_any_value(ok, *, policy_value: PolicyValue):
    """Return True if OOB should be treated as corruption (policy value)."""
    return jnp.any(oob_mask_value(ok, policy_value=policy_value))


__all__ = [
    "SafetyPolicy",
    "DEFAULT_SAFETY_POLICY",
    "PolicyValue",
    "POLICY_VALUE_CORRUPT",
    "POLICY_VALUE_CLAMP",
    "POLICY_VALUE_DROP",
    "policy_to_value",
    "oob_mask",
    "oob_any",
    "oob_mask_value",
    "oob_any_value",
]
