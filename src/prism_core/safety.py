from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from typing import TypeAlias

from prism_core.errors import (
    PrismPolicyModeError,
    PrismSafetyModeError,
    PrismPolicyBindingError,
)


class SafetyMode(str, Enum):
    CORRUPT = "corrupt"
    CLAMP = "clamp"
    DROP = "drop"


def coerce_safety_mode(mode: SafetyMode | str) -> SafetyMode:
    if isinstance(mode, SafetyMode):
        return mode
    if isinstance(mode, str):
        if mode == SafetyMode.CORRUPT.value:
            return SafetyMode.CORRUPT
        if mode == SafetyMode.CLAMP.value:
            return SafetyMode.CLAMP
        if mode == SafetyMode.DROP.value:
            return SafetyMode.DROP
    raise PrismSafetyModeError(mode=mode)


class PolicyMode(str, Enum):
    STATIC = "static"
    VALUE = "value"


def coerce_policy_mode(mode: PolicyMode | str, *, context: str | None = None) -> PolicyMode:
    """Normalize a policy mode to PolicyMode, raising on unknown values."""
    if isinstance(mode, PolicyMode):
        return mode
    if isinstance(mode, str):
        if mode == PolicyMode.STATIC.value:
            return PolicyMode.STATIC
        if mode == PolicyMode.VALUE.value:
            return PolicyMode.VALUE
    raise PrismPolicyModeError(mode=mode, allowed=(PolicyMode.STATIC.value, PolicyMode.VALUE.value), context=context)


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """Safety policy for out-of-bounds handling.

    mode:
      - "corrupt": propagate OOB as semantic corruption
      - "clamp": clamp indices, do not corrupt
      - "drop": ignore OOB (for masked/scatter-drop semantics)
    """

    mode: SafetyMode | str = SafetyMode.CORRUPT

    def __post_init__(self):
        object.__setattr__(self, "mode", coerce_safety_mode(self.mode))


DEFAULT_SAFETY_POLICY = SafetyPolicy()
PolicyValue: TypeAlias = jnp.ndarray
POLICY_VALUE_CORRUPT = jnp.int32(0)
POLICY_VALUE_CLAMP = jnp.int32(1)
POLICY_VALUE_DROP = jnp.int32(2)


def policy_to_value(policy: SafetyPolicy) -> PolicyValue:
    """Encode SafetyPolicy as a JAX value for JIT-safe policy handling."""
    if policy.mode == SafetyMode.CLAMP:
        return POLICY_VALUE_CLAMP
    if policy.mode == SafetyMode.DROP:
        return POLICY_VALUE_DROP
    return POLICY_VALUE_CORRUPT


POLICY_VALUE_DEFAULT = policy_to_value(DEFAULT_SAFETY_POLICY)


@dataclass(frozen=True, slots=True)
class PolicyBinding:
    """Resolved policy binding for static vs value policy modes."""

    mode: PolicyMode
    policy: SafetyPolicy | None = None
    policy_value: PolicyValue | None = None


def resolve_policy_binding(
    *,
    policy: SafetyPolicy | None = None,
    policy_value: PolicyValue | None = None,
    context: str | None = None,
    default_policy: bool = True,
) -> PolicyBinding:
    """Resolve a policy binding, enforcing that only one of policy/value is set."""
    if policy is not None and policy_value is not None:
        raise PrismPolicyBindingError(
            "received both safety policy and policy_value",
            context=context,
            policy_mode="ambiguous",
        )
    if policy_value is not None:
        return PolicyBinding(PolicyMode.VALUE, policy=None, policy_value=policy_value)
    if policy is None:
        if not default_policy:
            return PolicyBinding(PolicyMode.STATIC, policy=None, policy_value=None)
        policy = DEFAULT_SAFETY_POLICY
    return PolicyBinding(PolicyMode.STATIC, policy=policy, policy_value=None)


def oob_mask(ok, *, policy: SafetyPolicy = DEFAULT_SAFETY_POLICY):
    """Return a corruption mask based on OOB policy."""
    if policy.mode == SafetyMode.CORRUPT:
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
    "SafetyMode",
    "coerce_safety_mode",
    "PolicyMode",
    "coerce_policy_mode",
    "SafetyPolicy",
    "DEFAULT_SAFETY_POLICY",
    "PolicyValue",
    "POLICY_VALUE_CORRUPT",
    "POLICY_VALUE_CLAMP",
    "POLICY_VALUE_DROP",
    "POLICY_VALUE_DEFAULT",
    "policy_to_value",
    "oob_mask",
    "oob_any",
    "oob_mask_value",
    "oob_any_value",
    "PolicyBinding",
    "resolve_policy_binding",
]
