from typing import Callable

import jax.numpy as jnp

from prism_core.compact import (
    CompactResult,
    CompactConfig,
    compact_mask_cfg,
)
from prism_core.di import call_with_optional_kwargs

DEFAULT_CANDIDATE_COMPACT_CONFIG = CompactConfig(
    index_dtype=jnp.int32, count_dtype=jnp.int32
)


def _candidate_indices(
    enabled, *, compact_cfg: CompactConfig | None = None
) -> CompactResult:
    if compact_cfg is None:
        compact_cfg = DEFAULT_CANDIDATE_COMPACT_CONFIG
    return compact_mask_cfg(enabled, cfg=compact_cfg)


def candidate_indices_cfg(
    enabled,
    *,
    candidate_indices_fn: Callable[..., CompactResult] | None = None,
    compact_cfg: CompactConfig | None = None,
):
    """Interface/Control wrapper for candidate index selection."""
    if candidate_indices_fn is None:
        return _candidate_indices(enabled, compact_cfg=compact_cfg)
    return call_with_optional_kwargs(
        candidate_indices_fn, {"compact_cfg": compact_cfg}, enabled
    )


__all__ = [
    "_candidate_indices",
    "candidate_indices_cfg",
    "DEFAULT_CANDIDATE_COMPACT_CONFIG",
]
