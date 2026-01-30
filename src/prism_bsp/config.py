from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from prism_coord.config import CoordConfig
from prism_core.compact import CompactConfig
from prism_core.guards import GuardConfig
from prism_core.safety import SafetyPolicy, PolicyBinding
from prism_ledger.config import InternConfig
from prism_core.protocols import (
    PolicyValue,
    SafeGatherFn,
    SafeGatherOkFn,
    SafeGatherOkValueFn,
    SafeGatherValueFn,
)
from prism_vm_core.protocols import (
    ApplyQFn,
    ArenaRootHashFn,
    CandidateIndicesFn,
    CommitStratumFn,
    CoordXorBatchFn,
    DamageMetricsUpdateFn,
    DamageTileSizeFn,
    EmitCandidatesFn,
    GuardMaxFn,
    GuardsEnabledFn,
    HostBoolValueFn,
    HostIntValueFn,
    HostRaiseFn,
    IdentityQFn,
    InternFn,
    LedgerRootsHashFn,
    NodeBatchFn,
    OpInteractFn,
    OpMortonFn,
    OpRankFn,
    OpSortWithPermFn,
    ScatterDropFn,
    ServoEnabledFn,
    ServoUpdateFn,
)


@dataclass(frozen=True, slots=True)
class Cnf2Flags:
    """CNF-2 gate toggles for DI.

    None means "defer to default gating".
    """

    enabled: bool | None = None
    slot1_enabled: bool | None = None


DEFAULT_CNF2_FLAGS = Cnf2Flags()

@dataclass(frozen=True, slots=True)
class Cnf2Config:
    """CNF-2 dependency injection bundle.

    Any field set to None defers to the call-site default. Call-site keyword
    arguments override config values (DI precedence).
    """

    flags: Cnf2Flags | None = None
    intern_cfg: InternConfig | None = None
    coord_cfg: CoordConfig | None = None
    intern_fn: InternFn | None = None
    node_batch_fn: NodeBatchFn | None = None
    coord_xor_batch_fn: CoordXorBatchFn | None = None
    emit_candidates_fn: EmitCandidatesFn | None = None
    candidate_indices_fn: CandidateIndicesFn | None = None
    compact_cfg: CompactConfig | None = None
    scatter_drop_fn: ScatterDropFn | None = None
    commit_stratum_fn: CommitStratumFn | None = None
    apply_q_fn: ApplyQFn | None = None
    identity_q_fn: IdentityQFn | None = None
    safe_gather_ok_fn: SafeGatherOkFn | None = None
    safe_gather_ok_value_fn: SafeGatherOkValueFn | None = None
    guard_cfg: GuardConfig | None = None
    host_bool_value_fn: HostBoolValueFn | None = None
    host_int_value_fn: HostIntValueFn | None = None
    guards_enabled_fn: GuardsEnabledFn | None = None
    ledger_roots_hash_host_fn: LedgerRootsHashFn | None = None
    safe_gather_policy: SafetyPolicy | None = None
    safe_gather_policy_value: PolicyValue | None = None
    cnf2_enabled_fn: Callable[[], bool] | None = None
    cnf2_slot1_enabled_fn: Callable[[], bool] | None = None
    cnf2_metrics_enabled_fn: Callable[[], bool] | None = None
    cnf2_metrics_update_fn: Callable[[int, int, int], None] | None = None


DEFAULT_CNF2_CONFIG = Cnf2Config()


@dataclass(frozen=True, slots=True)
class ArenaInteractConfig:
    """Arena interact DI bundle."""

    safe_gather_fn: SafeGatherFn | None = None
    safe_gather_value_fn: SafeGatherValueFn | None = None
    safe_gather_policy: SafetyPolicy | None = None
    safe_gather_policy_value: PolicyValue | None = None
    policy_binding: PolicyBinding | None = None
    guard_cfg: GuardConfig | None = None
    scatter_drop_fn: ScatterDropFn | None = None
    guard_max_fn: GuardMaxFn | None = None


DEFAULT_ARENA_INTERACT_CONFIG = ArenaInteractConfig()


@dataclass(frozen=True, slots=True)
class ArenaCycleConfig:
    """Arena cycle DI bundle."""

    op_rank_fn: OpRankFn | None = None
    servo_enabled_fn: ServoEnabledFn | None = None
    servo_update_fn: ServoUpdateFn | None = None
    op_morton_fn: OpMortonFn | None = None
    op_sort_and_swizzle_with_perm_fn: OpSortWithPermFn | None = None
    op_sort_and_swizzle_morton_with_perm_fn: OpSortWithPermFn | None = None
    op_sort_and_swizzle_blocked_with_perm_fn: OpSortWithPermFn | None = None
    op_sort_and_swizzle_hierarchical_with_perm_fn: OpSortWithPermFn | None = None
    op_sort_and_swizzle_servo_with_perm_fn: OpSortWithPermFn | None = None
    safe_gather_fn: SafeGatherFn | None = None
    safe_gather_value_fn: SafeGatherValueFn | None = None
    safe_gather_policy: SafetyPolicy | None = None
    safe_gather_policy_value: PolicyValue | None = None
    policy_binding: PolicyBinding | None = None
    guard_cfg: GuardConfig | None = None
    arena_root_hash_fn: ArenaRootHashFn | None = None
    damage_tile_size_fn: DamageTileSizeFn | None = None
    damage_metrics_update_fn: DamageMetricsUpdateFn | None = None
    op_interact_fn: OpInteractFn | None = None
    interact_cfg: ArenaInteractConfig | None = None


DEFAULT_ARENA_CYCLE_CONFIG = ArenaCycleConfig()


@dataclass(frozen=True, slots=True)
class IntrinsicConfig:
    """Intrinsic cycle DI bundle."""

    intern_cfg: InternConfig | None = None
    intern_fn: InternFn | None = None
    node_batch_fn: NodeBatchFn | None = None
    host_raise_fn: HostRaiseFn | None = None


DEFAULT_INTRINSIC_CONFIG = IntrinsicConfig()

__all__ = [
    "Cnf2Flags",
    "DEFAULT_CNF2_FLAGS",
    "Cnf2Config",
    "DEFAULT_CNF2_CONFIG",
    "ArenaInteractConfig",
    "DEFAULT_ARENA_INTERACT_CONFIG",
    "ArenaCycleConfig",
    "DEFAULT_ARENA_CYCLE_CONFIG",
    "IntrinsicConfig",
    "DEFAULT_INTRINSIC_CONFIG",
]
