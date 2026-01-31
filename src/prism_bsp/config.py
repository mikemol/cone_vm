from __future__ import annotations

from dataclasses import dataclass, replace, field
from functools import partial
from typing import Callable, TypeAlias, Any

from prism_coord.config import CoordConfig
from prism_core.compact import CompactConfig
from prism_core.guards import GuardConfig, resolve_safe_gather_ok_fn, resolve_safe_gather_ok_value_fn
from prism_core import jax_safe as _jax_safe
from prism_core.safety import (
    SafetyPolicy,
    PolicyBinding,
    StaticPolicyBinding,
    ValuePolicyBinding,
    oob_any,
    oob_any_value,
)
from prism_core.modes import Cnf2Mode, coerce_cnf2_mode, ValidateMode, require_validate_mode
from prism_core.errors import PrismCnf2ModeConflictError, PrismPolicyBindingError
from prism_core.di import call_with_optional_kwargs
from prism_ledger.config import InternConfig
from prism_core.protocols import (
    PolicyValue,
    SafeGatherFn,
    SafeGatherOkBoundFn,
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
from prism_vm_core.candidates import candidate_indices_cfg
from prism_coord.coord import coord_xor_batch
from prism_ledger.intern import intern_nodes
from prism_vm_core.gating import _cnf2_enabled, _cnf2_slot1_enabled
from prism_metrics.metrics import _cnf2_metrics_enabled, _cnf2_metrics_update
from prism_vm_core.guards import _guards_enabled
from prism_vm_core.domains import _host_bool_value, _host_int_value
from prism_vm_core.hashes import _ledger_roots_hash_host
from prism_semantics.commit import commit_stratum_bound, commit_stratum_value


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

    cnf2_mode: Cnf2Mode | None = None
    flags: Cnf2Flags | None = None
    intern_cfg: InternConfig | None = None
    coord_cfg: CoordConfig | None = None
    intern_fn: InternFn | None = None
    node_batch_fn: NodeBatchFn | None = None
    coord_xor_batch_fn: CoordXorBatchFn | None = None
    emit_candidates_fn: EmitCandidatesFn | None = None
    candidate_indices_fn: CandidateIndicesFn | None = None
    candidate_fns: "Cnf2CandidateFns" | None = None
    compact_cfg: CompactConfig | None = None
    scatter_drop_fn: ScatterDropFn | None = None
    commit_stratum_fn: CommitStratumFn | None = None
    apply_q_fn: ApplyQFn | None = None
    identity_q_fn: IdentityQFn | None = None
    policy_fns: "Cnf2PolicyFns" | None = None
    runtime_fns: "Cnf2RuntimeFns" = field(
        default_factory=lambda: DEFAULT_CNF2_RUNTIME_FNS
    )
    safe_gather_ok_fn: SafeGatherOkFn | None = None
    safe_gather_ok_bound_fn: SafeGatherOkBoundFn | None = None
    safe_gather_ok_value_fn: SafeGatherOkValueFn | None = None
    guard_cfg: GuardConfig | None = None
    host_bool_value_fn: HostBoolValueFn | None = None
    host_int_value_fn: HostIntValueFn | None = None
    guards_enabled_fn: GuardsEnabledFn | None = None
    ledger_roots_hash_host_fn: LedgerRootsHashFn | None = None
    safe_gather_policy: SafetyPolicy | None = None
    safe_gather_policy_value: PolicyValue | None = None
    policy_binding: PolicyBinding | None = None


@dataclass(frozen=True, slots=True)
class Cnf2ResolvedInputs:
    """Resolved CNF-2 inputs with all config overrides applied."""

    runtime_fns: "Cnf2RuntimeFns"
    guard_cfg: GuardConfig | None
    intern_cfg: InternConfig | None
    intern_fn: InternFn
    node_batch_fn: NodeBatchFn
    coord_xor_batch_fn: CoordXorBatchFn
    emit_candidates_fn: EmitCandidatesFn
    candidate_indices_fn: CandidateIndicesFn
    scatter_drop_fn: ScatterDropFn
    commit_stratum_fn: CommitStratumFn
    apply_q_fn: ApplyQFn
    identity_q_fn: IdentityQFn
    safe_gather_ok_fn: SafeGatherOkFn
    safe_gather_ok_value_fn: SafeGatherOkValueFn
    host_bool_value_fn: HostBoolValueFn
    host_int_value_fn: HostIntValueFn
    guards_enabled_fn: GuardsEnabledFn
    ledger_roots_hash_host_fn: LedgerRootsHashFn
    cnf2_mode: Cnf2Mode | None


@dataclass(frozen=True, slots=True)
class Cnf2CandidateInputs:
    """Resolved candidate helper inputs."""

    emit_candidates_fn: EmitCandidatesFn
    candidate_indices_fn: CandidateIndicesFn
    scatter_drop_fn: ScatterDropFn


@dataclass(frozen=True, slots=True)
class Cnf2CandidateFns:
    """Bundle of candidate helper functions observed as a forwarding group."""

    emit_candidates_fn: EmitCandidatesFn | None = None
    candidate_indices_fn: CandidateIndicesFn | None = None
    scatter_drop_fn: ScatterDropFn | None = None


@dataclass(frozen=True, slots=True)
class Cnf2RuntimeFns:
    """Bundle of runtime control hooks observed as a forwarding group."""

    cnf2_enabled_fn: Callable[[], bool] = _cnf2_enabled
    cnf2_slot1_enabled_fn: Callable[[], bool] = _cnf2_slot1_enabled
    cnf2_metrics_enabled_fn: Callable[[], bool] = _cnf2_metrics_enabled
    cnf2_metrics_update_fn: Callable[[int, int, int], None] = _cnf2_metrics_update


DEFAULT_CNF2_RUNTIME_FNS = Cnf2RuntimeFns()


@dataclass(frozen=True, slots=True)
class Cnf2PolicyFns:
    """Bundle of policy-sensitive core functions observed as a forwarding group."""

    commit_stratum_fn: CommitStratumFn | None = None
    apply_q_fn: ApplyQFn | None = None
    identity_q_fn: IdentityQFn | None = None
    safe_gather_ok_fn: SafeGatherOkFn | None = None
    safe_gather_ok_bound_fn: SafeGatherOkBoundFn | None = None
    safe_gather_ok_value_fn: SafeGatherOkValueFn | None = None


def resolve_cnf2_candidate_inputs(
    cfg: Cnf2Config | None,
    *,
    emit_candidates_fn: EmitCandidatesFn,
    candidate_indices_fn: CandidateIndicesFn,
    candidate_indices_default: CandidateIndicesFn,
    scatter_drop_fn: ScatterDropFn,
) -> Cnf2CandidateInputs:
    if cfg is not None:
        if cfg.candidate_fns is not None:
            bundle = cfg.candidate_fns
            if bundle.emit_candidates_fn is not None:
                emit_candidates_fn = bundle.emit_candidates_fn
            if bundle.candidate_indices_fn is not None:
                candidate_indices_fn = bundle.candidate_indices_fn
            if bundle.scatter_drop_fn is not None:
                scatter_drop_fn = bundle.scatter_drop_fn
        if cfg.emit_candidates_fn is not None:
            emit_candidates_fn = cfg.emit_candidates_fn
        if cfg.candidate_indices_fn is not None:
            candidate_indices_fn = cfg.candidate_indices_fn
        if cfg.compact_cfg is not None and candidate_indices_fn is candidate_indices_default:
            candidate_indices_fn = partial(
                candidate_indices_cfg, compact_cfg=cfg.compact_cfg
            )
        if cfg.scatter_drop_fn is not None:
            scatter_drop_fn = cfg.scatter_drop_fn
    return Cnf2CandidateInputs(
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
    )


@dataclass(frozen=True, slots=True)
class Cnf2InternInputs:
    """Resolved intern helper inputs."""

    compact_candidates_fn: Callable
    intern_fn: InternFn
    node_batch_fn: NodeBatchFn


def resolve_cnf2_intern_inputs(
    cfg: Cnf2Config | None,
    *,
    compact_candidates_fn: Callable,
    compact_candidates_default: Callable,
    candidate_indices_default: CandidateIndicesFn,
    intern_fn: InternFn,
    intern_fn_default: InternFn,
    intern_cfg: InternConfig | None,
    node_batch_fn: NodeBatchFn,
    node_batch_default: NodeBatchFn,
) -> Cnf2InternInputs:
    if cfg is not None:
        if intern_cfg is None:
            intern_cfg = cfg.intern_cfg
        if intern_fn is intern_fn_default and cfg.intern_fn is not None:
            intern_fn = cfg.intern_fn
        if node_batch_fn is node_batch_default and cfg.node_batch_fn is not None:
            node_batch_fn = cfg.node_batch_fn
        if cfg.candidate_indices_fn is not None or cfg.compact_cfg is not None:
            candidate_indices_fn = cfg.candidate_indices_fn or candidate_indices_default
            if cfg.compact_cfg is not None and candidate_indices_fn is candidate_indices_default:
                candidate_indices_fn = partial(
                    candidate_indices_cfg, compact_cfg=cfg.compact_cfg
                )
            if compact_candidates_fn is compact_candidates_default:
                compact_candidates_fn = partial(
                    compact_candidates_default,
                    candidate_indices_fn=candidate_indices_fn,
                )
    if intern_cfg is not None and intern_fn is intern_fn_default:
        intern_fn = partial(intern_fn_default, cfg=intern_cfg)
    return Cnf2InternInputs(
        compact_candidates_fn=compact_candidates_fn,
        intern_fn=intern_fn,
        node_batch_fn=node_batch_fn,
    )


def resolve_cnf2_inputs(
    cfg: Cnf2Config | None,
    *,
    guard_cfg: GuardConfig | None,
    intern_cfg: InternConfig | None,
    intern_fn: InternFn,
    node_batch_fn: NodeBatchFn,
    coord_xor_batch_fn: CoordXorBatchFn,
    emit_candidates_fn: EmitCandidatesFn,
    candidate_indices_fn: CandidateIndicesFn,
    scatter_drop_fn: ScatterDropFn,
    commit_stratum_fn: CommitStratumFn,
    apply_q_fn: ApplyQFn,
    identity_q_fn: IdentityQFn,
    safe_gather_ok_fn: SafeGatherOkFn,
    safe_gather_ok_value_fn: SafeGatherOkValueFn,
    host_bool_value_fn: HostBoolValueFn,
    host_int_value_fn: HostIntValueFn,
    guards_enabled_fn: GuardsEnabledFn,
    ledger_roots_hash_host_fn: LedgerRootsHashFn,
    runtime_fns: Cnf2RuntimeFns,
) -> Cnf2ResolvedInputs:
    """Resolve CNF-2 config overrides into concrete inputs."""

    def _maybe_override(current, default, override):
        if override is None:
            return current
        if current is default:
            return override
        return current

    intern_default = intern_fn
    node_batch_default = node_batch_fn
    coord_xor_default = coord_xor_batch_fn
    emit_candidates_default = emit_candidates_fn
    candidate_indices_default = candidate_indices_fn
    scatter_drop_default = scatter_drop_fn
    commit_stratum_default = commit_stratum_fn
    apply_q_default = apply_q_fn
    identity_q_default = identity_q_fn
    safe_gather_ok_default = safe_gather_ok_fn
    safe_gather_ok_value_default = safe_gather_ok_value_fn
    host_bool_default = host_bool_value_fn
    host_int_default = host_int_value_fn
    guards_enabled_default = guards_enabled_fn
    ledger_roots_hash_default = ledger_roots_hash_host_fn
    runtime_default = runtime_fns

    cnf2_mode = None
    if cfg is not None:
        if cfg.policy_fns is not None:
            policy_bundle = cfg.policy_fns
            if policy_bundle.commit_stratum_fn is not None:
                commit_stratum_fn = policy_bundle.commit_stratum_fn
            if policy_bundle.apply_q_fn is not None:
                apply_q_fn = policy_bundle.apply_q_fn
            if policy_bundle.identity_q_fn is not None:
                identity_q_fn = policy_bundle.identity_q_fn
            if policy_bundle.safe_gather_ok_fn is not None:
                safe_gather_ok_fn = policy_bundle.safe_gather_ok_fn
            if policy_bundle.safe_gather_ok_bound_fn is not None:
                if safe_gather_ok_fn is safe_gather_ok_default:
                    safe_gather_ok_fn = policy_bundle.safe_gather_ok_bound_fn
            if policy_bundle.safe_gather_ok_value_fn is not None:
                safe_gather_ok_value_fn = policy_bundle.safe_gather_ok_value_fn
        runtime_bundle = cfg.runtime_fns
        if runtime_fns is runtime_default:
            runtime_fns = runtime_bundle
        if cfg.policy_binding is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_core received cfg.policy_binding; bind at wrapper",
                context="cycle_candidates_core",
                policy_mode="ambiguous",
            )
        cnf2_mode = cfg.cnf2_mode
        guard_cfg = cfg.guard_cfg if guard_cfg is None else guard_cfg
        intern_cfg = intern_cfg if intern_cfg is not None else cfg.intern_cfg
        if cfg.coord_cfg is not None and coord_xor_batch_fn is coord_xor_batch:
            coord_xor_batch_fn = partial(coord_xor_batch, cfg=cfg.coord_cfg)
        intern_fn = _maybe_override(intern_fn, intern_default, cfg.intern_fn)
        node_batch_fn = _maybe_override(node_batch_fn, node_batch_default, cfg.node_batch_fn)
        coord_xor_batch_fn = _maybe_override(
            coord_xor_batch_fn, coord_xor_default, cfg.coord_xor_batch_fn
        )
        emit_candidates_fn = _maybe_override(
            emit_candidates_fn, emit_candidates_default, cfg.emit_candidates_fn
        )
        candidate_indices_fn = _maybe_override(
            candidate_indices_fn, candidate_indices_default, cfg.candidate_indices_fn
        )
        if cfg.compact_cfg is not None and candidate_indices_fn is candidate_indices_default:
            candidate_indices_fn = partial(
                candidate_indices_cfg, compact_cfg=cfg.compact_cfg
            )
        scatter_drop_fn = _maybe_override(
            scatter_drop_fn, scatter_drop_default, cfg.scatter_drop_fn
        )
        safe_gather_ok_fn = _maybe_override(
            safe_gather_ok_fn, safe_gather_ok_default, cfg.safe_gather_ok_fn
        )
        safe_gather_ok_value_fn = _maybe_override(
            safe_gather_ok_value_fn, safe_gather_ok_value_default, cfg.safe_gather_ok_value_fn
        )
        if cfg.commit_stratum_fn is not None and getattr(safe_gather_ok_fn, "_prism_policy_bound", False):
            if cfg.commit_stratum_fn is not commit_stratum_fn:
                raise PrismPolicyBindingError(
                    "cycle_candidates_core received cfg.commit_stratum_fn with policy-bound safe_gather_ok_fn",
                    context="cycle_candidates_core",
                    policy_mode="static",
                )
        commit_stratum_fn = _maybe_override(
            commit_stratum_fn, commit_stratum_default, cfg.commit_stratum_fn
        )
        apply_q_fn = _maybe_override(apply_q_fn, apply_q_default, cfg.apply_q_fn)
        identity_q_fn = _maybe_override(identity_q_fn, identity_q_default, cfg.identity_q_fn)
        host_bool_value_fn = _maybe_override(
            host_bool_value_fn, host_bool_default, cfg.host_bool_value_fn
        )
        host_int_value_fn = _maybe_override(
            host_int_value_fn, host_int_default, cfg.host_int_value_fn
        )
        guards_enabled_fn = _maybe_override(
            guards_enabled_fn, guards_enabled_default, cfg.guards_enabled_fn
        )
        ledger_roots_hash_host_fn = _maybe_override(
            ledger_roots_hash_host_fn,
            ledger_roots_hash_default,
            cfg.ledger_roots_hash_host_fn,
        )
        if cfg.flags is not None:
            if cfg.flags.enabled is not None:
                enabled_value = bool(cfg.flags.enabled)
                runtime_fns = replace(
                    runtime_fns, cnf2_enabled_fn=lambda: enabled_value
                )
            if cfg.flags.slot1_enabled is not None:
                slot1_value = bool(cfg.flags.slot1_enabled)
                runtime_fns = replace(
                    runtime_fns, cnf2_slot1_enabled_fn=lambda: slot1_value
                )
        if cfg.safe_gather_ok_value_fn is not None and getattr(safe_gather_ok_fn, "_prism_policy_bound", False):
            raise PrismPolicyBindingError(
                "cycle_candidates_core received cfg.safe_gather_ok_value_fn with policy-bound safe_gather_ok_fn",
                context="cycle_candidates_core",
                policy_mode="ambiguous",
            )
    if cnf2_mode is not None:
        mode = coerce_cnf2_mode(cnf2_mode, context="cycle_candidates_core")
        if mode != Cnf2Mode.AUTO:
            if (
                cfg is not None
                and cfg.flags is not None
            ) or runtime_fns is not runtime_default:
                raise PrismCnf2ModeConflictError(
                    "cycle_candidates_core received cnf2_mode alongside cnf2_flags or runtime_fns overrides",
                    context="cycle_candidates_core",
                )
            enabled_value = mode in (Cnf2Mode.BASE, Cnf2Mode.SLOT1)
            slot1_value = mode == Cnf2Mode.SLOT1
            runtime_fns = replace(
                runtime_fns,
                cnf2_enabled_fn=lambda: enabled_value,
                cnf2_slot1_enabled_fn=lambda: slot1_value,
            )
    if intern_cfg is not None and intern_fn is intern_nodes:
        intern_fn = partial(intern_nodes, cfg=intern_cfg)
    if intern_cfg is not None and coord_xor_batch_fn is coord_xor_batch:
        coord_xor_batch_fn = partial(coord_xor_batch, intern_cfg=intern_cfg)
    if commit_stratum_fn is commit_stratum_bound and not getattr(
        safe_gather_ok_fn, "_prism_policy_bound", False
    ):
        raise PrismPolicyBindingError(
            "cycle_candidates_core received commit_stratum_bound without policy-bound safe_gather_ok_fn",
            context="cycle_candidates_core",
            policy_mode="static",
        )
    if runtime_fns.cnf2_metrics_enabled_fn is not None and not runtime_fns.cnf2_metrics_enabled_fn():
        runtime_fns = replace(runtime_fns, cnf2_metrics_update_fn=lambda *_: None)
    if node_batch_fn is None:
        raise ValueError("node_batch_fn is required")
    if emit_candidates_fn is None:
        raise ValueError("emit_candidates_fn is required")
    if candidate_indices_fn is None:
        raise ValueError("candidate_indices_fn is required")
    if scatter_drop_fn is None:
        raise ValueError("scatter_drop_fn is required")
    if commit_stratum_fn is None:
        raise ValueError("commit_stratum_fn is required")
    if apply_q_fn is None:
        raise ValueError("apply_q_fn is required")
    if identity_q_fn is None:
        raise ValueError("identity_q_fn is required")
    return Cnf2ResolvedInputs(
        runtime_fns=runtime_fns,
        guard_cfg=guard_cfg,
        intern_cfg=intern_cfg,
        intern_fn=intern_fn,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        commit_stratum_fn=commit_stratum_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=safe_gather_ok_fn,
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        cnf2_mode=cnf2_mode,
    )


DEFAULT_CNF2_CONFIG = Cnf2Config()


@dataclass(frozen=True, slots=True)
class Cnf2StaticBoundConfig:
    """CNF-2 config with required static PolicyBinding (no policy duplication)."""

    cfg: Cnf2Config
    policy_binding: StaticPolicyBinding

    def as_cfg(self) -> Cnf2Config:
        return replace(
            self.cfg,
            policy_binding=None,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )

    def bind_cfg(
        self,
        *,
        safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
        guard_cfg: GuardConfig | None = None,
        commit_stratum_fn: CommitStratumFn | None = None,
    ) -> tuple[Cnf2Config, SafetyPolicy]:
        """Return a cfg with policy-bound safe_gather_ok_fn + commit_stratum_bound."""
        if self.cfg.safe_gather_ok_fn is not None or self.cfg.safe_gather_ok_bound_fn is not None:
            raise PrismPolicyBindingError(
                "Cnf2StaticBoundConfig received safe_gather_ok_fn; bind at edge",
                context="Cnf2StaticBoundConfig.bind_cfg",
                policy_mode="static",
            )
        if self.cfg.safe_gather_ok_value_fn is not None:
            raise PrismPolicyBindingError(
                "Cnf2StaticBoundConfig received safe_gather_ok_value_fn",
                context="Cnf2StaticBoundConfig.bind_cfg",
                policy_mode="static",
            )
        if getattr(safe_gather_ok_fn, "_prism_policy_bound", False):
            raise PrismPolicyBindingError(
                "Cnf2StaticBoundConfig received policy-bound safe_gather_ok_fn",
                context="Cnf2StaticBoundConfig.bind_cfg",
                policy_mode="static",
            )
        if commit_stratum_fn is None:
            commit_stratum_fn = commit_stratum_bound
        if commit_stratum_fn is not commit_stratum_bound:
            raise PrismPolicyBindingError(
                "Cnf2StaticBoundConfig requires commit_stratum_bound",
                context="Cnf2StaticBoundConfig.bind_cfg",
                policy_mode="static",
            )
        if guard_cfg is None:
            guard_cfg = self.cfg.guard_cfg
        bound_ok_fn = resolve_safe_gather_ok_fn(
            safe_gather_ok_fn=safe_gather_ok_fn,
            policy=self.policy_binding.policy,
            guard_cfg=guard_cfg,
        )
        cfg = replace(
            self.cfg,
            policy_binding=None,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
            safe_gather_ok_bound_fn=bound_ok_fn,
            safe_gather_ok_fn=None,
            safe_gather_ok_value_fn=None,
            commit_stratum_fn=commit_stratum_fn,
            guard_cfg=None,
        )
        return cfg, self.policy_binding.policy

    @classmethod
    def bind(
        cls, cfg: Cnf2Config, policy_binding: StaticPolicyBinding
    ) -> "Cnf2StaticBoundConfig":
        return cls(cfg=cfg, policy_binding=policy_binding)


@dataclass(frozen=True, slots=True)
class Cnf2ValueBoundConfig:
    """CNF-2 config with required value PolicyBinding (no policy duplication)."""

    cfg: Cnf2Config
    policy_binding: ValuePolicyBinding

    def as_cfg(self) -> Cnf2Config:
        return replace(
            self.cfg,
            policy_binding=None,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )

    def bind_cfg(
        self,
        *,
        safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
        guard_cfg: GuardConfig | None = None,
        commit_stratum_fn: CommitStratumFn | None = None,
    ) -> tuple[Cnf2Config, PolicyValue]:
        """Return a cfg with policy-value + guarded safe_gather_ok_value_fn."""
        if self.cfg.safe_gather_ok_fn is not None or self.cfg.safe_gather_ok_bound_fn is not None:
            raise PrismPolicyBindingError(
                "Cnf2ValueBoundConfig received safe_gather_ok_fn/bound_fn",
                context="Cnf2ValueBoundConfig.bind_cfg",
                policy_mode="value",
            )
        if self.cfg.safe_gather_ok_value_fn is not None:
            raise PrismPolicyBindingError(
                "Cnf2ValueBoundConfig received safe_gather_ok_value_fn",
                context="Cnf2ValueBoundConfig.bind_cfg",
                policy_mode="value",
            )
        if commit_stratum_fn is None:
            commit_stratum_fn = commit_stratum_value
        if commit_stratum_fn is not commit_stratum_value:
            raise PrismPolicyBindingError(
                "Cnf2ValueBoundConfig requires commit_stratum_value",
                context="Cnf2ValueBoundConfig.bind_cfg",
                policy_mode="value",
            )
        if guard_cfg is None:
            guard_cfg = self.cfg.guard_cfg
        bound_ok_value_fn = resolve_safe_gather_ok_value_fn(
            safe_gather_ok_value_fn=safe_gather_ok_value_fn,
            guard_cfg=guard_cfg,
        )
        cfg = replace(
            self.cfg,
            policy_binding=None,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
            safe_gather_ok_value_fn=bound_ok_value_fn,
            safe_gather_ok_fn=None,
            safe_gather_ok_bound_fn=None,
            commit_stratum_fn=commit_stratum_fn,
            guard_cfg=None,
        )
        return cfg, self.policy_binding.policy_value

    @classmethod
    def bind(
        cls, cfg: Cnf2Config, policy_binding: ValuePolicyBinding
    ) -> "Cnf2ValueBoundConfig":
        return cls(cfg=cfg, policy_binding=policy_binding)


Cnf2BoundConfig: TypeAlias = Cnf2StaticBoundConfig | Cnf2ValueBoundConfig


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
class SwizzleWithPermFns:
    """Bundle of swizzle-with-perm functions observed as a forwarding group."""

    with_perm: OpSortWithPermFn | None = None
    morton_with_perm: OpSortWithPermFn | None = None
    blocked_with_perm: OpSortWithPermFn | None = None
    hierarchical_with_perm: OpSortWithPermFn | None = None
    servo_with_perm: OpSortWithPermFn | None = None


@dataclass(frozen=True, slots=True)
class SwizzleWithPermFnsBound:
    """Required swizzle-with-perm bundle (no None fields)."""

    with_perm: OpSortWithPermFn
    morton_with_perm: OpSortWithPermFn
    blocked_with_perm: OpSortWithPermFn
    hierarchical_with_perm: OpSortWithPermFn
    servo_with_perm: OpSortWithPermFn


@dataclass(frozen=True, slots=True)
class ArenaCycleConfig:
    """Arena cycle DI bundle."""

    op_rank_fn: OpRankFn | None = None
    servo_enabled_fn: ServoEnabledFn | None = None
    servo_update_fn: ServoUpdateFn | None = None
    op_morton_fn: OpMortonFn | None = None
    swizzle_with_perm_fns: SwizzleWithPermFns | None = None
    swizzle_with_perm_value_fns: SwizzleWithPermFns | None = None
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
class ArenaSortConfig:
    """Arena sort/schedule parameters bundled as data.

    morton is an optional precomputed array; type left as Any to avoid
    importing jax into config modules.
    """

    do_sort: bool = True
    use_morton: bool = False
    block_size: int | None = None
    morton: Any | None = None
    l2_block_size: int | None = None
    l1_block_size: int | None = None
    do_global: bool = False


DEFAULT_ARENA_SORT_CONFIG = ArenaSortConfig()


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
    "Cnf2StaticBoundConfig",
    "Cnf2ValueBoundConfig",
    "Cnf2BoundConfig",
    "ArenaInteractConfig",
    "DEFAULT_ARENA_INTERACT_CONFIG",
    "SwizzleWithPermFns",
    "SwizzleWithPermFnsBound",
    "ArenaCycleConfig",
    "DEFAULT_ARENA_CYCLE_CONFIG",
    "ArenaSortConfig",
    "DEFAULT_ARENA_SORT_CONFIG",
    "IntrinsicConfig",
    "DEFAULT_INTRINSIC_CONFIG",
    "Cnf2ResolvedInputs",
    "resolve_cnf2_inputs",
    "Cnf2CandidateInputs",
    "Cnf2CandidateFns",
    "Cnf2RuntimeFns",
    "DEFAULT_CNF2_RUNTIME_FNS",
    "Cnf2PolicyFns",
    "resolve_cnf2_candidate_inputs",
    "Cnf2InternInputs",
    "resolve_cnf2_intern_inputs",
    "resolve_validate_mode",
    "make_cnf2_post_q_handler_static",
    "make_cnf2_post_q_handler_value",
]


def resolve_validate_mode(
    validate_mode: ValidateMode,
    *,
    guards_enabled_fn,
    context: str,
) -> ValidateMode:
    mode = require_validate_mode(validate_mode, context=context)
    if guards_enabled_fn() and mode == ValidateMode.NONE:
        mode = ValidateMode.STRICT
    return mode


def _apply_q_optional_ok(apply_q_fn: ApplyQFn, q_map, ids):
    result = call_with_optional_kwargs(
        apply_q_fn, {"return_ok": True}, q_map, ids
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


def make_cnf2_post_q_handler_static(apply_q_fn: ApplyQFn):
    def _handler(ledger2, q_map, next_frontier):
        meta = getattr(q_map, "_prism_meta", None)
        if meta is not None and meta.safe_gather_policy_value is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates core (static) received policy_value metadata",
                context="cycle_candidates_core",
                policy_mode="static",
            )
        post_ids, ok = _apply_q_optional_ok(apply_q_fn, q_map, next_frontier)
        if meta is not None and meta.safe_gather_policy is not None and ok is not None:
            corrupt = oob_any(ok, policy=meta.safe_gather_policy)
            ledger2 = ledger2._replace(corrupt=ledger2.corrupt | corrupt)
        return ledger2, post_ids

    return _handler


def make_cnf2_post_q_handler_value(apply_q_fn: ApplyQFn):
    def _handler(ledger2, q_map, next_frontier):
        meta = getattr(q_map, "_prism_meta", None)
        if meta is not None and meta.safe_gather_policy is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates core (value) received policy metadata",
                context="cycle_candidates_core",
                policy_mode="value",
            )
        post_ids, ok = _apply_q_optional_ok(apply_q_fn, q_map, next_frontier)
        if meta is not None and meta.safe_gather_policy_value is not None and ok is not None:
            corrupt = oob_any_value(ok, policy_value=meta.safe_gather_policy_value)
            ledger2 = ledger2._replace(corrupt=ledger2.corrupt | corrupt)
        return ledger2, post_ids

    return _handler
