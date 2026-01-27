# Prism VM Evolution Implementation Plan

This plan implements the features described in `in/in-4.md` through
`in/in-7.md`, plus the CNF-2 and canonical interning semantics in
`in/in-9.md` through `in/in-14.md`, the milestone-gated testing workflow in
`in/in-15.md`, the homomorphic collapse contract in `in/in-16.md`, and the
conceptual contract in `in/in-17.md`, the Min(Prism) projection commutation
track in `in/in-18.md`, the topos/sheaf formalization in `in/in-19.md`, the
hyperlattice/Cayley-Dickson framing in `in/in-20.md`, and the semantic
justification patches in `in/in-21.md` through `in/in-25.md` (gauge symmetry,
canonical novelty, hyperoperator fixed points, ordinal boundary), plus the
Agda proof roadmap in `in/in-26.md`, as a concrete evolution of the current
`prism_vm.py` implementation. The commutation glossary in `in/glossary.md` is
normative for ambiguous terms and test obligations.

Each `in/in-*.md` note now includes a short NOTE header indicating whether it
has been refined, consolidated, or obsoleted. See `audit_in_versions.md` for
the current cross-version audit.

## Goals
- Establish the Ledger + candidate pipeline as the canonical execution path.
- Converge on CNF-2 symmetric rewrite semantics with a fixed-arity candidate
  pipeline (slot1 enabled from m2; hyperstrata visibility enforced under test guards).
- Enforce univalence: full-key equality for interning and no truncation aliasing.
- Introduce self-hosted CD coordinates as interned CNF-2 objects and define
  aggregation as coordinate-space normalization.
- Preserve a stable, testable baseline while the new mode is built out.
- Stage arena scheduling and 2:1 BSP locality as performance-only concerns.

## Non-Goals (Initial MVP)
- Full hierarchical arena migration across levels (L1/L2/global).
- Perfect allocator for all graph rewrite rules beyond ADD.
- Full macro system or higher-order quotation semantics.
- Persistent event-log storage or snapshotting for CQRS beyond in-memory.
- Treating 2:1 locality as a semantic invariant before functional correctness
  is locked down.
- Ordinal-indexed termination proofs or proof-theoretic strength claims.

## Baseline Summary (Current Code)
`prism_vm.py` uses a stable heap (`Manifest`) with:
- `opcode`, `arg1`, `arg2`, `active_count` arrays.
- Hash-consing via `trace_cache` (hint-only; no retroactive pointer rewrites).
- Static optimization and branchless dispatch (`optimize_ptr`,
  `dispatch_kernel`).
- `kernel_add` and a stub `kernel_mul`.

## Architecture Decision
Ledger + candidate pipeline is the canonical execution path. Baseline
`PrismVM` stays as an oracle. Arena scheduling is performance-only and must be
denotation-invariant with respect to the Ledger CNF-2 engine; minimal scheduler
variants are required by m3 for invariance tests, while performance-grade
locality begins at m4+.
Ledger denotation is the spec; ID equality is engine-local until baseline
migrates to the Ledger interner.

## Foundational Commitments (Active Now)
These are already in effect in code/tests and are treated as non-negotiable.
- Canonical identity is ledger interning; interning is semantic compression
  (no reclamation of unique structure).
- Univalence is enforced by full key-byte equality and a semantic id cap.
- CORRUPT is a hard semantic error (alias risk); OOM is a resource boundary.
- Sorting/swizzling and scheduling are renormalization only and must not
  affect denotation after `q`.
- BSPˢ is a gauge symmetry: any BSPˢ effect must be erased by `q`.

## Semantic Commitments (Staged)
- CNF-2 symmetric rewrite semantics are the target for BSP. Every rewrite site
  emits exactly two candidate slots (enabled or disabled).
- Slot1 continuation is enabled at m2; hyperstrata visibility is enforced under
  test guards (m3 normative) to justify continuation.
- m1 uses the intrinsic cycle path; CNF-2 pipeline is gated off until m2.
- Univalence is enforced by full-key equality (hash as hint only) and
  no truncation aliasing in key encoding.
- CD coordinates are interned CNF-2 objects (`OP_COORD_*`), and coordinate
  equality is pointer equality.
- Evaluator (Manifest/Arena) and Canonicalizer (Ledger) are linked by a total
  quotient map `q`; denotation is defined by projecting provisional nodes
  through `q` and must commute with scheduling (see `in/in-16.md`). `q` is an
  irreversible coarse-graining boundary, not an evaluator or scheduling step.
- Reserved glyphs/roles are normative: `q` is meaning-forming projection,
  `σ` denotes BSPˢ permutation (gauge), and `ρ` denotes pointer/id remaps
  induced by renormalization. Never describe `σ` or `ρ` as `q`.
- Glossary adjunction/coherence discipline is normative: any new polysemous
  term must declare axes, commutation equations, and test obligations.
- Entropy terms are semantic descriptors (Arena vs canonical vs hyperoperator
  vs gauge); do not add counters unless a test explicitly requires it.
- Hyperstrata visibility rule is normative from m2 for CNF-2 slot1: pre-step rows
  are immutable during a cycle, and within-cycle visibility is defined by the
  `(s,t)` staging order; the frozen read model is the hyperstrata collapse corollary.
- 2:1 BSP swizzle/locality is staged as a performance invariant:
  - m1-m2: rank-only sort (or no sort) is acceptable for correctness tests.
  - m3: denotation invariance must hold across unsorted/rank/morton/block
    scheduler variants (even if slow).
  - m4+: 2:1 swizzle is mandatory for the performance profile, but must not
    change denotation (same canonical ids/decoded normal forms).
- Baseline `PrismVM` is a regression implementation through m3; comparisons are
  on decoded normal forms and termination behavior, not raw ids.
- Canonical novelty monotonicity and hyperoperator representation fixed points
  are semantic justifications (m3-m4); they do not imply termination and do
  not require operational counters (see `in/in-22.md` through `in/in-25.md`).
  These claims are scoped to the pre-CORRUPT (id-cap) regime.
- Min(Prism) witness principle: semantic invariants proved under projection
  `π_K` are treated as representative at scale (m3+).

## Staged Commitments (Not Yet Enforced)
- No-copy sharing and alpha-equivalence collapse (see placeholder tests).
- Binding without names (structural or coordinate-based encoding).
- Damage/locality instrumentation as a performance-only concern (m4+).

## Univalence Contract (Fixed-Width, No-Ambiguity Semantics)

### Definition

Univalence = no ambiguity of identity. A proposed node maps to exactly one
canonical node iff its full encoded key bytes are identical. Hashing, sorting,
or indexing strategies may accelerate lookup, but equality is decided only by
full key-byte equality.

Univalence does not require the absence of hash collisions; it requires that
collisions never introduce ambiguity. Full-key comparison (or structurally
collision-free indexing) is mandatory.

### Canonical Key

For every interned node, define a canonical key `K` after canonicalization:

```
K = encode(op)
  || encode(rep(a1))
  || encode(rep(a2))
  || encode(extra(op))
```

Where:

- `rep(id)` = canonical representative of `id`
  - equals `id` in m1–m3
  - becomes `find(id)` if rewrite-union is introduced later
- Commutative ops (`ADD`, later others):
  - sort `(rep(a1), rep(a2))` before encoding
- `extra(op)` is empty unless explicitly defined (e.g. future flags)

Two nodes are equal iff their full `K` byte sequences are identical.

### Fixed-Width Encoding Mode (m1–m3)

In m1–m3, `K` uses a fixed-width encoding:

- `op` -> 1 byte
- `rep(a1)` -> 16 bits
- `rep(a2)` -> 16 bits
- total: 5 bytes

This encoding is injective only over a bounded id domain.

Therefore, the id bound is a semantic invariant, not a memory limit.

### Representational Cap (Hard Semantic Invariant)

Define (naming is capacity vs legal id):

- `LEDGER_CAPACITY = 65535` (backing array length / capacity)
- `MAX_ID = 65534` (max legal id; `LEDGER_CAPACITY - 1`)
- `MAX_COUNT = MAX_ID + 1` (next-free upper bound; equals `LEDGER_CAPACITY`)
- `id = 0` reserved for `NULL`
- `id = 1` reserved for `ZERO`

Invariant:

- All reachable ids used in key encoding must satisfy `id <= MAX_ID`.
- `ledger.count` is a next-free pointer and may equal `MAX_COUNT` (full).
- Any allocation that would make `count > MAX_COUNT` is CORRUPT.

Violating this invariant would cause key aliasing and destroy univalence.

Therefore:

- Exceeding `MAX_ID` is CORRUPT, not OOM.
- Continuing execution past this point is semantically undefined and forbidden.

### Deterministic Corruption Handling (JAX-Safe)

To maintain determinism and avoid undefined behavior in JIT-compiled code:

Device-side (checked packing / interning):

- If any of the following occur:
  - `count > MAX_COUNT`
  - `a1 > MAX_ID`
  - `a2 > MAX_ID`
- Then:
  - set a persistent `corrupt = True` flag in the Ledger/interner state
  - clamp offending values to a safe sentinel (e.g. `0`) to keep execution defined
  - continue execution without crashing or branching on host state

Host-side:

- After `block_until_ready()`:
  - if `corrupt == True`, raise a hard runtime error:

```
RuntimeError("CORRUPT: key encoding alias risk (id width exceeded)")
```

This guarantees:

- no silent ambiguity
- deterministic failure
- identical behavior across CPU/GPU backends

### Indexing Rule

- Hashes, buckets, sorted lanes, or Morton orderings are acceleration only.
- Equality is decided only by full key-byte comparison.
- Any index backend must be rebuildable from the canonical event stream and must
  preserve full-key equality semantics.

### Relationship to COORD Primitives

The fixed-width id cap is intentional and semantic.

Growth beyond this bound is expected to occur via structure, not scalar ids:

- COORD primitives (`OP_COORD_*`) represent arbitrarily large coordinate spaces
  as interned CNF-2 DAGs
- Coordinate equality is pointer equality
- Coordinate normalization happens before parent nodes are packed and interned

Thus:

- semantic expressiveness can grow without widening id fields
- the id bound remains a correctness invariant, not a scalability bottleneck

### Future Extension: Variable-Length Key Encoding (Explicit Milestone)

Variable-length key encoding is not permitted implicitly.

A future milestone may introduce:

- a new `KeyCodecVarLen`
- varint or structural encoding of child ids
- a radix/trie index backend

This is a semantic upgrade, not an optimization, and must:

- preserve the Univalence Contract
- preserve determinism
- pass the same univalence and injectivity test suite

Until that milestone is explicitly landed, fixed-width hard-cap mode is the law.

### Summary (Non-Negotiables)

- Univalence means no ambiguity, not “no hash collisions”
- Fixed-width keys are acceptable only with a semantic id cap
- Exceeding the cap is CORRUPT, not OOM
- Partial allocation of "new" proposals is invalid; fixed-width mode must
  treat any unmet new allocation as CORRUPT (no silent spawn clipping).
- COORD enables structural growth without widening ids
- Var-length encoding, if added, is a deliberate semantic transition

## Denotation Contract
Define a shared denotation interface used for cross-engine comparisons:

- `denote(ptr, store) -> normal_form_ptr` and `pretty(ptr) -> string/tuple`.
- Soundness: `pretty(denote_engine(expr)) == pretty(denote_ledger(expr))`.
- Idempotence: `denote(denote(x)) == denote(x)`.
- ID equivalence is engine-local until baseline migrates to the Ledger interner.
- Termination: within the same step budget, either both converge or both do not;
  non-termination or timeout is treated as a mismatch.
- CORRUPT dominance: if `corrupt == True` at any point, the denotation result
  must be `("corrupt", ...)`; clamped sentinel values must never be reported
  as `("ok", ...)`.
- OOM dominance: if `oom == True` at any point, the denotation result must be
  `("oom", ...)`; sentinel ids returned by stop paths are not semantic values.
- Univalence alignment: any engine-level compare or normalization must preserve
  the Univalence Contract (full key-byte equality, corrupt trap on overflow).
- Structural invariants (hashes, pointer topology checks) are implementation
  diagnostics only; semantic claims must be stated via `pretty(denote(q(...)))`.
- Homomorphic projection: define `q(provisional_node) -> canonical_id` and
  compare denotations only after projection; evaluator steps must commute with
  `q` up to canonical rewrite (see `in/in-16.md`).
- Normal-form reporting: `pretty` should return tagged outcomes (e.g.,
  `("ok", decoded)`, `("oom", ...)`, `("corrupt", ...)`, `("timeout", steps)`).

## Locked Decisions
- Read model backend (m1): keep the current ordered index (sorted lanes +
  binary search), maintained via full-array merge in m1. Tests assert full-key
  equality and do not assume any specific index structure so trie or hash-bucket
  indexes can replace it later.
- Univalence encoding (m1): hard-cap mode with 5-byte key; enforce
  `LEDGER_CAPACITY = 65535`, `MAX_ID = 65534`, and checked packing for `a1/a2`.
- Aggregation scope (m4): coordinate aggregation applies to `OP_ADD` only;
  `OP_MUL` remains Peano-style until explicitly lifted.

## Workstreams and Tasks (Tests-First Policy)
For each step below:
- Add **pytest** tests for expected behavior and a **null hypothesis** case.
- Add **program fixtures** (in `tests/` or a dedicated `programs/`) mirroring the
  pytest coverage for black-box validation.
- Commit tests first, then implement, then commit again.

Section **0** is Ledger-only (m1). Sections **1-2** land in m2 to introduce the
CNF-2 pipeline and the `q` boundary. Sections **4-9** supply scheduler variants
needed for m3 denotation invariance; performance-grade locality remains m4+.

### 0) Canonical Interner + Univalence (Ledger-first)
Objective: treat the Ledger as the canonical read model and enforce univalence.

Tests (before implementation):
- Pytest: `test_ledger_full_key_equality` (expected: hash collision does not merge distinct nodes).
- Pytest: `test_key_width_no_alias` (null: distinct child ids never alias in key encoding).
- Pytest: `test_intern_corrupt_flag_trips` (expected: corrupt flag is set and host raises).
- Pytest: `test_intern_nodes_early_out_on_oom_returns_zero_ids` (null: no mutation; count/key prefix unchanged).
- Pytest: `test_intern_nodes_early_out_on_corrupt_returns_zero_ids` (null: no mutation; count/key prefix unchanged).
- Pytest: `test_corrupt_is_sticky_and_non_mutating` (expected: corrupt blocks future interns; no mutation).
- Pytest: `test_gather_guard_negative_index_raises` (expected: guarded gather trips on idx < 0).
- Pytest: `test_gather_guard_oob_raises` (expected: guarded gather trips on idx >= size).
- Pytest: `test_gather_guard_valid_indices_noop` (null: valid gather remains intact).
- Pytest: `test_scatter_guard_negative_index_raises` (expected: guarded scatter trips on idx < 0).
- Pytest: `test_scatter_guard_oob_raises` (expected: guarded scatter trips on idx > size).
- Pytest: `test_scatter_guard_allows_sentinel_drop` (null: sentinel index is allowed).
- Pytest: `test_scatter_guard_valid_index_writes` (expected: valid scatter writes).
- Program: `tests/ledger_univalence.txt` (expected: deterministic canonical ids).
- Program null: `tests/ledger_noop.txt` (expected: no changes).

Tasks:
- Ensure interning uses full-key equality; hash is a hint only.
- Add a canonicalize-before-pack hook for any derived representations.
- Guard against truncation aliasing (m1 uses hard-cap mode).
- Enforce `LEDGER_CAPACITY = 65535`, define `MAX_ID = 65534`, reserve `0/1` for
  NULL/ZERO, and hard-trap on `count > MAX_COUNT` (lazy on allocation; eager if
  count already exceeds the bound).
- Implement a deterministic JAX trap:
  - Maintain a `corrupt` flag (or `oom`) in the interner state.
  - In checked packing, set `corrupt` on bounds violations (`a1/a2 > MAX_ID`)
    and clamp unsafe values to sentinel zeros to keep computation defined.
  - Host must raise if `corrupt` is true after `block_until_ready()`.
- Treat index arrays as rebuildable read models (event log optional).
- Add a reference interner contract test suite that validates injectivity and
  full-key equality across interner backends.

### 0b) Foundational Semantics: No-copy + Alpha-Equivalence (Staged)
Objective: pin down sharing and binding semantics early, even before lambda
encoding is fully implemented.

Tests (before implementation):
- Pytest: `test_no_copy_sharing` (expected: repeated use does not allocate duplicates; xfail until lambda encoding exists).
- Pytest: `test_alpha_equivalence_collapses` (expected: `lambda x. x` == `lambda y. y`; xfail until lambda encoding exists).
- Program: `tests/no_copy.txt` (expected: reuse does not grow structure; placeholder).
- Program: `tests/alpha_equiv.txt` (expected: alpha-equivalent forms match; placeholder).

Tasks:
- Add placeholder tests with clear xfail reasons tied to lambda encoding.
- Define a minimal lambda encoding surface (even if stubbed) to route the tests.
- Flip xfail to pass when the encoding lands.

### 1) CNF-2 Candidate Pipeline (Rewrite -> Compact -> Intern) (m2)
Objective: enforce fixed-arity rewrite semantics and explicit strata discipline.

Tests (before implementation):
- Pytest: `test_cnf2_fixed_arity` (expected: exactly 2 candidate slots per site).
- Pytest: `test_cnf2_slot_layout_indices` (expected: slot0 at `2*i`, slot1 at `2*i+1`; compaction preserves slot0 mapping).
- Pytest: `test_candidate_compaction_enabled_only` (null: disabled slots dropped).
- Pytest: `test_candidate_intern_sees_only_enabled` (expected: disabled slots
  are never interned).
- Program: `tests/cnf2_basic.txt` (expected: same normal form as baseline).

Tasks:
- Add a `CandidateBuffer` structure:
  - `enabled`, `opcode`, `arg1`, `arg2` arrays sized to `2 * |frontier|`.
- Implement CNF-2 rewrite emission (always two slots; predicates control enable).
- Slot semantics:
  - `slot0` = local rewrite
  - `slot1` = continuation / wrapper / second-stage
- Hyperstrata order (visibility boundary):
  - `slot0` writes stratum0 (`s=0`).
  - `slot1` reads only pre-step rows + stratum0, writes stratum1 (`s=1`).
  - wrapper emission reads only pre-step + prior strata, writes stratum2 (`s=2`).
- Emission invariant: buffer shape is `2 * |frontier|` every cycle.
- Candidate emission must be frontier-permutation invariant (up to slot layout)
  and independent of BSPˢ scheduling choices.
- m2 invariant: slot1 is enabled; slot1 nodes must only reference ids
  `< start(stratum1)` (hyperstrata visibility rule under guards).
- m3 invariant: hyperstrata visibility rule is fully normative and enforced
  across all strata (slot0, slot1, wrap).
- Implement compaction (enabled mask + prefix sums).
- Intern compacted candidates via `intern_nodes` (can be fused in m3).
- Optional debug: track origin site indices for strata violation tracing.

### 2) Strata Discipline + `q` Boundary (m2)
Objective: enforce explicit strata boundaries, prevent within-tier references,
and project provisional nodes through `q` at the stratum boundary.

Tests (before implementation):
- Pytest: `test_strata_no_within_tier_refs` (expected: new nodes only reference
  prior strata).
- Pytest: `test_stratum_commit_projects_q` (expected: provisional nodes map to
  canonical ids and swizzle is stable).
- Pytest: `test_commit_stratum_count_mismatch_fails` (expected: guard trips or corrupt flag is set deterministically).
- Program: `tests/strata_basic.txt` (expected: same normal form as baseline).
- Program null: `tests/strata_noop.txt` (expected: no changes).

Tasks:
- Add a lightweight `Stratum` record (offset + count) for new ids.
- Strict strata (default): nodes in `Si` may only reference ids `< start(Si)`.
- During a BSPᵗ superstep, rewrite eligibility and candidate payloads must be
  computed from the pre-step read model and must not observe within-step
  identity creation unless within-tier mode is explicitly enabled and validated.
- Future mode: allow within-stratum references only if they are to earlier ids
  in the same stratum (topological order), gated by explicit opt-in.
- Add validators that enforce the selected rule, scanning only up to
  `ledger.count`.
- Wire validators into `cycle_candidates` (guarded by a debug flag); when
  slot1 is enabled, validate stratum0/stratum1 and each wrapper micro-stratum.
- Add a `commit_stratum` boundary:
  - Validate strata, project through `q`, intern, and swizzle provisional ids.
  - Treat projection as the denotation boundary for evaluator emissions.

### 3) Coordinate Semantics (Self-Hosted CD, m4)
Objective: add CD coordinates as interned CNF-2 objects and define parity/aggregation.

Tests (before implementation):
- Pytest: `test_coord_opcodes_exist` (expected: `OP_COORD_*` registered).
- Pytest: `test_coord_pointer_equality` (expected: identical coordinates intern to same id).
- Pytest: `test_coord_xor_parity_cancel` (expected: parity cancellation is idempotent).
- Pytest: `test_coord_norm_idempotent` (expected: canonicalization is stable).
- Pytest: `test_coord_norm_confluent_small` (expected: small expressions converge).
- Pytest: `test_coord_norm_probe_only_runs_for_pairs` (expected: probe counts only coord pairs).
- Pytest: `test_coord_norm_probe_skips_non_coord_batch` (null: probe stays zero).
- Pytest: `test_coord_xor_batch_uses_single_intern_call` (expected: batch path bounds interning).
- Pytest: `test_coord_norm_batch_matches_host` (expected: batch path matches host).
- Pytest: `test_coord_add_aggregates_in_cycle_candidates` (expected: coord add lifts to XOR result).
- Pytest: `test_coord_mul_does_not_aggregate` (null: coord mul does not lift).
- Program: `tests/coord_basic.txt` (expected: canonical ids for coordinates).
- Program null: `tests/coord_noop.txt` (expected: no rewrite for non-coord ops).

Tasks:
- Add `OP_COORD_ZERO`, `OP_COORD_ONE`, `OP_COORD_PAIR`.
- Representation invariant: coordinates are stored only as interned
  `OP_COORD_*` DAGs; no linear bitstrings/digit arrays are stored anywhere.
- `OP_COORD_PAIR` is ordered (no commutative canonicalization unless staged).
- Coordinate normalization is part of canonicalization-before-pack (key-safe
  normalization), not a semantic rewrite stage.
- Coordinate normalization operates on `rep(id)` children (future Canonicalₑ)
  before packing keys and interning.
- Implement coordinate construction and XOR/parity rewrite rules in the BSP path.
- Aggregation scope (m4): coordinate lifting applies to `OP_ADD` only; `OP_MUL`
  remains Peano-style until explicitly lifted.
- Normalization order:
  1. Build coordinate CNF-2 objects.
  2. Normalize parity/XOR to a canonical coordinate object.
  3. Only then pack keys and intern any coordinate-carrying node.

### 3b) Damage / Locality Instrumentation (m4+)
Objective: add performance-only instrumentation for damage/locality without
affecting denotation.

Tests (before implementation):
- Pytest: `test_damage_metrics_no_semantic_effect` (expected: enabling metrics does not change denotation).
- Pytest: `test_damage_metrics_disabled_noop` (null: metrics remain zero/empty when disabled).
- Program: `tests/damage_metrics.txt` (expected: metrics are emitted under debug flag).
- Program null: `tests/damage_metrics_off.txt` (expected: no metrics emitted).

Tasks:
- Add per-cycle metrics (delta canonical nodes, reuse rate, damage rate).
- Define "damage" in terms of locality boundary crossings (tile/halo or block) for m4
  legacy metrics; m5 replaces this with spectral entropy (see `in/in-27.md`).
- Gate instrumentation behind a debug flag; treat it as a pure read-only pass.
- Add a small metrics summary to telemetry without altering scheduling.

### 3c) Telemetry Baselines (Host + Trace, record-only) (m4+)
Objective: capture host/CPU telemetry baselines without enforcing budgets yet.

Tests (before implementation):
- Pytest: `test_audit_host_performance_intrinsic` (expected: host perf script emits JSON).
- Pytest: `test_audit_memory_stability_intrinsic` (expected: memory script emits JSON).
- Pytest: `test_collect_host_metrics_summary` (expected: host summary table).
- Pytest: `test_collect_telemetry_baselines` (expected: baseline registry table).
- Pytest: `test_capture_trace_dry_run` (null: trace capture dry-run succeeds).
- Pytest: `test_trace_analyze_cpu_mode` (expected: CPU traces report without gating).

Tasks:
- Add `scripts/audit_host_performance.py` (cProfile → JSON, record-only).
- Add `scripts/audit_memory_stability.py` (tracemalloc → JSON, record-only).
- Add `scripts/capture_trace.py` to generate JAX traces for a small workload.
- Extend `scripts/trace_analyze.py`:
  - detect CPU traces,
  - support `--report-only` and `--json-out`.
- Add `scripts/collect_host_metrics.py` and `scripts/collect_telemetry_baselines.py`
  to aggregate artifacts into Markdown summaries.
- Record run metadata (python/jax/env/flags) into `telemetry_metadata_*.json`
  and include it in the telemetry baselines table.
- Wire CI (m4) to emit:
  - `artifacts/host_perf_*.json`
  - `artifacts/host_memory_*.json`
  - `artifacts/trace_cpu_report.json`
  and include summaries in `collected_report/telemetry_baselines.md`.
Notes:
- No performance budgets yet; record-only until baselines stabilize.
- GPU telemetry remains optional; host/CPU paths must not require pynvml.

### 3d) Geometric Servo Architecture (m5)
Objective: replace host-tuned linear paging with an on-device, **BSPˢ** entropy-driven
servo (Renormˢ) that coarse-grains Morton sorting without affecting denotation.

Tests (before implementation):
- Pytest: `test_spectral_probe_tree_peak` (expected: `H[10] > 0.8`).
- Pytest: `test_spectral_probe_noise_spread` (expected: `Entropy(H) > 3.0` bits).
- Pytest: `test_lung_capacity_dilate_contract` (expected: thresholds `P_buffer > 0.25`, `P_buffer < 0.10`, `D_active < 0.40`).
- Pytest: `test_blind_packing` (expected: `Entropy(H) < 1.5`, legacy `damage_rate(tile=512) < 0.01`).
- Pytest: `test_servo_denotation_invariance` (expected: servo on/off yields identical denotation).
- Pytest: `test_servo_sort_stable_tiebreaker` (expected: equal masked keys preserve index order).

Tasks:
- Extend `Arena` schema with a `servo` state tensor (uint32 mask + reserved slots).
- Implement `_blind_spectral_probe` (Entropyₐ):
  - filter `RANK_HOT`,
  - compute MSB(A ⊕ B) bins,
  - `bincount` into a fixed-size spectrum,
  - normalize to a probability distribution,
  - use `lax.stop_gradient` to keep it non-differentiable.
- Implement spatial hysteresis ("lung") as a memoryless controller over Entropyₐ.
- Apply masked Morton sorting (Renormˢ):
  - `key = morton(ids) & servo_mask`,
  - **stable** sort via composite key `(masked_key, original_index)` on GPU.
- Gate servo enablement behind `PRISM_ENABLE_SERVO` or milestone ≥ m5 (default off).
- Servo updates are BSPˢ gauge transforms; must commute with `q` and preserve denotation:
  - `q ∘ Servo = q`
  - `q ∘ Renormˢ = q`
- Keep m4 damage metrics as legacy; m5 entropy metrics are performance-only.

### 4) Data Model: Arena vs Manifest
Objective: introduce the fluid arena state and keep it isolated from the
existing `Manifest` for now.

Tests (before implementation):
- Pytest: `test_arena_init_zero_seed` (expected: `OP_ZERO` seeded at index 1).
- Pytest: `test_arena_init_null_free` (null: all other rows rank == FREE).
- Program: `tests/arena_init.txt` (expected: no crashes, optional debug print).
- Program null: `tests/arena_empty.txt` (expected: no allocations).

Tasks:
- Add an `Arena` NamedTuple with:
  - `opcode`, `arg1`, `arg2`, `rank` arrays
  - `count` (next-free pointer; do not overwrite with active count)
- Add `init_arena()` that seeds `OP_ZERO` and reserves `OP_NULL = 0`.
- Define ranks:
  - `RANK_HOT = 0`, `RANK_WARM = 1`, `RANK_COLD = 2`, `RANK_FREE = 3`
- Add new opcodes:
  - `OP_SORT = 99` (optional explicit primitive)
  - keep existing `OP_ZERO`, `OP_SUC`, `OP_ADD`, `OP_MUL`.

### 5) Rank Classification (2-bit Scheduler)
Objective: implement `op_rank` to classify nodes into HOT/WARM/COLD/FREE.

Tests (before implementation):
- Pytest: `test_rank_classification` (expected: ADD -> HOT, ZERO -> COLD).
- Pytest: `test_rank_null_is_free` (null: OP_NULL stays FREE).
- Program: `tests/rank_basic.txt` (expected: HOT/COLD counts match).
- Program null: `tests/rank_noop.txt` (expected: no rank changes for NULL).

Tasks:
- Implement `op_rank(arena)` in JAX:
  - `OP_NULL` -> FREE
  - Instructions (`OP_ADD`, `OP_MUL`, `OP_SORT`) -> HOT
  - Data (`OP_ZERO`, `OP_SUC`) -> COLD
- Keep rules simple at first; refine once `op_interact` exists.

### 6) Sort + Swizzle
Objective: implement the `Rank -> Sort -> Swizzle` pipeline.

Tests (before implementation):
- Pytest: `test_swizzle_preserves_edges` (expected: graph connectivity preserved).
- Pytest: `test_swizzle_null_pointer_stays_zero` (null: ptr 0 remains 0).
- Program: `tests/sort_swizzle.txt` (expected: same decode pre/post cycle).
- Program null: `tests/sort_swizzle_empty.txt` (expected: no changes).

Tasks:
- Implement `op_sort_and_swizzle(arena)`:
  - Compute rank (or accept precomputed rank).
  - Compute `perm` (stable) and `inv_perm`.
  - Permute `opcode`, `arg1`, `arg2`, `rank`.
  - Swizzle pointers via `inv_perm`, preserving `0` as NULL.
  - Preserve `count` as next-free pointer; track active count separately if needed.
- Use `jax.numpy.argsort` initially.
- Later: replace with 4-bin partition (prefix sums) for speed.
- Staging note: rank-only sort is sufficient for m1-m2 correctness. 2:1
  swizzle is optional until m4.

### 7) Interaction Kernel (Rewrite Rules)
Objective: implement `op_interact` to handle ADD rewrites in the fluid model.

Tests (before implementation):
- Pytest: `test_interact_add_zero` (expected: ADD(ZERO,y) -> y).
- Pytest: `test_interact_add_suc` (expected: ADD(SUC(x),y) expands).
- Pytest null: `test_interact_non_hot_noop` (null: non-HOT nodes unchanged).
- Program: `tests/interact_add.txt` (expected: correct reduction after cycle).
- Program null: `tests/interact_noop.txt` (expected: no reduction).

Tasks:
- Implement `op_interact(arena)` with masks:
  - `ADD(ZERO, y) -> y` (in-place rewrite or indirection).
  - `ADD(SUC(x), y) -> SUC(ADD(x, y))` (requires allocation).
- Allocation strategy:
  - Build `spawn_counts` per HOT node (0 or 2).
  - Use prefix sums for unique offsets into FREE region.
  - Scatter new nodes with `at[indices].set(...)`.
- Add bounds checks:
  - Prevent `count + total_spawn` from exceeding capacity.

### 8) Control Loop and Root Tracking
Objective: orchestrate the cycle and keep pointers valid across sorts.

Tests (before implementation):
- Pytest: `test_cycle_root_remap` (expected: root remains valid across cycles).
- Pytest null: `test_cycle_without_sort_keeps_root` (null: no swizzle => same ptr).
- Program: `tests/cycle_root.txt` (expected: same value after N cycles).
- Program null: `tests/cycle_noop.txt` (expected: no state changes).

Tasks:
- Implement `cycle()`:
  1. `op_rank`
  2. `op_sort_and_swizzle`
  3. `op_interact`
- Track `root_ptr`:
  - Store root separately (host) and remap via `inv_perm`.
  - Optionally store root in a dedicated arena field.
- Add a `run_cycles(n)` helper for benchmarking and tests.

### 9) Morton / 2:1 BSP Locality
Objective: add the 2:1 BSP ordering as a secondary sort key.

Tests (before implementation):
- Pytest: `test_morton_key_stable` (expected: stable ordering in same rank).
- Pytest null: `test_morton_disabled_matches_rank_sort` (null: rank-only same).
- Program: `tests/morton_order.txt` (expected: consistent ordering).
- Program null: `tests/morton_rank_only.txt` (expected: identical to rank-only).
- Pytest: `test_morton_denotation_invariant` (expected: same decoded result with
  and without 2:1 swizzle).

Tasks:
- Add `swizzle_2to1_dev(x, y)` (device) and `swizzle_2to1_host(...)` (host).
- Add `op_morton(arena)`:
  - Define a coordinate scheme (initially derive from index or a stored
    coordinate buffer).
  - Default scheme: index-derived `x=idx`, `y=0` (`PRISM_ARENA_COORD_SCHEME=index`).
  - Optional scheme: grid (`PRISM_ARENA_COORD_SCHEME=grid`,
    `PRISM_ARENA_COORD_GRID_LOG2=<k>`).
- Compose sort key:
  - `sort_key = (rank << high_bits) | morton`.
- Ensure stable ordering inside ranks when Morton keys collide.
- BSPˢ permutations must be bijections and must satisfy `perm[0]=0` and
  `inv_perm[0]=0`.

### 10) Hierarchical Arenas (Phase 2)
Objective: implement multi-level arenas to constrain shatter.

Tests (before implementation):
- Pytest: `test_block_local_sort` (expected: local ordering preserved).
- Pytest null: `test_single_block_same_as_global` (null: no diff for 1 block).
- Program: `tests/hierarchy_basic.txt` (expected: stable decode).
- Program null: `tests/hierarchy_single_block.txt` (expected: same as baseline).

Tasks:
- Partition arena into blocks (e.g., fixed block size).
- Run rank/sort locally per block, then merge.
- Add metadata arrays for block offsets and free ranges.
- This is a later milestone once fluid arena is correct.

### 11) Parser, REPL, and Telemetry
Objective: maintain usability and instrumentation.

Tests (before implementation):
- Pytest: `test_repl_mode_switch` (expected: mode flag selects BSP).
- Pytest null: `test_repl_default_is_baseline` (null: default unchanged).
- Program: `tests/repl_bsp.txt` (expected: BSP path exercised).
- Program null: `tests/repl_baseline.txt` (expected: current behavior).

Tasks:
- Add a mode switch in `repl()`:
  - `--mode=bsp` or environment flag.
- Update `parse` for BSP:
  - Linear allocation into arena, optional `(sort ...)`.
- Update telemetry:
  - Counts of HOT/WARM/COLD/FREE.
  - Per-cycle allocation delta.
- Provide debug hooks to compare results with the baseline VM.

## Milestones (Contract-Aligned)
- **m1: Ledger intrinsic core + deterministic keys**
  - Ledger interning is the reference path (`cycle_intrinsic + intern_nodes`).
  - Full-key equality and hard-cap univalence are enforced.
  - Milestone-gated tests are part of the deliverable (see `in/in-15.md`).
- **m2: Strata boundary + total `q` projection**
  - CNF-2 fixed-arity pipeline is enabled (slot1 enabled).
  - Evaluator emits strata; no within-tier references.
  - `commit_stratum`: validate → project `q` → intern → swizzle ids.
- **m3: Canonical rewrites + denotation invariance harness**
  - Canonical rewrites live in Ledger space (rewrite+intern).
  - Slot1 continuation is validated under the hyperstrata visibility rule.
  - Denotation invariance across unsorted, rank, and morton/block schedulers.
- **m4: Coordinates as interned objects + aggregation**
  - `OP_COORD_*` objects and idempotent parity normalization.
  - Coordinate aggregation applies to `OP_ADD` only.
- **m5: Full homomorphic collapse (production contract)**
  - Evaluator is write-model, Ledger is read-model.
  - Scheduling affects performance only; meaning is measured after `q`.
  - Autonomic servo (entropy probe + lung + masked Morton) replaces host-tuned paging.
- **m6: Hierarchical arenas (optional)**
  - Local block sort and merge once the contract is stable.

## Acceptance Gates
- **m1 gate:** univalence + no aliasing + baseline equivalence on small suite.
- **m2 gate:** strata validator passes + `q` projection total on emitted strata.
- **m3 gate:** denotation invariance across unsorted/rank/morton/block schedulers,
  plus pre-step immutability enforced as a hyperstrata visibility rule.
- **m4 gate:** coordinate normalization idempotence + parity cancellation + instrumentation is denotation-invariant.
- **m5 gate:** full-suite denotation invariance + univalence stress tests.

## Invariant Checklist (m1–m5)
m1:
- Key-byte univalence holds under hard-cap mode (`MAX_ID` checks + corrupt trap).
- Deterministic interning for identical inputs within a single engine.
- Baseline vs ledger equivalence on small add-zero suite.
- Guarded gathers and scatters reject out-of-range indices under test guards.
- Interning early-outs when `oom` or `corrupt` is set.
- Corrupt is sticky; stop paths leave `count` and key-prefix arrays unchanged.
m2:
- CNF-2 emission is fixed-arity (2 slots per site) with slot1 enabled.
- Compaction never interns disabled payloads.
- Stratum boundary enforces no within-tier refs; `q` projection is total.
m3:
- Denotation matches ledger intrinsic on the shared suite.
- Denotation invariance across unsorted/rank/morton/block schedulers.
- Pre-step immutability holds under CNF-2 cycles (hyperstrata visibility rule).
- Slot1 continuation respects hyperstrata visibility (reads pre-step + stratum0 only).
m4:
- Coordinate normalization is idempotent and confluent on small inputs.
- Coordinate lifting limited to `OP_ADD`.
- Damage/locality instrumentation does not change denotation when enabled.
m5:
- Arena scheduling (rank/sort/morton on/off) preserves denotation end-to-end.
- Servo mask/entropy control is performance-only and preserves denotation.

## Next Commit Checklist (m1 landing)
1. Add `tests/harness.py` with shared parse/run/normalize helpers.
2. Add tests `test_univalence_no_alias_guard`, `test_add_zero_equivalence_baseline_vs_ledger` (both operand orders), and `test_intern_deterministic_ids_single_engine`.
3. Implement key-width fix and hard-trap on `count > MAX_COUNT`.
4. Ensure `cycle_intrinsic` reduces the add-zero cases.
5. Only then introduce CNF-2 pipeline tests (m2).

## Testing Plan
### Milestone Test Selection
- CLI: use per-milestone configs (e.g., `pytest -c pytest.m2.ini`) or set
  `PRISM_MILESTONE=m2` to activate the milestone gate in `conftest.py`.
- VS Code: edit `.vscode/pytest.env` to set `PRISM_MILESTONE=m2`, then refresh
  the Testing panel; gating reads the env (or `.pytest-milestone`) in
  `conftest.py` and uses `pytest.ini`.
- Gating: tests are marked `m1`..`m6`; the milestone gate skips any test with
  a higher marker than the selected milestone.
- See `in/in-15.md` for the milestone-gated testing workflow and VS Code
  integration details.
Unit tests (ordered by dependency):
- Root remap correctness after any permutation/sort.
- Strata discipline: newly created nodes only reference prior strata.
- Lossless univalence/dedup (interning returns canonical ids).
- Key-width aliasing guard (full-key equality, no truncation merges).
- Gather/scatter guards reject out-of-range indices (test-guard mode).
- No-copy sharing (xfail until lambda encoding exists).
- Alpha-equivalence collapse (xfail until lambda encoding exists).
- Coordinate parity normalization (pointer equality for canonical coordinates).
- CNF-2 fixed arity: two candidate slots per rewrite site.
- CNF-2 slot layout indices are stable (slot0/slot1 mapping).
- 2:1 denotation invariance once swizzle is mandatory.

Integration tests:
- Compare baseline `PrismVM` eval vs `PrismVM_BSP` after N cycles for:
  - `(add (suc zero) (suc zero))`
  - `(add zero (suc (suc zero)))`
- Shared semantics harness (`tests/harness.py`) to parse/run/normalize across
  engines and compare decoded normal forms.
- Cross-engine equivalence harness (shared suite):
  - Baseline `PrismVM`
  - Ledger `cycle_intrinsic`
  - Ledger CNF-2 pipeline
  - (m4+) Arena-scheduled pipeline
  - Assert decoded normal forms match.
  - Optionally assert canonical ids match within each engine.

Performance checks:
- Cycle time (rank/sort/swizzle/interact).
- Allocation pressure vs capacity.
- Damage/locality metrics (when enabled).

## Risks and Mitigations
- **JAX scatter costs**: minimize scatter writes, batch with prefix sums.
- **Full-array ledger scans**: fixed-shape interning touches `LEDGER_CAPACITY`-sized
  buffers even when `count` is small (intentional for JIT stability in m1).
  m4 mitigations: introduce a smaller `MAX_CAPACITY` for production runs,
  maintain per-op counts incrementally to avoid full `_bincount_256`, and/or
  stage prefix-only scans via dynamic slice + pad.
- **Pointer invalidation**: treat root pointer as part of state, remap every
  sort, and verify in tests.
- **Capacity overflow**: add a guard that halts or triggers a sort/compaction.
- **Key aliasing**: avoid truncation in key encoding or assert limits at intern.
- **Interner rebuild cost**: avoid full-table merges per tiny batch; use staging
  buffers and periodic read-model rebuilds after m3.
- **JIT recompilations**: keep shapes static (fixed LEDGER_CAPACITY).

## Deferred Implementation Notes (Code Annotations)
This section mirrors deferred notes/TODOs in `prism_vm.py` for double-entry
tracking against the roadmap.

- JAX op dtype normalization (int32) is assumed; tighten if drift appears.
- `_scatter_drop` uses sentinel drop semantics; add a strict variant when ready.
- `safe_gather_1d` runs raw gathers outside test mode; add deterministic clamp
  or strictness when performance allows.
- Add an explicit zero-row (id=1) invariant guard.
- `init_ledger` relies on `_pack_key` being defined later; reorder helpers if
  init moves to import time.
- `validate_stratum_no_within_refs_jax` does a full-shape scan; host-slice
  validation is deferred to keep JIT shapes static.
- `_apply_stratum_q` assumes `canon_ids` length equals `stratum.count`; if
  batching changes, use `stratum.count` and add a guard.
- `_lookup_node_id` while-loop tuple unpacking (`pos, _`) is potentially
  confusing; clarify intent if refactored.
- Add CNF-2 observability counters for `rewrite_child`/`changed`/`wrap_emit`.
- CNF-2 candidate layout invariant: slot0 at `2*i`, slot1 at `2*i+1`; document
  and preserve in `cycle_candidates`.
- Consider updating the wrapper frontier directly in `rewrite_child` cases to
  avoid extra cycles when the child normalizes.
- Host-only coord helpers (e.g., `coord_xor`, leaf helpers) do per-scalar device
  reads; batch or cache if this becomes a perf cliff.
- `_coord_norm_id_jax` repeats lookups per step; batch coord normalization.
- Coord normalization uses `vmap` over a `cond`/loop; refactor to a single
  SIMD-style loop over the coord subset.
- Opcode buckets are precursors to per-op merges; global merge remains an m1
  tradeoff.
- Overflow checks depend on `requested_new`; add a secondary guard on `num_new`.
- Overflow is treated as CORRUPT in m1 (semantic id cap == capacity); split
  OOM handling if semantics decouple.
- `_merge_sorted_keys` is a global merge (m1 tradeoff); optimize per-op merges.
- Add a guard for `new_count` vs backing array length if `max_count` decouples.
- `intern_nodes` stop path performs a read-only lookup fallback (implemented).
- `_active_prefix_count` clamps to size; add an explicit overflow guard.
- Add value-bound guards for swizzled args in test mode.
- Remove the duplicate "JAX Kernels" section header.

## Prioritized Punch-List (Current State)
Ordered by semantic risk first, then determinism/observability, then performance.

**P0 — Semantic safety / correctness (complete)**
- No-silent-clipping guard in `intern_nodes` (secondary guard on `num_new`). ✅
- Explicit zero-row (id=1) invariant guard. ✅
- `_apply_stratum_q` length guard for `canon_ids` vs `stratum.count`. ✅
- Read-only lookup fallback for `intern_nodes` stop path (CORRUPT/OOM). ✅

**P1 — Determinism / observability**
- Strict scatter variant (no drop sentinel) for tests/guards.
- Deterministic gather behavior outside test mode (clamp or strict policy).
- CNF-2 observability counters (`rewrite_child`, `changed`, `wrap_emit`).

**P2 — Milestone integrity / test fidelity**
- Host-slice validation for `validate_stratum_no_within_refs`.
- Value-bound guards for swizzled args in test mode.

**P3 — Performance / scalability**
- Per-op merges in interner to avoid full-array merge per batch.
- Per-op counts to avoid full `_bincount_256` each pass.
- Prefix-only scans via dynamic slice + pad (avoid full `LEDGER_CAPACITY` sweep).
- Batch coord normalization (replace `vmap(cond)` with SIMD-style loop).
- Batch/cache host-only coord helpers to avoid per-scalar device reads.

**P4 — Hygiene / clarity**
- Clarify `_lookup_node_id` tuple unpacking.
- Overflow guard for `_active_prefix_count`.
- Remove duplicate "JAX Kernels" header (if not already resolved).

**P5 — Roadmap extensions**
- Min(Prism) harness + projection commutation (in-18).
- Agda proof roadmap execution (in-26).
- M6–M10 interaction-combinator engine (in-8 pivot).

## Deliverables
- `prism_vm.py`: new `PrismVM_BSP` and arena ops.
- `tests/`: new fixtures for cycle-based evaluation.
- `README.md`: documented modes and expected behavior.

## Roadmap Extension: in-8 Pivot (Tensor/Rule-Table Engine)
This is a later-stage pivot that replaces or augments the BSP pipeline with a
branchless interaction-combinator engine based on a NodeTypes/Ports/FreeStack
layout and rule-table rewrites.

Prereqs:
- Complete m1 through m5 (BSP cycle verified and stable).
- Decide whether this is a separate backend or a full replacement.
- Settle port encoding (3 ports vs 4-slot encoding) and invariants.

Planned steps (tests-first, pytest + program fixtures):
- M6: Data model for NodeTypes/Ports/FreeStack and port encoding invariants.
- M7: Active-pair matching + stream compaction (null: no active pairs).
- M8: Rule table and wiring templates (annihilation/commutation/erasure).
- M9: Allocation via prefix sums + FreeStack reuse/overflow guards.
- M10: Match/alloc/rewire/commit kernel pipeline and end-to-end reductions.
