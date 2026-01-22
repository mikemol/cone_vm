# Prism VM Evolution Implementation Plan

This plan implements the features described in `in/in-4.md` through
`in/in-7.md`, plus the CNF-2 and canonical interning semantics in
`in/in-9.md` through `in/in-14.md`, the milestone-gated testing workflow in
`in/in-15.md`, and the homomorphic collapse contract in `in/in-16.md`, as a
concrete evolution of the current `prism_vm.py` implementation.

Each `in/in-*.md` note now includes a short NOTE header indicating whether it
has been refined, consolidated, or obsoleted. See `audit_in_versions.md` for
the current cross-version audit.

## Goals
- Establish the Ledger + candidate pipeline as the canonical execution path.
- Converge on CNF-2 symmetric rewrite semantics with a fixed-arity candidate
  pipeline (slot 1 disabled until continuation/strata support).
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

## Baseline Summary (Current Code)
`prism_vm.py` uses a stable heap (`Manifest`) with:
- `opcode`, `arg1`, `arg2`, `active_count` arrays.
- Hash-consing via `trace_cache`.
- Static optimization and branchless dispatch (`optimize_ptr`,
  `dispatch_kernel`).
- `kernel_add` and a stub `kernel_mul`.

## Architecture Decision
Ledger + candidate pipeline is the canonical execution path. Baseline
`PrismVM` stays as an oracle. Arena scheduling is performance-only and must be
denotation-invariant with respect to the Ledger CNF-2 engine; it begins at M4+.
Ledger denotation is the spec; ID equality is engine-local until baseline
migrates to the Ledger interner.

## Semantic Commitments (Staged)
- CNF-2 symmetric rewrite semantics are the target for BSP. Every rewrite site
  emits exactly two candidate slots (enabled or disabled).
- Keep candidate slot 1 disabled until continuation/strata support lands.
- Univalence is enforced by full-key equality (hash as hint only) and
  no truncation aliasing in key encoding.
- CD coordinates are interned CNF-2 objects (`OP_COORD_*`), and coordinate
  equality is pointer equality.
- Evaluator (Manifest/Arena) and Canonicalizer (Ledger) are linked by a total
  quotient map `q`; denotation is defined by projecting provisional nodes
  through `q` and must commute with scheduling (see `in/in-16.md`).
- 2:1 BSP swizzle/locality is staged as a performance invariant:
  - M1-M3: rank-only sort (or no sort) is acceptable for correctness tests.
  - M4+: 2:1 swizzle is mandatory for the performance profile, but must not
    change denotation (same canonical ids/decoded normal forms).
- Baseline `PrismVM` is a regression implementation through M3; comparisons are
  on decoded normal forms and termination behavior, not raw ids.

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
  - equals `id` in M1–M3
  - becomes `find(id)` if rewrite-union is introduced later
- Commutative ops (`ADD`, later others):
  - sort `(rep(a1), rep(a2))` before encoding
- `extra(op)` is empty unless explicitly defined (e.g. future flags)

Two nodes are equal iff their full `K` byte sequences are identical.

### Fixed-Width Encoding Mode (M1–M3)

In M1–M3, `K` uses a fixed-width encoding:

- `op` -> 1 byte
- `rep(a1)` -> 16 bits
- `rep(a2)` -> 16 bits
- total: 5 bytes

This encoding is injective only over a bounded id domain.

Therefore, the id bound is a semantic invariant, not a memory limit.

### Representational Cap (Hard Semantic Invariant)

Define:

- `MAX_NODES = 65535`
- `MAX_ID = 65534`
- `id = 0` reserved for `NULL`
- `id = 1` reserved for `ZERO`

Invariant:

All reachable ids used in key encoding must satisfy `id <= MAX_ID`.

Violating this invariant would cause key aliasing and destroy univalence.

Therefore:

- Exceeding `MAX_ID` is CORRUPT, not OOM.
- Continuing execution past this point is semantically undefined and forbidden.

### Deterministic Corruption Handling (JAX-Safe)

To maintain determinism and avoid undefined behavior in JIT-compiled code:

Device-side (checked packing / interning):

- If any of the following occur:
  - `count > MAX_ID`
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
- Univalence alignment: any engine-level compare or normalization must preserve
  the Univalence Contract (full key-byte equality, corrupt trap on overflow).
- Homomorphic projection: define `q(provisional_node) -> canonical_id` and
  compare denotations only after projection; evaluator steps must commute with
  `q` up to canonical rewrite (see `in/in-16.md`).

## Locked Decisions
- Read model backend (M1): keep the current ordered index (sorted lanes +
  binary search). Tests assert full-key equality and do not assume any specific
  index structure so trie or hash-bucket indexes can replace it later.
- Univalence encoding (M1): hard-cap mode with 5-byte key; enforce
  `MAX_NODES = 65535`, `MAX_ID = 65534`, and checked packing for `a1/a2`.
- Aggregation scope (M3): coordinate aggregation applies to `OP_ADD` only;
  `OP_MUL` remains Peano-style until explicitly lifted.

## Workstreams and Tasks (Tests-First Policy)
For each step below:
- Add **pytest** tests for expected behavior and a **null hypothesis** case.
- Add **program fixtures** (in `tests/` or a dedicated `programs/`) mirroring the
  pytest coverage for black-box validation.
- Commit tests first, then implement, then commit again.

Sections **0-3** are Ledger-only (M1-M3). Arena scheduling starts at **4** and
is performance-only (M4+).

### 0) Canonical Interner + Univalence (Ledger-first)
Objective: treat the Ledger as the canonical read model and enforce univalence.

Tests (before implementation):
- Pytest: `test_ledger_full_key_equality` (expected: hash collision does not merge distinct nodes).
- Pytest: `test_key_width_no_alias` (null: distinct child ids never alias in key encoding).
- Pytest: `test_intern_corrupt_flag_trips` (expected: corrupt flag is set and host raises).
- Program: `tests/ledger_univalence.txt` (expected: deterministic canonical ids).
- Program null: `tests/ledger_noop.txt` (expected: no changes).

Tasks:
- Ensure interning uses full-key equality; hash is a hint only.
- Add a canonicalize-before-pack hook for any derived representations.
- Guard against truncation aliasing (M1 uses hard-cap mode).
- Enforce `MAX_NODES = 65535`, define `MAX_ID = 65534`, reserve `0/1` for
  NULL/ZERO, and hard-trap on `count > MAX_ID` before any allocation or intern.
- Implement a deterministic JAX trap:
  - Maintain a `corrupt` flag (or `oom`) in the interner state.
  - In checked packing, set `corrupt` on bounds violations (`a1/a2 > MAX_ID`)
    and clamp unsafe values to sentinel zeros to keep computation defined.
  - Host must raise if `corrupt` is true after `block_until_ready()`.
- Treat index arrays as rebuildable read models (event log optional).
- Add a reference interner contract test suite that validates injectivity and
  full-key equality across interner backends.

### 1) CNF-2 Candidate Pipeline (Rewrite -> Compact -> Intern)
Objective: enforce fixed-arity rewrite semantics and explicit strata discipline.

Tests (before implementation):
- Pytest: `test_cnf2_fixed_arity` (expected: exactly 2 candidate slots per site).
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
- Emission invariant: buffer shape is `2 * |frontier|` every cycle.
- M2 invariant: `enabled[slot1] == 0` always; slot1 payloads are ignored.
- Implement compaction (enabled mask + prefix sums).
- Intern compacted candidates via `intern_nodes` (can be fused in M2).
- Optional debug: track origin site indices for strata violation tracing.

### 2) Strata Discipline (Ledger-only)
Objective: enforce explicit strata boundaries and prevent within-tier references.

Tests (before implementation):
- Pytest: `test_strata_no_within_tier_refs` (expected: new nodes only reference
  prior strata).
- Program: `tests/strata_basic.txt` (expected: same normal form as baseline).
- Program null: `tests/strata_noop.txt` (expected: no changes).

Tasks:
- Add a lightweight `Stratum` record (offset + count) for new ids.
- Strict strata (default): nodes in `Si` may only reference ids `< start(Si)`.
- Future mode: allow within-stratum references only if they are to earlier ids
  in the same stratum (topological order), gated by explicit opt-in.
- Add validators that enforce the selected rule, scanning only up to
  `ledger.count`.
- Wire validators into `cycle_candidates` (guarded by a debug flag).

### 3) Coordinate Semantics (Self-Hosted CD)
Objective: add CD coordinates as interned CNF-2 objects and define parity/aggregation.

Tests (before implementation):
- Pytest: `test_coord_opcodes_exist` (expected: `OP_COORD_*` registered).
- Pytest: `test_coord_pointer_equality` (expected: identical coordinates intern to same id).
- Pytest: `test_coord_xor_parity_cancel` (expected: parity cancellation is idempotent).
- Pytest: `test_coord_norm_idempotent` (expected: canonicalization is stable).
- Pytest: `test_coord_norm_confluent_small` (expected: small expressions converge).
- Program: `tests/coord_basic.txt` (expected: canonical ids for coordinates).
- Program null: `tests/coord_noop.txt` (expected: no rewrite for non-coord ops).

Tasks:
- Add `OP_COORD_ZERO`, `OP_COORD_ONE`, `OP_COORD_PAIR`.
- Representation invariant: coordinates are stored only as interned
  `OP_COORD_*` DAGs; no linear bitstrings/digit arrays are stored anywhere.
- Implement coordinate construction and XOR/parity rewrite rules in the BSP path.
- Aggregation scope (M3): coordinate lifting applies to `OP_ADD` only; `OP_MUL`
  remains Peano-style until explicitly lifted.
- Normalization order:
  1. Build coordinate CNF-2 objects.
  2. Normalize parity/XOR to a canonical coordinate object.
  3. Only then pack keys and intern any coordinate-carrying node.

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
- Staging note: rank-only sort is sufficient for M1-M3 correctness. 2:1
  swizzle is optional until M4.

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
- Compose sort key:
  - `sort_key = (rank << high_bits) | morton`.
- Ensure stable ordering inside ranks when Morton keys collide.

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

## Milestones
- **M1-M3 are Ledger-only.** Arena scheduling starts at M4+ and must be
  denotation-invariant with respect to the Ledger CNF-2 engine.
- **M1: Correctness spine (eager canonicalization)**
  - Ledger interning is the reference path (`cycle_intrinsic + intern_nodes`).
  - Baseline `PrismVM` remains strict oracle.
  - Tests: root remap, univalence/dedup, key-width guard.
  - Minimal E2E reductions:
    - `(add zero (suc zero)) -> (suc zero)`
    - `(add (suc zero) zero) -> (suc zero)`
- **M2: CNF-2 surface (candidate buffer)**
  - Two-slot candidate emission + compaction (intern can be fused).
  - Slot 1 remains disabled until continuation/strata support.
  - Tests: fixed arity, enabled-only compaction, baseline equivalence.
- **M3: Strata discipline**
  - M3a: explicit stratum boundary with validators (no within-tier refs).
  - M3a: canonicalize-before-pack hook in place before coord usage.
  - M3b: introduce `OP_COORD_*` and parity/normalization rules.
- **M4: Performance spine (Arena scheduling)**
  - Arena rank/sort/swizzle/morton ordering as an optimization layer.
  - Mandatory 2:1 swizzle/morton ordering.
  - Denotation-invariance tests across swizzle modes (normal forms, not ids).
- **M5: Hierarchical arenas (optional)**
  - Local block sort and merge.

## Acceptance Gates
- **M1 gate:** univalence + no aliasing + baseline equivalence on small suite.
- **M2 gate:** CNF-2 fixed-arity emission + compaction correctness + baseline equivalence.
- **M3 gate:** strata validator passes on randomized small programs + coord idempotence.
- **M4 gate:** arena/morton on/off does not change denotation on randomized suite.

## Invariant Checklist (M1–M4)
M1:
- Key-byte univalence holds under hard-cap mode (`MAX_ID` checks + corrupt trap).
- Deterministic interning for identical inputs within a single engine.
- Baseline vs ledger equivalence on small add-zero suite.
M2:
- CNF-2 emission is fixed-arity (2 slots per site) with slot1 disabled.
- Compaction never interns disabled payloads.
- Denotation matches ledger intrinsic on the shared suite.
M3:
- Strata validator enforces no within-tier refs (strict mode).
- Coordinate normalization is idempotent and confluent on small inputs.
- Coordinate lifting limited to `OP_ADD`.
M4:
- Arena scheduling (rank/sort/morton on/off) preserves denotation.

## Next Commit Checklist (M1 landing)
1. Add `tests/harness.py` with shared parse/run/normalize helpers.
2. Add tests `test_univalence_no_alias_guard`, `test_add_zero_equivalence_baseline_vs_ledger` (both operand orders), and `test_intern_deterministic_ids_single_engine`.
3. Implement key-width fix and hard-trap on `count > 65534`.
4. Ensure `cycle_intrinsic` reduces the add-zero cases.
5. Only then introduce CNF-2 pipeline tests (M2).

## Testing Plan
### Milestone Test Selection
- CLI: use per-milestone configs (e.g., `pytest -c pytest.m2.ini`) or set
  `PRISM_MILESTONE=m2` to activate the milestone gate in `conftest.py`.
- VS Code: edit `.vscode/pytest.env` to set `PRISM_MILESTONE=m2`, then refresh
  the Testing panel; gating reads the env (or `.pytest-milestone`) in
  `conftest.py` and uses `pytest.ini`.
- Gating: tests are marked `m1`..`m5`; the milestone gate skips any test with
  a higher marker than the selected milestone.
- See `in/in-15.md` for the milestone-gated testing workflow and VS Code
  integration details.
Unit tests (ordered by dependency):
- Root remap correctness after any permutation/sort.
- Strata discipline: newly created nodes only reference prior strata.
- Lossless univalence/dedup (interning returns canonical ids).
- Key-width aliasing guard (full-key equality, no truncation merges).
- Coordinate parity normalization (pointer equality for canonical coordinates).
- CNF-2 fixed arity: two candidate slots per rewrite site.
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
  - (M4+) Arena-scheduled pipeline
  - Assert decoded normal forms match.
  - Optionally assert canonical ids match within each engine.

Performance checks:
- Cycle time (rank/sort/swizzle/interact).
- Allocation pressure vs capacity.

## Risks and Mitigations
- **JAX scatter costs**: minimize scatter writes, batch with prefix sums.
- **Pointer invalidation**: treat root pointer as part of state, remap every
  sort, and verify in tests.
- **Capacity overflow**: add a guard that halts or triggers a sort/compaction.
- **Key aliasing**: avoid truncation in key encoding or assert limits at intern.
- **Interner rebuild cost**: avoid full-table merges per tiny batch; use staging
  buffers and periodic read-model rebuilds after M3.
- **JIT recompilations**: keep shapes static (fixed MAX_NODES).

## Deliverables
- `prism_vm.py`: new `PrismVM_BSP` and arena ops.
- `tests/`: new fixtures for cycle-based evaluation.
- `README.md`: documented modes and expected behavior.

## Roadmap Extension: in-8 Pivot (Tensor/Rule-Table Engine)
This is a later-stage pivot that replaces or augments the BSP pipeline with a
branchless interaction-combinator engine based on a NodeTypes/Ports/FreeStack
layout and rule-table rewrites.

Prereqs:
- Complete M1 through M5 (BSP cycle verified and stable).
- Decide whether this is a separate backend or a full replacement.
- Settle port encoding (3 ports vs 4-slot encoding) and invariants.

Planned steps (tests-first, pytest + program fixtures):
- M6: Data model for NodeTypes/Ports/FreeStack and port encoding invariants.
- M7: Active-pair matching + stream compaction (null: no active pairs).
- M8: Rule table and wiring templates (annihilation/commutation/erasure).
- M9: Allocation via prefix sums + FreeStack reuse/overflow guards.
- M10: Match/alloc/rewire/commit kernel pipeline and end-to-end reductions.
