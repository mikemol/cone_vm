# Prism VM Evolution Implementation Plan

This plan implements the features described in `in/in-4.md`, `in/in-5.md`,
`in/in-6.md`, and `in/in-7.md` as a concrete evolution of the current
`prism_vm.py` implementation.

## Goals
- Introduce a fluid arena with rank-based sorting and pointer swizzling.
- Add a 2-bit rank scheduler with stable locality guarantees.
- Implement a rank/sort/swizzle/interaction cycle.
- Add Morton or 2:1 BSP locality as a secondary ordering key (staged).
- Converge on CNF-2 symmetric rewrite semantics with a fixed-arity candidate
  pipeline and explicit strata discipline.
- Preserve a stable, testable baseline while the new mode is built out.

## Non-Goals (Initial MVP)
- Full hierarchical arena migration across levels (L1/L2/global).
- Perfect allocator for all graph rewrite rules beyond ADD.
- Full macro system or higher-order quotation semantics.
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
Add a **new execution mode** (`PrismVM_BSP`) while keeping the existing VM.
This protects the current behavior and makes cross-validation possible.

## Semantic Commitments (Staged)
- CNF-2 symmetric rewrite semantics are the target for BSP. Every rewrite site
  emits exactly two candidate slots (enabled or disabled).
- 2:1 BSP swizzle/locality is staged as a performance invariant:
  - M1-M3: rank-only sort (or no sort) is acceptable for correctness tests.
  - M4+: 2:1 swizzle is mandatory for the performance profile, but must not
    change denotation (same canonical ids/decoded normal forms).
- Baseline `PrismVM` remains the oracle for functional results through M3.

## Workstreams and Tasks (Tests-First Policy)
For each step below:
- Add **pytest** tests for expected behavior and a **null hypothesis** case.
- Add **program fixtures** (in `tests/` or a dedicated `programs/`) mirroring the
  pytest coverage for black-box validation.
- Commit tests first, then implement, then commit again.

### 1) Data Model: Arena vs Manifest
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
  - `count` (active node boundary)
- Add `init_arena()` that seeds `OP_ZERO` and reserves `OP_NULL = 0`.
- Define ranks:
  - `RANK_HOT = 0`, `RANK_WARM = 1`, `RANK_COLD = 2`, `RANK_FREE = 3`
- Add new opcodes:
  - `OP_SORT = 99` (optional explicit primitive)
  - keep existing `OP_ZERO`, `OP_SUC`, `OP_ADD`, `OP_MUL`.

### 2) Rank Classification (2-bit Scheduler)
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

### 3) Sort + Swizzle
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
  - Update `count` to number of non-FREE nodes.
- Use `jax.numpy.argsort` initially.
- Later: replace with 4-bin partition (prefix sums) for speed.
- Staging note: rank-only sort is sufficient for M1-M3 correctness. 2:1
  swizzle is optional until M4.

### 4) Interaction Kernel (Rewrite Rules)
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

### 5) Control Loop and Root Tracking
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

### 6) CNF-2 Candidate Pipeline (Rewrite -> Compact -> Intern)
Objective: enforce fixed-arity rewrite semantics and explicit strata discipline.

Tests (before implementation):
- Pytest: `test_cnf2_fixed_arity` (expected: exactly 2 candidate slots per site).
- Pytest: `test_candidate_compaction_enabled_only` (null: disabled slots dropped).
- Pytest: `test_strata_no_within_tier_refs` (expected: new nodes only reference
  prior strata).
- Program: `tests/cnf2_basic.txt` (expected: same normal form as baseline).

Tasks:
- Add a `CandidateBuffer` structure:
  - `enabled`, `opcode`, `arg1`, `arg2` arrays sized to `2 * |frontier|`.
- Implement CNF-2 rewrite emission (always two slots; predicates control enable).
- Implement compaction (enabled mask + prefix sums).
- Intern compacted candidates via `intern_nodes` (can be fused in M2).
- Add a lightweight `Stratum` record (offset + count) for new ids, and validators
  to enforce no within-stratum dependencies.

### 7) Morton / 2:1 BSP Locality
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

### 8) Hierarchical Arenas (Phase 2)
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

### 9) Parser, REPL, and Telemetry
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
- **M1: Correctness spine (eager canonicalization)**
  - `Arena`, `init_arena`, `op_rank` compile and run.
  - `cycle_intrinsic + intern_nodes` used as the BSP reference path.
  - Tests: root remap, univalence/dedup.
- **M2: CNF-2 surface (candidate buffer)**
  - Two-slot candidate emission + compaction (intern can be fused).
  - Tests: fixed arity, enabled-only compaction, baseline equivalence.
- **M3: Strata discipline**
  - Explicit stratum boundary with validators (no within-tier refs).
  - Baseline remains strict oracle.
- **M4: Performance semantics**
  - Mandatory 2:1 swizzle/morton ordering.
  - Denotation-invariance tests across swizzle modes.
- **M5: Hierarchical arenas (optional)**
  - Local block sort and merge.

## Testing Plan
Unit tests (ordered by dependency):
- Root remap correctness after any permutation/sort.
- Strata discipline: newly created nodes only reference prior strata.
- Lossless univalence/dedup (interning returns canonical ids).
- CNF-2 fixed arity: two candidate slots per rewrite site.
- 2:1 denotation invariance once swizzle is mandatory.

Integration tests:
- Compare baseline `PrismVM` eval vs `PrismVM_BSP` after N cycles for:
  - `(add (suc zero) (suc zero))`
  - `(add zero (suc (suc zero)))`

Performance checks:
- Cycle time (rank/sort/swizzle/interact).
- Allocation pressure vs capacity.

## Risks and Mitigations
- **JAX scatter costs**: minimize scatter writes, batch with prefix sums.
- **Pointer invalidation**: treat root pointer as part of state, remap every
  sort, and verify in tests.
- **Capacity overflow**: add a guard that halts or triggers a sort/compaction.
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
