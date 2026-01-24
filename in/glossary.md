# Commuting Acronyms & Polysemous Terms (Normative)

## 0. Rule of Polysemy

Polysemy is permitted only when:

1. the meanings lie on orthogonal axes, and
2. the interaction is declared to commute, and
3. there is a test obligation for the commutation.

If any of (1-3) are absent, the reuse is invalid.

Goal: engineer convergence, not avoid ambiguity.

### Test Obligations (meta)

- see term-specific test obligations in sections 1-15

---

## 1. BSP — Bulk Synchronous Parallel / Binary Space Partitioning

### BSPᵗ — Bulk Synchronous Parallel (Temporal)

**Axis:** execution time / synchronization

* supersteps + barrier
* strata commits are BSPᵗ barriers

### BSPˢ — Binary Space Partitioning (Spatial)

**Axis:** memory/layout/locality

* Morton / 2:1 swizzle
* blocked / hierarchical arenas

### Desired Commutation (Denotation Invariance)

Let `q` be the quotient/projection into canonical Ledger IDs, and `denote` produce a normal form in Ledger space.

```
pretty(denote(q(BSPᵗ ∘ BSPˢ(P)))) = pretty(denote(q(BSPᵗ(P))))
```

### Why We Want This

* BSPᵗ controls **when identity is created**
* BSPˢ controls **where provisional data lives**
* the quotient map `q` erases spatial accidentals
* BSPᵗ and BSPˢ operate entirely in the Arena (presheaf) layer; their effects must be erased by `q`

### Failure Modes

* locality changes which rewrite is visible
* layout affects key material (illegal)
* barrier placement changes identity creation (illegal unless reflected only in provisional space erased by `q`)

### Normative Rule

BSPᵗ and BSPˢ may be composed in any order **iff** the equality above holds for the denotation harness.

### Test Obligations

- (m3) `tests/test_arena_denotation_invariance.py::test_arena_denotation_invariance_random_suite`
- (m3) `tests/test_morton.py::test_morton_key_stable`
- (m3) `tests/test_morton.py::test_morton_disabled_matches_rank_sort`
- (m6) `tests/test_hierarchy.py::test_block_local_sort`
- (m6) `tests/test_hierarchy.py::test_single_block_same_as_global`

---

## 2. CD — Cayley–Dickson / Coordinate Decomposition

### CDₐ — Cayley–Dickson (Algebra)

**Axis:** algebra/semantics

* parity/cancellation laws
* growth-by-structure (not scalar width)

### CDᵣ — Coordinate Decomposition (Representation)

**Axis:** graph encoding

* `OP_COORD_ZERO|ONE|PAIR`
* interned coordinate DAGs

### Desired Commutation (Coord Canonicality)

Let `coord_norm` be the coordinate normalization procedure that must run **before key encoding** for any coordinate-carrying node. Let `coord_key` be the fully normalized coordinate’s canonical Ledger key or ID.

```
coord_key(CDₐ(CDᵣ(x))) = coord_key(CDᵣ(CDₐ(x)))
```

### Why We Want This

* algebraic meaning must not depend on tree shape
* representation must not encode “accidental” algebra

### Failure Modes

* algebraically equal coordinates intern differently
* partial coord normalization leaks into key packing
* tree shape changes cancellation outcome

### Normative Rule

CDₐ and CDᵣ commute **iff** coordinate normalization is:

* idempotent, and
* confluent on the staged scope, and
* applied before packing any parent key that depends on a coordinate.

### Test Obligations

- (m4) `tests/test_coord_ops.py::test_coord_opcodes_exist`
- (m4) `tests/test_coord_ops.py::test_coord_pointer_equality`
- (m4) `tests/test_coord_ops.py::test_coord_xor_parity_cancel`
- (m4) `tests/test_coord_ops.py::test_coord_norm_idempotent`
- (m4) `tests/test_coord_ops.py::test_coord_norm_confluent_small`
- (m4) `tests/test_coord_ops.py::test_coord_norm_commutes_with_xor`
- (m4) `tests/test_coord_norm_probe.py::test_coord_norm_probe_only_runs_for_pairs`
- (m4) `tests/test_coord_norm_probe.py::test_coord_norm_probe_skips_non_coord_batch`
- (m4) `tests/test_coord_batch.py::test_coord_xor_batch_uses_single_intern_call`
- (m4) `tests/test_coord_batch.py::test_coord_norm_batch_matches_host`
- (m4) `tests/test_coord_fixtures.py::test_coord_basic_fixture`
- (m4) `tests/test_coord_fixtures.py::test_coord_noop_fixture`

---

# Audit of Other Risky Terms (with Desired Commutation)

Below are the other terms that historically drift — not because commutation is bad, but because it was implicit. We make it explicit.

---

## 3. Canonical / Canonicalization

### Meanings (must be qualified unless it is Canonicalᵢ)

* **Canonicalᵢ**: canonical identity via Ledger interning (full-key equality)
* **Canonicalʳ**: reduced form (rewrite result) — *not* “canonicalization”
* **Canonicalₑ**: equivalence-class representative (future union-find)

### Axes

* Identity vs Rewrite vs Equivalence

### Desired Commutation

```
pretty(denote(q(rewrite(propose(x))))) = pretty(denote(q(propose(rewrite(x)))))
```

### Failure Mode

* “Canonicalization” used to mean eager simplification
* identity created during rewrite
* rewrite order affects IDs

### Normative Rule

> **Canonicalization = interning by full key-byte equality.**
> Rewrite *proposes*; canonicalization *decides*.
> Canonicalization is the enforcement of global coherence (sheaf condition).

(Any other meaning must be explicitly qualified.)

### Test Obligations

- (m1) `tests/test_ledger_intern.py::test_intern_nodes_dedup_batch`
- (m1) `tests/test_ledger_intern.py::test_intern_nodes_reuses_existing`
- (m1) `tests/test_ledger_intern.py::test_intern_nodes_order_invariant`
- (m1) `tests/test_m1_gate.py::test_intern_deterministic_ids_single_engine`
- (m1) `tests/test_m1_gate.py::test_ledger_full_key_equality`
- (m1) `tests/test_m1_gate.py::test_key_width_no_alias`
- (m1) `tests/test_m1_gate.py::test_key_width_no_alias_under_canonicalize`
- (ungated) `tests/test_invariants.py::test_add_commutative_interning`
- (ungated) `tests/test_invariants.py::test_mul_commutative_interning`
- (ungated) `tests/test_invariants.py::test_add_commutative_baseline_cons`
- (ungated) `tests/test_invariants.py::test_mul_commutative_baseline_cons`

---

## 4. Collapse

### Meanings (must be qualified)

* **Collapseʰ**: homomorphic collapse via `q` (projection into canonical IDs)
* **Collapseᵍ**: graph reduction/elimination (a rewrite strategy)
* **Collapseˡ**: logical identification (proof collapse)

### Axes

* Semantics vs Structure vs Proof

### Desired Commutation (Homomorphic Projection)

Let `eval` be evaluator steps in provisional space and `evalₗ` Ledger-space evaluation:

```
pretty(denote(q(eval(P)))) = pretty(denote(evalₗ(q(P))))
```

### Failure Mode

* “Collapse” interpreted as erasing structure
* context clues lost (your original warning)
* isomorphism assumed where only homomorphism exists

### Normative Rule

> Collapse means **sheafification / gluing of presheaf-local structure**, not erasure of structure unless explicitly stated as a rewrite.

### Test Obligations

- (m2) `tests/test_commit_stratum.py::test_commit_stratum_identity`
- (m2) `tests/test_commit_stratum.py::test_commit_stratum_applies_prior_q_to_children`
- (m2) `tests/test_strata.py::test_stratum_no_within_refs_passes`
- (m2) `tests/test_strata.py::test_stratum_no_within_refs_detects_self_ref`
- (m2) `tests/test_strata_random_programs.py::test_strata_validator_random_programs`
- (ungated) `tests/test_invariants.py::test_validate_stratum_no_within_refs_jax_ok`
- (ungated) `tests/test_invariants.py::test_validate_stratum_no_within_refs_jax_bad`

---

## 5. Normalize / Normal Form

### Meanings (must be qualified)

* **Normalizeᵣ**: rewrite normalization (reduction by rules)
* **Normalizeᵢ**: interning normalization (identity by full key)
* **Normalize𝚌**: coordinate normalization (parity/XOR etc.)

### Axes

* Rewrite vs Identity vs Algebra

### Normative Order Constraint (Key Safety)

* identity-relevant normalization must occur **before key encoding**
* rewrite normalization may be staged, but denotation comparisons are performed in Ledger space after `q`

### Commutation Constraints

```
Normalizeᵢ ∘ Normalizeᵢ = Normalizeᵢ
Normalizeᵢ ∘ Normalize𝚌 = Normalizeᵢ    (when coords are fully normalized pre-pack)
```

### Failure Mode

* normalization order affects identity
* partial normalization encoded in keys
* “normal form” conflated across layers

### Test Obligations

- (m1) `tests/test_m1_gate.py::test_add_zero_equivalence_baseline_vs_ledger`
- (m1) `tests/test_bsp_equiv.py::test_bsp_matches_baseline_add_zero`
- (m1) `tests/test_small_arith.py::test_small_add_mul_baseline_vs_bsp`
- (m2) `tests/test_candidate_cycle.py::test_cycle_candidates_add_zero`
- (m2) `tests/test_candidate_cycle.py::test_cycle_candidates_add_suc`
- (m2) `tests/test_candidate_cycle.py::test_cycle_candidates_mul_zero`
- (ungated) `tests/test_invariants.py::test_optimize_ptr_zero_rules`

---

## 6. Aggregate

### Meanings (must be qualified)

* **Aggregateᵣ**: fold/reduce (computation)
* **Aggregate𝚌**: coordinate-space combination (semantic)
* **Aggregateₚ**: performance batching (no semantic effect)

### Axes

* Computation vs Algebra vs Performance

### Desired Commutation

```
canon( Aggregate𝚌 (x) ) = Aggregate𝚌 ( canon(x) )
```

### Failure Mode

* aggregation treated as arithmetic when it’s semantic
* batching affects identity
* partial aggregates interned prematurely

### Normative Rule

> Semantic aggregation (Aggregate𝚌) must be applied in the canonicalizer path (before key encoding / during interning rules), not as an external batching trick.

### Test Obligations

- (m4) `tests/test_coord_ops.py::test_coord_xor_parity_cancel`
- (m4) `tests/test_coord_ops.py::test_coord_pair_dedup`
- (m4) `tests/test_coord_batch.py::test_coord_xor_batch_uses_single_intern_call`

---

## 7. Scheduler / Ordering

### Meanings (must be qualified)

* **Schedulerₜ**: temporal evaluation order (which sites fire when)
* **Schedulerₛ**: spatial layout/permutation (where nodes sit)

### Axes

* Time vs Space

### Desired Commutation

```
pretty(denote(q(scheduleₜ ∘ scheduleₛ(P))))
=
pretty(denote(q(scheduleₜ(P))))
```

### Failure Mode

* order of evaluation affects identity
* locality changes rewrite visibility
* barrier placement changes results

### Normative Rule

> Scheduling is free **iff** denotation after `q` is invariant.

### Test Obligations

- (m3) `tests/test_arena_denotation_invariance.py::test_arena_denotation_invariance_random_suite`
- (m3) `tests/test_sort_swizzle.py::test_swizzle_preserves_edges`
- (m3) `tests/test_sort_swizzle.py::test_swizzle_null_pointer_stays_zero`
- (m3) `tests/test_sort_swizzle.py::test_sort_swizzle_root_remap`
- (m3) `tests/test_cycle.py::test_cycle_root_remap`
- (m3) `tests/test_cycle.py::test_cycle_without_sort_keeps_root`
- (m3) `tests/test_morton.py::test_morton_key_stable`
- (m3) `tests/test_morton.py::test_morton_disabled_matches_rank_sort`
- (m6) `tests/test_hierarchy.py::test_block_local_sort`
- (m6) `tests/test_hierarchy.py::test_single_block_same_as_global`

---

## 8. Identity / Pointer

### Meanings (must be qualified)

* **Pointerₑ**: evaluator-local address (Manifest/Arena)
* **IDₗ**: Ledger canonical ID (semantic identity)
* **IDₑ**: equivalence representative (future)

### Axes

* Implementation vs Semantics vs Proof

### Desired Commutation

```
q(pointerₑ) = IDₗ
```

### Failure Mode

* pointer equality used as semantic equality
* cross-engine pointer comparison
* implicit assumptions about stability

### Normative Rule

> Only `IDₗ` is semantic identity. All other identities are provisional and must be compared only after `q`.

### Test Obligations

- (m2) `tests/test_commit_stratum.py::test_commit_stratum_identity`
- (m2) `tests/test_commit_stratum.py::test_commit_stratum_applies_prior_q_to_children`
- (m2) `tests/test_strata.py::test_stratum_no_within_refs_passes`
- (m2) `tests/test_strata.py::test_stratum_no_within_refs_detects_self_ref`

---

## 9. HLO (XLA IR)

### Meanings in Play

* **HLO**: XLA High-Level Optimizer IR emitted after lowering JAXPR
* **HLO size**: compile-time graph size and complexity

### Axes

* Compilation-time cost vs Runtime work

### Desired Commutation

```
pretty(denote(q(P))) = pretty(denote(q(compile(P))))
```

(Correctness must not depend on the compiler, even when the compile graph is huge.)

### Failure Mode

* `vmap` + `while_loop` + search lowers to a massive HLO even when most lanes are no-ops
* runtime guards (`cond`) skip work but do not shrink compile-time HLO
* host recursion that calls jitted interning causes many tiny compilations

### Normative Rule

> If a function contains `vmap + while_loop + lookup`, apply it only to the smallest possible subset (gather -> normalize -> scatter). Keep host recursion as a slow reference path, and provide a batched/jitted path for hot use.

### Test Obligations

- (m4) `tests/test_coord_norm_probe.py::test_coord_norm_probe_only_runs_for_pairs`
- (m4) `tests/test_coord_norm_probe.py::test_coord_norm_probe_skips_non_coord_batch`
- (m4) `tests/test_coord_batch.py::test_coord_xor_batch_uses_single_intern_call`
- (m4) `tests/test_coord_batch.py::test_coord_norm_batch_matches_host`

---

## 10. Garbage Collection / Interning (Semantic Compression)

### Meanings (must be qualified)

* **GCᵣ**: resource reclamation
* **GCᵢ**: semantic compression via interning (dedup)

### Axes

* Resource management vs Semantic identity

### Desired Commutation

```
pretty(denote(q(rebuild_from_roots(L)))) = pretty(denote(q(L)))
```

### Failure Mode

* canonical IDs are reclaimed or reassigned
* “GC” used to mask semantic aliasing
* rebuild changes denotation

### Normative Rule

> Interning is semantic compression; optional rebuilds are allowed only as renormalization that preserves denotation.

### Test Obligations

- (m1) `tests/test_ledger_intern.py::test_intern_nodes_dedup_batch`
- (m1) `tests/test_ledger_intern.py::test_intern_nodes_reuses_existing`
- (m1) `tests/test_m1_gate.py::test_intern_deterministic_ids_single_engine`
- (m2) `tests/test_candidate_intern.py::test_intern_candidates_dedup`

---

## 11. Damage / Locality

### Meanings in Play

* **Damageₛ**: spatial boundary crossing (tile/halo escalation)
* **Damageₑ**: semantic rewrite impact

### Axes

* Locality vs Meaning

### Desired Commutation

```
pretty(denote(q(damage_escalate ∘ local_step(P)))) = pretty(denote(q(local_step(P))))
```

### Failure Mode

* damage sets influence identity creation
* locality changes which rewrites fire

### Normative Rule

> Damage/locality is a performance signal only. It must not affect denotation and must be erasable by `q`.

### Test Obligations

- (m3) `tests/test_arena_denotation_invariance.py::test_arena_denotation_invariance_random_suite`
- (m3) `tests/test_morton.py::test_morton_key_stable`

---

## 12. Renormalization / Sorting

### Meanings in Play

* **Renormˢ**: layout reorder (sort/swizzle)
* **Normalizeᵣ**: semantic reduction (already defined above)

### Axes

* Layout vs Semantics

### Desired Commutation

```
pretty(denote(q(renorm(P)))) = pretty(denote(q(P)))
```

### Failure Mode

* sorting changes keys or rewrite outcomes
* root pointer/remap errors leak into meaning

### Normative Rule

> Sorting/swizzling are renormalization passes only; preserve edges and NULL, and validate invariance after `q`.

### Test Obligations

- (m3) `tests/test_sort_swizzle.py::test_swizzle_preserves_edges`
- (m3) `tests/test_sort_swizzle.py::test_swizzle_null_pointer_stays_zero`
- (m3) `tests/test_sort_swizzle.py::test_sort_swizzle_root_remap`
- (m3) `tests/test_cycle.py::test_cycle_root_remap`
- (m3) `tests/test_cycle.py::test_cycle_without_sort_keeps_root`

---

## 13. OOM / CORRUPT

### Meanings in Play

* **OOM**: resource exhaustion (capacity)
* **CORRUPT**: semantic undefinedness (alias risk)

### Axes

* Resource limits vs Semantic validity

### Desired Commutation

```
denote(q(P)) is undefined iff CORRUPT
```

### Failure Mode

* key-width overflow treated as OOM
* execution proceeds after alias risk

### Normative Rule

> CORRUPT is a hard semantic error; OOM is an admissible resource boundary.

### Test Obligations

- (m1) `tests/test_m1_gate.py::test_intern_corrupt_flag_trips`
- (m1) `tests/test_m1_gate.py::test_intern_corrupt_flag_trips_on_a1_overflow`
- (m1) `tests/test_m1_gate.py::test_intern_corrupt_flag_trips_on_a2_overflow`
- (m1) `tests/test_m1_gate.py::test_intern_corrupt_flag_trips_on_negative_child_id`
- (m1) `tests/test_m1_gate.py::test_intern_corrupt_flag_trips_on_opcode_out_of_range`
- (m1) `tests/test_m1_gate.py::test_corrupt_is_sticky_and_non_mutating`
- (m1) `tests/test_m1_gate.py::test_intern_raises_on_corrupt_host`
- (ungated) `tests/test_invariants.py::test_ledger_capacity_guard`
- (ungated) `tests/test_invariants.py::test_intern_nodes_early_out_on_oom_returns_zero_ids`
- (ungated) `tests/test_invariants.py::test_intern_nodes_early_out_on_corrupt_returns_zero_ids`
- (ungated) `tests/test_invariants.py::test_kernel_add_oom`
- (ungated) `tests/test_invariants.py::test_kernel_mul_oom`
- (ungated) `tests/test_invariants.py::test_op_interact_oom`

---

## 14. Duplication / Sharing (No-copy)

### Meanings in Play

* **Copy**: allocate new structure
* **Share**: reuse canonical identity in multiple contexts

### Axes

* Operational steps vs Semantic identity

### Desired Commutation

```
use(x, x) should not allocate a duplicate of x
```

### Failure Mode

* primitive copy creates new nodes for existing structure
* superlinear growth from repeated use

### Normative Rule

> Duplication is expressed by sharing canonical IDs; no-copy is an operational axiom.

### Test Obligations

- (m1) `tests/test_ledger_intern.py::test_intern_nodes_dedup_batch`
- (m1) `tests/test_ledger_intern.py::test_intern_nodes_reuses_existing`
- (m2) `tests/test_candidate_intern.py::test_intern_candidates_dedup`

---

## 15. Binding / Names (Alpha-Equivalence)

### Meanings in Play

* **Nominal**: names and lookup
* **Structural**: wiring or coordinates

### Axes

* Names vs Structure

### Desired Commutation

```
compile(λx. x) == compile(λy. y)
```

### Failure Mode

* names leak into keys or identity
* alpha-equivalent terms intern differently

### Normative Rule

> Binding is structural; alpha-equivalence must collapse before interning.

### Test Obligations

- (planned) no pytest coverage yet

---

## 16. Arena / Frontier (Presheaf Semantics)

### Meanings (must be qualified)

* **Arenaₚ**: presheaf of staged constructions (frontier)
* **Arenaₘ**: manifest / device representation of the frontier

### Axes

* Staging vs Meaning
* Locality vs Coherence

### Normative Interpretation

> The Arena is the frontier.
> It is a GF(2)-valued presheaf of local constructions prior to semantic collapse.

Arena contents:

* may duplicate
* may overlap
* may depend on order or locality
* have no semantic identity until projected by `q`

### Erasure by `q`

```
q(Arenaₚ) = Ledger
```

All Arena-only distinctions (order, multiplicity, hyperstrata, locality) are erased.

### Failure Mode

* Arena artifacts influencing canonical IDs
* local ordering affecting meaning

### Normative Rule

> Arena semantics must be presheaf-local and must not survive sheafification.

### Test Obligations

- (m3) `tests/test_arena_denotation_invariance.py::test_arena_denotation_invariance_random_suite`
- (m2) `tests/test_candidate_cycle.py::test_cycle_candidates_does_not_mutate_preexisting_rows`

---

## 17. `q` — Sheafification / Gluing

### Meanings (must be qualified)

* **qₕ**: homomorphic projection (existing usage)
* **qₛ**: associated sheaf functor (topos semantics)

These are the same operation, viewed on different axes.

### Axes

* Construction vs Meaning
* Local vs Global

### Normative Interpretation

> `q` is the associated sheaf functor:
> it glues presheaf-local Arena data into globally coherent Ledger meaning.

Properties:

* idempotent
* total
* order-erasing
* GF(2)-cancellative
* structure-preserving

### Desired Commutation

```
q ∘ Arena_step = Ledger_step ∘ q
```

### Failure Mode

* partial projection
* non-idempotent collapse
* ordering-sensitive results

### Normative Rule

> `q` is the only meaning-forming operation in Prism.

### Test Obligations

- (m2) `tests/test_commit_stratum.py::test_commit_stratum_identity`
- (m2) `tests/test_commit_stratum.py::test_commit_stratum_applies_prior_q_to_children`
- (m2) `tests/test_commit_stratum.py::test_commit_stratum_q_map_totality_on_mixed_ids`

---

## 18. Ledger (Sheaf Object / Manifold)

### Meanings (must be qualified)

* **Ledgerₛ**: distinguished sheaf of canonical structure
* **Ledgerᵣ**: concrete interning table (implementation)

### Axes

* Semantics vs Representation

### Normative Interpretation

> The Ledger is a sheaf, not a log.
> Its contents are globally coherent semantic objects.

Canonical IDs are global elements of the Ledger sheaf.

### Erasure by `q`

Nothing: the Ledger is post-erasure.

### Failure Mode

* Ledger IDs encoding staging or locality
* canonical IDs depending on Arena history

### Normative Rule

> Only Ledger IDs carry semantic meaning.

### Test Obligations

- (m1) `tests/test_m1_gate.py::test_ledger_full_key_equality`
- (m1) `tests/test_m1_gate.py::test_intern_deterministic_ids_single_engine`

---

## 19. Boolean Logic (GF(2) Semantics)

### Meanings (must be qualified)

* **Booleanₗ**: internal logic of the Prism topos
* **GF(2)**: algebraic carrier of semantics

### Axes

* Logic vs Computation

### Normative Interpretation

> Prism's internal logic is classical (Boolean), even if its construction is staged or partial.

Implications:

* Law of Excluded Middle holds after `q`
* duplication annihilates (`x ⊕ x = 0`)
* no semantic accumulation by magnitude

### Desired Commutation

```
q(x ⊕ x) = 0
```

### Failure Mode

* weighted semantics
* order-dependent truth

### Normative Rule

> All semantic meaning is GF(2)-stable.

### Test Obligations

- (m1) `tests/test_m1_gate.py::test_intern_deterministic_ids_single_engine`
- (m4) `tests/test_coord_ops.py::test_coord_xor_parity_cancel`

---

## 20. Hyperpair / Cayley-Dickson Step

### Meanings (must be qualified)

* **CD-step**: Cayley-Dickson doubling
* **Hyperpair**: structural pairing at the semantic level

These are the same operator.

### Axes

* Dimension vs Structure

### Normative Interpretation

> The hyperpair is the Cayley-Dickson step.
> There is exactly one pairing operator.

Higher-dimensional values arise by structural recursion, not new semantics.

### Desired Commutation

```
op(CD(x₁,x₂), CD(y₁,y₂)) = CD(op(x₁,y₁), op(x₂,y₂))
```

(for dimension-preserving ops)

### Failure Mode

* introducing parallel "pair" semantics
* non-structural dimensional growth

### Normative Rule

> All higher-order behavior is structural depth in Σ.

### Test Obligations

- (m4+) planned, not yet in pytest

---

## 21. Hyperstrata (Staging, Not Semantics)

### Meanings (must be qualified)

* **Hyperstrataₚ**: staging indices in the Arena
* **Hyperstrataₛ**: invalid (must not exist)

### Axes

* Time vs Meaning

### Normative Interpretation

> Hyperstrata refine construction order, not semantic identity.

They:

* live only in the Arena
* are erased by `q`
* enforce immutability and causality

### Erasure by `q`

```
q((s,t)-staged data) = semantic value
```

### Failure Mode

* hyperstrata leaking into keys or IDs

### Normative Rule

> Hyperstrata are presheaf-local only.
> Pre-step ledger segment `[0, start_count)` is read-only during a cycle; interning is append-only relative to this base.

### Test Obligations

- (m2) `tests/test_candidate_cycle.py::test_cycle_candidates_does_not_mutate_preexisting_rows`
- (m2) `tests/test_candidate_cycle.py::test_cycle_candidates_validate_stratum_trips_on_within_refs`
- (m1) `tests/test_ledger_intern.py::test_intern_nodes_never_mutates_pre_step_segment`

---

## 22. Hyperlattice (Semantic Structure)

### Meanings (must be qualified)

* **Hyperlatticeᵢ**: internal lattice of subobjects (Boolean)
* **Hyperlatticeₑ**: refinement order on canonical IDs

### Axes

* Logic vs Structure

### Normative Interpretation

> Canonical IDs form a semantic hyperlattice:
> a recursively generated, GF(2)-stable lattice induced by CNF-2 and CD structure.

Joins:

* Arena accumulation -> `q`

Meets:

* shared canonical substructure

### Desired Commutation

```
q(joinₚ(x,y)) = joinₛ(q(x), q(y))
```

### Failure Mode

* non-idempotent joins
* semantic dependence on multiplicity

### Normative Rule

> All semantic composition is lattice-stable after `q`.

### Test Obligations

- (m2-m3) projection commutation and denotation invariance
- (planned) no pytest coverage yet

---

# Meta-Rule: How to Use This Going Forward

Whenever a term or acronym is reused:

1. name the axes
2. state the commutation equation
3. state what is erased by `q` (if anything)
4. attach a test obligation marker (`m1..m6`)

If any of those cannot be stated clearly, the reuse is invalid.

---

## Optional Next Steps

* Add axis tags (`ᵗ`, `ˢ`, `ₐ`, `ᵣ`) in code comments where ambiguity matters
* Add glossary references in tests (“this test enforces BSPᵗ/BSPˢ commutation”)
* Add a short “forbidden reinterpretations” appendix listing known past drift cases
