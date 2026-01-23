Perfect — this is the right framing: **we want commutation and convergence**, but we must make them **explicit, scoped, and testable**, not accidental. Below is a **drop-in expansion** for the glossary plus an **audit of other risky terms**, each annotated with:

* **Axes** (what dimensions are being crossed)
* **Desired commutation** (what *should* commute)
* **Failure mode** (what drift looks like)
* **Normative rule** (how to keep it safe)

You can paste this directly after the glossary you already have.

---

# Commuting Acronyms & Polysemous Terms (Normative)

> **Principle:**
> Polysemy is permitted *only* when meanings lie on orthogonal axes **and** their interaction is explicitly declared to commute.
>
> **Goal:** not to avoid ambiguity, but to **engineer convergence**.

---

## 1. BSP — Bulk Synchronous Parallel / Binary Space Partitioning

### BSPᵗ — Bulk Synchronous Parallel

**Axis:** *Temporal / Execution*

* Supersteps with barriers
* Global synchronization points
* Strata commits are BSPᵗ barriers

### BSPˢ — Binary Space Partitioning

**Axis:** *Spatial / Memory / Layout*

* Partitioning nodes, memory, or indices
* Morton / 2:1 swizzle
* Blocked / hierarchical arenas

### Desired Commutation

```
denote( BSPᵗ ∘ BSPˢ (P) ) = denote( BSPᵗ (P) )
```

### Why We Want This

* BSPᵗ controls **when identity is created**
* BSPˢ controls **where provisional data lives**
* The quotient map `q` erases spatial accidentals

### Failure Mode

* Spatial locality affects which rewrite fires
* Memory order leaks into semantic keys
* Barrier placement changes identity creation

### Normative Rule

> BSPᵗ and BSPˢ may be composed in any order **iff** denotation after projection through `q` is invariant.

---

## 2. CD — Cayley–Dickson / Coordinate Decomposition

### CDₐ — Cayley–Dickson (Algebra)

**Axis:** *Algebraic / Semantic*

* Doubling constructions
* Parity, conjugation, cancellation
* Growth by structure, not scalar width

### CDᵣ — Coordinate Decomposition (Representation)

**Axis:** *Structural / Graph Encoding*

* Binary trees / DAGs
* `OP_COORD_ZERO | ONE | PAIR`
* CNF-2 representation

### Desired Commutation

```
canon( CDₐ ∘ CDᵣ (x) ) = canon( CDᵣ ∘ CDₐ (x) )
```

### Why We Want This

* Algebraic meaning must not depend on tree shape
* Representation must not encode “accidental” algebra

### Failure Mode

* Two algebraically equal coordinates intern differently
* Partial normalization leaks into key packing
* Tree shape affects parity or cancellation

### Normative Rule

> CDₐ and CDᵣ commute **iff** coordinate normalization is idempotent and confluent *before* key encoding.

---

# Audit of Other Risky Terms (with Desired Commutation)

Below are the other terms that historically drift — **not because commutation is bad**, but because it was *implicit*. We make it explicit.

---

## 3. Canonical / Canonicalization

### Meanings in Play

* **Canonicalᵢ**: interned identity (Ledger)
* **Canonicalʳ**: reduced / simplified form (rewrite intuition)
* **Canonicalᵣ**: representative under equivalence (future union-find)

### Axes

* Identity vs Rewrite vs Equivalence

### Desired Commutation

```
canonᵢ ∘ rewrite = rewrite ∘ canonᵢ   (up to denotation)
```

### Failure Mode

* “Canonicalization” used to mean eager simplification
* Identity created during rewrite
* Rewrite order affects IDs

### Normative Rule

> **Canonicalization = interning by full key equality.**
> Rewrite *proposes*; canonicalization *decides*.

(Other meanings must be explicitly qualified.)

---

## 4. Collapse

### Meanings in Play

* **Collapseʰ**: homomorphic collapse via `q`
* **Collapseᵍ**: graph reduction / node elimination
* **Collapseˡ**: logical identification (proof collapse)

### Axes

* Semantics vs Structure vs Proof

### Desired Commutation

```
q ∘ eval = evalₗ ∘ q
```

### Failure Mode

* “Collapse” interpreted as erasing structure
* Context clues lost (your original warning)
* Isomorphism assumed where only homomorphism exists

### Normative Rule

> Collapse means **projection of meaning**, not erasure of structure or context.

---

## 5. Normalize / Normal Form

### Meanings in Play

* **Normalizeᵣ**: rewrite to a reduced term
* **Normalizeᵢ**: intern to canonical ID
* **Normalize𝚌**: coordinate normalization (parity/XOR)

### Axes

* Rewrite vs Identity vs Algebra

### Desired Commutation

```
normalizeᵢ ∘ normalizeᵣ = normalizeᵢ
normalizeᵢ ∘ normalize𝚌 = normalizeᵢ
```

### Failure Mode

* Normalization order affects identity
* Partial normalization encoded in keys
* “Normal form” conflated across layers

### Normative Rule

> Only identity-relevant normalization may affect keys.
> All others must commute *before* interning.

---

## 6. Aggregate

### Meanings in Play

* **Aggregateᵣ**: fold / reduce
* **Aggregate𝚌**: coordinate-space combination
* **Aggregateₚ**: performance batching

### Axes

* Computation vs Algebra vs Performance

### Desired Commutation

```
canon( Aggregate𝚌 (x) ) = Aggregate𝚌 ( canon(x) )
```

### Failure Mode

* Aggregation treated as arithmetic when it’s semantic
* Batching affects identity
* Partial aggregates interned prematurely

### Normative Rule

> Aggregation that affects meaning must occur **inside** canonicalization, not around it.

---

## 7. Scheduler / Ordering

### Meanings in Play

* **Schedulerₜ**: temporal order of rewrites
* **Schedulerₛ**: spatial layout / permutation

### Axes

* Time vs Space

### Desired Commutation

```
denote( scheduleₜ ∘ scheduleₛ (P) )
=
denote( scheduleₜ (P) )
```

### Failure Mode

* Order of evaluation affects identity
* Locality changes rewrite visibility
* Barrier placement changes results

### Normative Rule

> Scheduling is free **iff** denotation after `q` is invariant.

---

## 8. Identity / Pointer

### Meanings in Play

* **Pointerₑ**: evaluator-local address
* **IDₗ**: ledger canonical ID
* **IDₑ**: equivalence-class representative (future)

### Axes

* Implementation vs Semantics vs Proof

### Desired Commutation

```
q(pointerₑ) = IDₗ
```

### Failure Mode

* Pointer equality used as semantic equality
* Cross-engine pointer comparison
* Implicit assumptions about stability

### Normative Rule

> Only ledger IDs carry semantic identity.
> All other identities are provisional.

---

## 9. HLO (XLA IR)

### Meanings in Play

* **HLO**: XLA High-Level Optimizer IR emitted after lowering JAXPR
* **HLO size**: compile-time graph size and complexity

### Axes

* Compilation-time cost vs Runtime work

### Desired Commutation

```
denote(P) == denote(compile(P))
```

(Correctness must not depend on the compiler, even when the compile graph is huge.)

### Failure Mode

* `vmap` + `while_loop` + search lowers to a massive HLO even when most lanes are no-ops
* Runtime guards (`cond`) skip work but do not shrink compile-time HLO
* Host recursion that calls jitted interning causes many tiny compilations

### Normative Rule

> If a function contains `vmap + while_loop + lookup`, apply it only to the smallest
> possible subset (gather -> normalize -> scatter). Keep host recursion as a slow
> reference path, and provide a batched/jitted path for hot use.

---

## 10. Garbage Collection / Interning (Semantic Compression)

### Meanings in Play

* **GCᵣ**: runtime reclamation (tracing/sweeping)
* **GCᵢ**: interning/dedup as semantic compression

### Axes

* Resource management vs Semantic identity

### Desired Commutation

```
pretty(denote(rebuild_from_roots(L))) == pretty(denote(L))
```

### Failure Mode

* Canonical IDs are reclaimed or reassigned
* "GC" used to mask semantic aliasing
* Rebuild changes denotation

### Normative Rule

> Interning is semantic compression; optional rebuilds are allowed only as
> renormalization that preserves denotation.

---

## 11. Damage / Locality

### Meanings in Play

* **Damageₛ**: spatial boundary crossing (tile/halo escalation)
* **Damageₑ**: semantic rewrite impact

### Axes

* Locality vs Meaning

### Desired Commutation

```
denote( damage_escalate ∘ local_step (P) ) = denote( local_step (P) )
```

### Failure Mode

* Damage sets influence identity creation
* Locality changes which rewrites fire

### Normative Rule

> Damage is a performance signal only; it must be erased by `q` and never
> affect denotation.

---

## 12. Renormalization / Sorting

### Meanings in Play

* **Renormˢ**: layout reorder (sort/swizzle)
* **Normalizeᵣ**: semantic reduction (already defined above)

### Axes

* Layout vs Semantics

### Desired Commutation

```
denote( renorm(P) ) = denote(P)
```

### Failure Mode

* Sorting changes keys or rewrite outcomes
* Root pointer/remap errors leak into meaning

### Normative Rule

> Sorting/swizzling are renormalization passes only; preserve edges and NULL,
> and validate invariance after `q`.

---

## 13. OOM / CORRUPT

### Meanings in Play

* **OOM**: resource exhaustion (capacity)
* **CORRUPT**: semantic undefinedness (alias risk)

### Axes

* Resource limits vs Semantic validity

### Desired Commutation

```
denote(P) is undefined iff CORRUPT
```

### Failure Mode

* Key-width overflow treated as OOM
* Execution proceeds after alias risk

### Normative Rule

> CORRUPT is a hard semantic error; OOM is an admissible resource boundary.

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

* Primitive copy creates new nodes for existing structure
* Superlinear growth from repeated use

### Normative Rule

> Duplication is expressed by sharing canonical IDs; no-copy is an operational axiom.

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

* Names leak into keys or identity
* Alpha-equivalent terms intern differently

### Normative Rule

> Binding is structural; alpha-equivalence must collapse before interning.

---

# Meta-Rule: How to Use This Going Forward

Whenever a term or acronym is reused:

1. **Name the axes.**
2. **State the desired commutation.**
3. **Say what erases what** (usually `q`).
4. **Add a test obligation** if the commutation matters.

If any of those cannot be stated clearly, the reuse is invalid.

---

## Optional Next Steps (if you want to harden this further)

* Add **axis tags** (`ᵗ`, `ˢ`, `ₐ`, `ᵣ`) in code comments where ambiguity matters
* Add **glossary references in tests** (“this test enforces BSPᵗ/BSPˢ commutation”)
* Add a short **“forbidden reinterpretations”** appendix listing known past drift cases

This glossary doesn’t freeze language — it freezes **assumptions**. That’s exactly what you need if commutation and convergence are *features*, not accidents.
