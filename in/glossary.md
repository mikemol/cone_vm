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
