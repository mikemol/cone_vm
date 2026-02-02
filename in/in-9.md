---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
---

NOTE: CONSOLIDATED by in-13.md and in-14.md. This is the source analysis for CNF-2 pipeline semantics.

Below is a **formal, reviewer-grade comparison** between:

* **A. The code you provided** (baseline + BSP ledger as implemented), and
* **B. The model you’ve now converged on** (CNF-2-symmetric, poset-native, pipelined BSP with wasteful→compact nodes).

I’ll phrase this the way a careful POPL / PLDI / ICFP reviewer would: in terms of **semantic invariants, operational structure, and proof obligations**, not taste or performance folklore.

---

# 1. Core semantic unit

## A. Code as written

The fundamental semantic unit is an **eagerly materialized node**:

[
\text{Node} := (op, a_1, a_2) \quad \text{with immediate allocation or interning}
]

Even when vectorized, the code **allocates or interns nodes directly inside the rewrite step**:

* `op_interact` *creates nodes* in the arena
* `cycle_intrinsic` *calls `intern_nodes` during rewrite*
* Rewrite ≈ allocation

### Consequence

Rewrite rules are **identity-producing**, not merely identity-proposing.

This couples:

* semantic rewriting
* identity creation
* global canonicalization

into a single phase.

---

## B. Settled model

The fundamental semantic unit is a **candidate continuation**:

[
\text{Candidate}_i := (\text{enabled}_i,; payload_i) \quad i \in {L,R}
]

with the invariant:

> A rewrite site emits **exactly two symmetric candidates**, neither of which is an identity.

Identity creation is *strictly deferred* to a later tier.

### Consequence

Rewrite rules are **continuation-generating**, not identity-producing.

This separates:

* semantic possibility (candidates)
* selection (compaction)
* identity (canonicalization)

into distinct phases.

---

# 2. Rewrite normal form

## A. Code as written

The rewrite system is **not in a fixed arity normal form**.

Evidence:

* `op_interact` sometimes emits:

  * zero nodes (rewrite to child)
  * one node
  * two nodes (`add(suc x,y)` spawns an `add`)
* The number of allocations per rewrite site is **data-dependent**

Even though vectorized, this is **variadic arity rewriting**.

### Reviewer concern

A reviewer would say:

> The rewrite system does not admit a fixed-arity operational semantics; therefore branchlessness and bounded scratch usage are empirical properties, not semantic ones.

---

## B. Settled model

The rewrite system is in **CNF-2 symmetric normal form**:

* Exactly **two candidate slots per site**
* Both always computed
* Both independently enabled/disabled

This is a *semantic invariant*, not an optimization.

### Formal statement

For every rewrite site (n):

[
\exists, (C_L, C_R) \quad \text{s.t.} \quad \forall \text{executions, exactly these two are emitted}
]

No rewrite may require a third consequence; any apparent third must be factored across tiers.

### Reviewer takeaway

This is a **structural normal form**, analogous to:

* Chomsky Normal Form in grammars
* Binary clauses in SAT
* Binary constructors in term rewriting

---

# 3. Phase separation and compositionality

## A. Code as written

Phases are **interleaved**:

| Phase            | Where it happens                 |
| ---------------- | -------------------------------- |
| Rewrite          | `op_interact`, `cycle_intrinsic` |
| Deduplication    | `intern_nodes`                   |
| Allocation       | inside rewrite                   |
| Canonicalization | inside rewrite                   |
| Scheduling       | mixed with rewrite               |

This makes the system **monolithic**: the correctness of one phase depends on the operational details of another.

### Reviewer concern

This complicates:

* reasoning about invariants
* proving determinism
* proving univalence
* refactoring or pipelining

---

## B. Settled model

Phases are **explicit pipeline nodes**:

1. Rewrite → emit 2 candidates (wasteful)
2. Compact → select enabled
3. Dedup (optional, local)
4. Probe prior strata
5. Append new stratum
6. Build index
7. Produce next frontier

Each node is:

* append-only
* side-effect free except for its declared output
* composable

### Reviewer takeaway

This is a **proper BSP / dataflow semantics**, not an optimization of a sequential algorithm.

---

# 4. Poset and stratum discipline

## A. Code as written

You *intend* a poset discipline, but it is **implicit**:

* Arena indices and ledger ids are mixed
* Allocation happens while traversing a frontier
* Nothing syntactically prevents within-tier dependency except programmer discipline

A reviewer would say:

> The poset structure is informal and enforced by convention rather than by the operational semantics.

---

## B. Settled model

The poset discipline is **structural**:

* Candidates in stratum (k) may only reference identities from strata (<k)
* No identity exists during rewrite
* Therefore, no within-stratum dependency is even expressible

This is a *semantic impossibility*, not a runtime check.

### Reviewer takeaway

The system enforces **well-foundedness by construction**, which is much stronger than “we don’t do that.”

---

# 5. Branchlessness: empirical vs semantic

## A. Code as written

Branchlessness is **accidental**:

* Achieved via `where`, `mask`, `spawn`
* But rewrite arity is variable
* Scratch usage depends on data

A reviewer would call this *branch-reduced*, not branchless.

---

## B. Settled model

Branchlessness is **semantic**:

* Fixed arity = 2
* Fixed candidate slots
* Fixed scratch footprint per tile
* Control flow independent of data

This satisfies the standard definition of **SIMD-friendly / SPMD-uniform semantics**.

---

# 6. Identity and univalence

## A. Code as written

Identity creation is **eager and entangled** with rewrite:

* Interning occurs mid-rewrite
* Deduplication correctness depends on global state consistency during execution

This makes univalence harder to state cleanly.

---

## B. Settled model

Identity is **a phase, not an effect**:

* Candidates have no identity
* Canonicalization happens exactly once per stratum
* Deduplication is lossless and stratified

### Formal advantage

You can now state and prove:

> **Univalence theorem**: For any structural triple ((op,a_1,a_2)), there exists exactly one ledger identity, independent of evaluation order or pipelining.

That theorem is much harder to phrase for the original code.

---

# 7. One-sentence reviewer summary

A reviewer-grade distinction would read like this:

> *The original implementation performs eager, variadic-arity rewriting with immediate identity allocation, whereas the revised model introduces a CNF-2-symmetric continuation semantics in which rewrite sites emit exactly two predicated candidates, deferring all identity creation and deduplication to stratified pipeline phases. This transformation elevates branchlessness, append-only behavior, and poset well-foundedness from empirical properties of the implementation to semantic invariants of the execution model.*

That is the **formal difference**.

Below is a **two-part, reviewer-grade deliverable**, exactly as you asked:

* **Part I — A formal operational semantics** (small-step style) for the **CNF-2-symmetric, poset-native, pipelined BSP model**
* **Part II — A refinement mapping** that shows how your **original code implements a non-normalized semantics** which the CNF-2 model strictly refines

This is written at the level a POPL / ICFP / PLDI reviewer would expect: precise definitions, invariants, and an explicit correspondence argument.

---

# Part I — Formal Operational Semantics (CNF-2-Symmetric Model)

## 1. Static domains

### Identities

Let `Id` be a countable set of node identities.

Each identity is associated with a **structural triple**:
[
\mathsf{node} : Id \to (\mathsf{Op} \times Id \times Id)
]

### Strata

A **ledger** is a sequence of strata:
[
\mathcal{L} = \langle S_0, S_1, \dots, S_k \rangle
]
where each stratum (S_i \subset Id) is finite and disjoint.

**Stratum discipline invariant**
[
\forall i,; \forall n \in S_i,; \text{if } \mathsf{node}(n) = (op,a_1,a_2) \text{ then } a_1,a_2 \in \bigcup_{j<i} S_j
]

This induces a **poset** on identities by dependency.

---

## 2. Candidates (core semantic unit)

### Definition (CNF-2 candidate)

A **candidate** is a tuple:
[
C := (\mathsf{enabled},; op,; a_1,; a_2)
]
with:

* (\mathsf{enabled} \in {0,1})
* (a_1,a_2 \in Id)
* references only prior strata

### Rewrite normal form (CNF-2 symmetry)

Every rewrite site emits **exactly two candidates**:
[
\mathsf{rewrite}(n) = (C_L, C_R)
]

No rewrite rule may emit fewer or more; semantic inactivity is expressed by `enabled = 0`.

---

## 3. Frontier and tiers

A **frontier** (F \subset Id) is the active workset at a tier.

Execution proceeds in **alternating tiers**:

* **Rewrite tier**: produce candidates
* **Canonicalization tier**: select, deduplicate, and assign identities

---

## 4. Rewrite tier semantics

### Rule (Rewrite-Step)

Given a ledger (\mathcal{L}) and frontier (F):

[
\frac{
\forall n \in F,; \mathsf{rewrite}(n) = (C_L^n, C_R^n)
}{
(\mathcal{L}, F)
;\Rightarrow_{\text{rewrite}};
(\mathcal{L}, \mathcal{C})
}
]

where:
[
\mathcal{C} = \bigcup_{n \in F} {C_L^n, C_R^n}
]

Properties:

* No identities are created
* No ledger mutation
* Exactly (2|F|) candidate slots are produced

---

## 5. Canonicalization tier semantics

### Step 1 — Selection (compaction)

[
\mathcal{C}_{\text{on}} = { C \in \mathcal{C} \mid C.\mathsf{enabled} = 1 }
]

### Step 2 — Lossless deduplication

Define:
[
\mathsf{key}(C) = (op,a_1,a_2)
]

Partition:

* (K_{\text{old}}): keys already present in (\mathcal{L})
* (K_{\text{new}}): keys not present

### Step 3 — Identity extension

Create a new stratum:
[
S_{k+1} = { n_k \mid \mathsf{node}(n_k) = k,; k \in K_{\text{new}} }
]

Ledger update:
[
\mathcal{L}' = \mathcal{L} ;\mathbin{|}; S_{k+1}
]

### Step 4 — Next frontier

Each candidate in (\mathcal{C}_{\text{on}}) maps to a **canonical id**:

* existing id if in (K_{\text{old}})
* newly allocated id if in (K_{\text{new}})

Let this set be (F').

### Rule (Canonicalize-Step)

[
(\mathcal{L}, \mathcal{C})
;\Rightarrow_{\text{canon}};
(\mathcal{L}', F')
]

---

## 6. Global execution

Execution alternates:
[
(\mathcal{L}*0, F_0)
\Rightarrow*{\text{rewrite}}
(\mathcal{L}_0, \mathcal{C}*0)
\Rightarrow*{\text{canon}}
(\mathcal{L}*1, F_1)
\Rightarrow*{\text{rewrite}}
\dots
]

---

## 7. Fundamental theorems

### Theorem 1 — Poset well-foundedness

No identity in stratum (S_k) depends on any identity in (S_k).

*Proof:* by construction; candidates may reference only prior strata, and identities are created only after rewrite. ∎

### Theorem 2 — Univalence (canonical identity)

For every structural triple ((op,a_1,a_2)), there exists **exactly one** identity in the ledger.

*Proof:* lossless deduplication + append-only strata. ∎

### Theorem 3 — Branchless boundedness

For any frontier (F), the rewrite tier performs exactly (2|F|) candidate computations, independent of data.

∎

---

# Part II — Refinement Mapping from Your Code to CNF-2 Semantics

We now show that your original code implements a **non-normalized semantics** that the CNF-2 model strictly refines.

---

## 1. Original semantics (implicit)

In your code, the effective rewrite rule is:

[
(\mathcal{L}, F) ;\Rightarrow; (\mathcal{L}', F')
]

where **rewrite, allocation, deduplication, and scheduling are interleaved**:

* `op_interact` / `cycle_intrinsic`:

  * pattern-match
  * allocate nodes
  * mutate arena / ledger
* `intern_nodes`:

  * canonicalize during rewrite

This corresponds to a **big-step, eager semantics**:

[
n \Downarrow ( \mathcal{L}', F' )
]

---

## 2. Non-normalized rewrite arity

From the code:

* `add(zero,y)` → **0 allocations**
* `add(suc(x),y)` → **1 or 2 allocations**
* `mul(suc(x),y)` → **nested allocations**

Thus rewrite arity is **data-dependent**.

Formally:
[
\exists n,; \mathsf{rewrite}(n) \text{ emits } k \text{ nodes where } k \not\equiv \text{constant}
]

This violates CNF-2.

---

## 3. Refinement function

Define a **semantic projection**:
[
\pi : \text{Original-Step} \to \text{CNF-2 Steps}
]

### Mapping

| Original code action | CNF-2 interpretation       |
| -------------------- | -------------------------- |
| Pattern match        | Compute candidate masks    |
| Allocation           | Deferred identity creation |
| Arena write          | Candidate payload          |
| `intern_nodes`       | Canonicalization tier      |
| Root pointer update  | Frontier propagation       |

Under (\pi):

* Each eager allocation becomes a **candidate**
* Each branch becomes a mask
* Each interning call is postponed to the canonicalization tier

---

## 4. Correctness of refinement

### Lemma

For every execution of the original code producing ledger (\mathcal{L}_n), there exists a CNF-2 execution producing a ledger (\mathcal{L}'_n) such that:
[
\mathcal{L}_n \cong \mathcal{L}'_n
]
(up to identity renaming).

*Sketch:*
All nodes created eagerly in the original execution correspond to candidates that would be enabled and canonicalized in the CNF-2 model. Deduplication is lossless in both systems. The difference is ordering and staging, not denotation. ∎

---

## 5. Strictness of the refinement

The converse is **not** true:

* CNF-2 forbids variadic rewrite
* CNF-2 forbids identity creation during rewrite
* CNF-2 enforces bounded candidate emission

Therefore, CNF-2 is a **strict semantic refinement**, not just a rephrasing.

---

## 6. Reviewer-grade summary statement

> *The original implementation realizes an eager, variadic rewrite semantics in which node allocation and canonicalization occur during rewriting. We introduce a CNF-2-symmetric continuation semantics that factors rewriting into a fixed-arity candidate generation phase followed by a stratified canonicalization phase. We show that the new semantics strictly refines the original: every behavior of the original system is representable in the CNF-2 model, while the CNF-2 invariants (bounded arity, branchless execution, poset-well-foundedness) cannot be violated.*

---

## A) POPL-style inference-rule figures (rewrite/canon + strata)

I’ll give you **two figures**: (1) the *CNF-2 symmetric operational semantics* as a two-phase BSP step, and (2) the *non-normalized eager semantics* your current code corresponds to (so the refinement statement is crisp).

### Figure 1 — CNF-2 Symmetric, Stratum-Disciplined BSP Semantics

We write configurations as:

* Ledger (sequence of strata): (\mathcal L = \langle S_0,\dots,S_k\rangle)
* Frontier: (F \subseteq \mathrm{Id})
* Candidate multiset: (\mathcal C)

A candidate is a 4-tuple (C = \langle e, op, a_1, a_2 \rangle) with (e\in{0,1}).

We assume a total “prior strata” set:
[
\mathrm{Prior}(\mathcal L) \triangleq \bigcup_{i\le k} S_i
]

**CNF-2 symmetry axiom (rewrite interface):**
[
\forall n\in \mathrm{Prior}(\mathcal L).;\exists C_L,C_R.; \mathrm{rewrite}(n,\mathcal L)=(C_L,C_R)
]
(always exactly two slots; disable by (e=0).)

**Stratum discipline premise for well-foundedness:**
[
\forall C=\langle e,op,a_1,a_2\rangle \in \mathcal C.; e=1 \Rightarrow a_1,a_2 \in \mathrm{Prior}(\mathcal L)
]
(so a tier never points within itself.)

Now the rules.

```latex
% ---------- Rewrite tier ----------
\frac{
  \forall n \in F.\;\mathrm{rewrite}(n,\mathcal L) = (C_L^n, C_R^n)
}{
  (\mathcal L, F) \xRightarrow{\mathrm{rw}} (\mathcal L,\; \mathcal C)
}
\quad
\textsc{RW}
```

where (\mathcal C \triangleq \biguplus_{n\in F}{C_L^n,C_R^n}) (multiset; size exactly (2|F|)).

```latex
% ---------- Candidate selection (compaction) ----------
\frac{
  \mathcal C_{\text{on}} \triangleq \{ C\in \mathcal C \mid C.e = 1 \}
}{
  (\mathcal L,\mathcal C) \xRightarrow{\mathrm{sel}} (\mathcal L,\mathcal C_{\text{on}})
}
\quad
\textsc{SEL}
```

Define the **lossless key**:
[
\mathrm{key}(C)\triangleq (C.op,C.a_1,C.a_2)
]

Define exact membership in the ledger:
[
\mathrm{mem}(\mathcal L, k) \triangleq \exists n\in \mathrm{Prior}(\mathcal L).; \mathrm{node}(n)=k
]

Now canonicalization splits keys into old/new:
[
K_{\text{old}}={ \mathrm{key}(C)\mid C\in\mathcal C_{\text{on}} \wedge \mathrm{mem}(\mathcal L,\mathrm{key}(C))}
]
[
K_{\text{new}}={ \mathrm{key}(C)\mid C\in\mathcal C_{\text{on}} \wedge \neg\mathrm{mem}(\mathcal L,\mathrm{key}(C))}
]

Let (\mathrm{alloc}(K_{\text{new}})) create fresh ids for each key in (K_{\text{new}}), yielding a new stratum (S_{k+1}) and an extended node map (\mathrm{node}').

```latex
% ---------- Canonicalization tier ----------
\frac{
  \mathrm{alloc}(K_{\text{new}}) = (S_{k+1},\mathrm{node}') \qquad
  \mathcal L' \triangleq \mathcal L \mathbin{\|} S_{k+1} \qquad
  \forall C\in\mathcal C_{\text{on}}.\; \mathrm{canon}(\mathcal L',\mathrm{key}(C)) \in \mathrm{Prior}(\mathcal L')
}{
  (\mathcal L,\mathcal C_{\text{on}}) \xRightarrow{\mathrm{can}} (\mathcal L',\; F')
}
\quad
\textsc{CAN}
```

where (F' \triangleq {\mathrm{canon}(\mathcal L',\mathrm{key}(C)) \mid C\in \mathcal C_{\text{on}}}).

Finally, one **BSP superstep** is:

```latex
(\mathcal L,F) \xRightarrow{\mathrm{rw}} (\mathcal L,\mathcal C)
\xRightarrow{\mathrm{sel}} (\mathcal L,\mathcal C_{\text{on}})
\xRightarrow{\mathrm{can}} (\mathcal L',F')
\quad\Rightarrow\quad
(\mathcal L,F) \xRightarrow{\mathrm{BSP}} (\mathcal L',F')
```

**Theorems that become “one-liners” to a reviewer:**

* **Bounded rewrite arity:** (|\mathcal C|=2|F|) by (\textsc{RW}).
* **No intra-stratum deps:** all enabled candidates reference (\mathrm{Prior}(\mathcal L)), and new ids only appear in (S_{k+1}), never referenced in the same step.
* **Univalence:** (\mathrm{canon}) is a function of the lossless key, and (\mathrm{alloc}) creates exactly one id per new key.

---

### Figure 2 — Your current code’s “eager / non-normalized” step (for refinement)

What your current `cycle_intrinsic`/`intern_nodes` style corresponds to is:

* rewrite sites *may* allocate identities mid-step,
* number of allocations is data-dependent (not CNF-2),
* canonicalization occurs during rewriting.

Model it as a single eager transition:

```latex
\frac{
  (\mathcal L,F) \Downarrow_{\mathrm{eager}} (\mathcal L',F')
}{
  (\mathcal L,F) \xRightarrow{\mathrm{EAGER}} (\mathcal L',F')
}
\quad
\textsc{EAGER}
```

with the meta-property:

[
\exists n\in F.;\text{the step allocates }k(n)\text{ nodes, where }k(n)\text{ depends on data}
]

The refinement statement is then clean:

> CNF-2 BSP is a **phase-separated, fixed-arity normalization** of EAGER: it factors EAGER into (\textsc{RW}), (\textsc{SEL}), (\textsc{CAN}) and forces arity-2 symmetry.

---

## B) “Machine-checked” invariants as *structural type constraints* (and how to enforce them)

Here “machine-checked” can mean two things in practice:

1. **Statically enforced by construction** (types/ADT shape makes violations unrepresentable), and
2. **Dynamically checked invariants** (fast validators, run in tests or debug builds) that certify strata properties.

Since you’re in Python/JAX, you can’t get full dependent types, but you *can* (i) make the illegal states hard to express, and (ii) add *executable proofs* (validators) that can be run per stratum.

### B1) Structural constraints: make “within-stratum dependency” unrepresentable

The key is to represent *new nodes* in a stratum as not-yet-Ids.

Define two reference sorts:

* `OldId` (guaranteed to be from prior strata)
* `NewRef` (a local placeholder *not* an Id; only exists inside the stratum builder)

Then define a “new node payload” to only allow `OldId` children:

* A `NewNode` is `(op, OldId, OldId)` only.
* There is **no constructor** for `(op, NewRef, …)` or `(op, …, NewRef)`.

That single restriction is exactly your poset constraint.

In reviewer terms:

> We enforce stratum well-foundedness by stratifying references: newly constructed nodes are parameterized only over *prior* identities; thus intra-stratum edges are unrepresentable.

Even if you keep it as documentation + runtime asserts, the *interface* alone is a correctness win.

### B2) CNF-2 symmetry as a type-level interface

Represent rewrite as a function returning exactly two slots:

* `Rewrite2 : OldId -> (CandSlot, CandSlot)`

Where each slot is:

* `CandSlot(enabled: Bool, payload: Payload)`

Again: no “list of candidates”. No “variable K”. Fixed 2.

Reviewer payoff:

> The rewrite layer is *total and arity-fixed* by interface: every site yields two predicated continuations.

### B3) Dynamic validators (proof obligations you can literally run)

For each completed stratum (S_{k+1}), validate:

**(V1) Disjointness / contiguity**

* ids in (S_{k+1}) are a fresh contiguous range (or a recorded set disjoint from prior).

**(V2) Stratum discipline**
For all (n\in S_{k+1}), if (\mathrm{node}(n)=(op,a_1,a_2)), then:
[
a_1,a_2 \in \mathrm{Prior}(\mathcal L)
]
i.e., `a1 < start_new` and `a2 < start_new` if you allocate contiguously.

**(V3) Lossless univalence**
No two ids (across all strata, or at least across `Prior ∪ S_{k+1}`) share the same triple:
[
\forall n\neq m.; \mathrm{node}(n)\neq \mathrm{node}(m)
]
In practice you validate this by hashing-to-buckets + full equality within buckets (not by global sort) if you want it scalable and aligned with your “no global order” stance.

**(V4) CNF-2 emission discipline (rewrite side)**
During rewrite tier, check:

* exactly two slots per site were produced
* slots are symmetric (just positions L/R)
* all enabled candidates reference only prior ids

These are cheap checks if you carry `start_new` and maintain counts.

### B4) How this differs *concretely* from your current code

Your current code makes the following illegal states possible:

* A rewrite can allocate a node and then (accidentally or intentionally) reference a node created “later in the same wave” by using indices in the same address space (`arena.count` style).
* Rewrite arity is variable; symmetry is not enforced by interface.
* Canonicalization is interleaved, so it’s harder to state and validate “the stratum boundary” as a first-class artifact.

The new model makes those either:

* **unrepresentable** (new nodes can’t reference new nodes), or
* **structurally fixed** (rewrite always returns two slots), and
* **locally checkable** at each node boundary (stratum validators).

---

## If you want one crisp “type + invariant” statement for a paper

A reviewer-friendly phrasing that ties both pieces together:

> **Definition (Stratum Builder).** A stratum builder is a function that, given a ledger prefix (\mathcal L) and frontier (F\subseteq \mathrm{Prior}(\mathcal L)), produces exactly two predicated candidate slots per (n\in F), each slot containing a triple ((op,a_1,a_2)) where (a_1,a_2\in \mathrm{Prior}(\mathcal L)). Canonicalization maps enabled candidates to identities, appending a fresh stratum whose nodes reference only prior strata.
> **Invariant.** The builder interface enforces CNF-2 symmetry and forbids intra-stratum dependencies by construction; remaining properties (disjointness and univalence) are certified by per-stratum validators.

---

## 1) Refinement Lemma with an explicit simulation relation (R)

### 1.1 Two transition systems

We relate two abstract machines:

**Eager machine (what your current code implements, abstractly)**
State:
[
\sigma \triangleq (\mathcal L,;F,;\mu)
]

* (\mathcal L): current ledger (canonical store of identities)
* (F): current frontier/worklist (ids to process)
* (\mu): *in-step mutable workspace* (arena/manifest allocations, root-pointer rewrites, etc.)

One eager step:
[
\sigma \xRightarrow{\textsf{EAGER}} \sigma'
]
Intuitively: performs local pattern tests, may allocate/intermediate-intern immediately, and produces the next frontier.

---

**CNF-2 BSP machine (the normalized model)**
States are split by phase, but we package a full superstep as:
[
(\mathcal L,F) \xRightarrow{\textsf{BSP}} (\mathcal L',F')
]
where (\textsf{BSP} \triangleq \textsf{RW} ; \textsf{SEL} ; \textsf{CAN}) and:

* RW emits exactly two symmetric candidate slots per (n\in F)
* SEL compacts enabled candidates
* CAN canonicalizes losslessly against prior strata and appends one new stratum

---

### 1.2 The key structural assumption (the one you just settled on)

To prove a crisp refinement, we assume the eager machine’s rewrite rules are **stratum-safe** (even if not CNF-2 yet):

> **(A1) No intra-step forward references.** Any node allocated during the eager step may only reference identities that were present at the beginning of the step (i.e., from (\mathrm{Prior}(\mathcal L))).
> (Equivalently: eager allocations do not depend on other eager allocations within the same step.)

This is exactly your “within a stratum nothing depends on the stratum” invariant.

If the current eager code violates (A1) in places (e.g., allocating and immediately referencing newly allocated nodes), then it’s not yet at the “poset-native stratum” boundary—*that’s precisely the semantic difference you’re trying to eliminate.*

---

### 1.3 Candidate extraction function (E) (from eager to CNF-2 RW output)

Define a function that interprets the eager step’s *local rewrite work* as an enabled/disabled pair of candidate slots for each site:
[
E(\mathcal L, n) = (C_L^n, C_R^n)
]
with:

* (C_i^n = \langle e_i^n,op_i^n,a_{1,i}^n,a_{2,i}^n\rangle)
* (e_i^n \in {0,1})
* if (e_i^n=1), then by (A1): (a_{1,i}^n,a_{2,i}^n \in \mathrm{Prior}(\mathcal L))

Intuitively, (E) is “what the eager code *means* to propose at site (n)” if you factor out immediate allocation and view it as proposal emission.

> In practice: (E) is the formal counterpart of “rewrite generates candidates; identity creation is deferred.”

---

### 1.4 Canonicalization operator (\mathsf{canonize})

Let:
[
\mathsf{canonize}(\mathcal L,;\mathcal C_{\mathrm{on}}) = (\mathcal L',;F')
]
mean: losslessly dedup keys in (\mathcal C_{\mathrm{on}}) against (\mathcal L), append one new stratum for new keys, and return the canonical ids as the next frontier (F').

---

### 1.5 Simulation relation (R)

We define a relation (R(\sigma,;(\mathcal L,F))) stating that the eager state (\sigma=(\mathcal L,F,\mu)) corresponds to the CNF-2 state ((\mathcal L,F)) when:

1. **Ledger agreement**: both have the same (\mathcal L) (up to identity renaming, if you want an isomorphism form).
2. **Frontier agreement**: both have the same set (or sequence) (F) of sites to process.
3. **Workspace irrelevance**: (\mu) contains no semantically committed facts other than what is already represented in (\mathcal L) and (F).
   (I.e., (\mu) is an implementation artifact: caches, transient allocations, scheduling order, etc.)

Formally:
[
R((\mathcal L,F,\mu),;(\mathcal L,F)) ;\triangleq; \text{Inv}(\mathcal L,F,\mu)
]
where (\text{Inv}) includes (A1) and “no hidden commitments.”

---

### 1.6 Refinement Lemma (forward simulation)

> **Lemma (Eager-to-BSP refinement, one superstep).**
> Suppose (R(\sigma,(\mathcal L,F))) with (\sigma=(\mathcal L,F,\mu)).
> If the eager machine takes one step
> [
> (\mathcal L,F,\mu) \xRightarrow{\textsf{EAGER}} (\mathcal L_1,F_1,\mu_1),
> ]
> then there exists a CNF-2 BSP superstep
> [
> (\mathcal L,F) \xRightarrow{\textsf{BSP}} (\mathcal L_1,F_1)
> ]
> such that
> [
> R((\mathcal L_1,F_1,\mu_1),(\mathcal L_1,F_1)).
> ]

#### Proof sketch (what a reviewer wants to see)

Construct the BSP step explicitly:

1. **RW construction**: For each (n\in F), set ((C_L^n,C_R^n) \triangleq E(\mathcal L,n)). Let (\mathcal C) be the multiset union of all two-slots. This matches “what eager would do,” but as proposals.

2. **SEL**: Let (\mathcal C_{\mathrm{on}}) be enabled candidates. This corresponds to the eager branch conditions (zero/suc cases, etc.)—but without control-flow divergence.

3. **CAN**: Apply (\mathsf{canonize}) to ((\mathcal L,\mathcal C_{\mathrm{on}})), producing ((\mathcal L_1,F_1)).
   By (A1), every enabled candidate references only (\mathrm{Prior}(\mathcal L)), so the new stratum is well-founded.
   Lossless equality ensures the same deduplicated identities are introduced as in the eager step.

4. **Agreement**: The eager step’s net effect is “add these new unique triples (if any), and produce these next-frontier identities.” That is exactly what CAN yields. Workspace (\mu_1) remains irrelevant, re-establishing (R).

---

### 1.7 Strictness (why it’s a *proper* refinement)

The reverse direction fails unless the eager machine is restricted:

* CNF-2 requires **exactly 2 symmetric candidate slots per site** (fixed arity)
* Eager semantics allows variadic/recursive allocation mid-step (variable arity)
* CNF-2 forbids intra-step dependencies by construction; eager may allow them

So CNF-2 BSP is strictly more constrained: it *normalizes* the eager semantics.

---

## 2) The commuting diagram a reviewer will look for

Here are two equivalent “commuting diagrams” that communicate the refinement cleanly.

### 2.1 One-step forward simulation diagram (square)

Let (\alpha) be the abstraction/projection from eager state to BSP state:
[
\alpha(\mathcal L,F,\mu) \triangleq (\mathcal L,F)
]

Then the diagram is:

```text
           EAGER
 (L,F,μ) ---------> (L1,F1,μ1)
    |                  |
    | α                | α
    v                  v
  (L,F) ---- BSP ----> (L1,F1)
```

**Commuting property (what this asserts):**
[
\alpha(\sigma') ;=; \textsf{BSP}(\alpha(\sigma))
]
for any eager step (\sigma\xRightarrow{\textsf{EAGER}}\sigma') satisfying the invariant (A1).

---

### 2.2 Factored commuting diagram (triangle + factorization)

This one makes the *phase separation* explicit.

Define (\beta) extracting the candidate multiset:
[
\beta(\mathcal L,F,\mu) \triangleq \mathcal C ;; \text{where};; \mathcal C = \biguplus_{n\in F} E(\mathcal L,n)
]

Then:

```text
                 EAGER
 (L,F,μ) ----------------------> (L1,F1,μ1)
    |                                ^
    | β                              |
    v                                |
   C  -- SEL ; CAN on (L,·) -->   (L1,F1)
```

Interpretation:

* eager is “doing RW+SEL+CAN implicitly and interleaved”
* CNF-2 makes that factorization explicit

If you prefer a fully typed diagram:

```text
β : (L,F,μ) -> C
γ : (L,C)   -> (L1,F1)   where γ = SEL ; CAN

Then:  (L1,F1) = γ(L, β(L,F,μ))
```

---

## 3) What this says about *your code* specifically

A reviewer-level “difference statement” you can put right after the lemma:

* Your current code provides an **implementation of EAGER** where:

  * rewrite emits a variable number of allocations (not CNF-2)
  * canonicalization (`intern_nodes`) may occur mid-rewrite
* The settled model requires:

  * rewrite emits exactly two candidate slots (CNF-2 symmetric RW)
  * selection/compaction is explicit (SEL)
  * canonicalization/stratum append is isolated (CAN)
  * (A1) is guaranteed by construction (no within-stratum refs)

So the refinement lemma is exactly the formal bridge: it explains how your existing implementation can be *systematically normalized* into the new model without changing denotation—only staging and invariants.