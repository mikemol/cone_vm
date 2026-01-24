# Hyperstrata and Hypervalue CNF-2 Proposal

Status: m3 semantic commitment (implementation staged; no code changes implied).
Milestone: m3 (hyperstrata visibility rule is normative; hypervalues remain staged).

This note proposes a two-dimensional strata model ("hyperstrata") and a
structural "hypervalue" representation using Cayley-Dickson recursion. The goal
is to preserve the BSP_t invariants (no within-stratum influence, deterministic
canonicalization) while enabling orthogonal dependency axes and CNF-2
composition at higher "hyperdepth."

## Background

Current CNF-2 cycle structure:
- A frontier of ledger ids is rewritten into a candidate buffer.
- Candidates are compacted and interned into new ledger rows.
- The cycle has three internal strata (slot0, slot1, wrap).
- A q-map projects provisional ids to canonical ids at each stratum boundary.
- Pre-step rows are treated as immutable for the duration of a cycle.

The current semantics rely on:
- No within-stratum references (new rows must reference only pre-step rows).
- Strata boundaries define when newly created ids become visible.
- Canonical identity (interning) uses full key equality.

## Motivation

Two drivers for the proposal:
1) Some dependencies are "orthogonal" to the existing strata axis and can be
   modeled as a second ordered axis rather than forcing deeper cycles.
2) Cayley-Dickson recursion provides a structured, arity-2 representation of
   orthogonal dimensions, which matches CNF-2 arity and slot layout.

The idea is to keep CNF-2 arity and slot layout, but allow a slot to carry a
hypervalue whose components represent orthogonal dimensions.

## Proposal Overview

### 1) Two-dimensional strata (hyperstrata)

Introduce a second ordered strata index. Each new row is conceptually tagged
with a coordinate:
  (s, t)
where:
  s = existing stratum index (slot0, slot1, wrap)
  t = micro-stratum index (orthogonal refinement axis)

Ordering options:
1) Lexicographic order:
   (s, t) < (s', t') iff s < s' OR (s == s' AND t < t')
   This embeds micro-strata inside each stratum.

2) Product/partial order:
   (s, t) <= (s', t') iff s <= s' AND t <= t'
   This supports antichain frontiers and "grid" semantics.

A practical default is lexicographic order for determinism and simplicity. The
partial order can be added later if needed.

### 2) Hypervalues via Cayley-Dickson recursion

Define a hypervalue type as nested pairs:

HV(0) = LedgerId
HV(k+1) = (HV(k), HV(k))

Arity stays 2 at every level. When a component is unused, encode it as ZERO_PTR
or NULL (by convention).

Each CNF-2 slot carries an HV(d) for some fixed hyperdepth d. Depth 0 is the
current scalar id behavior.

### 3) CNF-2 as a hyperoperator

At depth 0, CNF-2 operates on scalar ids as it does today.
At depth d>0, lift each operation to hypervalues.

Two candidate lifting rules:

Option A: Componentwise lift (simple, safe)
  lift(op, (a0, a1), (b0, b1)) = (op(a0, b0), op(a1, b1))

Option B: Cayley-Dickson mixing for mul (requires extra ops)
  mul((a,b),(c,d)) = (mul(a,c) - mul(conj(d), b),
                     mul(d,a) + mul(b,conj(c)))
This is a true CD multiplication, but it needs subtraction and conjugation.
If those ops are not present, this option is deferred.

For m2/m3, Option A is the default. Option B can be introduced at a later
milestone if the algebra is desired.

### 4) Read model and immutability

The immutability contract becomes explicit:
- Pre-step rows (indices < start_count) are not mutated during a cycle.
- All candidate emission reads are from the pre-step ledger.
- Strata define when newly created rows become visible.

Two-dimensional strata do not remove the need for this contract; they refine it.
The rule becomes:
- A row may only reference rows with strictly smaller (s, t) coordinates.

## Representation Choices

### A) Hypervalues as nested pairs of ids (external)
Store HV(d) as Python tuples of arrays or as struct-of-arrays:
- For d=1: (id_left, id_right) arrays.
- For d=2: pairs of pairs, etc.

Pros:
- No new opcodes.
- Explicit structure.
Cons:
- Wider candidate buffers, more plumbing in JAX kernels.

### B) Hypervalues as ledger nodes
Introduce OP_HYPER_PAIR (or reuse OP_COORD_PAIR if semantics allow) and intern
pair nodes in the ledger.

Pros:
- Candidate buffers stay scalar (ids of hyperpair nodes).
- Canonicalization handled by existing interning.
Cons:
- Extra ledger pressure; semantics of OP_HYPER_PAIR must be defined.

Recommendation: start with (A) for clarity, then consider (B) for performance.

## Strata Semantics with Hypervalues

Within a single cycle, for each stratum s:
1) Emit candidates from frontier (read-only).
2) Intern candidates to create new rows tagged at (s, t=0).
3) If needed, perform micro-strata steps t=1..T within the same s to resolve
   intra-stratum dependencies (each step reads from earlier t only).
4) Commit via q-map, then advance to s+1.

Micro-strata can be fixed depth (T fixed per cycle) or bounded by a guard.

## Q-map and Projection Rules

Define q-maps over hypervalues as componentwise projections:
  q_map(HV(d)) applies q_map to every leaf id.

If both axes are used:
  q_{s,t} = q_{s,t-1} composed with a stratum-local projection
  q_{s+1,0} = q_{s,t_max} composed with the next-stratum projection

Totality rule:
- q_map must be defined for all ids, including ids outside the committed range.
- For ids not in the stratum range, q_map returns the input id.

## Invariants

Core invariants to preserve:
- No within-stratum references: a row at (s,t) may only reference ids created at
  coordinates < (s,t).
- Pre-step immutability: rows < start_count are not mutated during a cycle.
- Canonical identity: interning uses full key equality; deterministic ids.
- Frontier permutation invariance: candidate emission and slot layout do not
  depend on frontier order.
- Q-map totality and compositionality across strata and micro-strata.

## Test Plan

New or extended tests should cover:
- Pre-step immutability: op/arg arrays unchanged for rows < start_count.
- Hypervalue totality: q_map applies componentwise to mixed ids.
- Strata ordering: no within-(s,t) references.
- Frontier permutation invariance for hypervalues (componentwise).
- Composition: q-map across (s,t) chain matches re-interned canonical ids.

## Migration Path

1) Formalize the immutability contract (tests already added).
2) Introduce hypervalue wrappers at depth 1 (componentwise lift).
3) Add optional micro-strata index t; keep lexicographic order.
4) Extend CNF-2 candidates to carry HV(d) in slot payloads.
5) (Optional) Add CD mixing rules for mul when supporting ops exist.

## Open Questions

- Which representation is preferable long-term: external HV tuples or ledger
  hyperpair nodes?
- Do we want partial-order (grid) semantics or strict lexicographic order?
- Should micro-strata be fixed depth or driven by a per-cycle guard?
- If CD mixing is desired, what is the minimal operator set needed?

---

Summary:
Two-dimensional strata and hypervalues can be added without changing the core
CNF-2 arity or slot layout. The key requirements are explicit pre-step
immutability and a clear read model. Hypervalues let CNF-2 act as a hyperoperator
over orthogonal dimensions, while strata and micro-strata provide the staging
needed to preserve determinism and canonical identity.
