---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
---

# Milestones

## m1 (2026-01-23) — completed
Tag: `m1`
Baseline gate (m1 suite, runs under the current baseline milestone):
- `mise exec -- pytest -c pytest.baseline.ini`

m1-only mode is deprecated. The m1 suite should run as baseline coverage, not
under m1-restricted semantics.

Expected xfails (in later milestones):
- `tests/test_coord_batch.py::test_coord_xor_batch_uses_single_intern_call` - m4: no batched coord_xor_batch / coord_norm_batch yet.
- `tests/test_coord_batch.py::test_coord_norm_batch_matches_host` - m4: no batched coord_norm_batch yet.

## Prioritized Punch-List (Current)
Ordered by semantic risk first, then verification depth, then hygiene.

**P0 — Semantic/core correctness** ✅
- Explicit `q` projection boundary for Arena/Manifest (glossary §17/§21). ✅
  (See `tests/test_q_projection.py`.)
- Hyperstrata / micro-strata visibility beyond the fixed 3-strata cycle. ✅
  (See `tests/test_candidate_cycle.py::test_cycle_candidates_wrap_microstrata_validate`.)

**P1 — Verification depth / semantic completeness** ✅
- Min(Prism) full harness: `canon_state` + bounded enumeration + projection commutation. ✅
- Event-sourced/CQRS interner model (read-model rebuild path; in-12/13). ✅
- Hyperlattice / lattice-stable join tests (in-20). ✅

**P2 — Hygiene / clarity / roadmap housekeeping** ✅
- Clarify `_lookup_node_id` tuple unpacking. ✅
- Overflow guard for `_active_prefix_count`. ✅
- Remove duplicate "JAX Kernels" header (if not already resolved). ✅
- Agda proof roadmap execution (in-26). ✅
- in-8 interaction-combinator engine pivot (tensor/rule-table path). ✅

**P3 — Formalization + next-architecture track** ✅
- in-19 staging/site topology: define/encode micro‑strata (`t`) + tile (`τ`) semantics and visibility rules. ✅
- CD morphisms / hyperpair semantics beyond coord opcodes; add lattice‑law tests for CD‑lifted ops. ✅
- Novelty / hyperoperator fixed‑point instrumentation or bounded checks (Min(Prism) expansion). ✅
- Agda proofs (actual theorems) for univalence/gauge/novelty/finite closure/fixed points. ✅
- in-8 interaction‑combinator engine implementation (rule table + port encoding + rewrite kernel). ✅

**P4 — Verification hardening + next backend** ✅
- No‑copy / alpha‑equivalence tests for ledger sharing (in‑17). ✅
- CQRS replay harness beyond Min(Prism) (optional audit mode). ✅
- Agda boundary theorems (no‑termination / negative capability). ✅
- Interaction‑combinator backend (in‑8) beyond roadmap: data model + kernel prototype. ✅