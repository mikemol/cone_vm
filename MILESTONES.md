# Milestones

## m1 (2026-01-23)
Tag: `m1`
Gate command:
- `mise exec -- pytest -c pytest.m1.ini`

Expected xfails (gated above m1):
- `tests/test_coord_batch.py::test_coord_xor_batch_uses_single_intern_call` - m4: no batched coord_xor_batch / coord_norm_batch yet.
- `tests/test_coord_batch.py::test_coord_norm_batch_matches_host` - m4: no batched coord_norm_batch yet.

## Prioritized Punch-List (Current)
Ordered by semantic risk first, then verification depth, then hygiene.

**P0 — Semantic/core correctness**
- Explicit `q` projection boundary for Arena/Manifest (glossary §17/§21).
- Hyperstrata / micro-strata visibility beyond the fixed 3-strata cycle.

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

**P3 — Formalization + next-architecture track (Current)**
- in-19 staging/site topology: define/encode micro‑strata (`t`) + tile (`τ`) semantics and visibility rules. ✅
- CD morphisms / hyperpair semantics beyond coord opcodes; add lattice‑law tests for CD‑lifted ops. ✅
- Novelty / hyperoperator fixed‑point instrumentation or bounded checks (Min(Prism) expansion). ✅
- Agda proofs (actual theorems) for univalence/gauge/novelty/finite closure/fixed points. ✅
- in-8 interaction‑combinator engine implementation (rule table + port encoding + rewrite kernel). ✅
