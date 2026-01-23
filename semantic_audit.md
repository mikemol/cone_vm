Manual semantic analysis based on `in/in-*.md`, `in/glossary.md`, `prism_vm.py`, and `IMPLEMENTATION_PLAN.md`.

## Definitions
- Intersection: shared semantic commitments (what both documents must be true about).
- Symmetric difference: commitments unique to each document.
- Wedge product: emergent constraint or design requirement that only appears when you make the two documents cohere.

## in/in-1.md
- Core semantics: host static analysis + hash consing on a manifest heap, static add/mul kernels, REPL telemetry that shows cache hits and kernel skips.
- Prior: none.
- vs `prism_vm.py`: Intersection: manifest-based IR, hash consing, static kernels, add-zero optimization, telemetry REPL; SymDiff: `prism_vm.py` adds BSP arena/ledger, rank/sort/swizzle, morton and blocked/hierarchical sorting; Wedge: baseline semantics must stay intact as a regression oracle while BSP evolves.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: baseline manifest + kernels; SymDiff: plan focuses on arena/rank/sort/cycle and tests; Wedge: plan should keep the baseline VM stable and testable alongside BSP work.

## in/in-2.md
- Core semantics: homoiconic IR where code and data are identical; universal `cons`; static kernels; interpreter dispatch with deferred execution.
- Prior `in/in-1.md`: Intersection: manifest heap, JAX kernels, deduplication; SymDiff: in-2 drops explicit static analysis emphasis and elevates code=data unification; Wedge: combine homoiconicity with in-1's optimizer so pre-dispatch reductions do not break the unified IR story.
- vs `prism_vm.py`: Intersection: cons-driven IR and JAX kernels; SymDiff: `prism_vm.py` adds static optimization plus BSP/ledger modes; Wedge: keep a single `cons` API that can feed both baseline and BSP while allowing optional optimizer pass.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: unified opcodes and manifest concept; SymDiff: plan does not cover homoiconicity or universal cons framing; Wedge: if the baseline parser is reused, its semantics should preserve the universal allocator idea to avoid drift.

## in/in-3.md
- Core semantics: exec-based kernel compiler, status field in manifest, telemetric REPL, dedup via cache; opcode set diverges from later docs.
- Prior `in/in-2.md`: Intersection: manifest heap and JAX kernel semantics; SymDiff: in-3 reintroduces dynamic codegen and adds status states; Wedge: a decision point exists between static-kernel purity and dynamic code synthesis.
- vs `prism_vm.py`: Intersection: heap + dedup + REPL intent; SymDiff: `prism_vm.py` is static-kernel and BSP-heavy, with no exec compiler or status field; Wedge: if any dynamic kernel work is desired, isolate it from the main path to keep determinism and XLA friendliness.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: desire for telemetry and dedup; SymDiff: plan is explicitly static-kernel and BSP-driven; Wedge: plan should explicitly mark exec-based compilation as deprecated or experimental.

## in/in-4.md
- Core semantics: conceptual design for 2-bit rank scheduling, morton order, BSP hierarchy, OP_SORT as defragmenter to counter shatter.
- Prior `in/in-3.md`: Intersection: JAX memory model focus; SymDiff: in-4 abandons compiler concerns and shifts to fluid arena and spatial locality; Wedge: requires a redesign of runtime around rank/sort/swizzle rather than instruction dispatch.
- vs `prism_vm.py`: Intersection: rank sorting, morton key, swizzle, OP_SORT notion; SymDiff: in-4 assumes explicit hierarchical arenas and coordinate assignment, which `prism_vm.py` only partially models; Wedge: to match the architecture, a concrete coordinate source and hierarchy boundaries must be defined.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: plan is derived from in-4 ideas (rank/sort/swizzle/morton); SymDiff: plan is pragmatic and defers hierarchy; Wedge: plan should clarify which locality guarantees are expected per milestone.

## in/in-5.md
- Core semantics: 2:1 alternating BSP geometry, hierarchical arenas, 2-bit rank enabling 4-bin sorting, local allocation to contain shatter.
- Prior `in/in-4.md`: Intersection: BSP + morton + rank packing; SymDiff: in-5 makes 2:1 geometry and local allocation concrete; Wedge: the sort key and allocator must preserve 2:1 locality, not just rank order.
- vs `prism_vm.py`: Intersection: swizzle_2to1, rank bins, sort+swizzle pipeline; SymDiff: `prism_vm.py` still uses argsort and a global count without local free management; Wedge: to realize 2:1 benefits, add binning or block-local allocation and reduce global churn.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: rank bins and later hierarchy; SymDiff: plan does not emphasize 2:1 specifics or allocator locality; Wedge: treat 2:1 as a correctness constraint, and align the plan with the CNF-2 symmetry/branchless fixed-arity requirements in `in/in-9.md`.

## in/in-6.md
- Core semantics: evaluation of 2:1 BSP tradeoffs; compute-vs-memory argument; Pallas/Triton swizzle; bucket sort via scan; hierarchical bitmask allocator.
- Prior `in/in-5.md`: Intersection: 2:1 BSP, hierarchy, rank bins; SymDiff: in-6 is explicitly about performance tradeoffs and tooling choices; Wedge: makes swizzle acceleration a requirement, not an optional optimization.
- vs `prism_vm.py`: Intersection: optional Pallas swizzle and morton key; SymDiff: `prism_vm.py` lacks bitmask allocator and does not enforce swizzle on GPU; Wedge: to match the evaluation claims, allocator and GPU swizzle path need to be first-class.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: morton and rank sorting goals; SymDiff: plan lacks explicit perf gates (Pallas/bitmask/scan); Wedge: add benchmarks to validate that swizzle cost stays below locality gains.

## in/in-7.md
- Core semantics: concrete fluid arena code for rank/sort/swizzle and add rewrite; cycle pipeline; swizzle_2to1 host/dev; root pointer caveat.
- Prior `in/in-6.md`: Intersection: 2:1 BSP, swizzle, rank bins; SymDiff: in-7 moves from analysis to executable skeleton and add-specific rewrite behavior; Wedge: correctness constraints (root remap, capacity checks) become mandatory.
- vs `prism_vm.py`: Intersection: op_rank, sort+swizzle, op_interact for add, swizzle_2to1; SymDiff: `prism_vm.py` adds ledger interning and blocked/hierarchical sorts; in-7 assumes single rank sort and naive allocation; Wedge: ensure root remapping and bounds checks align with the richer `prism_vm.py` pipeline.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: directly matches plan steps 1-5; SymDiff: plan adds explicit tests and root-tracking discipline; Wedge: encode in-7's rewrite behavior into tests, especially add(suc) allocation and pointer swizzle.

## in/in-8.md
- Core semantics: interaction combinators on GPU using branchless tensor rewrites; rule tensors; port-encoded adjacency; SoA with free stack; strong confluence.
- Prior `in/in-7.md`: Intersection: GPU focus, branchless vectorized rewriting, SoA memory; SymDiff: in-8 shifts from Peano add to interaction combinators and matrix view, with no rank/sort cycle; Wedge: adopting in-8 implies a new data model (ports and rule tensors), not just a new kernel.
- vs `prism_vm.py`: Intersection: SoA arrays and scatter/where style updates; SymDiff: `prism_vm.py` is opcode/arg graph with rank/sort, not port-encoded IC rewrite; Wedge: migration requires a new memory layout and rewrite kernel, not an incremental extension.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: high-level goal of GPU-friendly rewriting; SymDiff: plan is anchored on add/mul and BSP sorting, not ICs or rule tensors; Wedge: treat in-8 as a parallel research track or add a new workstream.

## in/in-9.md
- Core semantics: CNF-2 symmetric candidate pipeline, strict phase separation, poset strata discipline, deferred identity and canonicalization; critique of eager allocation.
- Prior `in/in-8.md`: Intersection: branchlessness and formal semantics focus; SymDiff: in-9 introduces candidate pipeline and poset discipline, while in-8 focuses on rule tensor rewiring; Wedge: rule tensors can emit two predicated candidates per active pair to satisfy CNF-2 semantics.
- vs `prism_vm.py`: Intersection: interest in canonicalization and determinism; SymDiff: `prism_vm.py` allocates/interns during rewrite and mixes phases; Wedge: refactor to rewrite -> candidate compaction -> `intern_nodes` stages to align with CNF-2.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: plan already stages new mode and emphasizes tests; SymDiff: plan does not adopt CNF-2 or strict phase separation; Wedge: if CNF-2 is the target model, add a candidate buffer tier and fixed-arity rewrite tests.

## in/in-10.md
- Core semantics: milestone delta from tree-local rewriting to CD-addressed aggregation; introduce CD coordinates as first-class data, interpret aggregation over GF(2) parity in coordinate space, and insert a canonicalization hook before interning; keep arena/ledger/BSP substrate and CNF-2 arity unchanged.
- Prior `in/in-9.md`: Intersection: CNF-2 fixed-arity candidate pipeline and strata discipline remain the semantic shell; SymDiff: in-10 shifts the payload semantics from tree adjacency to coordinate-based aggregation with parity cancellation and idempotent normalization; Wedge: candidate emission must operate on aggregates and canonicalize before key packing so parity equivalence is observable to interning.
- vs `prism_vm.py`: Intersection: interning as canonicalization boundary, BSP frontier cycling, and ledger append-only semantics; SymDiff: code has no CD coordinate representation, no pre-intern canonicalize hook, and treats ADD/MUL as tree rewrites; Wedge: add a coordinate field or derived cache, insert a canonicalize step before `_pack_key`, and update candidate emission to reflect aggregate semantics rather than structural rewrites.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: staged milestones and test-driven evolution; SymDiff: plan does not include CD coordinates, GF(2) aggregation, or canonicalization hooks; Wedge: introduce a new milestone or sub-track for CD-addressed aggregation with acceptance tests (coordinate parity equivalence, idempotent canon, shape-insensitive interning) while keeping CNF-2 invariants intact.

## in/in-11.md
- Core semantics: self-hosted CD coordinates represented as interned ledger nodes; new coordinate opcodes (`OP_COORD_ZERO`, `OP_COORD_ONE`, `OP_COORD_PAIR`); XOR/parity defined as structural rewrite over coordinate DAGs; coordinate equality becomes pointer equality; strata align with CD lift depth.
- Prior `in/in-10.md`: Intersection: CD coordinates and parity-normalized aggregation as the payload semantics; SymDiff: in-11 makes coordinates first-class ledger values and specifies concrete opcode-level representation plus XOR rewrite rules; Wedge: canonicalization can be realized by interning coordinate DAGs and using pointer equality, eliminating external coordinate metadata.
- vs `prism_vm.py`: Intersection: ledger interning and binary `(opcode, arg1, arg2)` rows; SymDiff: code lacks coordinate opcodes, coordinate constructors, and XOR rewrite semantics; Wedge: add `OP_COORD_*` and coordinate rewrite rules while keeping the ledger/candidate pipeline unchanged.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: CNF-2 and strata discipline; SymDiff: plan does not cover self-hosted coordinates or new opcodes; Wedge: add a milestone for coordinate opcodes and parity-cancellation tests (pointer-equality and idempotent XOR).

## in/in-12.md
- Core semantics: event-sourced + CQRS canonical interner; append-only intern log, deterministic read-model indexes, univalence via full-key equality (hash collisions are harmless), canonical key bytes built from canonical child representatives, and explicit warnings about key-width aliasing.
- Prior `in/in-11.md`: Intersection: canonicality and pointer-identity commitments; SymDiff: in-12 shifts focus from coordinate semantics to interner mechanics (event log, index/trie, union-find, key encoding); Wedge: self-hosted coordinates still require key-byte canonicalization to preserve univalence across structural equivalences.
- vs `prism_vm.py`: Intersection: interning boundary and packed-key indexing; SymDiff: code uses 16-bit child packing and full-array resort, lacks event log or full-key collision handling; Wedge: add key-width constraints or widening plus a canonicalize-before-pack step, and treat index arrays as a rebuildable view.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: canonicalization and ledger-centered execution; SymDiff: plan does not cover CQRS/event sourcing, collision handling, or key-width aliasing; Wedge: add a milestone for univalence guarantees (full-key compare/trie) and key encoding limits/tests.

## in/in-13.md
- Core semantics: reconciles in-10/11/12 into a single architecture: event-sourced canonical interner (ledger read model), CNF-2 nodes as the only payload, CD coordinates as interned objects (not metadata), aggregation as coordinate-space normalization, and univalence via full-key equality (hash as hint only).
- Prior `in/in-12.md`: Intersection: event-sourced interner, univalence, full-key equality, key-width safety; SymDiff: in-13 reintroduces CD coordinate semantics and makes them self-hosted CNF-2 objects with an acceptance checklist; Wedge: the interner must canonicalize coordinate objects before key packing so parity/aggregation normalization collapses to pointer identity.
- vs `prism_vm.py`: Intersection: ledger interning and `(opcode,arg1,arg2)` row shape; SymDiff: no coordinate opcodes, no event log/read model separation, and no explicit full-key equality or key-width guards; Wedge: add `OP_COORD_*`, coordinate normalization rules, and a canonicalize-before-pack boundary with full-key compare to uphold univalence.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: CNF-2/strata discipline and ledger as canonical store; SymDiff: plan lacks CD coordinate opcodes, event-sourced read model, and acceptance checklist; Wedge: add a milestone for coordinate self-hosting plus univalence tests (full-key equality, no truncation aliasing, parity cancellation).

## in/in-14.md
- Core semantics: reconcile the plan by separating semantic spine (Ledger + CNF-2 + intern) from performance spine (Arena scheduling), define a single univalence contract (key bytes + equality + no truncation), reorder milestones so M1-M3 are Ledger-only, make add rewrite symmetry explicit via commutative canonicalization, add cross-engine equivalence + small randomized tests, keep CQRS shape in-memory, and pin down slot-1 semantics.
- Prior `in/in-13.md`: Intersection: univalence and canonicalization commitments, CNF-2 as semantic payload, coordinate canonicalize-before-pack; SymDiff: in-14 focuses on plan governance (semantic vs performance layering, milestone reordering, test harnesses) and on making the univalence contract explicit; Wedge: M1-M3 must prove semantics in the Ledger-only engine before Arena work can be treated as denotation-invariant optimization.
- vs `prism_vm.py`: Intersection: ledger interning, candidate pipeline, commutative canonicalization in `intern_nodes`; SymDiff: code still mixes Arena and Ledger in one file, lacks a published univalence contract, cross-engine equivalence harness, and event-log style rebuild checks; Wedge: codify univalence and add equivalence/property tests so Arena scheduling can be validated as performance-only.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: Ledger-only M1-M3, explicit univalence contract, fixed-arity CNF-2 slot semantics, and cross-engine equivalence harness; SymDiff: plan still treats CQRS/event-log replay as optional and does not mandate a randomized generator or a formal add-rewrite spec table; Wedge: add a small seeded generator and an optional replay check (rebuild read-model from intern batches) to keep the CQRS shape auditable without persistent storage.

## in/in-15.md
- Core semantics: milestone-gated testing with pytest markers, a conftest gate that skips tests beyond the selected milestone, per-milestone pytest configs, and a VS Code workflow that selects the milestone via `PRISM_MILESTONE` in `.vscode/pytest.env`; no xfail as a correctness crutch.
- Prior `in/in-14.md`: Intersection: acceptance gates and harness-first testing; SymDiff: in-15 operationalizes gating in the IDE/CLI and makes milestone selection deterministic; Wedge: acceptance gates become enforceable tooling rather than a document-only policy.
- vs `prism_vm.py`: Intersection: none (workflow-only); SymDiff: in-15 defines test discovery and milestone selection rather than runtime semantics; Wedge: development workflow must ensure milestone gating does not mask regressions in the canonical engine.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: milestone acceptance gates and explicit test expectations; SymDiff: plan lacked a concrete VS Code testing workflow; Wedge: link the plan to in-15 and treat milestone selection as part of the test contract.

## in/in-16.md
- Core semantics: evaluator (Manifest/Arena) and canonicalizer (Ledger) are two projections of one system, linked by a total quotient map `q`; CNF-2 bounded emission; strata discipline (no within-tier refs); coordinates as interned canonical objects; scheduler freedom with denotation invariance; convergence defined by fixed points in canonical space.
- Prior `in/in-15.md`: Intersection: milestone gating and invariants-first workflow remain; SymDiff: in-16 shifts from test tooling to semantic unification (homomorphism + quotient map) and formalizes evaluator/canonicalizer roles; Wedge: milestone gates now need to validate the homomorphic projection, not just feature presence.
- vs `prism_vm.py`: Intersection: Ledger interning, CNF-2 nodes, coordinate opcodes, strata validators, scheduler invariance tests, and arena/ledger split; SymDiff: code lacks an explicit `q` projection from Manifest/Arena nodes into Ledger IDs and does not enforce evaluator->canonical projection as a first-class contract; Wedge: introduce a projection boundary (stratum commit) that maps provisional nodes to canonical IDs and use it to define denotation for arena/manifest paths.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: semantic spine vs performance spine, univalence contract, CNF-2 bounded emission, strata discipline, denotation invariance; SymDiff: plan treats ledger as canonical engine but does not explicitly define the quotient map or homomorphism laws; Wedge: add an explicit `q` projection spec (or "stratum commit" step) tying evaluator outputs to ledger interned identity so schedule invariance has a formal anchor.

## in/in-17.md
- Core semantics: CA-as-micro-ISA framing; interning as semantic compression (GC = dedup), no-copy axiom for sharing, binding without names, CNF-2 fixed arity as a local rule format, sorting/swizzling as renormalization, explicit OOM vs CORRUPT boundary, and a thermodynamic framing for finite phase space; damage/locality and hierarchical interning are performance tracks gated by denotation invariance.
- Prior `in/in-16.md`: Intersection: denotation after `q`, scheduler invariance, CNF-2 bounded emission, and coordinate normalization before interning; SymDiff: in-17 adds thermodynamic boundary (OOM vs CORRUPT), no-copy and alpha-equivalence obligations, damage-perimeter locality claims, and CA substrate emphasis; Wedge: elevate no-copy and OOM/CORRUPT distinction into explicit test obligations (alpha-equivalence, reuse scaling, and failure-mode classification).
- vs `prism_vm.py`: Intersection: ledger interning with CORRUPT vs OOM, coordinate opcodes, and layout-only sorting/swizzle; SymDiff: code lacks explicit damage tracking, CA lattice semantics, alpha-equivalence tests, or hierarchical/tiled interning; Wedge: add tests for alpha-equivalence and reuse scaling, and optionally instrument reuse/damage metrics without changing denotation.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: univalence contract, CORRUPT semantics, scheduler invariance, and `q` as a boundary concept; SymDiff: plan does not yet formalize the no-copy axiom, alpha-equivalence tests, or thermodynamic framing; Wedge: add no-copy + alpha-equivalence checks and document OOM vs CORRUPT as semantic vs resource failures in acceptance gates.

## in/glossary.md
- Core semantics: glossary defines commutation constraints for overloaded terms (BSP, CD, canonicalization, collapse, normalize, aggregate, scheduler, identity) and adds compiler-facing hazards (HLO size) with a batching/subsetting rule; mandates explicit axes plus `q`-based projection rules.
- Prior: none (glossary).
- vs `prism_vm.py`: Intersection: scheduler invariance tests and coordinate opcodes exist; SymDiff: code has no explicit `q` projection boundary or glossary-aligned axis tags; Wedge: add commutation-focused tests (schedule invariance after `q`, coord normalization before interning) or doc hooks so terminology stays operational.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: plan includes `q` projection, scheduler invariance, and coordinate normalization; SymDiff: plan does not yet cite the glossary or require commutation tests; Wedge: reference the glossary in plan/test obligations and treat commutation as a named acceptance requirement.
