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

## in/in-18.md
- Core semantics: Min(Prism) as a finite, semantics-preserving submodel for exhaustive verification; canonicalize raw states via `canon_state`, bound the ledger, enumerate all reachable semantic states, and lift invariants to larger systems via projection `π_K` and simulation; introduces D.5.O (projection commutation test obligation).
- Prior `in/in-17.md`: Intersection: univalence, CNF-2 bounded arity, strata discipline, and `q` as semantic boundary; SymDiff: in-18 adds finite-state abstraction, exhaustive enumeration, and formal refinement proof; Wedge: codify a canonical state representation and projection law so invariants can be proven once and lifted.
- vs `prism_vm.py`: Intersection: ledger canonicalization, CNF-2 cycles, strata validators, and stop-path semantics; SymDiff: code lacks `canon_state`, `π_K`, and Min(Prism) harness; Wedge: add a minimal `min_prism` test harness or offline tool to enumerate bounded semantics and enforce D.5.O.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: milestone gates and acceptance testing; SymDiff: plan does not define Min(Prism) or exhaustive verification; Wedge: add a milestone-adjacent verification track (e.g., m2-min) and document projection commutation as a required check.

## in/in-19.md
- Core semantics: defines the Prism site (C,J) with staging contexts `(n,s,t,τ)`, morphisms as forgetful maps, Arena as presheaf topos, `q` as associated sheaf functor, a Boolean Grothendieck topology enforcing gluing of compatible local sections, and a frozen read-model corollary via hyperstrata collapse.
- Prior `in/in-18.md`: Intersection: formalization of semantic core, `q` as gluing, and bounded/staged views; SymDiff: in-19 introduces explicit categorical structure (site, topology, sheaf condition); Wedge: align implementation staging indices and strata with the formal site and ensure `q` satisfies the sheaf condition.
- vs `prism_vm.py`: Intersection: BSP cycles, strata indices, and `q`-style projection via stratum commits; SymDiff: code does not encode the site or topology explicitly and has no micro-strata (`t`) or tile (`τ`) semantics; Wedge: if micro-strata or locality strata are introduced, define their visibility rules to match the site morphisms.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: `q` projection, strata discipline, denotation invariance, and hyperstrata visibility; SymDiff: plan still does not formalize the site/topology; Wedge: add a short formal note that identifies the staging contexts used in tests and how they satisfy the sheaf condition.

## in/in-20.md
- Core semantics: Ledger as a distinguished sheaf with canonical IDs as global elements; defines Cayley-Dickson pairing as an internal morphism and the semantic hyperlattice as a refinement preorder with join/meet induced by Arena accumulation + `q`.
- Prior `in/in-19.md`: Intersection: sheaf semantics, Boolean logic, and `q` as gluing; SymDiff: in-20 elevates Ledger to a specific sheaf object and introduces hyperlattice structure and CD pairing rules; Wedge: ensure CD pairing and interning preserve lattice laws and Booleanity after `q`.
- vs `prism_vm.py`: Intersection: ledger interning, coordinate opcodes, and candidate cycles; SymDiff: code does not model hyperlattice refinement or CD morphisms explicitly; Wedge: add invariants/tests for lattice-stable joins and CD-lifted operations when hyperpair semantics are introduced.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: Ledger-centric semantics and CNF-2 arity; SymDiff: plan does not mention hyperlattice or sheaf-theoretic Ledger role; Wedge: incorporate hyperlattice invariants as future acceptance criteria once CD lifting becomes active.

## in/in-21.md
- Core semantics: BSPᵗ vs BSPˢ as time/gauge split; `q` as irreversible coarse-graining; multiple entropies (Arena, canonical novelty, hyperoperator); adjunction/coherence discipline; hyperoperator stabilization framed as semantic consequences; Min(Prism) projection compatibility.
- Prior `in/in-20.md`: Intersection: `q`-centric sheaf semantics and Ledger as canonical meaning; SymDiff: in-21 adds gauge symmetry, entropy distinctions, and adjunction/coherence as mandatory semantic discipline; Wedge: treat BSPˢ invariance as a gauge law and ensure any semantic claim is stated after `q`.
- vs `prism_vm.py`: Intersection: denotation-invariance tests and `q`-style projection via stratum commits; SymDiff: code does not encode entropy or adjunction language; Wedge: avoid any observable dependence on BSPˢ layout and treat projection commutation as the coarse-graining law.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: denotation invariance, hyperstrata visibility, and BSPˢ gauge symmetry; SymDiff: plan still does not encode entropy/adjunction framing; Wedge: keep coarse-graining language and avoid overloading operational semantics with entropy claims.

## in/in-22.md
- Core semantics: canonical novelty as a semantic monotone; saturation implies representation stability (not termination); novelty is BSPˢ invariant; hyperoperator novelty saturates; Min(Prism) projection preserves novelty within bounds.
- Prior `in/in-21.md`: Intersection: semantic stability and gauge invariance; SymDiff: in-22 formalizes novelty as a monotone and defines saturation/fixed semantic universe; Wedge: clarify that stabilization is semantic only and does not imply runtime convergence.
- vs `prism_vm.py`: Intersection: interning by full-key equality implies monotone novelty; SymDiff: no explicit novelty counters or saturation checks exist; Wedge: optional tests can track novelty monotonicity or saturation without adding runtime instrumentation, and docs may need to note the hard `MAX_ID` cap (CORRUPT) as a precondition for any saturation claims.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: univalence, canonical interning, and novelty/fixed-point justification language; SymDiff: plan does not require novelty instrumentation or tests; Wedge: treat novelty as semantic rationale and keep any checks optional.

## in/in-23.md
- Core semantics: representation fixed points for interned hyperoperators; hyperoperator preorder with ascending-chain condition; CNF-2 local finiteness; fixed points are BSPˢ-invariant and BSPᵗ schedule-independent; Min(Prism) can detect fixed-point status; explicit non-claims about termination.
- Prior `in/in-22.md`: Intersection: canonical novelty saturation; SymDiff: in-23 lifts novelty to hyperoperator fixed points with an order-theoretic framing; Wedge: keep fixed point claims representational (not evaluative).
- vs `prism_vm.py`: Intersection: interning and CNF-2 candidate emission provide the machinery for closure; SymDiff: code does not expose hyperoperator sets or fixed-point checks; Wedge: optional Min(Prism) tests could validate hyperoperator closure without runtime changes, and fixed-point claims should be scoped to the pre-CORRUPT (id-cap) regime.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: canonical interning, CNF-2 arity, and hyperoperator fixed-point justification; SymDiff: plan does not define hyperoperator sets or fixed-point tests; Wedge: keep fixed points as doc-level justification unless Min(Prism) tests are added.

## in/in-24.md
- Core semantics: worked semantic sketch for TREE-class behavior; Arena explosion vs Ledger stabilization; representation fixed point after novelty saturation; explicit non-claims about termination/evaluation; hyperlattice connection; Min(Prism) finite witness principle.
- Prior `in/in-23.md`: Intersection: hyperoperator fixed points and novelty saturation; SymDiff: in-24 provides an illustrative mapping and explicit non-claims about termination/evaluation; Wedge: keep this as narrative justification rather than a new proof obligation.
- vs `prism_vm.py`: Intersection: interning and CNF-2 candidate emission support the sketch’s mechanics; SymDiff: no explicit TREE-class mapping exists in code; Wedge: if tests are added, they should be Min(Prism)-style structure checks, not runtime claims, and the sketch should acknowledge the semantic id hard-cap in current implementations.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: representation stability and fixed-point justification; SymDiff: plan does not include the worked TREE sketch; Wedge: treat the sketch as documentation, not a milestone gate.

## in/in-25.md
- Core semantics: explicit boundary between ordinal-indexed rewrite (termination via ordinal descent) and Prism’s canonical fixed points (representation stability without termination); rejects ordinal claims for Prism; emphasizes orthogonality.
- Prior `in/in-24.md`: Intersection: representation stability without termination; SymDiff: in-25 adds proof-theory boundary language and explicit non-claims about ordinal descent; Wedge: guard against reviewers inferring termination from fixed-point semantics.
- vs `prism_vm.py`: Intersection: no ordinal machinery exists in code; SymDiff: none (pure documentation boundary); Wedge: ensure docs and tests avoid ordinal/termination claims, and clarify that “finite representation” is conditional on the univalence cap (CORRUPT on overflow).
- vs `IMPLEMENTATION_PLAN.md`: Intersection: semantic focus on representation and ordinals as non-goals; SymDiff: plan does not elaborate the boundary details; Wedge: keep ordinal descent out of tests and acceptance criteria.

## in/in-26.md
- Core semantics: proof roadmap for Agda formalization of the semantic kernel (Sigma, canonical keys, univalence, q, BSPˢ gauge invariance, novelty monotonicity, finite closure, fixed points, Min(Prism), and explicit non-goals); focuses on semantic leverage per proof hour.
- Prior `in/in-25.md`: Intersection: boundary discipline and explicit non-claims; SymDiff: in-26 shifts from semantic claims to formalization priorities and module structure; Wedge: ensure the proof plan stays aligned with the glossary’s axes/commutation rules rather than implementation details.
- vs `prism_vm.py`: Intersection: semantic kernel concepts map to existing interning/q semantics; SymDiff: code does not expose Agda artifacts; Wedge: treat in-26 as verification roadmap, not an implementation requirement.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: Min(Prism), novelty/fixed-point justification, and non-goals (no ordinals/termination); SymDiff: plan does not yet capture the Agda module roadmap; Wedge: link the plan’s verification track to the in-26 proof milestones when formalization begins.

## in/glossary.md
- Core semantics: glossary defines commutation constraints for overloaded terms (BSP, CD, canonicalization, collapse, normalize, aggregate, scheduler, identity) and adds Arena/`q`/Ledger sheaf framing, hyperstrata, hyperlattice, gauge symmetry, canonical novelty, hyperoperator fixed points, adjunction/coherence discipline, and an entropy taxonomy; includes explicit pre-step immutability and `q` as coarse-graining.
- Prior: none (glossary).
- vs `prism_vm.py`: Intersection: scheduler invariance tests, coordinate opcodes, and strata discipline exist; SymDiff: code has no explicit `q` projection boundary or hyperstrata/hyperlattice invariants; Wedge: add commutation-focused tests (schedule invariance after `q`, coord normalization before interning) and document how `q` is enforced in code paths.
- vs `IMPLEMENTATION_PLAN.md`: Intersection: plan includes `q` projection, scheduler invariance, coordinate normalization, and hyperstrata visibility; SymDiff: plan still does not cite the glossary or require hyperlattice commutation tests; Wedge: reference the glossary in plan/test obligations and treat commutation as a named acceptance requirement.
