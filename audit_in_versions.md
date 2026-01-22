# Audit: in/in-*.md

Methodology:
- Tokenization: `[A-Za-z_][A-Za-z0-9_]*`, lowercased.
- Stopwords: English common words + Python keywords (see script).
- Sets are unique tokens after filtering.
- Wedge product: intersection of adjacent token bigram sets (ordered pairs).

## in/in-1.md
- Unique tokens: 306
- Unique bigrams: 723
- Prior version: none
- Compare: prism_vm.py
  - Intersection: 192 | top: jnp, self, int32, manifest, opcode, a1, ptr, a2, dtype, arg1, count, int
  - Symmetric difference: 750 (only in in/in-1.md: 114, only in prism_vm.py: 636)
    - Only in in/in-1.md: node, instruction, data, memory, optimized_ptr, pointer, python, telemetry, type, types, construct, construction
    - Only in prism_vm.py: arena, ledger, astype, size, none, morton, perm, block_size, uint32, oom, mode, enabled
  - Wedge product (bigram intersection): 396 | top: jnp int32, dtype jnp, a1 a2, self manifest, jnp ndarray, int self, jnp array, op a1, parse tokens, print f, manifest opcode, jnp zeros
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 100 | top: add, manifest, null, ptr, zero, suc, x, opcode, arg1, print, implement, new
  - Symmetric difference: 643 (only in in/in-1.md: 206, only in IMPLEMENTATION_PLAN.md: 437)
    - Only in in/in-1.md: self, jnp, int, op, ir, f, a1, a2, tokens, analysis, token, idx
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, pytest, program, rank, txt, arena, sort, swizzle, bsp, rewrite, tasks
  - Wedge product (bigram intersection): 22 | top: manifest opcode, suc zero, add zero, zero suc, x y, add suc, arg1 arg2, opcode arg1, op_add op_mul, op_zero op_suc, suc x, e g

## in/in-2.md
- Unique tokens: 338
- Unique bigrams: 771
- Prior version: in/in-1.md
  - Intersection: 190 | top: self, manifest, ptr, jnp, op, ir, a1, int, opcode, f, suc, a2
  - Symmetric difference: 264 (only in in/in-2.md: 148, only in in/in-1.md: 116)
    - Only in in/in-2.md: synthesis, memo, gpu, operand, registry, b, active_flag, addition, arg, child, compile, creates
    - Only in in/in-1.md: analysis, trace_cache, optimized_ptr, prism, rows, analyze_and_optimize, deduplication, exec_allocs, ir_allocs, mid_rows, optimization, telemetry
  - Wedge product (bigram intersection): 311 | top: self manifest, manifest opcode, print f, int self, a1 a2, dtype jnp, jnp int32, parse tokens, op a1, jnp ndarray, self parse, time perf_counter
- Compare: prism_vm.py
  - Intersection: 156 | top: jnp, self, int32, manifest, a1, opcode, a2, ptr, dtype, count, arg1, op
  - Symmetric difference: 854 (only in in/in-2.md: 182, only in prism_vm.py: 672)
    - Only in in/in-2.md: data, node, code, instruction, synthesis, pointer, python, memo, loop, operand, registry, b
    - Only in prism_vm.py: arena, ledger, astype, size, none, morton, perm, block_size, uint32, oom, mode, enabled
  - Wedge product (bigram intersection): 244 | top: jnp int32, dtype jnp, a1 a2, self manifest, jnp ndarray, jnp array, op a1, int self, parse tokens, jnp zeros, manifest opcode, self parse
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 91 | top: add, manifest, null, ptr, opcode, suc, data, zero, arg1, new, node, arg2
  - Symmetric difference: 693 (only in in/in-2.md: 247, only in IMPLEMENTATION_PLAN.md: 446)
    - Only in in/in-2.md: self, jnp, ir, a1, op, a2, f, cons, tokens, instruction, int, interpreter
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, pytest, program, rank, txt, arena, sort, swizzle, bsp, implement, implementation
  - Wedge product (bigram intersection): 13 | top: manifest opcode, arg1 arg2, opcode arg1, hash consing, op_add op_mul, op_zero op_suc, suc x, e g, instructions op_add, jax numpy, namedtuple opcode, normal form

## in/in-3.md
- Unique tokens: 284
- Unique bigrams: 690
- Prior version: in/in-2.md
  - Intersection: 150 | top: self, manifest, jnp, opcode, f, arg1, op, ptr, arg2, a1, int, print
  - Symmetric difference: 322 (only in in/in-3.md: 134, only in in/in-2.md: 188)
    - Only in in/in-3.md: func_name, trace_cache, _parse_expr, name, row, _alloc_memoized, _alloc_raw, alloc, _compile_kernel, allocate, candidate_idx, end_count
    - Only in in/in-2.md: ir, data, cons, node, interpreter, op_add, synthesis, vm, python, memo, execution, heap
  - Wedge product (bigram intersection): 203 | top: self manifest, print f, dtype jnp, int self, manifest opcode, arg1 arg2, jnp int32, jnp ndarray, opcode arg1, a1 a2, jnp zeros, max_rows dtype
- Compare: prism_vm.py
  - Intersection: 144 | top: jnp, self, int32, opcode, manifest, arg1, dtype, a1, a2, count, arg2, ptr
  - Symmetric difference: 824 (only in in/in-3.md: 140, only in prism_vm.py: 684)
    - Only in in/in-3.md: func_name, _parse_expr, name, row, _alloc_memoized, _alloc_raw, alloc, pointer, _compile_kernel, candidate_idx, compile, compiler
    - Only in prism_vm.py: arena, ledger, astype, size, none, morton, perm, block_size, uint32, oom, mode, enabled
  - Wedge product (bigram intersection): 193 | top: jnp int32, dtype jnp, self manifest, a1 a2, jnp ndarray, jnp array, int self, jnp zeros, print f, manifest opcode, b_ops b_a1, self parse
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 73 | top: add, expected, null, arg1, opcode, manifest, arg2, suc, zero, state, new, print
  - Symmetric difference: 675 (only in in/in-3.md: 211, only in IMPLEMENTATION_PLAN.md: 464)
    - Only in in/in-3.md: self, jnp, f, int, op, tokens, idx, cache, func_name, token, signature, _parse_expr
    - Only in IMPLEMENTATION_PLAN.md: tests, pytest, program, rank, txt, arena, sort, swizzle, bsp, implement, implementation, rewrite
  - Wedge product (bigram intersection): 14 | top: arg1 arg2, opcode arg1, suc zero, manifest opcode, x y, add suc, hash consing, op_zero op_suc, suc x, zero suc, arg2 active_count, e g

## in/in-4.md
- Unique tokens: 403
- Unique bigrams: 724
- Prior version: in/in-3.md
  - Intersection: 35 | top: self, manifest, jnp, active, memory, state, cache, jax, index, array, set, status
  - Symmetric difference: 617 (only in in/in-4.md: 368, only in in/in-3.md: 249)
    - Only in in/in-4.md: arena, morton, rank, sort, nodes, arenas, bit, graph, bsp, hierarchy, node, op_sort
    - Only in in/in-3.md: arg1, f, opcode, arg2, print, int, active_count, op, suc, tokens, zero, idx
  - Wedge product (bigram intersection): 3 | top: e g, jax array, x y
- Compare: prism_vm.py
  - Intersection: 62 | top: jnp, arena, self, manifest, astype, size, morton, set, perm, rank, x, state
  - Symmetric difference: 1107 (only in in/in-4.md: 341, only in prism_vm.py: 766)
    - Only in in/in-4.md: memory, arenas, graph, hierarchy, node, index, level, order, shatter, space, binary, bits
    - Only in prism_vm.py: int32, ledger, dtype, opcode, a1, a2, count, ptr, arg1, none, arg2, op
  - Wedge product (bigram intersection): 12 | top: astype jnp, arena rank, x y, perm jnp, argsort sort_key, jnp argsort, block size, rank astype, bsp arena, jit op_morton, morton astype, rank bit
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 107 | top: rank, arena, sort, bsp, morton, nodes, active, swizzle, manifest, arenas, implementation, tasks
  - Symmetric difference: 726 (only in in/in-4.md: 296, only in IMPLEMENTATION_PLAN.md: 430)
    - Only in in/in-4.md: memory, hierarchy, space, binary, bits, close, dormant, waiting, alternating, cache, connected, d
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, add, null, pytest, program, txt, implement, rewrite, baseline, cycle, objective
  - Wedge product (bigram intersection): 12 | top: sort key, bit rank, x y, e g, arena rank, block size, blocks e, interaction combinator, pointer swizzling, rank based, secondary sort, sort_key rank

## in/in-5.md
- Unique tokens: 360
- Unique bigrams: 639
- Prior version: in/in-4.md
  - Intersection: 131 | top: arena, memory, nodes, bsp, sort, rank, bit, active, arenas, morton, shatter, x
  - Symmetric difference: 501 (only in in/in-5.md: 229, only in in/in-4.md: 272)
    - Only in in/in-5.md: address, square, become, range, using, bottom, corresponds, cycle, dtype, new, partition, region
    - Only in in/in-4.md: manifest, op_sort, key, close, data, dormant, architecture, fluid, status, based, child, composition
  - Wedge product (bigram intersection): 37 | top: bsp tree, shatter effect, x y, bit rank, e g, alternating bsp, arena hierarchy, l1 arenas, spatial locality, active nodes, binary space, cache line
- Compare: prism_vm.py
  - Intersection: 77 | top: jnp, arena, dtype, morton, perm, uint32, x, rank, bsp, y, zeros_like, state
  - Symmetric difference: 1034 (only in in/in-5.md: 283, only in prism_vm.py: 751)
    - Only in in/in-5.md: memory, children, shatter, bits, address, l2, locality, tree, arenas, hierarchy, node, space
    - Only in prism_vm.py: int32, self, ledger, opcode, a1, a2, astype, count, manifest, ptr, size, arg1
  - Wedge product (bigram intersection): 23 | top: dtype jnp, jnp uint32, jnp zeros_like, arena rank, x y, arena perm, perm jnp, jnp argsort, arena inv_perm, y z, swizzle_2to1 x, z jnp
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 99 | top: arena, bsp, rank, sort, y, nodes, x, hot, swizzle, free, cycle, implementation
  - Symmetric difference: 699 (only in in/in-5.md: 261, only in IMPLEMENTATION_PLAN.md: 438)
    - Only in in/in-5.md: memory, children, z, jnp, bits, address, tree, hierarchy, space, square, alternating, become
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, add, null, pytest, program, txt, implement, rewrite, tasks, baseline, objective
  - Wedge product (bigram intersection): 14 | top: x y, new nodes, e g, hierarchical arenas, hot nodes, warm cold, arena rank, bit rank, blocks e, classify nodes, free region, jax numpy

## in/in-6.md
- Unique tokens: 397
- Unique bigrams: 596
- Prior version: in/in-5.md
  - Intersection: 143 | top: bsp, arena, memory, jax, nodes, x, bit, graph, shatter, address, children, y
  - Symmetric difference: 471 (only in in/in-6.md: 254, only in in/in-5.md: 217)
    - Only in in/in-6.md: hvm, latency, based, bottleneck, compute, graphblas, scan, superior, vs, alu, analysis, approach
    - Only in in/in-5.md: jnp, tree, become, blocks, range, bottom, coordinates, corresponds, dtype, execution, l1, new
  - Wedge product (bigram intersection): 55 | top: x y, alternating bsp, shatter effect, address swizzling, bit interleaving, bsp block, bsp layout, linear arena, address space, bit rank, cache line, e g
- Compare: prism_vm.py
  - Intersection: 61 | top: arena, size, morton, ops, idx, jax, rank, x, bsp, array, zero, y
  - Symmetric difference: 1103 (only in in/in-6.md: 336, only in prism_vm.py: 767)
    - Only in in/in-6.md: memory, graph, address, alternating, hvm, linear, reduction, shatter, standard, divergence, hierarchical, latency
    - Only in prism_vm.py: jnp, int32, self, ledger, dtype, opcode, a1, a2, astype, count, manifest, ptr
  - Wedge product (bigram intersection): 9 | top: x y, x x, cold free, hot warm, jax numpy, pallas triton, warm cold, bsp arena, swizzle x
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 92 | top: bsp, rank, arena, jax, sort, swizzle, implementation, implement, free, graph, nodes, rewrite
  - Symmetric difference: 750 (only in in/in-6.md: 305, only in IMPLEMENTATION_PLAN.md: 445)
    - Only in in/in-6.md: memory, gpu, address, alternating, hvm, standard, cache, divergence, latency, space, bottleneck, contiguous
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, add, null, pytest, program, txt, tasks, baseline, objective, new, stable
  - Wedge product (bigram intersection): 10 | top: x y, cold free, hot warm, warm cold, bit rank, jax numpy, e g, higher order, stream compaction, use jax

## in/in-7.md
- Unique tokens: 508
- Unique bigrams: 1135
- Prior version: in/in-6.md
  - Intersection: 128 | top: arena, rank, x, y, jax, free, nodes, sort, bsp, memory, bit, shatter
  - Symmetric difference: 649 (only in in/in-7.md: 380, only in in/in-6.md: 269)
    - Only in in/in-7.md: jnp, self, add, suc, a1, a2, count, new, opcode, mask_suc, ptr, arg1
    - Only in in/in-6.md: gpu, hvm, divergence, hierarchical, bottleneck, compute, graphblas, level, pallas, superior, vs, allows
  - Wedge product (bigram intersection): 23 | top: x y, address space, alternating bsp, free space, shatter effect, bit rank, bsp layout, calculate offsets, cold free, hot warm, jax numpy, warm cold
- Compare: prism_vm.py
  - Intersection: 180 | top: jnp, arena, self, int32, a1, a2, opcode, count, dtype, ptr, arg1, manifest
  - Symmetric difference: 976 (only in in/in-7.md: 328, only in prism_vm.py: 648)
    - Only in in/in-7.md: bits, node, shatter, space, child1_idx, fluid, root, every, linear, pointers, address, index
    - Only in prism_vm.py: ledger, astype, size, none, block_size, oom, enabled, shape, max_key, base_next, is_mul_suc, is_add_suc
  - Wedge product (bigram intersection): 254 | top: jnp int32, dtype jnp, jnp uint32, a1 a2, jnp ndarray, self arena, arena rank, jnp arange, jnp zeros_like, jnp array, mode drop, x y
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 168 | top: arena, add, rank, y, sort, null, x, free, suc, swizzle, nodes, new
  - Symmetric difference: 709 (only in in/in-7.md: 340, only in IMPLEMENTATION_PLAN.md: 369)
    - Only in in/in-7.md: jnp, self, a1, a2, bits, mask_suc, z, idx, op, space, child1_idx, dtype
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, pytest, program, txt, implement, tasks, baseline, objective, candidate, ordering, cnf
  - Wedge product (bigram intersection): 49 | top: x y, rank sort, sort swizzle, add suc, arena rank, new nodes, add zero, fluid arena, suc zero, y y, add x, suc add

## in/in-8.md
- Unique tokens: 1170
- Unique bigrams: 2604
- Prior version: in/in-7.md
  - Intersection: 186 | top: interaction, jax, arena, nodes, jnp, node, active, rank, graph, x, y, new
  - Symmetric difference: 1306 (only in in/in-8.md: 984, only in in/in-7.md: 322)
    - Only in in/in-8.md: n, https, tensor, port, ports, accessed, _1, january, delta, _2, gamma, matrix
    - Only in in/in-7.md: self, sort, suc, a1, a2, count, opcode, mask_suc, ptr, swizzle, arg1, hot
  - Wedge product (bigram intersection): 16 | top: new nodes, x y, new node, prefix sum, free nodes, must point, nodes free, structure arrays, active nodes, allocate new, execution kernel, jax jit
- Compare: prism_vm.py
  - Intersection: 112 | top: jnp, arena, dtype, set, size, jax, n, op, uint32, ops, shape, nodes
  - Symmetric difference: 1774 (only in in/in-8.md: 1058, only in prism_vm.py: 716)
    - Only in in/in-8.md: interaction, https, tensor, port, node, graph, ports, accessed, _1, b, january, delta
    - Only in prism_vm.py: int32, self, ledger, opcode, a1, a2, astype, count, manifest, ptr, arg1, none
  - Wedge product (bigram intersection): 5 | top: x y, shape n, jax jit, jax numpy, lax scan
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 175 | top: interaction, n, jax, tensor, port, nodes, add, node, active, graph, ports, e
  - Symmetric difference: 1357 (only in in/in-8.md: 995, only in IMPLEMENTATION_PLAN.md: 362)
    - Only in in/in-8.md: https, accessed, _1, b, january, logic, memory, _2, gamma, matrix, gpu, _0
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, null, pytest, program, txt, sort, swizzle, bsp, tasks, baseline, objective
  - Wedge product (bigram intersection): 21 | top: new nodes, active pair, rewrite rules, active pairs, annihilation commutation, e g, x y, commutation erasure, use jax, branchless interaction, interaction combinator, alloc rewire

## in/in-9.md
- Unique tokens: 798
- Unique bigrams: 2300
- Prior version: in/in-8.md
  - Intersection: 249 | top: n, rewrite, text, node, c, nodes, new, e, jax, step, two, b
  - Symmetric difference: 1470 (only in in/in-9.md: 549, only in in/in-8.md: 921)
    - Only in in/in-9.md: mathcal, l, f, mathrm, eager, cnf, stratum, candidate, prior, reviewer, bsp, candidates
    - Only in in/in-8.md: interaction, https, tensor, port, graph, ports, accessed, _1, january, delta, logic, memory
  - Wedge product (bigram intersection): 24 | top: new nodes, rewrite rules, control flow, e g, exactly one, k k, new node, x y, newly allocated, allocated nodes, creates nodes, execution makes
- Compare: prism_vm.py
  - Intersection: 98 | top: arena, ledger, f, a1, a2, count, manifest, op, n, set, size, stratum
  - Symmetric difference: 1430 (only in in/in-9.md: 700, only in prism_vm.py: 730)
    - Only in in/in-9.md: mathcal, l, rewrite, mathrm, text, c, eager, cnf, step, candidate, code, exactly
    - Only in prism_vm.py: jnp, int32, self, dtype, opcode, astype, ptr, arg1, none, arg2, morton, int
  - Wedge product (bigram intersection): 14 | top: x y, arena count, enabled candidates, zero suc, add suc, add zero, ledger intern_nodes, within tier, bsp ledger, validate stratum, canonical ids, f ledger
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 167 | top: rewrite, n, cnf, eager, tests, stratum, add, new, bsp, candidate, step, two
  - Symmetric difference: 1001 (only in in/in-9.md: 631, only in IMPLEMENTATION_PLAN.md: 370)
    - Only in in/in-9.md: mathcal, l, f, mathrm, text, c, identity, k, reviewer, one, a_1, a_2
    - Only in IMPLEMENTATION_PLAN.md: expected, null, pytest, program, rank, txt, swizzle, implement, tasks, cycle, objective, stable
  - Wedge product (bigram intersection): 45 | top: exactly two, candidate slots, cnf symmetric, fixed arity, prior strata, current code, normal form, slots per, x y, rewrite site, two slots, add suc

## in/in-10.md
- Unique tokens: 376
- Unique bigrams: 638
- Prior version: in/in-9.md
  - Intersection: 133 | top: rewrite, text, cnf, stratum, new, code, identity, candidate, semantics, two, bsp, canonicalization
  - Symmetric difference: 908 (only in in/in-10.md: 243, only in in/in-9.md: 665)
    - Only in in/in-10.md: cd, aggregation, coordinate, parity, tree, milestone, adjacency, aggregate, coordinates, cut, diff, elimination
    - Only in in/in-9.md: mathcal, l, f, n, mathrm, c, eager, step, exactly, prior, k, candidates
  - Wedge product (bigram intersection): 15 | top: normal form, address space, arena ledger, rewrite rules, core semantic, local rewrite, rewrite allocation, candidate emission, execution model, frontier propagation, one sentence, stratum invariant
- Compare: prism_vm.py
  - Intersection: 46 | top: arena, ledger, opcode, arg1, arg2, op, morton, ops, shape, x, rank, array
  - Symmetric difference: 1112 (only in in/in-10.md: 330, only in prism_vm.py: 782)
    - Only in in/in-10.md: cd, aggregation, coordinate, parity, tree, milestone, explicit, invariant, canonicalization, adjacency, aggregate, coordinates
    - Only in prism_vm.py: jnp, int32, self, dtype, a1, a2, astype, count, manifest, ptr, size, set
  - Wedge product (bigram intersection): 8 | top: arg1 arg2, opcode arg1, op arg1, tier references, within tier, x x, op_add op_mul, arg2 op
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 95 | top: rank, arena, bsp, coordinate, sort, rewrite, new, swizzle, explicit, cnf, invariant, local
  - Symmetric difference: 723 (only in in/in-10.md: 281, only in IMPLEMENTATION_PLAN.md: 442)
    - Only in in/in-10.md: cd, aggregation, parity, tree, adjacency, aggregate, coordinates, cut, elimination, idempotent, space, address
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, add, null, pytest, program, txt, implement, implementation, tasks, baseline, cycle
  - Wedge product (bigram intersection): 11 | top: arg1 arg2, rank sort, opcode arg1, sort swizzle, normal form, op_add op_mul, rewrite rules, candidate emission, semantics explicit, swizzle morton, within tier

## in/in-11.md
- Unique tokens: 291
- Unique bigrams: 449
- Prior version: in/in-10.md
  - Intersection: 117 | top: cd, coordinate, aggregation, parity, xor, milestone, coordinates, local, tree, x, new, ledger
  - Symmetric difference: 433 (only in in/in-11.md: 174, only in in/in-10.md: 259)
    - Only in in/in-11.md: op_coord_zero, exactly, interned, program, self, vm, already, arithmetic, coord_ptr, p, recursion, structural
    - Only in in/in-10.md: explicit, frontier, adjacency, aggregate, diff, addressed, aggregates, changes, code, form, key, normalization
  - Wedge product (bigram intersection): 30 | top: cd coordinate, cayley dickson, cd coordinates, cut elimination, address space, arg1 arg2, canonicalization hook, equality pointer, finitely observable, local rewrite, pointer equality, x x
- Compare: prism_vm.py
  - Intersection: 46 | top: int32, self, ledger, a1, opcode, a2, size, arg1, op, arg2, ops, vm
  - Symmetric difference: 1027 (only in in/in-11.md: 245, only in prism_vm.py: 782)
    - Only in in/in-11.md: coordinate, xor, cd, text, coordinates, equality, parity, pointer, exactly, interned, milestone, program
    - Only in prism_vm.py: jnp, arena, dtype, astype, count, manifest, ptr, set, none, morton, int, active_count
  - Wedge product (bigram intersection): 7 | top: a1 a2, op a1, arg1 arg2, opcode arg1, x x, within tier, a2 op
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 73 | top: program, coordinate, bsp, rewrite, new, x, cnf, nodes, pointer, strata, vm, exactly
  - Symmetric difference: 682 (only in in/in-11.md: 218, only in IMPLEMENTATION_PLAN.md: 464)
    - Only in in/in-11.md: xor, cd, text, coordinates, equality, ledger, op_coord_zero, parity, depth, interned, self, already
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, add, null, pytest, rank, txt, arena, sort, swizzle, implement, implementation
  - Wedge product (bigram intersection): 6 | top: arg1 arg2, opcode arg1, rewrite rules, new opcodes, tier refs, within tier

## in/in-12.md
- Unique tokens: 352
- Unique bigrams: 633
- Prior version: in/in-11.md
  - Intersection: 64 | top: hash, canonical, equality, rewrite, op, x, intern, ledger, new, a1, a2, exactly
  - Symmetric difference: 515 (only in in/in-12.md: 288, only in in/in-11.md: 227)
    - Only in in/in-12.md: key, id, bytes, collision, full, ambiguity, collisions, event, index, log, deterministic, bucket
    - Only in in/in-11.md: coordinate, xor, cd, text, coordinates, op_coord_zero, parity, depth, interned, local, milestone, program
  - Wedge product (bigram intersection): 3 | top: a1 a2, op a1, gives literal
- Compare: prism_vm.py
  - Intersection: 52 | top: ledger, a1, a2, count, set, op, ops, key, x, array, args, jax
  - Symmetric difference: 1076 (only in in/in-12.md: 300, only in prism_vm.py: 776)
    - Only in in/in-12.md: id, hash, bytes, collision, ambiguity, collisions, event, index, log, deterministic, bucket, rewrite
    - Only in prism_vm.py: jnp, int32, arena, self, dtype, opcode, astype, manifest, ptr, size, arg1, none
  - Wedge product (bigram intersection): 2 | top: a1 a2, op a1
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 91 | top: key, add, hash, rewrite, sort, free, full, implementation, canonical, new, index, intern
  - Symmetric difference: 707 (only in in/in-12.md: 261, only in IMPLEMENTATION_PLAN.md: 446)
    - Only in in/in-12.md: id, bytes, collision, ambiguity, collisions, event, log, deterministic, bucket, equality, gives, one
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, null, pytest, program, rank, txt, arena, swizzle, bsp, implement, tasks
  - Wedge product (bigram intersection): 1 | top: e g

## in/in-13.md
- Unique tokens: 258
- Unique bigrams: 414
- Prior version: in/in-12.md
  - Intersection: 133 | top: key, id, hash, canonical, equality, full, ambiguity, collision, collisions, event, univalence, bytes
  - Symmetric difference: 344 (only in in/in-13.md: 125, only in in/in-12.md: 219)
    - Only in in/in-13.md: coordinate, coordinates, cd, aggregation, cnf, interned, normalization, parity, canon, objects, space, acceptance
    - Only in in/in-12.md: gives, children, different, h, tables, two, arrays, derived, encode, events, hashing, order
  - Wedge product (bigram intersection): 63 | top: full key, collision free, key bytes, a1 a2, hash collisions, op a1, read model, canonicality without, child id, event log, event sourced, key equality
- Compare: prism_vm.py
  - Intersection: 41 | top: int32, arena, self, ledger, a1, opcode, a2, arg1, op, arg2, shape, x
  - Symmetric difference: 1004 (only in in/in-13.md: 217, only in prism_vm.py: 787)
    - Only in in/in-13.md: coordinate, equality, coordinates, cd, id, univalence, aggregation, becomes, hash, ambiguity, cnf, collisions
    - Only in prism_vm.py: jnp, dtype, astype, count, manifest, ptr, size, set, none, morton, int, active_count
  - Wedge product (bigram intersection): 7 | top: a1 a2, op a1, arg1 arg2, opcode arg1, a2 ledger, canonical ledger, local rewrites
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 73 | top: arena, coordinate, rewrite, implementation, cnf, free, key, nodes, new, univalence, full, pointer
  - Symmetric difference: 649 (only in in/in-13.md: 185, only in IMPLEMENTATION_PLAN.md: 464)
    - Only in in/in-13.md: equality, coordinates, cd, id, ledger, aggregation, becomes, ambiguity, collisions, event, collision, interned
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, add, null, pytest, program, rank, txt, sort, swizzle, bsp, implement
  - Wedge product (bigram intersection): 4 | top: arg1 arg2, opcode arg1, e g, normal form
