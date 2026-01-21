# Audit: in/in-*.md

Methodology:
- Tokenization: `[A-Za-z_][A-Za-z0-9_]*`, lowercased.
- Stopwords: English common words + Python keywords (see script).
- Sets are unique tokens after filtering.
- Wedge product: intersection of adjacent token bigram sets (ordered pairs).

## in/in-1.md
- Unique tokens: 313
- Unique bigrams: 737
- Prior version: none
- Compare: prism_vm.py
  - Intersection: 197 | top: jnp, self, manifest, int32, ptr, where, dtype, active_count, op, int, opcode, a1
  - Symmetric difference: 477 (only in in/in-1.md: 116, only in prism_vm.py: 361)
    - Only in in/in-1: node, instruction, data, memory, optimized_ptr, pointer, python, s, now, telemetry, type, types
    - Only in prism_vm.py: arena, uint32, none, morton, size, block_size, ledger, perm, astype, rank, mode, arange
  - Wedge product (bigram intersection): 426 | top: jnp int32, jnp where, dtype jnp, self manifest, a1 a2, jnp ndarray, parse tokens, print f, jnp array, int self, manifest opcode, op a1
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 95 | top: add, manifest, null, ptr, zero, suc, x, opcode, print, arg1, kernel, static
  - Symmetric difference: 581 (only in in/in-1.md: 218, only in IMPLEMENTATION_PLAN.md: 363)
    - Only in in/in-1: self, jnp, int, op, ir, f, a1, a2, tokens, analysis, token, idx
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, pytest, program, rank, arena, txt, sort, cycle, bsp, tasks, free
  - Wedge product (bigram intersection): 21 | top: manifest opcode, suc zero, add zero, zero suc, add suc, x y, arg1 arg2, opcode arg1, op_add op_mul, op_zero op_suc, suc x, e g

## in/in-2.md
- Unique tokens: 343
- Unique bigrams: 778
- Prior version: in/in-1.md
  - Intersection: 191 | top: self, manifest, ptr, jnp, op, ir, a1, int, opcode, f, suc, a2
  - Symmetric difference: 274 (only in in/in-2.md: 152, only in in/in-1.md: 122)
    - Only in in/in-2: synthesis, memo, gpu, operand, registry, b, active_flag, addition, arg, child, compile, creates
    - Only in in/in-1: analysis, trace_cache, optimized_ptr, prism, rows, s, analyze_and_optimize, deduplication, exec_allocs, ir_allocs, mid_rows, optimization
  - Wedge product (bigram intersection): 311 | top: self manifest, manifest opcode, print f, int self, a1 a2, dtype jnp, jnp int32, parse tokens, op a1, jnp ndarray, self parse, time perf_counter
- Compare: prism_vm.py
  - Intersection: 157 | top: jnp, self, manifest, int32, ptr, where, dtype, op, opcode, active_count, a1, a2
  - Symmetric difference: 587 (only in in/in-2.md: 186, only in prism_vm.py: 401)
    - Only in in/in-2: data, node, code, instruction, synthesis, pointer, python, memo, loop, operand, registry, b
    - Only in prism_vm.py: arena, uint32, none, morton, size, block_size, ledger, perm, astype, rank, mode, arange
  - Wedge product (bigram intersection): 269 | top: jnp int32, jnp where, dtype jnp, self manifest, a1 a2, jnp ndarray, parse tokens, jnp array, self parse, op a1, print f, manifest opcode
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 84 | top: add, manifest, null, ptr, opcode, suc, data, zero, node, arg1, arg2, at
  - Symmetric difference: 633 (only in in/in-2.md: 259, only in IMPLEMENTATION_PLAN.md: 374)
    - Only in in/in-2: self, jnp, ir, a1, op, a2, f, cons, tokens, instruction, int, interpreter
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, pytest, program, rank, arena, txt, sort, cycle, implement, implementation, bsp
  - Wedge product (bigram intersection): 12 | top: manifest opcode, arg1 arg2, opcode arg1, hash consing, op_add op_mul, op_zero op_suc, suc x, e g, instructions op_add, jax numpy, namedtuple opcode, null ptr

## in/in-3.md
- Unique tokens: 287
- Unique bigrams: 693
- Prior version: in/in-2.md
  - Intersection: 151 | top: self, manifest, jnp, opcode, f, arg1, op, ptr, arg2, a1, int, print
  - Symmetric difference: 328 (only in in/in-3.md: 136, only in in/in-2.md: 192)
    - Only in in/in-3: func_name, trace_cache, _parse_expr, name, row, _alloc_memoized, _alloc_raw, alloc, _compile_kernel, allocate, candidate_idx, end_count
    - Only in in/in-2: ir, data, cons, node, interpreter, op_add, synthesis, vm, python, memo, execution, heap
  - Wedge product (bigram intersection): 206 | top: self manifest, print f, dtype jnp, int self, manifest opcode, arg1 arg2, jnp int32, jnp ndarray, opcode arg1, a1 a2, jnp zeros, max_rows dtype
- Compare: prism_vm.py
  - Intersection: 145 | top: jnp, self, manifest, int32, where, dtype, opcode, active_count, arg1, op, ptr, int
  - Symmetric difference: 555 (only in in/in-3.md: 142, only in prism_vm.py: 413)
    - Only in in/in-3: func_name, _parse_expr, name, row, _alloc_memoized, _alloc_raw, alloc, pointer, _compile_kernel, candidate_idx, compile, compiler
    - Only in prism_vm.py: arena, uint32, none, morton, size, block_size, ledger, perm, astype, rank, vm, mode
  - Wedge product (bigram intersection): 215 | top: jnp int32, jnp where, dtype jnp, self manifest, jnp ndarray, print f, a1 a2, jnp array, jnp zeros, int self, self parse, manifest opcode
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 70 | top: add, expected, null, arg1, manifest, opcode, arg2, suc, zero, state, print, active_count
  - Symmetric difference: 605 (only in in/in-3.md: 217, only in IMPLEMENTATION_PLAN.md: 388)
    - Only in in/in-3: self, jnp, f, int, op, tokens, idx, cache, func_name, token, signature, _parse_expr
    - Only in IMPLEMENTATION_PLAN.md: tests, pytest, program, rank, arena, txt, sort, cycle, implement, implementation, bsp, tasks
  - Wedge product (bigram intersection): 14 | top: arg1 arg2, opcode arg1, suc zero, manifest opcode, x y, add suc, hash consing, op_zero op_suc, suc x, zero suc, arg2 active_count, e g

## in/in-4.md
- Unique tokens: 409
- Unique bigrams: 736
- Prior version: in/in-3.md
  - Intersection: 38 | top: self, manifest, jnp, active, memory, state, cache, jax, at, index, array, set
  - Symmetric difference: 620 (only in in/in-4.md: 371, only in in/in-3.md: 249)
    - Only in in/in-4: arena, morton, rank, sort, nodes, arenas, bit, graph, bsp, hierarchy, node, op_sort
    - Only in in/in-3: arg1, f, opcode, arg2, print, int, active_count, op, suc, tokens, zero, idx
  - Wedge product (bigram intersection): 3 | top: e g, jax array, x y
- Compare: prism_vm.py
  - Intersection: 58 | top: jnp, arena, self, morton, manifest, where, size, perm, rank, astype, at, set
  - Symmetric difference: 851 (only in in/in-4.md: 351, only in prism_vm.py: 500)
    - Only in in/in-4: memory, arenas, graph, hierarchy, node, index, key, level, shatter, space, within, binary
    - Only in prism_vm.py: int32, dtype, uint32, none, active_count, block_size, ptr, ledger, opcode, op, a1, a2
  - Wedge product (bigram intersection): 12 | top: astype jnp, arena rank, x y, perm jnp, argsort sort_key, jnp argsort, block size, rank astype, bsp arena, jit op_morton, morton astype, rank bit
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 101 | top: arena, rank, sort, morton, active, bsp, nodes, manifest, arenas, graph, implementation, bit
  - Symmetric difference: 665 (only in in/in-4.md: 308, only in IMPLEMENTATION_PLAN.md: 357)
    - Only in in/in-4: memory, hierarchy, space, within, binary, bits, but, close, dormant, waiting, alternating, cache
    - Only in IMPLEMENTATION_PLAN.md: tests, add, expected, null, pytest, program, txt, cycle, implement, objective, stable, suc
  - Wedge product (bigram intersection): 14 | top: sort key, bit rank, x y, e g, arena rank, at index, block size, blocks e, graph structure, interaction combinator, pointer swizzling, rank based

## in/in-5.md
- Unique tokens: 362
- Unique bigrams: 641
- Prior version: in/in-4.md
  - Intersection: 135 | top: arena, memory, nodes, bsp, sort, rank, bit, active, arenas, morton, shatter, x
  - Symmetric difference: 501 (only in in/in-5.md: 227, only in in/in-4.md: 274)
    - Only in in/in-5: address, square, become, range, bottom, corresponds, cycle, dtype, partition, region, top, which
    - Only in in/in-4: manifest, op_sort, key, close, data, dormant, architecture, fluid, status, based, child, composition
  - Wedge product (bigram intersection): 40 | top: bsp tree, shatter effect, x y, bit rank, e g, alternating bsp, arena hierarchy, l1 arenas, node at, spatial locality, active nodes, at index
- Compare: prism_vm.py
  - Intersection: 71 | top: jnp, arena, where, dtype, uint32, morton, perm, at, x, rank, y, bsp
  - Symmetric difference: 778 (only in in/in-5.md: 291, only in prism_vm.py: 487)
    - Only in in/in-5: memory, children, shatter, bits, address, l2, locality, tree, arenas, hierarchy, node, space
    - Only in prism_vm.py: self, int32, manifest, none, size, active_count, block_size, ptr, ledger, opcode, op, a1
  - Wedge product (bigram intersection): 24 | top: jnp uint32, dtype jnp, arena rank, x y, jnp zeros_like, arena perm, perm jnp, jnp argsort, arena inv_perm, y z, swizzle_2to1 x, z jnp
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 90 | top: arena, bsp, rank, sort, y, x, hot, nodes, cycle, free, block, implementation
  - Symmetric difference: 640 (only in in/in-5.md: 272, only in IMPLEMENTATION_PLAN.md: 368)
    - Only in in/in-5: memory, children, z, jnp, bits, address, tree, hierarchy, space, square, alternating, become
    - Only in IMPLEMENTATION_PLAN.md: tests, add, expected, null, pytest, program, txt, implement, tasks, objective, ordering, suc
  - Wedge product (bigram intersection): 16 | top: x y, hot nodes, at index, e g, hierarchical arenas, warm cold, arena rank, at indices, bit rank, blocks e, classify nodes, free region

## in/in-6.md
- Unique tokens: 400
- Unique bigrams: 602
- Prior version: in/in-5.md
  - Intersection: 146 | top: bsp, arena, memory, jax, nodes, x, bit, graph, shatter, address, children, y
  - Symmetric difference: 470 (only in in/in-6.md: 254, only in in/in-5.md: 216)
    - Only in in/in-6: hvm, latency, s, based, bottleneck, compute, graphblas, scan, superior, alu, analysis, approach
    - Only in in/in-5: jnp, tree, become, blocks, range, bottom, but, coordinates, corresponds, dtype, execution, l1
  - Wedge product (bigram intersection): 56 | top: x y, alternating bsp, shatter effect, address swizzling, at index, bit interleaving, bsp block, bsp layout, linear arena, address space, bit rank, cache line
- Compare: prism_vm.py
  - Intersection: 58 | top: arena, where, morton, size, at, idx, rank, jax, x, bsp, zero, array
  - Symmetric difference: 842 (only in in/in-6.md: 342, only in prism_vm.py: 500)
    - Only in in/in-6: memory, graph, address, alternating, hvm, layout, linear, reduction, shatter, standard, divergence, hierarchical
    - Only in prism_vm.py: jnp, self, int32, dtype, uint32, manifest, none, active_count, block_size, ptr, ledger, perm
  - Wedge product (bigram intersection): 9 | top: x y, x x, cold free, hot warm, jax numpy, pallas triton, warm cold, bsp arena, swizzle x
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 84 | top: bsp, rank, arena, jax, sort, implementation, free, graph, cycle, implement, swizzle, bit
  - Symmetric difference: 690 (only in in/in-6.md: 316, only in IMPLEMENTATION_PLAN.md: 374)
    - Only in in/in-6: memory, gpu, address, alternating, hvm, standard, cache, divergence, latency, s, space, bottleneck
    - Only in IMPLEMENTATION_PLAN.md: tests, add, expected, null, pytest, program, txt, tasks, objective, stable, ordering, suc
  - Wedge product (bigram intersection): 11 | top: x y, cold free, hot warm, warm cold, at index, bit rank, jax numpy, e g, graph structure, higher order, stream compaction

## in/in-7.md
- Unique tokens: 514
- Unique bigrams: 1159
- Prior version: in/in-6.md
  - Intersection: 129 | top: arena, rank, x, y, jax, at, free, nodes, where, sort, bsp, memory
  - Symmetric difference: 656 (only in in/in-7.md: 385, only in in/in-6.md: 271)
    - Only in in/in-7: jnp, self, add, suc, a1, a2, count, opcode, mask_suc, ptr, arg1, op
    - Only in in/in-6: gpu, hvm, divergence, hierarchical, bottleneck, compute, graphblas, level, pallas, superior, allows, analysis
  - Wedge product (bigram intersection): 25 | top: x y, address space, alternating bsp, at index, free space, shatter effect, bit rank, bsp layout, calculate offsets, cold free, hot warm, jax numpy
- Compare: prism_vm.py
  - Intersection: 173 | top: jnp, arena, self, where, int32, dtype, rank, uint32, manifest, ptr, morton, a1
  - Symmetric difference: 726 (only in in/in-7.md: 341, only in prism_vm.py: 385)
    - Only in in/in-7: bits, node, shatter, space, child1_idx, fluid, root, every, linear, now, pointers, address
    - Only in prism_vm.py: none, size, block_size, ledger, astype, l1_block_size, inp, shape, l2_block_size, active_blocks, arg, sort_key
  - Wedge product (bigram intersection): 265 | top: jnp where, jnp int32, dtype jnp, jnp uint32, arena rank, self arena, jnp ndarray, a1 a2, jnp arange, x y, parse tokens, jnp array
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 155 | top: arena, add, rank, y, sort, null, x, free, suc, at, nodes, cycle
  - Symmetric difference: 662 (only in in/in-7.md: 359, only in IMPLEMENTATION_PLAN.md: 303)
    - Only in in/in-7: jnp, self, where, a1, a2, bits, mask_suc, z, idx, op, space, child1_idx
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, pytest, program, txt, implement, tasks, objective, ordering, baseline, keep, md
  - Wedge product (bigram intersection): 49 | top: x y, sort swizzle, add suc, arena rank, rank sort, add zero, fluid arena, suc zero, y y, add x, suc add, suc x

## in/in-8.md
- Unique tokens: 1175
- Unique bigrams: 2634
- Prior version: in/in-7.md
  - Intersection: 189 | top: interaction, jax, arena, nodes, jnp, node, active, at, rank, graph, x, y
  - Symmetric difference: 1311 (only in in/in-8.md: 986, only in in/in-7.md: 325)
    - Only in in/in-8: n, https, tensor, port, ports, accessed, _1, january, delta, _2, gamma, matrix
    - Only in in/in-7: self, sort, suc, a1, a2, count, opcode, mask_suc, ptr, swizzle, arg1, hot
  - Wedge product (bigram intersection): 19 | top: jnp where, x y, prefix sum, at index, free nodes, must now, nodes free, now point, structure arrays, active nodes, allocate nodes, at x
- Compare: prism_vm.py
  - Intersection: 96 | top: jnp, arena, where, n, dtype, uint32, jax, size, at, op, set, nodes
  - Symmetric difference: 1541 (only in in/in-8.md: 1079, only in prism_vm.py: 462)
    - Only in in/in-8: interaction, https, tensor, port, node, graph, ports, accessed, _1, b, january, delta
    - Only in prism_vm.py: self, int32, manifest, none, morton, active_count, block_size, ptr, ledger, perm, opcode, a1
  - Wedge product (bigram intersection): 7 | top: jnp where, x y, shape n, jax jax, jax jit, jax numpy, lax scan
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 149 | top: interaction, n, jax, tensor, port, nodes, add, node, graph, active, ports, e
  - Symmetric difference: 1335 (only in in/in-8.md: 1026, only in IMPLEMENTATION_PLAN.md: 309)
    - Only in in/in-8: https, accessed, _1, b, january, logic, memory, _2, gamma, matrix, gpu, _0
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, null, pytest, program, txt, sort, bsp, tasks, objective, stable, hot
  - Wedge product (bigram intersection): 19 | top: active pair, rewrite rules, active pairs, annihilation commutation, e g, x y, commutation erasure, branchless interaction, interaction combinator, alloc rewire, allocation allocation, at index

## in/in-9.md
- Unique tokens: 813
- Unique bigrams: 2351
- Prior version: in/in-8.md
  - Intersection: 256 | top: n, rewrite, text, node, c, nodes, e, jax, step, two, b, exactly
  - Symmetric difference: 1476 (only in in/in-9.md: 557, only in in/in-8.md: 919)
    - Only in in/in-9: mathcal, l, f, mathrm, eager, cnf, stratum, candidate, prior, reviewer, bsp, candidates
    - Only in in/in-8: interaction, https, tensor, port, graph, ports, accessed, _1, january, delta, logic, memory
  - Wedge product (bigram intersection): 24 | top: rewrite rules, control flow, e g, exactly one, k k, x y, exactly once, newly allocated, allocate nodes, allocated nodes, creates nodes, during execution
- Compare: prism_vm.py
  - Intersection: 82 | top: arena, f, where, n, ledger, op, manifest, size, a1, at, a2, set
  - Symmetric difference: 1207 (only in in/in-9.md: 731, only in prism_vm.py: 476)
    - Only in in/in-9: mathcal, l, rewrite, mathrm, text, c, eager, cnf, stratum, step, candidate, code
    - Only in prism_vm.py: jnp, self, int32, dtype, uint32, none, morton, active_count, block_size, ptr, perm, opcode
  - Wedge product (bigram intersection): 13 | top: x y, arena count, zero suc, add suc, add zero, ledger intern_nodes, bsp ledger, nodes where, canonical ids, don t, f ledger, one bsp
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 120 | top: rewrite, n, tests, add, step, bsp, code, model, node, semantics, arena, nodes
  - Symmetric difference: 1031 (only in in/in-9.md: 693, only in IMPLEMENTATION_PLAN.md: 338)
    - Only in in/in-9: mathcal, l, f, mathrm, text, c, eager, cnf, stratum, identity, candidate, exactly
    - Only in IMPLEMENTATION_PLAN.md: expected, null, pytest, program, rank, txt, cycle, implement, tasks, objective, stable, hot
  - Wedge product (bigram intersection): 12 | top: x y, add suc, rewrite rules, suc x, add zero, root pointer, zero suc, zero y, arena manifest, e g, node boundary, optional local

## in/in-10.md
- Unique tokens: 375
- Unique bigrams: 635
- Prior version: in/in-9.md
  - Intersection: 131 | top: rewrite, text, cnf, stratum, new, code, identity, candidate, semantics, bsp, canonicalization, reviewer
  - Symmetric difference: 912 (only in in/in-10.md: 244, only in in/in-9.md: 668)
    - Only in in/in-10: cd, aggregation, coordinate, parity, tree, milestone, adjacency, aggregate, coordinates, cut, diff, elimination
    - Only in in/in-9: mathcal, l, f, n, mathrm, c, eager, step, exactly, prior, k, candidates
  - Wedge product (bigram intersection): 11 | top: normal form, address space, arena ledger, rewrite rules, core semantic, local rewrite, rewrite allocation, candidate emission, execution model, frontier propagation, stratum invariant
- Compare: prism_vm.py
  - Intersection: 41 | top: arena, ledger, opcode, arg1, morton, arg2, op, x, rank, shape, op_add, ops
  - Symmetric difference: 884 (only in in/in-10.md: 334, only in prism_vm.py: 550)
    - Only in in/in-10: cd, aggregation, coordinate, parity, tree, milestone, explicit, invariant, canonicalization, adjacency, aggregate, coordinates
    - Only in prism_vm.py: jnp, int32, self, dtype, size, uint32, manifest, none, perm, active_count, block_size, ptr
  - Wedge product (bigram intersection): 7 | top: arg1 arg2, opcode arg1, op arg1, x x, arg2 op, op_add op_mul, tier references
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 93 | top: rank, arena, bsp, coordinate, sort, rewrite, new, swizzle, explicit, cnf, invariant, local
  - Symmetric difference: 726 (only in in/in-10.md: 282, only in IMPLEMENTATION_PLAN.md: 444)
    - Only in in/in-10: cd, aggregation, parity, tree, adjacency, aggregate, coordinates, cut, elimination, idempotent, space, address
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, add, null, pytest, program, txt, implement, implementation, tasks, baseline, cycle
  - Wedge product (bigram intersection): 10 | top: arg1 arg2, rank sort, opcode arg1, sort swizzle, normal form, op_add op_mul, rewrite rules, candidate emission, semantics explicit, swizzle morton

## in/in-11.md
- Unique tokens: 288
- Unique bigrams: 451
- Prior version: in/in-10.md
  - Intersection: 116 | top: cd, coordinate, aggregation, parity, xor, milestone, coordinates, local, tree, x, new, ledger
  - Symmetric difference: 431 (only in in/in-11.md: 172, only in in/in-10.md: 259)
    - Only in in/in-11: op_coord_zero, exactly, interned, program, self, vm, already, arithmetic, coord_ptr, p, recursion, structural
    - Only in in/in-10: explicit, frontier, adjacency, aggregate, diff, addressed, aggregates, changes, code, form, key, normalization
  - Wedge product (bigram intersection): 31 | top: cd coordinate, cayley dickson, cd coordinates, cut elimination, address space, arg1 arg2, canonicalization hook, equality pointer, finitely observable, local rewrite, pointer equality, x x
- Compare: prism_vm.py
  - Intersection: 40 | top: int32, self, ledger, size, opcode, arg1, arg2, a1, a2, op, vm, x
  - Symmetric difference: 799 (only in in/in-11.md: 248, only in prism_vm.py: 551)
    - Only in in/in-11: coordinate, xor, cd, now, text, coordinates, equality, op_coord_zero, parity, pointer, depth, exactly
    - Only in prism_vm.py: jnp, arena, dtype, uint32, manifest, none, perm, morton, active_count, block_size, ptr, astype
  - Wedge product (bigram intersection): 5 | top: a1 a2, op a1, arg1 arg2, opcode arg1, x x
- Compare: IMPLEMENTATION_PLAN.md
  - Intersection: 72 | top: program, coordinate, bsp, rewrite, new, x, cnf, nodes, now, pointer, strata, vm
  - Symmetric difference: 681 (only in in/in-11.md: 216, only in IMPLEMENTATION_PLAN.md: 465)
    - Only in in/in-11: xor, cd, text, coordinates, equality, ledger, op_coord_zero, parity, depth, interned, self, already
    - Only in IMPLEMENTATION_PLAN.md: tests, expected, add, null, pytest, rank, txt, arena, sort, swizzle, implement, implementation
  - Wedge product (bigram intersection): 5 | top: arg1 arg2, opcode arg1, rewrite rules, new opcodes, tier refs
