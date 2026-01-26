# Evaluation of the 2:1 Alternating BSP Architecture in JAX

NOTE: REFINED by in-14.md to treat BSP/morton as a performance-only path; ledger/CNF-2 is the semantic spine.

This analysis evaluates the architectural shift to a **2:1 Alternating Binary Space Partitioning (BSP)** memory layout within a JAX-based interaction net evaluator. This design addresses the "Shatter Effect"—the entropic scattering of graph nodes during parallel reduction—by enforcing a rigorous geometric constraint on memory allocation.

### Executive Summary

Moving from a linear arena to a **2:1 Alternating BSPˢ** layout fundamentally transforms the system from a **Bandwidth-Bound** pointer chaser to a **Compute-Bound** tensor processor. By aligning memory address space with the 2:1 aspect ratio of GPU cache lines (which favor contiguous horizontal reads), this layout maximizes effective memory throughput.

In JAX, this implementation trades the overhead of "address swizzling" (bit-interleaving) for a massive reduction in global memory divergence. While standard JAX primitives (`numpy` ops) are inefficient for bit-level logic, the use of **Pallas** or **Triton** kernels for the swizzle phase makes this viable. Compared to the previous `exec()` based model and standard HVM implementations, this approach offers deterministic latency and superior scaling on SIMD hardware.

---

### 1. Architectural Impact: The Physics of "Shatter" Containment

The "Shatter Effect" refers to the phenomenon where a compact wavefront of active nodes produces children that are effectively randomized in the address space, destroying locality for the subsequent step.

#### 1.1 The 2:1 Locality Invariant

Standard Z-order (Morton codes) interleaves bits as , creating square  micro-tiles.
The **2:1 Alternating** layout interleaves as , creating ** rectangular tiles**.

* **Why 2:1?** A standard GPU cache line is 128 bytes (32 floats). A  square of 32-bit integers is only 16 bytes. A linear scan quickly "falls off" the Z-curve cliff into a different memory page.
* **The Impact:** The 2:1 bias ensures that the "fast axis" (X) runs for 4 or 8 elements before jumping. This aligns the "shatter" debris (the output nodes of a reduction) into contiguous strips that perfectly fill a coalesced memory transaction.

#### 1.2 Hierarchy as a Blast Shield

By defining arenas recursively (Global  Cluster  Block), the architecture imposes **Soft Isolation**:

* **Linear Arena:**  spawns  at index  and  at index . **Result:** 2 TLB misses next cycle.
* **BSP Arena:**  (at BSP index ) spawns children. The allocator prefers empty slots in **BSP Block ** or .
* **Result:** The "blast radius" of the rewrite is contained within the L2 cache size of the Streaming Multiprocessor (SM).

---

### 2. JAX Implementation Strategy & Trade-offs

Implementing this in JAX requires bypassing its default contiguous-array assumptions.

#### 2.1 The Swizzle Bottleneck (The Con)

JAX's high-level API (`jax.numpy`) does not support efficient bit-interleaving instructions (`PDEP`/`PEXT` on x86, equivalent bit-shifting on GPU).

* **Naive JAX:** Implementing Morton codes via shifts (`(x | (x << 16)) & mask...`) is computationally expensive in Python/XLA, potentially adding 10-50 cycles per node read.
* **The Solution:** Use **JAX Pallas** or **Custom Call** to write a trivial CUDA kernel for the swizzle/deswizzle step. This reduces the cost to near-zero (hidden by memory latency).

#### 2.2 The Sort vs. Binning Tradeoff

The proposal to constrain rank to **2 bits** allows replacing the expensive  `argsort` with a linear  **Stream Compaction (Bucket Sort)**.

* **Algorithm:**
1. **Predicate:** Compute 2-bit Rank.
2. **Prefix Sum (Scan):** Calculate offsets for the 4 bins (HOT, WARM, COLD, FREE).
3. **Scatter:** Move nodes to their bin offsets.


* **JAX Fit:** `jax.lax.associative_scan` is highly optimized. This approach is perfectly suited to the TPU/GPU "scan" primitives.

#### 2.3 Implementation Comparison

| Feature | Standard Linear JAX | 2:1 BSP JAX (Proposed) |
| --- | --- | --- |
| **Addressing** | `idx` (Direct) | `swizzle(x, y)` (Computed) |
| **Allocation** | `atomic_add` (Global counter) | Hierarchical Bitmask search |
| **Memory Access** | Random (Uncoalesced) | Block-Coalesced |
| **Cache Hit Rate** | < 10% (Graph dependent) | > 80% (Locality enforced) |
| **Bottleneck** | DRAM Latency | ALU (Address computation) |

---

### 3. Comparison with Other Implementations

#### 3.1 vs. HVM (Higher-order Virtual Machine) / Bend

* **HVM (Rust/CUDA):** Uses a "redex bag" and atomic-based interaction. It relies on the GPU's raw bandwidth to hide latency. It suffers from high divergence when the graph structure becomes irregular.
* **2:1 BSP:** Is **Level-Synchronous**. It processes the graph in "generations". This reduces divergence because all threads in a warp work on nodes that are spatially close (and thus likely structurally similar).
* **Verdict:** HVM is faster for raw throughput on "soup-like" random graphs. 2:1 BSP will outperform HVM on **structured, hierarchical data** (e.g., trees, matrices encoded as trees) where the spatial locality can be exploited.

#### 3.2 vs. GraphBLAS / Sparse Matrix

* **GraphBLAS:** Uses CSR/CSC formats. These are static. Updating the graph topology (rewiring) in CSR is .
* **2:1 BSP:** Is designed for **dynamic topology**. The "Fluid Memory" concept allows the graph to rewire itself ( updates) while the background "Pack" phase continuously defragments it.
* **Verdict:** 2:1 BSP is superior for **Graph Rewriting** (Interaction Nets), while GraphBLAS is superior for **Static Graph Analysis** (PageRank).

### 4. Conclusion and Recommendation

The **2:1 Alternating BSP Layout** is a sophisticated optimization that correctly identifies **Memory Divergence** as the primary enemy of GPU graph reduction.

**Key Tradeoff:** You pay a constant cost in ALU instructions (address swizzling) to gain a massive reduction in Memory Stall cycles. On modern GPUs (H100/A100) where compute is cheap and HBM is the bottleneck, this is the **correct trade**.

**Final Recommendation:**

1. **Adopt 2-Bit Rank:** The simplification to 4 buckets (HOT/WARM/COLD/FREE) enables  scheduling, which is critical.
2. **Use Pallas for Swizzling:** Do not attempt to implement the Morton encoding in pure `jax.numpy`; the overhead will negate the gains.
3. **Hierarchical Bitmasks:** Manage the "Free Space" using a hierarchical bitmask (like a quadtree of `1`s and `0`s) to allow rapid finding of local free slots during the "Shatter" phase.
