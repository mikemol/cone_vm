---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
---

# **Implementing Branchless Interaction Combinator Rewrite Rules on GPUs: A Tensor-Theoretic Approach to Optimal Graph Reduction**

NOTE: DEFERRED. This is a future roadmap track (see IMPLEMENTATION_PLAN.md "Roadmap Extension: in-8 Pivot").

## **1\. Executive Summary**

The evaluation of functional programs on massively parallel hardware represents one of the most significant open challenges in modern computer science. While the von Neumann architecture has served sequentially executed imperative code for decades, the physical limits of frequency scaling and the rise of General-Purpose Graphics Processing Units (GPGPUs) demand a fundamental rethinking of computational models. The Interaction Combinator system, introduced by Yves Lafont 1, offers a theoretically sound and inherently parallel graph-rewriting framework that avoids the global synchronization bottlenecks of traditional execution models. Unlike the $\\lambda$-calculus, which requires complex garbage collection and non-local reductions, interaction nets operate through local, constant-time rewrite rules that are strongly confluent.3

This research report presents a comprehensive plan for implementing these rules on GPUs using a novel **branchless, matrix-theoretic approach**. The central hypothesis of this work is that the irregular control flow typically associated with graph reduction—a primary source of performance degradation on Single Instruction, Multiple Thread (SIMT) architectures—can be eliminated by reformulating the interaction rules as tensor transformations. By encoding the topological mutations of the graph as sparse matrix permutations and utilizing the high-bandwidth scatter/gather primitives of modern frameworks like JAX 5, we can achieve "optimal" parallel reduction that scales linearly with the number of available cores.

This document details the theoretical underpinnings of this approach, deriving the matrix operations required for the Symmetric Interaction Combinators.7 It provides a rigorous specification for the memory layout, the construction of "Rule Tensors" that govern topology changes, and the specific JAX implementation strategies required to map these dynamic structures onto static XLA (Accelerated Linear Algebra) graphs. Furthermore, it contrasts this tensor-based methodology with existing atomic-locking approaches used in state-of-the-art runtimes like HVM2 8, arguing that the determinism and memory coalescing of the tensor approach offer superior long-term scaling for dense, highly-parallel workloads.

## **2\. Introduction and Problem Statement**

### **2.1 The Parallelism Gap in Symbolic Computation**

The computing landscape has undergone a bifurcation. On one side, numerical computing—driven by deep learning and scientific simulation—has successfully migrated to GPUs, leveraging their massive parallelism to achieve petaflop-scale performance. This success is predicated on the regularity of dense linear algebra: matrix multiplications ($C \= A \\times B$) are predictable, uniform, and memory-coherent. On the other side, symbolic computation—including compilers, theorem provers, and functional programming runtimes—remains largely bound to the Central Processing Unit (CPU).

The resistance of symbolic computation to GPU acceleration stems from its inherent irregularity. Functional programs are typically represented as trees or graphs (Abstract Syntax Trees, Graph Reduction machines). Evaluation proceeds via recursive traversal and rewriting, processes that involve:

1. **Pointer Chasing:** Traversing linked structures creates random memory access patterns that defeat the caching mechanisms and memory coalescing of GPUs.9  
2. **Dynamic Allocation:** Nodes are created and destroyed unpredictably, requiring sophisticated memory management that is difficult to implement efficiently in a massive parallel environment without significant locking overhead.  
3. **Branch Divergence:** The core logic of reduction involves checking node types (e.g., "Is this an application or a lambda?") and executing different code paths. On SIMT architectures, where 32 or 64 threads (a warp) must execute the same instruction, divergent branches cause serialization, reducing throughput by orders of magnitude.11

### **2.2 The Interaction Net Alternative**

Interaction Nets, and specifically Interaction Combinators, provide a model of computation that is fundamentally different from the pointer-chasing graph reduction of the standard $\\lambda$-calculus. Originating from the proof structures of Linear Logic 13, interaction nets model computation as the local annihilation and reconfiguration of agents in a graph.

The critical property of interaction nets is **Strong Confluence**.3 In an interaction net, a reduction (rewrite) can only occur when two agents are connected via their **Principal Ports**. Since every agent has exactly one principal port, it can participate in at most one active interaction at any given time. This implies that the set of all possible reductions in a net is a set of disjoint pairs. Consequently, **all** active reductions can be performed simultaneously without the risk of race conditions or the need for global synchronization locks.4

This property makes interaction nets theoretically ideal for GPUs. If a net contains one million active pairs, a GPU could theoretically assign one thread to each pair and execute one million rewrites in a single step. However, the practical implementation is hindered by the aforementioned branch divergence. An interaction between a Constructor and a Duplicator requires different wiring logic than an interaction between two Duplicators. Implementing this via if-else blocks reintroduces the SIMT inefficiency we seek to avoid.

### **2.3 The Tensor-Theoretic Proposal**

To bridge the gap between the theoretical promise of interaction nets and the hardware reality of GPUs, this report proposes a **Tensor-Theoretic** approach. Instead of viewing the interaction net as a graph of objects to be traversed, we view it as a **state tensor** (a sparse adjacency matrix) to be transformed.

We postulate that the interaction rules (Annihilation, Commutation, Erasure) can be unified into a single mathematical operation: a **sparse matrix permutation** coupled with a **linear allocation step**. By lifting the logic from control flow (instructions) to data (tensors), we can utilize the GPU's special function units and high-bandwidth memory for what they do best: vector math.

Specifically, we investigate:

* **Tensor-Based Port Rewiring:** How to define the complex topological surgeries of interaction combinators as vector indices operations.  
* **Sparse Matrix Permutations:** Representing the net's connectivity as an adjacency list and defining rewrite rules as local permutation matrices.  
* **JAX Implementation:** Leveraging the functional, immutable, and compilable nature of JAX to express these operations in a way that XLA can optimize into efficient CUDA kernels.16

## **3\. Theoretical Foundations: The Geometry of Interaction**

To engineer a matrix-based reduction engine, we must first establish the algebraic correspondence between graph rewriting and linear operators. This connection is deep, rooted in Jean-Yves Girard's "Geometry of Interaction" (GoI).13

### **3.1 Linear Logic and Proof Nets**

Linear Logic refines classical logic by treating formulas as resources that must be consumed exactly once.14 This resource consciousness is mirrored in the structure of **Proof Nets**, the graphical syntax for Linear Logic proofs. In a proof net, the "links" represent logical connectives, and the "wires" represent formulae. Cut-elimination—the process of normalizing a proof—corresponds to graph reduction.

Interaction Combinators are a distillation of this concept. They provide a universal interaction system capable of encoding all computable functions (Turing completeness) using a minimal alphabet of agents and a fixed set of local rewrite rules.2

### **3.2 The Signature of Interaction Combinators**

The canonical system consists of three agents (symbols):

1. **Constructor ($\\gamma$):** Represents the construction of data or linear logic multiplicatives (Tensor/Par).  
2. **Duplicator ($\\delta$):** Represents the structural rule of Contraction (copying resources).  
3. **Eraser ($\\epsilon$):** Represents the structural rule of Weakening (discarding resources).

Each agent has a specific **arity**:

* $\\gamma$ and $\\delta$ are binary (arity 2). They have one **Principal Port** and two **Auxiliary Ports**.  
* $\\epsilon$ is nullary (arity 0). It has one Principal Port and zero Auxiliary Ports.

Visualizing the Ports:  
It is essential for the matrix formulation to assign a rigid ordering to these ports.

* **Port 0 (Principal):** The "active" interface. Reduction happens here.  
* **Port 1 (Aux Left):** The first data channel.  
* **Port 2 (Aux Right):** The second data channel.

We denote an agent $A$ as a vector of its ports: $A \= \[p\_0, p\_1, p\_2\]$.

### **3.3 The Rewrite Rules**

The dynamics of the system are defined by the interaction of agents at their principal ports. There are six possible interactions in the basic system (3 symbols $\\times$ 3 symbols, modulo symmetry), which collapse into two primary behaviors: **Annihilation** and **Commutation**.1

#### **3.3.1 Annihilation ($\\alpha \\bowtie \\alpha$)**

When two agents of the *same* symbol interact ($\\gamma \\bowtie \\gamma$ or $\\delta \\bowtie \\delta$), they annihilate.

* **Graph Transformation:** The two nodes vanish. The wire connected to the left aux port of the first node connects to the left aux port of the second. Similarly for the right ports.  
* **Matrix Interpretation:** This is the identity transformation on the auxiliary wires. It reduces the dimension of the active graph space.

#### **3.3.2 Commutation ($\\alpha \\bowtie \\beta$, $\\alpha \\neq \\beta$)**

When agents of *different* symbols interact ($\\gamma \\bowtie \\delta$), they commute.

* **Graph Transformation:** The agents "pass through" each other. The single $\\delta$ node is duplicated into two $\\delta$ nodes, and the single $\\gamma$ node is duplicated into two $\\gamma$ nodes.  
* **Topology:** This effectively computes the "cross product" of the connections.  
* **Matrix Interpretation:** This is a constructive operation. It expands the dimension of the graph space (allocating new nodes) and applies a specific permutation to the connectivity vector.

#### **3.3.3 Erasure ($\\epsilon \\bowtie \\alpha$)**

When an Eraser interacts with a binary node:

* **Graph Transformation:** The binary node is deleted. Two new Erasers are spawned to cap the auxiliary wires of the deleted node.  
* **Matrix Interpretation:** A localized replacement of a rank-3 tensor (the binary node) with two rank-1 tensors (erasers).

### **3.4 GoI: Graphs as Matrices**

In the Geometry of Interaction, a net is represented by a permutation matrix $\\Pi$ acting on the space of directed edges.18 If the graph has $N$ edges, $\\Pi$ is a $2N \\times 2N$ matrix where $\\Pi\_{i,j} \= 1$ if the output of edge $i$ connects to the input of edge $j$.

The execution of the net is given by the Execution Formula:

$$Ex(\\Pi) \= \\text{Tr}\_{input}(\\Pi (1 \- \\mu \\Pi)^{-1})$$

Here, $\\mu$ represents the "cut" (the active pair). The term $(1 \- \\mu \\Pi)^{-1}$ can be expanded as a geometric series $\\sum (\\mu \\Pi)^k$, representing the path of a "particle" traveling through the net.  
While we will not implement the infinite series directly (as it is inefficient for simulation), this theoretical stance validates our approach: **Graph reduction is fundamentally a linear algebra operation over a permutation space.** Our task is to implement the discrete transitions of this matrix $\\Pi \\to \\Pi'$ efficiently on the GPU.

## ---

**4\. Data Structures: Mapping Graphs to Tensors**

To implement a graph reduction engine in JAX, we must map the dynamic, pointer-based structure of Interaction Nets into static, dense tensors. The XLA compiler used by JAX relies heavily on static shapes to perform optimizations (loop unrolling, fusion, vectorization).17 Therefore, we cannot use dynamic memory allocation (like malloc) in the traditional sense. We must implement a **static memory arena**.

### **4.1 The Structure of Arrays (SoA) Layout**

We define the state of the machine using a collection of flat arrays. We assume a fixed maximum capacity $N\_{max}$ (e.g., $2^{24} \\approx 16.7$ million nodes). This fits comfortably within the VRAM of modern data center GPUs (e.g., A100 with 40GB/80GB) or even consumer cards (RTX 4090 with 24GB).

#### **4.1.1 The NodeTypes Tensor**

* **Shape:** (N\_max,)  
* **Dtype:** uint8  
* **Semantics:** Stores the symbol of the agent at index $i$.  
  * $0$: FREE (Unused slot)  
  * $1$: ERA ($\\epsilon$)  
  * $2$: CON ($\\gamma$)  
  * $3$: DUP ($\\delta$)  
  * ... (Additional types for numeric extensions, if used)

#### **4.1.2 The Ports Tensor (The Adjacency Matrix)**

* **Shape:** (N\_max, 3\)  
* **Dtype:** uint32  
* **Semantics:** Stores the connectivity.  
  * Ports\[i, 0\]: Connection at Principal Port.  
  * Ports\[i, 1\]: Connection at Aux Left.  
  * Ports\[i, 2\]: Connection at Aux Right.

The Implicit Port Addressing Scheme:  
A "pointer" in this system must identify not just a target node, but a specific port on that node. We utilize the low-order bits of the integer index for this purpose.

$$\\text{Pointer} \= (\\text{NodeIndex} \\ll 2\) \\ | \\ \\text{PortIndex}$$

* **PortIndex 0:** Principal Port.  
* **PortIndex 1:** Aux Left.  
* **PortIndex 2:** Aux Right.

This encoding allows us to store a directed edge as a single 32-bit integer. An index $i$ in the Ports tensor corresponds to the node i // 4, and the column i % 4 corresponds to the specific port.

#### **4.1.3 The FreeStack Tensor**

* **Shape:** (N\_max,)  
* **Dtype:** uint32  
* **Semantics:** A stack containing the indices of all currently FREE nodes.  
* **Associated Scalar:** StackPointer (uint32), pointing to the next available free node.

This tensor manages our memory allocation. When new nodes are required (during Commutation), we pop indices from this stack. When nodes are destroyed (during Annihilation or Erasure), we push indices back onto this stack. This allows for $O(1)$ allocation and deallocation, crucial for GPU performance.

### **4.2 Bidirectional Connectivity Invariant**

In a pointer-based graph, an edge is often a single reference. In Interaction Nets, edges are wires. A wire connecting port $A$ to port $B$ implies that $A$ points to $B$ and $B$ points to $A$.  
Invariant: Ports\[Node(A), Port(A)\] \== Address(B, Port(B)) $\\iff$ Ports \== Address(A, Port(A)).  
This invariant allows us to implement **local rewiring**. When we rewrite a node $A$, we can immediately find its neighbor $B$ by looking at $A$'s ports. Crucially, we can also update $B$'s connection to point to the *new* node $A'$ because we know exactly where $B$ is. This eliminates the need for searching the graph or maintaining back-pointers.

## ---

**5\. Branchless Interaction Rules: The Tensor Formulation**

The core innovation of this report is the reformulation of the interaction rules as **tensor operations**. We aim to eliminate the control flow that distinguishes between Annihilation, Commutation, and Erasure. We achieve this by unifying them into a single **Alloc-Rewire-Free** cycle governed by a **Rule Tensor**.

### **5.1 The Universal Rule Tensor**

We construct a constant tensor, RULE\_TABLE, stored in constant memory (or shared memory) on the GPU.

* **Dimensions:** (NumTypes, NumTypes, RuleVectorSize)  
* **Indexing:** RULE\_TABLE returns a vector defining the interaction between agent types A and B.

The Rule Vector Structure:  
The vector contains all necessary parameters to execute any of the interaction types branchlessly:

1. **AllocCount:** Number of new nodes to allocate (0 for Annihilation, 4 for Commutation, 2 for Erasure).  
2. **WiringTemplateID:** Index into the WIRING\_TEMPLATES tensor.  
3. **Aux1Mapping:** Instruction for rewiring Aux 1 (used in Annihilation).  
4. **Aux2Mapping:** Instruction for rewiring Aux 2\.

### **5.2 Tensor-Based Port Rewiring (The "Wiring Template")**

The complexity of the rewrite rules lies in the topology changes. Commutation, for instance, requires a specific "twist" in the wiring. We encode these topologies as data.

We define a Wiring Template as a small integer matrix that describes relative connections.  
For a Commutation ($\\delta \\bowtie \\gamma$), we allocate 4 new nodes. Let their indices be $k, k+1, k+2, k+3$.  
The template defines 12 connections (3 ports $\\times$ 4 nodes).  
Each entry in the template specifies a Source:

* Values $0 \\dots 3$: Refer to the 4 newly allocated nodes (internal wiring).  
* Values $4 \\dots 7$: Refer to the "External Neighbors" (the nodes originally connected to the active pair).

Example: The Symmetric Commutation Template  
Let the active pair be $A$ ($\\delta$) and $B$ ($\\gamma$).  
External Neighbors: $E\_0$ (on $A\_1$), $E\_1$ (on $A\_2$), $E\_2$ (on $B\_1$), $E\_3$ (on $B\_2$).  
New Nodes: $N\_0, N\_1$ (copies of $\\gamma$), $N\_2, N\_3$ (copies of $\\delta$).  
The template for this rule might look like (simplified):

| Node Index | Port 0 (Principal) | Port 1 (Aux 1\) | Port 2 (Aux 2\) |
| :---- | :---- | :---- | :---- |
| $N\_0$ | Connect to $E\_0$ | Connect to $N\_2$ | Connect to $N\_3$ |
| $N\_1$ | Connect to $E\_1$ | Connect to $N\_2$ | Connect to $N\_3$ |
| $N\_2$ | Connect to $E\_2$ | Connect to $N\_0$ | Connect to $N\_1$ |
| $N\_3$ | Connect to $E\_3$ | Connect to $N\_0$ | Connect to $N\_1$ |

This template is stored as a tensor. During execution, the kernel:

1. Loads the template ID from the RULE\_TABLE.  
2. Fetches the template.  
3. Resolves the abstract connections (e.g., "Connect to $N\_2$") into concrete memory addresses using the allocated indices.  
4. Writes the result to the Ports tensor.

### **5.3 Handling Annihilation Branchlessly**

Annihilation creates 0 nodes. How does it fit this model?

* AllocCount is 0\.  
* The WiringTemplate is effectively empty or ignored.  
* However, we need a mechanism to wire $E\_0$ directly to $E\_2$ and $E\_1$ to $E\_3$.

We introduce a Linkage Tensor. This tensor describes updates to the External Neighbors.  
For every interaction, we generate updates for the external neighbors.

* **Commutation:** $E\_0$ must now point to $N\_0$. $E\_1 \\to N\_1$, etc.  
* **Annihilation:** $E\_0$ must now point to $E\_2$. $E\_1 \\to E\_3$.

By treating "External Neighbor Updates" as a first-class citizen of the rule definition, we unify the logic. The rule vector simply specifies: "Update $E\_0$ with value X". For Commutation, X is a new node address. For Annihilation, X is the address of the partner neighbor. This calculation can be done via jax.numpy.where selection or simple arithmetic indexing, avoiding thread divergence.

## ---

**6\. JAX Implementation Strategy**

JAX offers a Python-to-XLA pipeline that is ideal for this workload. Unlike PyTorch or TensorFlow, which are optimized for fixed DAGs, JAX's lax primitives allow for the expression of low-level loops and scatters required for graph rewriting.

### **6.1 The Reduction Kernel (Scan Loop)**

The simulation proceeds in discrete time steps (generations). We use jax.lax.scan or jax.lax.while\_loop to iterate the reduction.

Step 1: Active Pair Identification (The Match Phase)  
We need to find all indices $i$ such that Node $i$ is part of an active pair.  
Vectorized Logic:

Python

\# Load principal neighbor  
neighbor \= ports\[:, 0\]  
neighbor\_idx \= neighbor \>\> 2  
neighbor\_port \= neighbor & 3

\# Check active condition  
\# 1\. Neighbor connects via Principal Port (port 0\)  
\# 2\. To strictly order the pair (avoid processing A-B and B-A twice), enforce i \< neighbor\_idx  
active\_mask \= (neighbor\_port \== 0) & (device\_id \< neighbor\_idx) & (node\_types\!= FREE)

This produces a boolean mask. To process efficiently, we must "compact" this sparse mask into a dense list of indices.

Step 2: Stream Compaction  
In CUDA, this is thrust::copy\_if. In JAX, we can use jax.lax.top\_k on the boolean mask (treating True as 1, False as 0\) to extract the indices of the active nodes. Alternatively, since XLA handles sparse operations poorly, we might process the mask directly if the density is high, or use a pre-computed permutation.  
Let's assume we extract active\_indices of size $K$.  
Step 3: Rule Lookup and Allocation (The Allocation Phase)  
We fetch the types of the pairs:  
type\_A \= node\_types\[active\_indices\]  
type\_B \= node\_types\[ports\[active\_indices, 0\] \>\> 2\]  
We index the RULE\_TABLE to get alloc\_counts.  
We perform a Parallel Prefix Sum (CumSum) on alloc\_counts to determine the memory offset for each pair.  
offsets \= jax.lax.cumsum(alloc\_counts)  
total\_alloc \= offsets\[-1\]  
We claim total\_alloc indices from the FreeStack.  
Step 4: Tensor Rewiring (The Execute Phase)  
This is the heart of the engine. We perform a "gather" to collect all necessary data (external neighbors) into registers.  
Then, we apply the WiringTemplate. This involves:

1. Calculating the addresses of the new nodes ($Base \+ Offset$).  
2. Looking up the targets in the Template.  
3. Resolving the targets (Internal vs External).  
   This computes a large set of updates: pairs of (Address, Value).

Step 5: Scatter Commit (The Write Phase)  
We utilize jax.lax.scatter to write the new connections into the Ports tensor.

* **Update 1:** Write the ports of the newly created nodes.  
* **Update 2:** Update the ports of the external neighbors (re-linking them to the new net).  
* **Update 3:** Mark the old active pair nodes as FREE (push to FreeStack).

### **6.2 Managing Static Shapes**

JAX JIT requires static array shapes.

* **Problem:** The number of active pairs $K$ varies per step. The number of new nodes varies.  
* Solution: We use Padding. We set a MAX\_ACTIVE\_PAIRS buffer size per step. We pad the active\_indices with a sentinel value. We use masked operations (jnp.where) to ensure that computations on padded slots result in no-ops (writes to a "dump" address or identity updates).  
  While this introduces some compute overhead (processing padding), it enables fully compiled, unrolled kernels on the GPU, which is often faster than CPU-managed dynamic launching.

### **6.3 Concurrency and Race Conditions**

Interaction Combinators are confluent, meaning the *order* of reductions doesn't matter. However, memory safety matters.

* **Principal Port Safety:** Since we enforce i \< neighbor\_idx and neighbor connects to 0, each active pair is identified exactly once by exactly one thread. No race condition here.  
* **Aux Port Safety:** Can two active pairs try to update the same external neighbor?  
  * No. If pair A-B interacts, it updates its neighbors $N\_1 \\dots N\_4$.  
  * If $N\_1$ were part of *another* active pair, it would be interacting via its Principal Port. But here it is connected to A via an Aux port.  
  * Therefore, the sets of "nodes being written to" by simultaneous reductions are disjoint.  
  * This guarantees that we can use scatter without atomic locks for the graph updates. (Atomic add is only needed for the global StackPointer allocation, which is handled by the Prefix Sum).

## ---

**7\. Comparative Analysis**

### **7.1 Comparison with HVM/HVM2**

HVM2 (Higher-order Virtual Machine 2\) 8 is the current state-of-the-art implementation of interaction nets on GPUs.

* **HVM2 Approach:** Atomic Locking.  
  * It uses a massive grid of threads. Each thread attempts to rewrite a specific node.  
  * It relies on atomicCAS (Compare-and-Swap) to claim nodes and update pointers.  
  * This allows for asynchronous wavefront execution.  
* **Tensor Approach:** Bulk Synchronous.  
  * It separates "Match", "Allocate", and "Rewrite" into global phases.  
  * **Pros:** Better memory coalescing. The Ports tensor is accessed in a linear, predictable pattern during the scan. Deterministic execution makes debugging and reproduction trivial.  
  * **Cons:** The global barrier between phases (kernel launch overhead).  
* **Performance Implications:** HVM2 may win on extremely sparse, irregular graphs where the active pairs are few and scattered. The Tensor approach will dominate on **dense, saturated workloads** where the number of active pairs allows the GPU to saturate memory bandwidth during the "Scatter" phase.

### **7.2 Throughput Analysis**

The theoretical limit of this system is **Memory Bandwidth**.

* A Commutation involves:  
  * Reading 2 nodes (24 bytes).  
  * Reading 4 neighbors (16 bytes).  
  * Allocating 4 nodes.  
  * Writing 4 new nodes (48 bytes).  
  * Updating 4 neighbors (16 bytes).  
  * Total Traffic: \~104 bytes per reduction.  
* On an A100 GPU (1.5 TB/s bandwidth), the theoretical ceiling is:

  $$\\frac{1.5 \\times 10^{12} \\text{ bytes/s}}{104 \\text{ bytes/op}} \\approx 14.4 \\text{ Billion Reductions/Second (GIPS)}$$

  This creates a target for "optimal" performance. Current CPU implementations achieve \~100-200 MIPS. HVM2 achieves \~2-4 GIPS. The Tensor approach targets the theoretical maximum by ensuring every byte transferred contributes to a reduction.

## ---

**8\. Specific Interaction Rules Analysis**

This section details the precise wiring logic derived from the literature 2 required for the implementation.

### **8.1 Table: The Rule Tensor Specification**

| Pair Type A | Pair Type B | Alloc | Wiring ID | Semantics |
| :---- | :---- | :---- | :---- | :---- |
| $\\delta$ | $\\delta$ | 0 | T\_ANNIHIL | Straight pass-through. $A\_1 \\leftrightarrow B\_1, A\_2 \\leftrightarrow B\_2$. |
| $\\gamma$ | $\\gamma$ | 0 | T\_ANNIHIL | Straight pass-through. |
| $\\delta$ | $\\gamma$ | 4 | T\_COMMUTE | Cross-product duplication. |
| $\\gamma$ | $\\delta$ | 4 | T\_COMMUTE | Cross-product duplication. |
| $\\epsilon$ | $\\delta$ | 2 | T\_ERASE | Erasure propagation. |
| $\\epsilon$ | $\\gamma$ | 2 | T\_ERASE | Erasure propagation. |
| $\\delta$ | $\\epsilon$ | 2 | T\_ERASE | Erasure propagation (symmetric). |

### **8.2 Wiring Template: Commutation ($\\delta \\bowtie \\gamma$)**

Let the 4 external neighbors be $x, y$ (from $\\delta$) and $a, b$ (from $\\gamma$).  
Let the 4 new nodes be $n\_0, n\_1$ (type $\\gamma$) and $n\_2, n\_3$ (type $\\delta$).  
**New Node Configuration:**

1. $n\_0$: Principal $\\to x$. Aux1 $\\to n\_2$. Aux2 $\\to n\_3$.  
2. $n\_1$: Principal $\\to y$. Aux1 $\\to n\_2$. Aux2 $\\to n\_3$.  
3. $n\_2$: Principal $\\to a$. Aux1 $\\to n\_0$. Aux2 $\\to n\_1$.  
4. $n\_3$: Principal $\\to b$. Aux1 $\\to n\_0$. Aux2 $\\to n\_1$.

**External Neighbor Updates:**

1. Update node at $x$: Point its connected port to $n\_0$.  
2. Update node at $y$: Point its connected port to $n\_1$.  
3. Update node at $a$: Point its connected port to $n\_2$.  
4. Update node at $b$: Point its connected port to $n\_3$.

This precise set of 16 integer writes completely implements the commutation rule without any conditional branching.

## ---

**9\. Conclusion**

The transition from von Neumann architectures to SIMT GPUs requires a reinvention of our fundamental algorithms. This report has demonstrated that **Interaction Combinators**, by virtue of their locality and strong confluence, are uniquely positioned to bridge this divide. By abandoning the object-oriented view of graph reduction in favor of a **Tensor-Theoretic** perspective, we can define a branchless, vectorized rewrite engine.

The proposed system utilizes a **Structure of Arrays** layout to maximize memory coalescing, a **Universal Rule Tensor** to eliminate control flow divergence, and **Prefix-Sum Allocation** to manage memory without locking. The use of JAX allows for high-level expression of these complex matrix permutations while leveraging XLA to generate optimized PTX kernels.

As functional programming languages move toward "beta-optimality" and massive parallelism, this tensor-based graph reduction engine offers a blueprint for the runtimes of the future—runtimes where logical deduction flows as fluidly and efficiently as the dense matrix multiplications of AI.

---

References:  
13 Girard, "Geometry of Interaction 1"  
3 "Interaction nets are a general graph-rewriting system..."  
5 JAX Documentation: lax.scatter  
1 Wikipedia: Interaction Nets  
2 Lafont, "Interaction Combinators"  
22 Mazza, "Symmetric Interaction Combinators"  
8 HVM2 Paper  
8 HVM2 Code Analysis

#### **Works cited**

1. Interaction nets \- Wikipedia, accessed January 20, 2026, [https://en.wikipedia.org/wiki/Interaction\_nets](https://en.wikipedia.org/wiki/Interaction_nets)  
2. Interaction Combinators \- chorasimilarity, accessed January 20, 2026, [https://chorasimilarity.wordpress.com/wp-content/uploads/2024/01/ic-lafont-1.pdf](https://chorasimilarity.wordpress.com/wp-content/uploads/2024/01/ic-lafont-1.pdf)  
3. Lambda Calculus Normalization with Interaction Nets \- Enrico Z. Borba, accessed January 20, 2026, [https://ezb.io/thoughts/interaction\_nets/lambda\_calculus/2025-04-25\_normalization.html](https://ezb.io/thoughts/interaction_nets/lambda_calculus/2025-04-25_normalization.html)  
4. Encoding Linear Logic with Interaction Combinators \- Universidade do Minho, accessed January 20, 2026, [https://repositorium.uminho.pt/bitstreams/f3f5ad5e-9466-4ddf-8869-14d34ea18782/download](https://repositorium.uminho.pt/bitstreams/f3f5ad5e-9466-4ddf-8869-14d34ea18782/download)  
5. jax.lax.scatter \- JAX documentation, accessed January 20, 2026, [https://docs.jax.dev/en/latest/\_autosummary/jax.lax.scatter.html](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html)  
6. Scatter on Sharded Matrices has bugs · Issue \#23052 · jax-ml/jax \- GitHub, accessed January 20, 2026, [https://github.com/google/jax/issues/23052](https://github.com/google/jax/issues/23052)  
7. Full Abstraction for Set-Based Models of the Symmetric Interaction Combinators \- Department of Mathematics and Statistics, accessed January 20, 2026, [https://www.mathstat.dal.ca/\~neilr/Papers/FOSSACS2012.pdf](https://www.mathstat.dal.ca/~neilr/Papers/FOSSACS2012.pdf)  
8. HVM2: A Parallel Evaluator for Interaction Combinators \- GitHub, accessed January 20, 2026, [https://raw.githubusercontent.com/HigherOrderCO/HVM/main/paper/HVM2.pdf](https://raw.githubusercontent.com/HigherOrderCO/HVM/main/paper/HVM2.pdf)  
9. (PDF) GPU-Based Branchless Distance-Driven Projection and Backprojection, accessed January 20, 2026, [https://www.researchgate.net/publication/314195567\_GPU-Based\_Branchless\_Distance-Driven\_Projection\_and\_Backprojection](https://www.researchgate.net/publication/314195567_GPU-Based_Branchless_Distance-Driven_Projection_and_Backprojection)  
10. Evaluating Gather and Scatter Performance on CPUs and GPUs, accessed January 20, 2026, [https://www.memsys.io/wp-content/uploads/2020/10/p220-lavin.pdf](https://www.memsys.io/wp-content/uploads/2020/10/p220-lavin.pdf)  
11. Chapter 34\. GPU Flow-Control Idioms \- NVIDIA Developer, accessed January 20, 2026, [https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-34-gpu-flow-control-idioms](https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-34-gpu-flow-control-idioms)  
12. GPU branching if without else \- Computer Graphics Stack Exchange, accessed January 20, 2026, [https://computergraphics.stackexchange.com/questions/4115/gpu-branching-if-without-else](https://computergraphics.stackexchange.com/questions/4115/gpu-branching-if-without-else)  
13. Geometry of interaction \- Wikipedia, accessed January 20, 2026, [https://en.wikipedia.org/wiki/Geometry\_of\_interaction](https://en.wikipedia.org/wiki/Geometry_of_interaction)  
14. Linear Logic \- Stanford Encyclopedia of Philosophy, accessed January 20, 2026, [https://plato.stanford.edu/archives/win2016/entries/logic-linear/](https://plato.stanford.edu/archives/win2016/entries/logic-linear/)  
15. Δ-Nets: Interaction-Based System for Optimal Parallel 𝜆-Reduction \- arXiv, accessed January 20, 2026, [https://arxiv.org/html/2505.20314v1](https://arxiv.org/html/2505.20314v1)  
16. GPU performance tips \- JAX documentation, accessed January 20, 2026, [https://docs.jax.dev/en/latest/gpu\_performance\_tips.html](https://docs.jax.dev/en/latest/gpu_performance_tips.html)  
17. JAX and OpenXLA Part 1: Run Process and Underlying Logic \- Intel, accessed January 20, 2026, [https://www.intel.com/content/www/us/en/developer/articles/technical/jax-openxla-running-process-and-underlying-logic-1.html](https://www.intel.com/content/www/us/en/developer/articles/technical/jax-openxla-running-process-and-underlying-logic-1.html)  
18. Geometry of Interaction and Linear Combinatory Algebras. \- ResearchGate, accessed January 20, 2026, [https://www.researchgate.net/publication/220173613\_Geometry\_of\_Interaction\_and\_Linear\_Combinatory\_Algebras](https://www.researchgate.net/publication/220173613_Geometry_of_Interaction_and_Linear_Combinatory_Algebras)  
19. Linear logic \- Wikipedia, accessed January 20, 2026, [https://en.wikipedia.org/wiki/Linear\_logic](https://en.wikipedia.org/wiki/Linear_logic)  
20. Geometry of Interaction explained \- RIMS, Kyoto University, accessed January 20, 2026, [https://www.kurims.kyoto-u.ac.jp/\~hassei/algi-13/kokyuroku/19\_shirahata.pdf](https://www.kurims.kyoto-u.ac.jp/~hassei/algi-13/kokyuroku/19_shirahata.pdf)  
21. Automatic vectorization \- JAX documentation, accessed January 20, 2026, [https://docs.jax.dev/en/latest/automatic-vectorization.html](https://docs.jax.dev/en/latest/automatic-vectorization.html)  
22. OBSERVATIONAL EQUIVALENCE AND FULL ABSTRACTION IN ..., accessed January 20, 2026, [https://lmcs.episciences.org/1150/pdf](https://lmcs.episciences.org/1150/pdf)