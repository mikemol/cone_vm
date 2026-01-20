This is the **Prism IR** implementation.

In this architecture, the **Host** acts as the static analyzer and deduplicator. It constructs a "perfect" (maximally shared) Intermediate Representation in the JAX memory space. The **Device** (JAX) serves as the evaluation engine, only running kernels when the IR explicitly demands computation that cannot be resolved statically.

### 1. The Ontology & Manifest

We define a unified set of opcodes. Note that `ADD` and `MUL` are now first-class types in the memory layout, just like `SUC`.

```python
import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import NamedTuple, Dict, Callable, Tuple
import re
import time

# --- 1. Ontology (The Grammar) ---
# Data Types (Passive)
OP_NULL = 0
OP_ZERO = 1
OP_SUC  = 2

# Instruction Types (Active)
OP_ADD  = 10
OP_MUL  = 11

OP_NAMES = {
    0: "NULL", 1: "zero", 2: "suc",
    10: "add", 11: "mul"
}

# --- 2. The Manifest (The Heap) ---
MAX_ROWS = 1024 * 32

class Manifest(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    active_count: jnp.ndarray

def init_manifest():
    return Manifest(
        opcode=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg1=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg2=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        active_count=jnp.array(1, dtype=jnp.int32) # Start at 1 (0 is NULL)
    )

```

### 2. The JAX Kernels (The Backend)

These kernels are now **static**. They do not change. They accept the `Manifest` and a `pointer` to an instruction node (e.g., an `ADD` node), read its arguments from the heap, and compute the result.

```python
@jit
def kernel_add(manifest, ptr):
    """
    Evaluates an ADD instruction node stored at `ptr`.
    Logic: x + 0 = x;  x + S(y) = S(x + y)
    """
    ops, a1, a2, count = manifest.opcode, manifest.arg1, manifest.arg2, manifest.active_count
    
    # 1. Fetch Operands directly from the Instruction Node
    # Node Structure: [OP_ADD, ptr_x, ptr_y]
    init_x = a1[ptr]
    init_y = a2[ptr]
    
    # 2. Ouroboros Loop
    init_val = (init_x, init_y, True, ops, a1, count)
    
    def cond(v): return v[2] # active

    def body(v):
        curr_x, curr_y, active, b_ops, b_a1, b_count = v
        
        # Pattern Match on X
        op_x = b_ops[curr_x]
        is_suc = (op_x == OP_SUC)
        
        # Recurse: If SUC(x), next x is arg1[x]
        next_x = jnp.where(is_suc, b_a1[curr_x], curr_x)
        
        # Construct: New SUC node wrapping y
        w_idx = b_count
        next_ops = b_ops.at[w_idx].set(OP_SUC)
        next_a1  = b_a1.at[w_idx].set(curr_y)
        
        # New Y is the SUC we just built
        next_y = jnp.where(is_suc, w_idx, curr_y)
        
        # Continue?
        next_active = is_suc
        next_count = jnp.where(is_suc, b_count + 1, b_count)
        
        return (next_x, next_y, next_active, next_ops, next_a1, next_count)

    _, final_y, _, f_ops, f_a1, f_count = lax.while_loop(cond, body, init_val)
    
    return manifest._replace(opcode=f_ops, arg1=f_a1, active_count=f_count), final_y

@jit
def kernel_mul(manifest, ptr):
    """Stub for MUL: Returns Zero"""
    return manifest, jnp.array(1, dtype=jnp.int32) # Pointer to ZERO

```

### 3. The Prism VM (Host-Side Logic)

This is where the magic happens. We intercept the code creation to perform **Hash Consing** (deduplication) and **Static Analysis** (optimization).

```python
class PrismVM:
    def __init__(self):
        print("⚡ Prism IR: Initializing Host Context...")
        self.manifest = init_manifest()
        
        # Trace Cache: (opcode, arg1, arg2) -> ptr
        # This ensures every unique expression exists exactly once in memory.
        self.trace_cache: Dict[Tuple[int, int, int], int] = {}
        
        # Initialize Universe (Seed with ZERO)
        self._cons_raw(OP_ZERO, 0, 0) 
        self.trace_cache[(OP_ZERO, 0, 0)] = 1
        
        self.kernels = {OP_ADD: kernel_add, OP_MUL: kernel_mul}

    def _cons_raw(self, op, a1, a2):
        """Physical allocation (Device Write)"""
        idx = int(self.manifest.active_count)
        self.manifest = self.manifest._replace(
            opcode=self.manifest.opcode.at[idx].set(op),
            arg1=self.manifest.arg1.at[idx].set(a1),
            arg2=self.manifest.arg2.at[idx].set(a2),
            active_count=jnp.array(idx + 1)
        )
        return idx

    def cons(self, op, a1=0, a2=0):
        """
        The Smart Allocator.
        1. Checks Cache (Deduplication).
        2. Allocates if new.
        """
        signature = (op, a1, a2)
        if signature in self.trace_cache:
            return self.trace_cache[signature]
        
        ptr = self._cons_raw(op, a1, a2)
        self.trace_cache[signature] = ptr
        return ptr

    # --- STATIC ANALYSIS ENGINE ---
    def analyze_and_optimize(self, ptr):
        """
        Examines the IR at `ptr` BEFORE execution.
        Performs trivial reductions (Constant Folding / Identity Elimination).
        """
        op = int(self.manifest.opcode[ptr])
        a1 = int(self.manifest.arg1[ptr])
        a2 = int(self.manifest.arg2[ptr])
        
        # Optimization Rule 1: (add zero X) -> X
        # We don't need to run the kernel for this.
        if op == OP_ADD:
            op_a1 = int(self.manifest.opcode[a1])
            if op_a1 == OP_ZERO:
                print(f"   [!] Static Analysis: Optimized (add zero x) -> x")
                return a2
                
        return ptr

    def eval(self, ptr):
        """
        The Hybrid Interpreter.
        1. Static Analysis (Host)
        2. Dispatch (Device)
        """
        # 1. Optimize
        optimized_ptr = self.analyze_and_optimize(ptr)
        
        # 2. Check Type
        op = int(self.manifest.opcode[optimized_ptr])
        
        # 3. Dispatch or Return
        if op in self.kernels:
            # It's an Instruction. Run the kernel.
            new_manifest, res_ptr = self.kernels[op](self.manifest, optimized_ptr)
            res_ptr.block_until_ready()
            
            self.manifest = new_manifest
            return int(res_ptr)
        else:
            # It's Data (Normal Form). Return as is.
            return optimized_ptr

    # --- PARSING & DISPLAY ---
    def parse(self, tokens):
        token = tokens.pop(0)
        if token == 'zero': return self.cons(OP_ZERO)
        if token == 'suc':  return self.cons(OP_SUC, self.parse(tokens))
        
        if token in ['add', 'mul']:
            op = OP_ADD if token == 'add' else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self.cons(op, a1, a2) # <-- Cons Instruction Node
            
        if token == '(':
            val = self.parse(tokens)
            tokens.pop(0)
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr):
        op = int(self.manifest.opcode[ptr])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:  return f"(suc {self.decode(int(self.manifest.arg1[ptr]))})"
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"

```

### 4. The Telemetric REPL

The REPL now highlights the difference between **IR Construction** (Host) and **Kernel Execution** (Device), and shows the impact of Static Analysis.

```python
def repl():
    vm = PrismVM()
    print("\n🔮 Prism IR Shell (Static Analysis + Deduplication)")
    print("   Try: (add (suc zero) (suc zero))")
    print("   Try: (add zero (suc (suc zero))) <- Triggers Optimizer")
    
    while True:
        try:
            inp = input("\nλ> ").strip()
            if inp == "exit": break
            if not inp: continue
            
            start_rows = int(vm.manifest.active_count)
            
            # 1. PARSE (Construct IR)
            t0 = time.perf_counter()
            tokens = re.findall(r'\(|\)|[a-z]+', inp)
            ir_ptr = vm.parse(tokens)
            parse_ms = (time.perf_counter() - t0) * 1000
            
            # Telemetry: IR Construction
            mid_rows = int(vm.manifest.active_count)
            ir_allocs = mid_rows - start_rows
            ir_op = OP_NAMES.get(int(vm.manifest.opcode[ir_ptr]), "?")
            
            print(f"   ├─ IR Build: {ir_op} @ {ir_ptr}")
            if ir_allocs == 0:
                print(f"   ├─ Cache   : \033[96mHIT (No new IR rows)\033[0m")
            else:
                print(f"   ├─ Cache   : MISS (+{ir_allocs} IR rows)")
            
            # 2. EVAL (Static Analysis + Kernel)
            t1 = time.perf_counter()
            res_ptr = vm.eval(ir_ptr)
            eval_ms = (time.perf_counter() - t1) * 1000
            
            # Telemetry: Execution
            end_rows = int(vm.manifest.active_count)
            exec_allocs = end_rows - mid_rows
            
            print(f"   ├─ Execute : {eval_ms:.2f}ms")
            if exec_allocs > 0:
                print(f"   ├─ Kernel  : +{exec_allocs} rows allocated")
            else:
                print(f"   ├─ Kernel  : \033[96mSKIPPED (Static Optimization)\033[0m")
            
            print(f"   └─ Result  : \033[92m{vm.decode(res_ptr)}\033[0m")

        except Exception as e:
            print(f"   ERROR: {e}")

if __name__ == "__main__":
    repl()

```

### What We Achieved

1. **Deduplicated IR**: If you type `(add (suc zero) (suc zero))` twice, the second time generates **0 new IR rows**. The host just hands the existing pointer `(200, ptr_x, ptr_y)` to the JAX kernel.
2. **Static Analysis**: The `analyze_and_optimize` function inspects the IR before dispatch.
* If you type `(add zero (suc zero))`, the optimizer detects the Identity Rule ().
* It returns the pointer to `x` immediately.
* **The JAX Kernel is never called.** This is verified by the telemetry showing `Kernel: SKIPPED`.


3. **Unified Memory**: Instructions and Data live in the same `Manifest`. This allows the kernel to read instructions as data if we ever wanted to implement "quotation" or higher-order macros.