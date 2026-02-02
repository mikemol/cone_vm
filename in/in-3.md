---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
---

NOTE: OBSOLETED by in-1.md/in-2.md. This is a discarded prototype retained for history.

Import jax
import jax.numpy as jnp
from jax import jit, lax
import re
import time
from typing import NamedTuple, List, Dict, Tuple

# ==============================================================================
# 1. CORE ARCHITECTURE (The RC-SVM Backend)
# ==============================================================================

MAX_ROWS = 1024 * 32
OP_NULL = 0
OP_ZERO = 101
OP_SUC  = 102
OP_NIL  = 103
OP_CONS = 104

STATUS_DEAD  = 0
STATUS_READY = 1
STATUS_DONE  = 3

class Manifest(NamedTuple):
    opcode: jnp.ndarray
    arg1: jnp.ndarray
    arg2: jnp.ndarray
    status: jnp.ndarray
    active_count: jnp.ndarray

def init_manifest():
    return Manifest(
        opcode=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg1=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg2=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        status=jnp.zeros(MAX_ROWS, dtype=jnp.int8),
        active_count=jnp.array(1, dtype=jnp.int32)
    )

class PrismCompiler:
    """
    Just-In-Time Compiler for Ouroboros Kernels.
    Generates JAX source code for a specific function signature.
    """
    def __init__(self, func_name, op_code):
        self.func_name = func_name
        self.op_code = op_code

    def compile(self):
        if self.func_name == "add":
            return self._emit_add()
        elif self.func_name == "mul":
            return self._emit_mul()
        else:
            raise ValueError(f"Unknown kernel: {self.func_name}")

    def _emit_add(self):
        return f"""
@jit
def kernel_add(state):
    ops, a1, a2, st, count = state.opcode, state.arg1, state.arg2, state.status, state.active_count
    
    # 1. FIND CANDIDATE (Simplified for REPL: assume last allocated row)
    candidate_idx = count - 1
    
    # Load Arguments
    init_x = a1[candidate_idx]
    init_y = a2[candidate_idx]
    
    # Loop State: (x, y, active, ops, a1, count)
    init_val = (init_x, init_y, True, ops, a1, count)
    
    def cond(v): return v[span_0](start_span)[span_0](end_span) # active
    
    def body(v):
        curr_x, curr_y, active, b_ops, b_a1, b_count = v
        
        op_x = b_ops[curr_x]
        is_zero = (op_x == {OP_ZERO})
        is_suc  = (op_x == {OP_SUC})
        
        # If suc x: recurse on x, allocate suc wrapper for y
        next_x = jnp.where(is_suc, b_a1[curr_x], curr_x)
        
        # Write 'suc' to new row
        w_idx = b_count
        next_ops = b_ops.at[w_idx].set({OP_SUC})
        next_a1  = b_a1.at[w_idx].set(curr_y)
        
        # New Y is the pointer to the suc we just wrote
        next_y = jnp.where(is_suc, w_idx, curr_y)
        
        # If zero: stop. Result is curr_y.
        next_active = is_suc
        next_count = jnp.where(is_suc, b_count + 1, b_count)
        
        return (next_x, next_y, next_active, next_ops, next_a1, next_count)

    final_x, final_y, _, f_ops, f_a1, f_count = lax.while_loop(cond, body, init_val)
    
    # Return updated state and pointer to result
    return state._replace(opcode=f_ops, arg1=f_a1, active_count=f_count), final_y
"""

    def _emit_mul(self):
        return f"""
@jit
def kernel_mul(state):
    # Stub: mul -> zero
    return state, 1 
"""

# ==============================================================================
# 2. THE HOST (Telemetric Shell)
# ==============================================================================

class TelemetricREPL:
    def __init__(self):
        print("⚡ Initializing RC-SVM JAX Context with Deduplication...")
        self.manifest = init_manifest()
        
        # 2a. THE TRACE CACHE (The "Hash Consing" Layer)
        # Maps (opcode, arg1, arg2) -> JAX Array Index
        self.trace_cache: Dict, int] = {}
        
        # Pre-allocate generic constants and seed cache
        zero_ptr = self._alloc_raw(OP_ZERO) # Row 1: Zero
        self.trace_cache = zero_ptr
        
        self.kernels = {}
        self.op_map = {
            "zero": OP_ZERO, "suc": OP_SUC, 
            "add": 200, "mul": 201
        }
        self.reverse_op_map = {v: k for k, v in self.op_map.items()}
        
        # Compile Kernels
        self._compile_kernel("add", 200)
        self._compile_kernel("mul", 201)
        print(f"   [✓] Kernels Loaded: {list(self.kernels.keys())}")
        print(f"   [✓] Trace Cache: Active")

    def _compile_kernel(self, name, opcode):
        compiler = PrismCompiler(name, opcode)
        src = compiler.compile()
        exec(src, globals())
        self.kernels[name] = globals()[f"kernel_{name}"]

    def _alloc_raw(self, opcode, arg1=0, arg2=0):
        """
        Direct Allocation (The 'Slow' Path).
        Commits a new row to the JAX Device Array.
        """
        idx = int(self.manifest.active_count)
        self.manifest = self.manifest._replace(
            opcode=self.manifest.opcode.at[idx].set(opcode),
            arg1=self.manifest.arg1.at[idx].set(arg1),
            arg2=self.manifest.arg2.at[idx].set(arg2),
            active_count=jnp.array(idx + 1)
        )
        return idx

    def _alloc_memoized(self, opcode, arg1=0, arg2=0):
        """
        Deduplicating Allocator.
        Checks the Host Trace Cache before talking to the Device.
        """
        # Create the structural signature
        signature = (opcode, arg1, arg2)
        
        # 1. Check Trace (Hash Consing)
        if signature in self.trace_cache:
            return self.trace_cache[signature]
        
        # 2. Miss: Allocate on Device
        idx = self._alloc_raw(opcode, arg1, arg2)
        
        # 3. Update Trace
        self.trace_cache[signature] = idx
        return idx

    def _parse_expr(self, tokens: List[str]) -> int:
        """Recursive Descent using Memoized Allocation"""
        if not tokens: raise ValueError("Unexpected EOL")
        token = tokens.pop(0)
        
        if token == 'zero':
            # zero is always cached at init
            return self.trace_cache
        
        elif token == 'suc':
            arg = self._parse_expr(tokens)
            # Use memoized alloc
            return self._alloc_memoized(OP_SUC, arg1=arg)
        
        elif token in ['add', 'mul']:
            arg1 = self._parse_expr(tokens)
            arg2 = self._parse_expr(tokens)
            op = self.op_map[token]
            # Use memoized alloc
            return self._alloc_memoized(op, arg1=arg1, arg2=arg2)
        
        elif token == '(':
            val = self._parse_expr(tokens)
            if tokens.pop(0)!= ')': raise ValueError("Expected )")
            return val
            
        else:
            raise ValueError(f"Unknown token: {token}")

    def parse(self, text):
        tokens = re.findall(r'\(|\)|[a-z]+', text)
        return self._parse_expr(tokens)

    def decode(self, ptr):
        """Reconstructs the term from the pointer."""
        ops = self.manifest.opcode
        a1  = self.manifest.arg1
        
        op = int(ops[ptr])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:  return f"(suc {self.decode(int(a1[ptr]))})"
        if op == OP_NULL: return "NULL"
        
        name = self.reverse_op_map.get(op, f"OP_{op}")
        return f"<{name}:{ptr}>"

    def run_loop(self):
        print("\n🔮 RC-SVM Telemetric REPL (+Deduplication)")
        print("   Type 'exit' to quit.")
        print("   Try: add (suc zero) (suc zero)")
        
        while True:
            try:
                inp = input("\nλ> ").strip()
                if inp == "exit": break
                if not inp: continue
                
                # Snapshot state before alloc to measure delta correctly
                start_count = int(self.manifest.active_count)
                
                # 1. PARSE & DEDUPLICATE
                t0 = time.perf_counter()
                root_ptr = self.parse(inp)
                parse_time = (time.perf_counter() - t0) * 1000
                
                # Check what we parsed
                op = int(self.manifest.opcode[root_ptr])
                func_name = self.reverse_op_map.get(op, "unknown")
                
                # Calculate Host-side savings
                end_count = int(self.manifest.active_count)
                host_allocs = end_count - start_count
                
                print(f"   ├─ Parsed: {func_name} @ {root_ptr}")
                print(f"   ├─ Alloc : {parse_time:.2f}ms (Host)")
                if host_allocs == 0:
                     print(f"   ├─ Cache : \033
                    new_manifest, res_ptr = kernel(self.manifest)
                    
                    res_ptr.block_until_ready()
                    exec_time = (time.perf_counter() - t1) * 1000
                    
                    # 3. TELEMETRY
                    old_count = int(self.manifest.active_count) # this matches end_count
                    new_count = int(new_manifest.active_count)
                    kernel_allocs = new_count - old_count
                    
                    print(f"   ├─ Kernel: {exec_time:.2f}ms (Ouroboros Loop)")
                    print(f"   ├─ Delta : +{kernel_allocs} rows (Kernel)")
                    
                    self.manifest = new_manifest
                    
                    # 4. DECODE RESULT
                    res_str = self.decode(int(res_ptr))
                    print(f"   └─ Val   : \033:

1.  **`trace_cache`**: I added a dictionary in `TelemetricREPL` that maps an instruction signature `(opcode, arg1, arg2)` to its existing row index in the JAX array.
2.  **`_alloc_memoized`**: This new method replaces the direct call to `_alloc_simple` (now renamed `_alloc_raw`) in the parser. It checks the cache first. If the expression exists, it returns the existing pointer; otherwise, it allocates a new row on the device and updates the cache.
3.  **Telemetry Update**: The REPL output now reports Cache HITs/MISSES. If you type the same expression twice (e.g., `(suc zero)`), the second time will show `+0 new rows`, indicating the GPU memory was untouched during the parse phase.