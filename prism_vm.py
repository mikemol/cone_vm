from jax import jit, lax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Callable, Tuple
import re
import time

# --- 1. Ontology (Opcodes) ---
OP_NULL = 0
OP_ZERO = 1
OP_SUC  = 2
OP_ADD  = 10
OP_MUL  = 11
OP_SORT = 99

OP_NAMES = {
    0: "NULL", 1: "zero", 2: "suc",
    10: "add", 11: "mul", 99: "sort"
}

MAX_ROWS = 1024 * 32
MAX_NODES = 1024 * 64

# --- Rank (2-bit Scheduler) ---
RANK_HOT = 0
RANK_WARM = 1
RANK_COLD = 2
RANK_FREE = 3

# --- 2. Manifest (Heap) ---
class Manifest(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    active_count: jnp.ndarray

class Arena(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    rank:   jnp.ndarray
    count:  jnp.ndarray

def init_manifest():
    return Manifest(
        opcode=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg1=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg2=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        active_count=jnp.array(1, dtype=jnp.int32)
    )

def init_arena():
    arena = Arena(
        opcode=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        arg1=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        arg2=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        rank=jnp.full(MAX_NODES, RANK_FREE, dtype=jnp.int8),
        count=jnp.array(1, dtype=jnp.int32),
    )
    arena = arena._replace(
        opcode=arena.opcode.at[1].set(OP_ZERO),
        arg1=arena.arg1.at[1].set(0),
        arg2=arena.arg2.at[1].set(0),
        count=jnp.array(2, dtype=jnp.int32),
    )
    return arena

@jit
def op_rank(arena):
    ops = arena.opcode
    is_free = ops == OP_NULL
    is_inst = ops >= 10
    new_rank = jnp.where(is_free, RANK_FREE, jnp.where(is_inst, RANK_HOT, RANK_COLD))
    return arena._replace(rank=new_rank.astype(jnp.int8))

@jit
def op_sort_and_swizzle(arena):
    size = arena.rank.shape[0]
    idx = jnp.arange(size, dtype=jnp.int32)
    sort_key = arena.rank.astype(jnp.int32) * (size + 1) + idx
    perm = jnp.argsort(sort_key)
    inv_perm = jnp.argsort(perm)
    new_ops = arena.opcode[perm]
    new_arg1 = arena.arg1[perm]
    new_arg2 = arena.arg2[perm]
    new_rank = arena.rank[perm]
    swizzled_arg1 = jnp.where(new_arg1 != 0, inv_perm[new_arg1], 0)
    swizzled_arg2 = jnp.where(new_arg2 != 0, inv_perm[new_arg2], 0)
    active_count = jnp.sum(new_rank != RANK_FREE).astype(jnp.int32)
    return Arena(new_ops, swizzled_arg1, swizzled_arg2, new_rank, active_count)

@jit
def op_interact(arena):
    ops = arena.opcode
    a1 = arena.arg1
    a2 = arena.arg2
    is_hot = arena.rank == RANK_HOT
    is_add = ops == OP_ADD
    op_x = ops[a1]
    is_zero = op_x == OP_ZERO
    is_suc = op_x == OP_SUC
    mask_zero = is_hot & is_add & is_zero
    mask_suc = is_hot & is_add & is_suc

    spawn_counts = jnp.where(mask_suc, 1, 0)
    offsets = jnp.cumsum(spawn_counts) - spawn_counts
    total_spawn = jnp.sum(spawn_counts).astype(jnp.int32)
    base_free = arena.count
    new_add_idx = base_free + offsets

    grandchild_x = a1[a1]

    new_ops = jnp.where(mask_suc, OP_SUC, ops)
    new_a1 = jnp.where(mask_suc, new_add_idx, a1)
    new_a2 = jnp.where(mask_suc, 0, a2)

    y_op = ops[a2]
    y_a1 = a1[a2]
    y_a2 = a2[a2]
    new_ops = jnp.where(mask_zero, y_op, new_ops)
    new_a1 = jnp.where(mask_zero, y_a1, new_a1)
    new_a2 = jnp.where(mask_zero, y_a2, new_a2)

    safe_idx = jnp.where(mask_suc, new_add_idx, 0)
    safe_op = jnp.where(mask_suc, OP_ADD, new_ops[0])
    safe_a1 = jnp.where(mask_suc, grandchild_x, new_a1[0])
    safe_a2 = jnp.where(mask_suc, a2, new_a2[0])

    final_ops = new_ops.at[safe_idx].set(safe_op, mode="drop")
    final_a1 = new_a1.at[safe_idx].set(safe_a1, mode="drop")
    final_a2 = new_a2.at[safe_idx].set(safe_a2, mode="drop")

    new_count = arena.count + total_spawn
    return Arena(final_ops, final_a1, final_a2, arena.rank, new_count)

# --- 3. JAX Kernels (Static) ---
# --- 3. JAX Kernels (Static) ---
@jit
def kernel_add(manifest, ptr):
    ops, a1, a2, count = manifest.opcode, manifest.arg1, manifest.arg2, manifest.active_count
    init_x = a1[ptr]
    init_y = a2[ptr]
    init_val = (init_x, init_y, True, ops, a1, count)

    def cond(v): return v[2]

    def body(v):
        curr_x, curr_y, active, b_ops, b_a1, b_count = v
        op_x = b_ops[curr_x]
        is_suc = (op_x == OP_SUC)
        next_x = jnp.where(is_suc, b_a1[curr_x], curr_x)
        w_idx = b_count
        next_ops = b_ops.at[w_idx].set(OP_SUC)
        next_a1  = b_a1.at[w_idx].set(curr_y)
        next_y = jnp.where(is_suc, w_idx, curr_y)
        next_active = is_suc
        next_count = jnp.where(is_suc, b_count + 1, b_count)
        return (next_x, next_y, next_active, next_ops, next_a1, next_count)

    _, final_y, _, f_ops, f_a1, f_count = lax.while_loop(cond, body, init_val)
    return manifest._replace(opcode=f_ops, arg1=f_a1, active_count=f_count), final_y

# Kernel stub for MUL
@jit
def kernel_mul(manifest, ptr):
    # Returns pointer to ZERO (1)
    return manifest, jnp.array(1, dtype=jnp.int32)

def _dispatch_identity(args):
    manifest, ptr = args
    return manifest, ptr

def _dispatch_add(args):
    manifest, ptr = args
    return kernel_add(manifest, ptr)

def _dispatch_mul(args):
    manifest, ptr = args
    return kernel_mul(manifest, ptr)

@jit
def optimize_ptr(manifest, ptr):
    ops, a1s, a2s = manifest.opcode, manifest.arg1, manifest.arg2
    op = ops[ptr]
    a1 = a1s[ptr]
    a2 = a2s[ptr]
    op_a1 = ops[a1]
    is_add = op == OP_ADD
    is_zero = op_a1 == OP_ZERO
    optimized = jnp.logical_and(is_add, is_zero)
    out_ptr = jnp.where(optimized, a2, ptr)
    return out_ptr, optimized

@jit
def dispatch_kernel(manifest, ptr):
    opt_ptr, opt_applied = optimize_ptr(manifest, ptr)
    op = manifest.opcode[opt_ptr]
    case_index = jnp.where(op == OP_ADD, 1, jnp.where(op == OP_MUL, 2, 0))
    new_manifest, res_ptr = lax.switch(
        case_index,
        (_dispatch_identity, _dispatch_add, _dispatch_mul),
        (manifest, opt_ptr),
    )
    return new_manifest, res_ptr, opt_applied


# --- 4. Prism VM (Host Logic) ---
class PrismVM:
    def __init__(self):
        print("⚡ Prism IR: Initializing Host Context...")
        self.manifest = init_manifest()
        # Trace Cache: (opcode, arg1, arg2) -> ptr
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
            active_count=jnp.array(idx + 1, dtype=jnp.int32)
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
        ptr_arr = jnp.array(ptr, dtype=jnp.int32)
        opt_ptr, opt_applied = optimize_ptr(self.manifest, ptr_arr)
        if bool(opt_applied):
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        return int(opt_ptr)

    def eval(self, ptr):
        """
        The Hybrid Interpreter.
        1. Static Analysis (Host)
        2. Dispatch (Device)
        """
        ptr_arr = jnp.array(ptr, dtype=jnp.int32)
        new_manifest, res_ptr, opt_applied = dispatch_kernel(self.manifest, ptr_arr)
        res_ptr.block_until_ready()
        self.manifest = new_manifest
        if bool(opt_applied):
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        return int(res_ptr)

    # --- PARSING & DISPLAY ---
    def parse(self, tokens):
        token = tokens.pop(0)
        if token == 'zero': return self.cons(OP_ZERO)
        if token == 'suc':  return self.cons(OP_SUC, self.parse(tokens))
        if token in ['add', 'mul']:
            op = OP_ADD if token == 'add' else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self.cons(op, a1, a2)
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


# --- 5. Telemetric REPL ---
def run_program_lines(lines, vm=None):
    if vm is None:
        vm = PrismVM()
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith('#'):
            continue
        start_rows = int(vm.manifest.active_count)
        t0 = time.perf_counter()
        tokens = re.findall(r'\(|\)|[a-z]+', inp)
        ir_ptr = vm.parse(tokens)
        parse_ms = (time.perf_counter() - t0) * 1000
        mid_rows = int(vm.manifest.active_count)
        ir_allocs = mid_rows - start_rows
        ir_op = OP_NAMES.get(int(vm.manifest.opcode[ir_ptr]), "?")
        print(f"   ├─ IR Build: {ir_op} @ {ir_ptr}")
        if ir_allocs == 0:
            print(f"   ├─ Cache   : \033[96mHIT (No new IR rows)\033[0m")
        else:
            print(f"   ├─ Cache   : MISS (+{ir_allocs} IR rows)")
        t1 = time.perf_counter()
        res_ptr = vm.eval(ir_ptr)
        eval_ms = (time.perf_counter() - t1) * 1000
        end_rows = int(vm.manifest.active_count)
        exec_allocs = end_rows - mid_rows
        print(f"   ├─ Execute : {eval_ms:.2f}ms")
        if exec_allocs > 0:
            print(f"   ├─ Kernel  : +{exec_allocs} rows allocated")
        else:
            print(f"   ├─ Kernel  : \033[96mSKIPPED (Static Optimization)\033[0m")
        print(f"   └─ Result  : \033[92m{vm.decode(res_ptr)}\033[0m")
    return vm

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
            run_program_lines([inp], vm)
        except Exception as e:
            print(f"   ERROR: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
        run_program_lines(lines)
    else:
        repl()
