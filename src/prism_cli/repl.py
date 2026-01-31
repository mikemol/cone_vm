import re
import time

# dataflow-bundle: block_size, do_global, do_sort, l1_block_size, l2_block_size, use_morton
# CLI sort/schedule bundle intentionally kept at the host interface.
# dataflow-bundle: block_size, use_morton
# Minimal CLI sort bundle for BSP entrypoints.
# dataflow-bundle: a1, a2
# Host-side pointer pair bundle in baseline/arena allocators.
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_core.modes import ValidateMode, coerce_validate_mode
from prism_core.safety import oob_mask
from prism_baseline.kernels import dispatch_kernel, kernel_add, kernel_mul, optimize_ptr
from prism_bsp.arena_step import cycle
from prism_bsp.config import ArenaSortConfig, DEFAULT_ARENA_SORT_CONFIG
from prism_bsp.intrinsic import cycle_intrinsic
from prism_metrics.gpu import _gpu_watchdog_create
from prism_metrics.metrics import (
    _damage_metrics_enabled,
    _damage_metrics_update,
    _damage_tile_size,
    cnf2_metrics_get,
    cnf2_metrics_reset,
    damage_metrics_get,
    damage_metrics_reset,
)
from prism_semantics.commit import apply_q
from prism_vm_core.types import (
    _arena_ptr,
    _committed_ids,
    _host_bool_value,
    _host_int_value,
    _host_raise_if_bad,
    _ledger_id,
    _manifest_ptr,
    _require_arena_ptr,
    _require_ledger_id,
    _require_manifest_ptr,
    ArenaPtr,
    LedgerId,
    ManifestPtr,
    OP_ADD,
    OP_MUL,
    OP_NAMES,
    OP_SUC,
    OP_ZERO,
    ZERO_PTR,
    NodeBatch,
)
from prism_vm_core.facade import (
    _key_order_commutative_host,
    _cnf2_enabled,
    _normalize_bsp_mode,
    cycle_candidates,
    init_arena,
    init_ledger,
    init_manifest,
    intern_nodes,
    node_batch,
)
from prism_core.modes import BspMode
from prism_vm_core.guards import _expect_token, _pop_token

_TEST_GUARDS = _jax_safe.TEST_GUARDS


class PrismVM:
    def __init__(self):
        print("⚡ Prism IR: Initializing Host Context...")
        self.manifest = init_manifest()
        self.active_count_host = _host_int_value(self.manifest.active_count)
        self.refresh_cache_on_eval = True
        self.trace_cache: Dict[Tuple[int, ManifestPtr, ManifestPtr], ManifestPtr] = {}
        self.canonical_ptrs: Dict[ManifestPtr, ManifestPtr] = {
            _manifest_ptr(0): _manifest_ptr(0)
        }
        zero_ptr = self._cons_raw(OP_ZERO, _manifest_ptr(0), _manifest_ptr(0))
        self.trace_cache[(OP_ZERO, _manifest_ptr(0), _manifest_ptr(0))] = zero_ptr
        self.canonical_ptrs[zero_ptr] = zero_ptr
        self.cache_filled_to = self.active_count_host
        self.kernels = {OP_ADD: kernel_add, OP_MUL: kernel_mul}

    def _cons_raw(self, op: int, a1: ManifestPtr, a2: ManifestPtr) -> ManifestPtr:
        _require_manifest_ptr(a1, "PrismVM._cons_raw a1")
        _require_manifest_ptr(a2, "PrismVM._cons_raw a2")
        a1_i, a2_i = _key_order_commutative_host(op, int(a1), int(a2))
        cap = int(self.manifest.opcode.shape[0])
        if self.active_count_host >= cap:
            self.manifest = self.manifest._replace(
                oom=jnp.array(True, dtype=jnp.bool_)
            )
            raise ValueError("Manifest capacity exceeded")
        idx = self.active_count_host
        self.active_count_host += 1
        self.manifest = self.manifest._replace(
            opcode=self.manifest.opcode.at[idx].set(op),
            arg1=self.manifest.arg1.at[idx].set(a1_i),
            arg2=self.manifest.arg2.at[idx].set(a2_i),
            active_count=jnp.array(self.active_count_host, dtype=jnp.int32),
        )
        return _manifest_ptr(idx)

    def _refresh_trace_cache(self, start_idx: int, end_idx: int) -> None:
        if end_idx <= start_idx:
            return
        ops = jax.device_get(self.manifest.opcode[start_idx:end_idx])
        a1s = jax.device_get(self.manifest.arg1[start_idx:end_idx])
        a2s = jax.device_get(self.manifest.arg2[start_idx:end_idx])
        for offset, (op, a1, a2) in enumerate(zip(ops, a1s, a2s)):
            ptr = _manifest_ptr(start_idx + offset)
            op_i = int(op)
            a1_i = int(self._canonical_ptr(_manifest_ptr(a1)))
            a2_i = int(self._canonical_ptr(_manifest_ptr(a2)))
            a1_i, a2_i = _key_order_commutative_host(op_i, a1_i, a2_i)
            signature = (op_i, _manifest_ptr(a1_i), _manifest_ptr(a2_i))
            if signature not in self.trace_cache:
                self.trace_cache[signature] = ptr
            self.canonical_ptrs[ptr] = ptr

    def _canonical_ptr(self, ptr: ManifestPtr) -> ManifestPtr:
        return self.canonical_ptrs.get(ptr, ptr)

    def cons(
        self,
        op: int,
        a1: ManifestPtr = ManifestPtr(0),
        a2: ManifestPtr = ManifestPtr(0),
    ) -> ManifestPtr:
        _require_manifest_ptr(a1, "PrismVM.cons a1")
        _require_manifest_ptr(a2, "PrismVM.cons a2")
        a1_i = int(self._canonical_ptr(a1))
        a2_i = int(self._canonical_ptr(a2))
        a1_i, a2_i = _key_order_commutative_host(op, a1_i, a2_i)
        signature = (op, _manifest_ptr(a1_i), _manifest_ptr(a2_i))
        if signature in self.trace_cache:
            return self.trace_cache[signature]
        ptr = self._cons_raw(op, _manifest_ptr(a1_i), _manifest_ptr(a2_i))
        self.trace_cache[signature] = ptr
        self.canonical_ptrs[ptr] = ptr
        return ptr

    def analyze_and_optimize(self, ptr: ManifestPtr) -> ManifestPtr:
        _require_manifest_ptr(ptr, "PrismVM.analyze_and_optimize ptr")
        ptr = self._canonical_ptr(ptr)
        ptr_arr = jnp.array(int(ptr), dtype=jnp.int32)
        opt_ptr, opt_reason = optimize_ptr(self.manifest, ptr_arr)
        opt_reason_i = _host_int_value(opt_reason)
        if opt_reason_i == 1:
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        elif opt_reason_i == 2:
            print("   [!] Static Analysis: Optimized (mul zero x) -> zero")
        return self._canonical_ptr(_manifest_ptr(_host_int_value(opt_ptr)))

    def eval(self, ptr: ManifestPtr) -> ManifestPtr:
        _require_manifest_ptr(ptr, "PrismVM.eval ptr")
        ptr = self._canonical_ptr(ptr)
        ptr_arr = jnp.array(int(ptr), dtype=jnp.int32)
        prev_count = self.active_count_host
        new_manifest, res_ptr, opt_reason = dispatch_kernel(self.manifest, ptr_arr)
        res_ptr.block_until_ready()
        self.manifest = new_manifest
        self.active_count_host = _host_int_value(self.manifest.active_count)
        if self.refresh_cache_on_eval and self.active_count_host > prev_count:
            self._refresh_trace_cache(prev_count, self.active_count_host)
            self.cache_filled_to = self.active_count_host
        if _host_bool_value(self.manifest.oom):
            raise RuntimeError("Manifest capacity exceeded during kernel execution")
        opt_reason_i = _host_int_value(opt_reason)
        if opt_reason_i == 1:
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        elif opt_reason_i == 2:
            print("   [!] Static Analysis: Optimized (mul zero x) -> zero")
        return self._canonical_ptr(_manifest_ptr(_host_int_value(res_ptr)))

    def parse(self, tokens) -> ManifestPtr:
        token = _pop_token(tokens)
        if token == "zero":
            return self.cons(OP_ZERO)
        if token == "suc":
            return self.cons(OP_SUC, self.parse(tokens))
        if token in ["add", "mul"]:
            op = OP_ADD if token == "add" else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self.cons(op, a1, a2)
        if token == "(":
            val = self.parse(tokens)
            _expect_token(tokens, ")")
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr: ManifestPtr) -> str:
        _require_manifest_ptr(ptr, "PrismVM.decode ptr")
        ptr_i = int(ptr)
        op = _host_int_value(self.manifest.opcode[ptr_i])
        if op == OP_ZERO:
            return "zero"
        if op == OP_SUC:
            return f"(suc {self.decode(_manifest_ptr(self.manifest.arg1[ptr_i]))})"
        if op == OP_ADD:
            return (
                f"(add {self.decode(_manifest_ptr(self.manifest.arg1[ptr_i]))} "
                f"{self.decode(_manifest_ptr(self.manifest.arg2[ptr_i]))})"
            )
        if op == OP_MUL:
            return (
                f"(mul {self.decode(_manifest_ptr(self.manifest.arg1[ptr_i]))} "
                f"{self.decode(_manifest_ptr(self.manifest.arg2[ptr_i]))})"
            )
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"


class PrismVM_BSP_Legacy:
    def __init__(self):
        print("⚡ Prism IR: Initializing BSP Arena (Legacy)...")
        self.arena = init_arena()

    def _alloc(
        self, op: int, a1: ArenaPtr = ArenaPtr(0), a2: ArenaPtr = ArenaPtr(0)
    ) -> ArenaPtr:
        cap = int(self.arena.opcode.shape[0])
        idx = _host_int_value(self.arena.count)
        if idx >= cap:
            self.arena = self.arena._replace(oom=jnp.array(True, dtype=jnp.bool_))
            raise ValueError("Arena capacity exceeded")
        _require_arena_ptr(a1, "PrismVM_BSP_Legacy._alloc a1")
        _require_arena_ptr(a2, "PrismVM_BSP_Legacy._alloc a2")
        a1_i, a2_i = _key_order_commutative_host(op, int(a1), int(a2))
        self.arena = self.arena._replace(
            opcode=self.arena.opcode.at[idx].set(op),
            arg1=self.arena.arg1.at[idx].set(a1_i),
            arg2=self.arena.arg2.at[idx].set(a2_i),
            count=jnp.array(idx + 1, dtype=jnp.int32),
        )
        return _arena_ptr(idx)

    def parse(self, tokens) -> ArenaPtr:
        token = _pop_token(tokens)
        if token == "zero":
            return self._alloc(OP_ZERO)
        if token == "suc":
            return self._alloc(OP_SUC, self.parse(tokens))
        if token in ["add", "mul"]:
            op = OP_ADD if token == "add" else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self._alloc(op, a1, a2)
        if token == "(":
            val = self.parse(tokens)
            _expect_token(tokens, ")")
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr: ArenaPtr, show_ids: bool = False) -> str:
        _require_arena_ptr(ptr, "PrismVM_BSP_Legacy.decode ptr")
        ptr_i = int(ptr)
        op = _host_int_value(self.arena.opcode[ptr_i])
        if op == OP_ZERO:
            return "zero"
        if op == OP_SUC:
            return (
                f"(suc {self.decode(_arena_ptr(self.arena.arg1[ptr_i]), show_ids=show_ids)})"
            )
        name = OP_NAMES.get(op, "?")
        if show_ids:
            return f"<{name}:{ptr}>"
        return f"<{name}>"


class PrismVM_BSP:
    def __init__(self):
        print("⚡ Prism IR: Initializing BSP Ledger...")
        self.ledger = init_ledger()

    def _intern(
        self, op: int, a1: LedgerId = LedgerId(0), a2: LedgerId = LedgerId(0)
    ) -> LedgerId:
        _require_ledger_id(a1, "PrismVM_BSP._intern a1")
        _require_ledger_id(a2, "PrismVM_BSP._intern a2")
        a1_i, a2_i = _key_order_commutative_host(op, int(a1), int(a2))
        ids, self.ledger = intern_nodes(
            self.ledger,
            node_batch(
                jnp.array([op], dtype=jnp.int32),
                jnp.array([a1_i], dtype=jnp.int32),
                jnp.array([a2_i], dtype=jnp.int32),
            ),
        )
        _host_raise_if_bad(
            self.ledger,
            "Ledger capacity exceeded during interning",
            oom_exc=ValueError,
        )
        return _ledger_id(ids[0])

    def parse(self, tokens) -> LedgerId:
        token = _pop_token(tokens)
        if token == "zero":
            return self._intern(OP_ZERO)
        if token == "suc":
            return self._intern(OP_SUC, self.parse(tokens))
        if token in ["add", "mul"]:
            op = OP_ADD if token == "add" else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self._intern(op, a1, a2)
        if token == "(":
            val = self.parse(tokens)
            _expect_token(tokens, ")")
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr: LedgerId) -> str:
        _require_ledger_id(ptr, "PrismVM_BSP.decode ptr")
        ptr_i = int(ptr)
        op = _host_int_value(self.ledger.opcode[ptr_i])
        if op == OP_ZERO:
            return "zero"
        if op == OP_SUC:
            return f"(suc {self.decode(_ledger_id(self.ledger.arg1[ptr_i]))})"
        if op == OP_ADD:
            return (
                f"(add {self.decode(_ledger_id(self.ledger.arg1[ptr_i]))} "
                f"{self.decode(_ledger_id(self.ledger.arg2[ptr_i]))})"
            )
        if op == OP_MUL:
            return (
                f"(mul {self.decode(_ledger_id(self.ledger.arg1[ptr_i]))} "
                f"{self.decode(_ledger_id(self.ledger.arg2[ptr_i]))})"
            )
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"


def make_vm(mode="baseline"):
    if mode == "arena":
        return PrismVM_BSP_Legacy()
    if mode == "bsp":
        return PrismVM_BSP()
    return PrismVM()


def run_program_lines(lines, vm=None):
    if vm is None:
        vm = PrismVM()
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith("#"):
            continue
        start_rows = _host_int_value(vm.manifest.active_count)
        t0 = time.perf_counter()
        tokens = re.findall(r"\(|\)|[a-z]+", inp)
        ir_ptr = vm.parse(tokens)
        ir_ptr_i = int(ir_ptr)
        parse_ms = (time.perf_counter() - t0) * 1000
        _ = parse_ms
        mid_rows = _host_int_value(vm.manifest.active_count)
        ir_allocs = mid_rows - start_rows
        ir_op = OP_NAMES.get(_host_int_value(vm.manifest.opcode[ir_ptr_i]), "?")
        print(f"   ├─ IR Build: {ir_op} @ {ir_ptr}")
        if ir_allocs == 0:
            print("   ├─ Cache   : \033[96mHIT (No new IR rows)\033[0m")
        else:
            print(f"   ├─ Cache   : MISS (+{ir_allocs} IR rows)")
        t1 = time.perf_counter()
        res_ptr = vm.eval(ir_ptr)
        eval_ms = (time.perf_counter() - t1) * 1000
        end_rows = _host_int_value(vm.manifest.active_count)
        exec_allocs = end_rows - mid_rows
        print(f"   ├─ Execute : {eval_ms:.2f}ms")
        if exec_allocs > 0:
            print(f"   ├─ Kernel  : +{exec_allocs} rows allocated")
        else:
            print("   ├─ Kernel  : \033[96mSKIPPED (Static Optimization)\033[0m")
        print(f"   └─ Result  : \033[92m{vm.decode(res_ptr)}\033[0m")
    return vm


def run_program_lines_arena(
    lines,
    vm=None,
    cycles=1,
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
):
    if vm is None:
        vm = PrismVM_BSP_Legacy()
    tile_size = _damage_tile_size(
        sort_cfg.block_size, sort_cfg.l2_block_size, sort_cfg.l1_block_size
    )
    watchdog = _gpu_watchdog_create()
    try:
        for inp in lines:
            inp = inp.strip()
            if not inp or inp.startswith("#"):
                continue
            if _damage_metrics_enabled():
                damage_metrics_reset()
            tokens = re.findall(r"\(|\)|[a-z]+", inp)
            root_ptr = vm.parse(tokens)
            for _ in range(max(1, cycles)):
                vm.arena, root_ptr = cycle(
                    vm.arena,
                    root_ptr,
                    sort_cfg=sort_cfg,
                )
                root_ptr = _arena_ptr(_host_int_value(root_ptr))
                if _host_bool_value(vm.arena.oom):
                    raise RuntimeError("Arena capacity exceeded during cycle")
            print(f"   ├─ Arena    : {_host_int_value(vm.arena.count)} nodes")
            if _damage_metrics_enabled():
                metrics = damage_metrics_get()
                print(
                    "   ├─ Damage   : "
                    f"cycles={metrics['cycles']} "
                    f"hot={metrics['hot_nodes']} "
                    f"edges={metrics['edge_cross']}/{metrics['edge_total']} "
                    f"rate={metrics['damage_rate']:.4f} "
                    f"tile={int(tile_size)}"
                )
            if watchdog is not None:
                stats = watchdog.poll()
                if stats is not None:
                    power = (
                        f"{stats['power_w']:.1f}W"
                        if stats["power_w"] is not None
                        else "na"
                    )
                    clock = (
                        f"{stats['sm_clock']}MHz"
                        if stats["sm_clock"] is not None
                        else "na"
                    )
                    print(
                        "   ├─ GPU      : "
                        f"util={stats['gpu_util']}% "
                        f"memio={stats['mem_io']}% "
                        f"vram={stats['vram_used_mb']}/{stats['vram_total_mb']}MB "
                        f"power={power} "
                        f"clock={clock}"
                    )
            print(f"   └─ Result   : \033[92m{vm.decode(root_ptr)}\033[0m")
    finally:
        if watchdog is not None:
            watchdog.close()
    return vm


def run_program_lines_bsp(
    lines,
    vm=None,
    cycles=1,
    do_sort=True,
    use_morton=False,
    block_size=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
    bsp_mode=BspMode.AUTO,
    validate_mode: ValidateMode = ValidateMode.NONE,
):
    validate_mode = coerce_validate_mode(
        validate_mode, context="run_program_lines_bsp"
    )
    if vm is None:
        vm = PrismVM_BSP()
    bsp_mode = _normalize_bsp_mode(bsp_mode)
    if bsp_mode == BspMode.CNF2 and not _cnf2_enabled():
        raise ValueError("bsp_mode='cnf2' disabled until m2")
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith("#"):
            continue
        tokens = re.findall(r"\(|\)|[a-z]+", inp)
        root_ptr = vm.parse(tokens)
        frontier = _committed_ids(jnp.array([int(root_ptr)], dtype=jnp.int32))
        for _ in range(max(1, cycles)):
            if bsp_mode == BspMode.INTRINSIC:
                vm.ledger, frontier_arr = cycle_intrinsic(vm.ledger, frontier.a)
                frontier = _committed_ids(frontier_arr)
            else:
                vm.ledger, frontier_prov, _, q_map = cycle_candidates(
                    vm.ledger, frontier, validate_mode=validate_mode
                )
                frontier, ok = apply_q(q_map, frontier_prov, return_ok=True)
                meta = getattr(q_map, "_prism_meta", None)
                if meta is not None and meta.safe_gather_policy is not None:
                    corrupt = jnp.any(
                        oob_mask(ok, policy=meta.safe_gather_policy)
                    )
                    vm.ledger = vm.ledger._replace(
                        corrupt=vm.ledger.corrupt | corrupt
                    )
            _host_raise_if_bad(vm.ledger, "Ledger capacity exceeded during cycle")
        root_ptr = frontier.a[0]
        root_ptr_int = _host_int_value(root_ptr)
        print(f"   ├─ Ledger   : {_host_int_value(vm.ledger.count)} nodes")
        print(f"   └─ Result  : \033[92m{vm.decode(_ledger_id(root_ptr_int))}\033[0m")
    return vm


def repl(
    mode="baseline",
    use_morton=False,
    block_size=None,
    bsp_mode=BspMode.AUTO,
    validate_mode: ValidateMode = ValidateMode.NONE,
):
    validate_mode = coerce_validate_mode(validate_mode, context="repl")
    if mode == "bsp":
        vm = PrismVM_BSP()
        bsp_mode = _normalize_bsp_mode(bsp_mode)
        mode_label = "CNF-2" if bsp_mode == BspMode.CNF2 else "Intrinsic"
        print(f"\n🔮 Prism IR Shell (BSP Ledger, {mode_label})")
        print("   Try: (add (suc zero) (suc zero))")
    elif mode == "arena":
        vm = PrismVM_BSP_Legacy()
        print("\n🔮 Prism IR Shell (Arena BSP, Legacy)")
        print("   Try: (add (suc zero) (suc zero))")
    else:
        vm = PrismVM()
        print("\n🔮 Prism IR Shell (Static Analysis + Deduplication)")
        print("   Try: (add (suc zero) (suc zero))")
        print("   Try: (add zero (suc (suc zero))) <- Triggers Optimizer")
    while True:
        try:
            inp = input("\nλ> ").strip()
            if inp == "exit":
                break
            if not inp:
                continue
            if mode == "bsp":
                run_program_lines_bsp(
                    [inp],
                    vm,
                    use_morton=use_morton,
                    block_size=block_size,
                    bsp_mode=bsp_mode,
                    validate_mode=validate_mode,
                )
            elif mode == "arena":
                sort_cfg = ArenaSortConfig(
                    do_sort=do_sort,
                    use_morton=use_morton,
                    block_size=block_size,
                    l2_block_size=l2_block_size,
                    l1_block_size=l1_block_size,
                    do_global=do_global,
                )
                run_program_lines_arena(
                    [inp],
                    vm,
                    sort_cfg=sort_cfg,
                )
            else:
                run_program_lines([inp], vm)
        except Exception as e:
            print(f"   ERROR: {e}")


def main():
    import sys

    args = sys.argv[1:]
    mode = "baseline"
    bsp_mode = BspMode.AUTO
    validate_mode: ValidateMode = ValidateMode.NONE
    cycles = 1
    do_sort = True
    use_morton = False
    block_size = None
    path = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--mode", "-m") and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
            continue
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--cycles" and i + 1 < len(args):
            cycles = int(args[i + 1])
            i += 2
            continue
        if arg.startswith("--cycles="):
            cycles = int(arg.split("=", 1)[1])
            i += 1
            continue
        if arg == "--bsp-mode" and i + 1 < len(args):
            bsp_mode = args[i + 1]
            i += 2
            continue
        if arg.startswith("--bsp-mode="):
            bsp_mode = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--validate-stratum":
            validate_mode = ValidateMode.STRICT
            i += 1
            continue
        if arg.startswith("--validate-stratum="):
            value = arg.split("=", 1)[1].strip().lower()
            if value in ("1", "true", "yes", "on"):
                validate_mode = ValidateMode.STRICT
            i += 1
            continue
        if arg == "--validate-mode" and i + 1 < len(args):
            validate_mode = coerce_validate_mode(
                args[i + 1], context="cli"
            )
            i += 2
            continue
        if arg.startswith("--validate-mode="):
            validate_mode = coerce_validate_mode(
                arg.split("=", 1)[1], context="cli"
            )
            i += 1
            continue
        if arg == "--no-sort":
            do_sort = False
            i += 1
            continue
        if arg == "--morton":
            use_morton = True
            i += 1
            continue
        if arg == "--block-size" and i + 1 < len(args):
            block_size = int(args[i + 1])
            i += 2
            continue
        if arg.startswith("--block-size="):
            block_size = int(arg.split("=", 1)[1])
            i += 1
            continue
        if path is None:
            path = arg
            i += 1
            continue
        i += 1
    if path:
        with open(path) as f:
            lines = f.readlines()
        if mode == "bsp":
            run_program_lines_bsp(
                lines,
                cycles=cycles,
                do_sort=do_sort,
                use_morton=use_morton,
                block_size=block_size,
                bsp_mode=bsp_mode,
                validate_mode=validate_mode,
            )
        elif mode == "arena":
            sort_cfg = ArenaSortConfig(
                do_sort=do_sort,
                use_morton=use_morton,
                block_size=block_size,
                l2_block_size=l2_block_size,
                l1_block_size=l1_block_size,
                do_global=do_global,
            )
            run_program_lines_arena(
                lines,
                cycles=cycles,
                sort_cfg=sort_cfg,
            )
        else:
            run_program_lines(lines)
    else:
        repl(
            mode=mode,
            use_morton=use_morton,
            block_size=block_size,
            bsp_mode=bsp_mode,
            validate_mode=validate_mode,
        )


__all__ = [
    "PrismVM",
    "PrismVM_BSP_Legacy",
    "PrismVM_BSP",
    "make_vm",
    "run_program_lines",
    "run_program_lines_arena",
    "run_program_lines_bsp",
    "repl",
    "main",
]
