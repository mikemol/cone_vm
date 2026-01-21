import argparse
import csv
import importlib
import io
import os
import random
import re
import sys
import time
from contextlib import redirect_stdout
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import numpy as np

import prism_vm as pv


class Workload(NamedTuple):
    name: str
    kind: str
    build: Callable[[], object]
    meta: Dict[str, object]


def _suppress_output():
    return redirect_stdout(io.StringIO())


def _load_prism_vm(swizzle_backend: str):
    global pv
    os.environ["PRISM_SWIZZLE_BACKEND"] = swizzle_backend
    if "prism_vm" in sys.modules:
        pv = importlib.reload(sys.modules["prism_vm"])
    else:
        import prism_vm as pv_module

        pv = pv_module
    return pv


def _parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in _parse_csv_list(raw):
        if not item.isdigit():
            continue
        value = int(item)
        if value > 0:
            values.append(value)
    return values


def _apply_workload_meta(row: Dict[str, object], workload: Workload) -> None:
    for key, value in workload.meta.items():
        row[f"workload_{key}"] = value


def _suc_chain(n: int) -> str:
    expr = "zero"
    for _ in range(n):
        expr = f"(suc {expr})"
    return expr


def _balanced_add(depth: int, leaf_suc: int) -> str:
    if depth <= 0:
        return _suc_chain(leaf_suc)
    left = _balanced_add(depth - 1, leaf_suc)
    right = _balanced_add(depth - 1, leaf_suc)
    return f"(add {left} {right})"


def _add_chain(length: int, leaf_suc: int) -> str:
    expr = _suc_chain(leaf_suc)
    for _ in range(length):
        expr = f"(add {expr} {_suc_chain(leaf_suc)})"
    return expr


def _reuse_tree(repeats: int) -> str:
    sub = "(add (suc zero) (suc zero))"
    expr = sub
    for _ in range(repeats):
        expr = f"(add {expr} {expr})"
    return expr


def _make_random_arena(
    seed: int,
    active_count: int,
    add_ratio: float,
    suc_ratio: float,
    spawn_ratio: float,
) -> pv.Arena:
    size = pv.MAX_NODES
    rng = random.Random(seed)
    opcode = np.zeros(size, dtype=np.int32)
    arg1 = np.zeros(size, dtype=np.int32)
    arg2 = np.zeros(size, dtype=np.int32)

    active_count = max(2, min(active_count, size - 1))
    opcode[0] = pv.OP_NULL
    opcode[1] = pv.OP_ZERO

    for i in range(2, active_count):
        r = rng.random()
        if r < add_ratio:
            opcode[i] = pv.OP_ADD
        elif r < add_ratio + suc_ratio:
            opcode[i] = pv.OP_SUC
        else:
            opcode[i] = pv.OP_ZERO

    suc_idx = [i for i in range(1, active_count) if opcode[i] == pv.OP_SUC]
    zero_idx = [i for i in range(1, active_count) if opcode[i] == pv.OP_ZERO]
    if not zero_idx:
        zero_idx = [1]
        opcode[1] = pv.OP_ZERO

    add_idx = [i for i in range(2, active_count) if opcode[i] == pv.OP_ADD]
    free_slots = size - active_count
    if not suc_idx or free_slots <= 0:
        spawn_count = 0
    else:
        max_spawn = min(len(add_idx), max(0, free_slots - 1))
        target_spawn = int(len(add_idx) * spawn_ratio)
        spawn_count = min(target_spawn, max_spawn)
    spawn_set = set(rng.sample(add_idx, spawn_count)) if spawn_count > 0 else set()

    for i in add_idx:
        if i in spawn_set and suc_idx:
            arg1[i] = rng.choice(suc_idx)
        else:
            arg1[i] = rng.choice(zero_idx)
        arg2[i] = rng.randint(0, active_count - 1)

    for i in range(2, active_count):
        if opcode[i] == pv.OP_ADD:
            continue
        if rng.random() < 0.5:
            arg1[i] = rng.randint(0, active_count - 1)
        if rng.random() < 0.5:
            arg2[i] = rng.randint(0, active_count - 1)

    return pv.Arena(
        opcode=jnp.array(opcode, dtype=jnp.int32),
        arg1=jnp.array(arg1, dtype=jnp.int32),
        arg2=jnp.array(arg2, dtype=jnp.int32),
        rank=jnp.full(size, pv.RANK_FREE, dtype=jnp.int8),
        count=jnp.array(active_count, dtype=jnp.int32),
    )


def _make_hierarchy_arena(
    l2_block_size: int,
    l1_block_size: int,
) -> pv.Arena:
    size = pv.MAX_NODES
    active_count = min(size - 1, l1_block_size * 2)
    opcode = np.zeros(size, dtype=np.int32)
    arg1 = np.zeros(size, dtype=np.int32)
    arg2 = np.zeros(size, dtype=np.int32)

    opcode[0] = pv.OP_NULL
    opcode[1] = pv.OP_ZERO
    for i in range(2, active_count):
        opcode[i] = pv.OP_ZERO

    hot_indices = [
        min(active_count - 1, l2_block_size + 3),
        min(active_count - 1, l2_block_size * 3 + 7),
        min(active_count - 1, l1_block_size + l2_block_size + 5),
        min(active_count - 1, l1_block_size + l2_block_size * 2 + 9),
    ]
    for idx in hot_indices:
        opcode[idx] = pv.OP_ADD
        arg1[idx] = 1
        arg2[idx] = 1

    return pv.Arena(
        opcode=jnp.array(opcode, dtype=jnp.int32),
        arg1=jnp.array(arg1, dtype=jnp.int32),
        arg2=jnp.array(arg2, dtype=jnp.int32),
        rank=jnp.full(size, pv.RANK_FREE, dtype=jnp.int8),
        count=jnp.array(active_count, dtype=jnp.int32),
    )


def _compute_arena_metrics(
    arena: pv.Arena,
    block_size: Optional[int],
    l2_block_size: Optional[int],
    l1_block_size: Optional[int],
) -> Dict[str, float]:
    opcode = np.asarray(arena.opcode)
    arg1 = np.asarray(arena.arg1)
    arg2 = np.asarray(arena.arg2)
    idx = np.arange(opcode.shape[0], dtype=np.int32)

    is_free = opcode == pv.OP_NULL
    is_inst = opcode >= 10
    hot = int(np.sum(is_inst))
    free = int(np.sum(is_free))
    cold = int(opcode.shape[0] - hot - free)
    warm = 0

    node_mask = ~is_free
    mask1 = node_mask & (arg1 != 0)
    mask2 = node_mask & (arg2 != 0)

    dist1 = np.abs(idx[mask1] - arg1[mask1])
    dist2 = np.abs(idx[mask2] - arg2[mask2])
    if dist1.size and dist2.size:
        dists = np.concatenate([dist1, dist2])
    elif dist1.size:
        dists = dist1
    elif dist2.size:
        dists = dist2
    else:
        dists = np.array([], dtype=np.int32)

    mean_dist = float(dists.mean()) if dists.size else 0.0
    p95_dist = float(np.percentile(dists, 95)) if dists.size else 0.0

    block_local_pct = None
    if block_size and block_size > 0:
        block_idx = idx // block_size
        within1 = block_idx[mask1] == (arg1[mask1] // block_size)
        within2 = block_idx[mask2] == (arg2[mask2] // block_size)
        if within1.size and within2.size:
            within = np.concatenate([within1, within2])
        elif within1.size:
            within = within1
        elif within2.size:
            within = within2
        else:
            within = np.array([], dtype=bool)
        block_local_pct = float(within.mean()) if within.size else 0.0

    hot_count = hot
    hot_in_global_head_l1_pct = 0.0
    hot_in_global_head_l2_pct = 0.0
    hot_in_l1_head_l2_pct = 0.0

    if l1_block_size and l1_block_size > 0 and hot_count > 0:
        hot_in_global_head_l1_pct = float(np.sum(is_inst[:l1_block_size])) / hot_count
    if l2_block_size and l2_block_size > 0 and hot_count > 0:
        hot_in_global_head_l2_pct = float(np.sum(is_inst[:l2_block_size])) / hot_count
    if (
        l1_block_size
        and l2_block_size
        and l1_block_size > 0
        and l2_block_size > 0
        and hot_count > 0
    ):
        l2_in_l1 = (idx[is_inst] % l1_block_size) // l2_block_size
        hot_in_l1_head_l2_pct = float(np.sum(l2_in_l1 == 0)) / hot_count

    return {
        "hot": hot,
        "warm": warm,
        "cold": cold,
        "free": free,
        "mean_edge_dist": mean_dist,
        "p95_edge_dist": p95_dist,
        "block_local_pct": block_local_pct if block_local_pct is not None else 0.0,
        "hot_in_global_head_l1_pct": hot_in_global_head_l1_pct,
        "hot_in_global_head_l2_pct": hot_in_global_head_l2_pct,
        "hot_in_l1_head_l2_pct": hot_in_l1_head_l2_pct,
    }


def _run_baseline_expr(expr: str) -> Dict[str, object]:
    vm = pv.PrismVM()
    init_count = int(vm.manifest.active_count)

    t0 = time.perf_counter()
    tokens = re.findall(r"\(|\)|[a-z]+", expr)
    ir_ptr = vm.parse(tokens)
    vm.manifest.opcode.block_until_ready()
    parse_ms = (time.perf_counter() - t0) * 1000
    start_count = int(vm.manifest.active_count)

    t1 = time.perf_counter()
    _ = vm.eval(ir_ptr)
    exec_ms = (time.perf_counter() - t1) * 1000
    end_count = int(vm.manifest.active_count)

    return {
        "parse_ms": parse_ms,
        "exec_ms": exec_ms,
        "parse_allocs": start_count - init_count,
        "exec_allocs": end_count - start_count,
        "final_count": end_count,
    }


def _run_bsp_expr(
    expr: str,
    cycles: int,
    do_sort: bool,
    use_morton: bool,
    block_size: Optional[int],
    metrics_block_size: Optional[int],
    l2_block_size: Optional[int],
    l1_block_size: Optional[int],
    do_global: bool,
    metrics_l2_block_size: Optional[int],
    metrics_l1_block_size: Optional[int],
) -> Dict[str, object]:
    vm = pv.PrismVM_BSP_Legacy()
    init_count = int(vm.arena.count)

    t0 = time.perf_counter()
    tokens = re.findall(r"\(|\)|[a-z]+", expr)
    root_ptr = vm.parse(tokens)
    vm.arena.opcode.block_until_ready()
    parse_ms = (time.perf_counter() - t0) * 1000
    start_count = int(vm.arena.count)

    pre_metrics = _compute_arena_metrics(
        vm.arena, metrics_block_size, metrics_l2_block_size, metrics_l1_block_size
    )

    t1 = time.perf_counter()
    for _ in range(cycles):
        vm.arena, root_ptr = pv.cycle(
            vm.arena,
            root_ptr,
            do_sort=do_sort,
            use_morton=use_morton,
            block_size=block_size,
            l2_block_size=l2_block_size,
            l1_block_size=l1_block_size,
            do_global=do_global,
        )
    vm.arena.opcode.block_until_ready()
    exec_ms = (time.perf_counter() - t1) * 1000

    end_count = int(vm.arena.count)
    post_metrics = _compute_arena_metrics(
        vm.arena, metrics_block_size, metrics_l2_block_size, metrics_l1_block_size
    )

    return {
        "parse_ms": parse_ms,
        "exec_ms": exec_ms,
        "parse_allocs": start_count - init_count,
        "exec_allocs": end_count - start_count,
        "final_count": end_count,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
    }


def _run_bsp_arena(
    arena: pv.Arena,
    root_ptr: int,
    cycles: int,
    do_sort: bool,
    use_morton: bool,
    block_size: Optional[int],
    metrics_block_size: Optional[int],
    l2_block_size: Optional[int],
    l1_block_size: Optional[int],
    do_global: bool,
    metrics_l2_block_size: Optional[int],
    metrics_l1_block_size: Optional[int],
) -> Dict[str, object]:
    pre_metrics = _compute_arena_metrics(
        arena, metrics_block_size, metrics_l2_block_size, metrics_l1_block_size
    )
    start_count = int(arena.count)

    t1 = time.perf_counter()
    for _ in range(cycles):
        arena, root_ptr = pv.cycle(
            arena,
            root_ptr,
            do_sort=do_sort,
            use_morton=use_morton,
            block_size=block_size,
            l2_block_size=l2_block_size,
            l1_block_size=l1_block_size,
            do_global=do_global,
        )
    arena.opcode.block_until_ready()
    exec_ms = (time.perf_counter() - t1) * 1000

    end_count = int(arena.count)
    post_metrics = _compute_arena_metrics(
        arena, metrics_block_size, metrics_l2_block_size, metrics_l1_block_size
    )

    return {
        "parse_ms": 0.0,
        "exec_ms": exec_ms,
        "parse_allocs": 0,
        "exec_allocs": end_count - start_count,
        "final_count": end_count,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
    }


def _parse_csv_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_modes(
    block_sizes: List[int],
    include_baseline: bool,
    hierarchy_l1_mult: int,
    include_hierarchy_global: bool,
    include_hierarchy_morton: bool,
) -> List[Dict[str, object]]:
    modes: List[Dict[str, object]] = []
    if include_baseline:
        modes.append({"name": "baseline", "kind": "baseline"})
    modes.append(
        {
            "name": "bsp-rank",
            "kind": "bsp",
            "do_sort": True,
            "use_morton": False,
            "block_size": None,
            "l2_block_size": None,
            "l1_block_size": None,
            "do_global": False,
        }
    )
    modes.append(
        {
            "name": "bsp-nosort",
            "kind": "bsp",
            "do_sort": False,
            "use_morton": False,
            "block_size": None,
            "l2_block_size": None,
            "l1_block_size": None,
            "do_global": False,
        }
    )
    modes.append(
        {
            "name": "bsp-morton",
            "kind": "bsp",
            "do_sort": True,
            "use_morton": True,
            "block_size": None,
            "l2_block_size": None,
            "l1_block_size": None,
            "do_global": False,
        }
    )
    for block_size in block_sizes:
        modes.append(
            {
                "name": f"bsp-block{block_size}",
                "kind": "bsp",
                "do_sort": True,
                "use_morton": False,
                "block_size": block_size,
                "l2_block_size": None,
                "l1_block_size": None,
                "do_global": False,
            }
        )
        modes.append(
            {
                "name": f"bsp-morton-block{block_size}",
                "kind": "bsp",
                "do_sort": True,
                "use_morton": True,
                "block_size": block_size,
                "l2_block_size": None,
                "l1_block_size": None,
                "do_global": False,
            }
        )

        l1_block_size = block_size * max(1, hierarchy_l1_mult)
        if l1_block_size > block_size and pv.MAX_NODES % l1_block_size == 0:
            modes.append(
                {
                    "name": f"bsp-hier-l2{block_size}-l1{l1_block_size}",
                    "kind": "bsp",
                    "do_sort": True,
                    "use_morton": False,
                    "block_size": None,
                    "l2_block_size": block_size,
                    "l1_block_size": l1_block_size,
                    "do_global": False,
                }
            )
            if include_hierarchy_global:
                modes.append(
                    {
                        "name": f"bsp-hier-global-l2{block_size}-l1{l1_block_size}",
                        "kind": "bsp",
                        "do_sort": True,
                        "use_morton": False,
                        "block_size": None,
                        "l2_block_size": block_size,
                        "l1_block_size": l1_block_size,
                        "do_global": True,
                    }
                )
            if include_hierarchy_morton:
                modes.append(
                    {
                        "name": f"bsp-hier-morton-l2{block_size}-l1{l1_block_size}",
                        "kind": "bsp",
                        "do_sort": True,
                        "use_morton": True,
                        "block_size": None,
                        "l2_block_size": block_size,
                        "l1_block_size": l1_block_size,
                        "do_global": False,
                    }
                )
                if include_hierarchy_global:
                    modes.append(
                        {
                            "name": f"bsp-hier-morton-global-l2{block_size}-l1{l1_block_size}",
                            "kind": "bsp",
                            "do_sort": True,
                            "use_morton": True,
                            "block_size": None,
                            "l2_block_size": block_size,
                            "l1_block_size": l1_block_size,
                            "do_global": True,
                        }
                    )
    return modes


def _build_workloads(
    seed: int,
    arena_counts: Optional[List[int]],
    hierarchy_pairs: List[Tuple[int, int]],
) -> List[Workload]:
    random.seed(seed)
    workloads: List[Workload] = [
        Workload("expr_small", "expr", lambda: "(add (suc zero) (suc zero))", {}),
        Workload("expr_reuse", "expr", lambda: _reuse_tree(3), {}),
        Workload("expr_balanced", "expr", lambda: _balanced_add(6, 2), {}),
        Workload("expr_chain", "expr", lambda: _add_chain(64, 1), {}),
    ]

    if arena_counts:
        for i, count in enumerate(arena_counts):
            workloads.append(
                Workload(
                    f"arena_dense_{count}",
                    "arena",
                    lambda c=count, s=seed + 1 + i: (
                        _make_random_arena(s, c, 0.45, 0.35, 0.35),
                        1,
                    ),
                    {"arena_count": count, "arena_kind": "dense"},
                )
            )
            workloads.append(
                Workload(
                    f"arena_sparse_{count}",
                    "arena",
                    lambda c=count, s=seed + 11 + i: (
                        _make_random_arena(s, c, 0.30, 0.25, 0.25),
                        1,
                    ),
                    {"arena_count": count, "arena_kind": "sparse"},
                )
            )
            workloads.append(
                Workload(
                    f"arena_shatter_{count}",
                    "arena",
                    lambda c=count, s=seed + 21 + i: (
                        _make_random_arena(s, c, 0.55, 0.30, 0.55),
                        1,
                    ),
                    {"arena_count": count, "arena_kind": "shatter"},
                )
            )
    else:
        workloads.extend(
            [
                Workload(
                    "arena_dense",
                    "arena",
                    lambda: (
                        _make_random_arena(seed + 1, 24000, 0.45, 0.35, 0.35),
                        1,
                    ),
                    {"arena_count": 24000, "arena_kind": "dense"},
                ),
                Workload(
                    "arena_sparse",
                    "arena",
                    lambda: (
                        _make_random_arena(seed + 2, 12000, 0.30, 0.25, 0.25),
                        1,
                    ),
                    {"arena_count": 12000, "arena_kind": "sparse"},
                ),
                Workload(
                    "arena_shatter",
                    "arena",
                    lambda: (
                        _make_random_arena(seed + 3, 20000, 0.55, 0.30, 0.55),
                        1,
                    ),
                    {"arena_count": 20000, "arena_kind": "shatter"},
                ),
            ]
        )

    if not hierarchy_pairs:
        hierarchy_pairs = [(256, 1024)]
    for l2_block_size, l1_block_size in hierarchy_pairs:
        workloads.append(
            Workload(
                f"arena_hierarchy_l2{l2_block_size}_l1{l1_block_size}",
                "arena",
                lambda l2=l2_block_size, l1=l1_block_size: (
                    _make_hierarchy_arena(l2, l1),
                    1,
                ),
                {"hierarchy_l2": l2_block_size, "hierarchy_l1": l1_block_size},
            )
        )

    return workloads


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in rows:
        req = str(row.get("swizzle_backend_requested", ""))
        eff = str(row.get("swizzle_backend_effective", req))
        backend_label = req if req == eff else f"{req}->{eff}"
        key = (row["workload"], row["mode"], backend_label)
        grouped.setdefault(key, []).append(float(row["exec_ms"]))
    print("Summary (mean exec_ms):")
    for (workload, mode, backend_label), values in sorted(grouped.items()):
        mean_ms = sum(values) / max(1, len(values))
        print(f"  {workload:14s} {mode:22s} {backend_label:14s} {mean_ms:9.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Prism VM modes and BSP variants.")
    parser.add_argument("--out", default="bench_results.csv", help="CSV output path.")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per mode/workload.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per mode/workload.")
    parser.add_argument("--cycles", type=int, default=3, help="BSP cycles per run.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random arenas.")
    parser.add_argument("--block-sizes", default="256", help="Comma list for block sizes.")
    parser.add_argument(
        "--arena-counts",
        default="",
        help="Comma list of arena active counts for random workloads.",
    )
    parser.add_argument("--workloads", default="", help="Comma list to filter workloads.")
    parser.add_argument("--modes", default="", help="Comma list to filter modes.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline mode.")
    parser.add_argument(
        "--swizzle-backends",
        default="jax",
        help="Comma list of swizzle backends to sweep (jax,pallas,triton).",
    )
    parser.add_argument(
        "--hierarchy-l1-mult",
        type=int,
        default=4,
        help="L1 block size multiplier relative to L2 (for hierarchy modes).",
    )
    parser.add_argument(
        "--hierarchy-no-global",
        action="store_true",
        help="Disable global migration stage in hierarchy modes.",
    )
    parser.add_argument(
        "--hierarchy-morton",
        action="store_true",
        help="Include Morton variants for hierarchy modes.",
    )
    parser.add_argument(
        "--metrics-block-size",
        type=int,
        default=256,
        help="Block size used for locality metrics.",
    )
    parser.add_argument(
        "--hierarchy-workload-l2",
        type=int,
        default=0,
        help="L2 block size for arena_hierarchy workload (defaults to first block size).",
    )
    parser.add_argument(
        "--hierarchy-workload-l1",
        type=int,
        default=0,
        help="L1 block size for arena_hierarchy workload (defaults to L2 * hierarchy-l1-mult).",
    )
    args = parser.parse_args()

    block_sizes = _parse_int_list(args.block_sizes)
    arena_counts = _parse_int_list(args.arena_counts)
    swizzle_backends = [b.lower() for b in _parse_csv_list(args.swizzle_backends)]
    if not swizzle_backends:
        swizzle_backends = ["jax"]
    hierarchy_pairs: List[Tuple[int, int]] = []
    l1_mult = max(1, args.hierarchy_l1_mult)
    if args.hierarchy_workload_l2 or args.hierarchy_workload_l1:
        hierarchy_l2 = args.hierarchy_workload_l2 or (block_sizes[0] if block_sizes else 256)
        hierarchy_l1 = args.hierarchy_workload_l1 or hierarchy_l2 * l1_mult
        hierarchy_pairs = [(hierarchy_l2, hierarchy_l1)]
    else:
        base_sizes = block_sizes if block_sizes else [256]
        for l2_size in base_sizes:
            l1_size = l2_size * l1_mult
            if pv.MAX_NODES % l1_size != 0:
                continue
            hierarchy_pairs.append((l2_size, l1_size))
    workloads = _build_workloads(args.seed, arena_counts, hierarchy_pairs)
    if args.workloads:
        wanted = set(_parse_csv_list(args.workloads))
        workloads = [w for w in workloads if w.name in wanted]

    modes = _build_modes(
        block_sizes,
        include_baseline=not args.no_baseline,
        hierarchy_l1_mult=max(1, args.hierarchy_l1_mult),
        include_hierarchy_global=not args.hierarchy_no_global,
        include_hierarchy_morton=args.hierarchy_morton,
    )
    if args.modes:
        wanted = set(_parse_csv_list(args.modes))
        modes = [m for m in modes if m["name"] in wanted]

    rows: List[Dict[str, object]] = []
    metrics_block_size = args.metrics_block_size if args.metrics_block_size > 0 else None

    for backend in swizzle_backends:
        pv_module = _load_prism_vm(backend)
        effective_backend = getattr(pv_module, "_SWIZZLE_BACKEND", backend)
        for workload in workloads:
            for mode in modes:
                l2_block_size = mode.get("l2_block_size")
                l1_block_size = mode.get("l1_block_size")
                do_global = mode.get("do_global", False)
                metrics_l2_block_size = l2_block_size or mode.get("block_size")
                metrics_l1_block_size = l1_block_size
                for _ in range(args.warmup):
                    with _suppress_output():
                        if workload.kind == "expr":
                            expr = workload.build()
                            if mode["kind"] == "baseline":
                                _ = _run_baseline_expr(expr)
                            else:
                                _ = _run_bsp_expr(
                                    expr,
                                    args.cycles,
                                    mode["do_sort"],
                                    mode["use_morton"],
                                    mode["block_size"],
                                    metrics_block_size,
                                    l2_block_size,
                                    l1_block_size,
                                    do_global,
                                    metrics_l2_block_size,
                                    metrics_l1_block_size,
                                )
                        else:
                            if mode["kind"] == "baseline":
                                continue
                            arena, root_ptr = workload.build()
                            _ = _run_bsp_arena(
                                arena,
                                root_ptr,
                                args.cycles,
                                mode["do_sort"],
                                mode["use_morton"],
                                mode["block_size"],
                                metrics_block_size,
                                l2_block_size,
                                l1_block_size,
                                do_global,
                                metrics_l2_block_size,
                                metrics_l1_block_size,
                            )

                for run_idx in range(args.runs):
                    with _suppress_output():
                        if workload.kind == "expr":
                            expr = workload.build()
                            if mode["kind"] == "baseline":
                                res = _run_baseline_expr(expr)
                            else:
                                res = _run_bsp_expr(
                                    expr,
                                    args.cycles,
                                    mode["do_sort"],
                                    mode["use_morton"],
                                    mode["block_size"],
                                    metrics_block_size,
                                    l2_block_size,
                                    l1_block_size,
                                    do_global,
                                    metrics_l2_block_size,
                                    metrics_l1_block_size,
                                )
                        else:
                            if mode["kind"] == "baseline":
                                continue
                            arena, root_ptr = workload.build()
                            res = _run_bsp_arena(
                                arena,
                                root_ptr,
                                args.cycles,
                                mode["do_sort"],
                                mode["use_morton"],
                                mode["block_size"],
                                metrics_block_size,
                                l2_block_size,
                                l1_block_size,
                                do_global,
                                metrics_l2_block_size,
                                metrics_l1_block_size,
                            )

                    backend_fields = {
                        "swizzle_backend_requested": backend,
                        "swizzle_backend_effective": effective_backend,
                    }

                    if workload.kind == "expr":
                        if mode["kind"] == "baseline":
                            row = {
                                "workload": workload.name,
                                "workload_kind": workload.kind,
                                "mode": mode["name"],
                                "do_sort": "",
                                "use_morton": "",
                                "block_size": "",
                                "l2_block_size": "",
                                "l1_block_size": "",
                                "do_global": "",
                                "metrics_block_size": metrics_block_size or 0,
                                "metrics_l2_block_size": metrics_l2_block_size or 0,
                                "metrics_l1_block_size": metrics_l1_block_size or 0,
                                "run": run_idx,
                                "cycles": 1,
                                "parse_ms": res["parse_ms"],
                                "exec_ms": res["exec_ms"],
                                "ms_per_cycle": res["exec_ms"],
                                "total_ms": res["parse_ms"] + res["exec_ms"],
                                "parse_allocs": res["parse_allocs"],
                                "exec_allocs": res["exec_allocs"],
                                "final_count": res["final_count"],
                            }
                            _apply_workload_meta(row, workload)
                            row.update(backend_fields)
                            rows.append(row)
                        else:
                            row = {
                                "workload": workload.name,
                                "workload_kind": workload.kind,
                                "mode": mode["name"],
                                "do_sort": mode["do_sort"],
                                "use_morton": mode["use_morton"],
                                "block_size": mode["block_size"] or 0,
                                "l2_block_size": l2_block_size or (mode["block_size"] or 0),
                                "l1_block_size": l1_block_size or 0,
                                "do_global": do_global,
                                "metrics_block_size": metrics_block_size or 0,
                                "metrics_l2_block_size": metrics_l2_block_size or 0,
                                "metrics_l1_block_size": metrics_l1_block_size or 0,
                                "run": run_idx,
                                "cycles": args.cycles,
                                "parse_ms": res["parse_ms"],
                                "exec_ms": res["exec_ms"],
                                "ms_per_cycle": res["exec_ms"] / max(1, args.cycles),
                                "total_ms": res["parse_ms"] + res["exec_ms"],
                                "parse_allocs": res["parse_allocs"],
                                "exec_allocs": res["exec_allocs"],
                                "final_count": res["final_count"],
                            }
                            _apply_workload_meta(row, workload)
                            row.update(backend_fields)
                            for prefix, metrics in [("pre", res["pre_metrics"]), ("post", res["post_metrics"])]:
                                for k, v in metrics.items():
                                    row[f"{prefix}_{k}"] = v
                            rows.append(row)
                    else:
                        if mode["kind"] == "baseline":
                            continue
                        row = {
                            "workload": workload.name,
                            "workload_kind": workload.kind,
                            "mode": mode["name"],
                            "do_sort": mode["do_sort"],
                            "use_morton": mode["use_morton"],
                            "block_size": mode["block_size"] or 0,
                            "l2_block_size": l2_block_size or (mode["block_size"] or 0),
                            "l1_block_size": l1_block_size or 0,
                            "do_global": do_global,
                            "metrics_block_size": metrics_block_size or 0,
                            "metrics_l2_block_size": metrics_l2_block_size or 0,
                            "metrics_l1_block_size": metrics_l1_block_size or 0,
                            "run": run_idx,
                            "cycles": args.cycles,
                            "parse_ms": res["parse_ms"],
                            "exec_ms": res["exec_ms"],
                            "ms_per_cycle": res["exec_ms"] / max(1, args.cycles),
                            "total_ms": res["parse_ms"] + res["exec_ms"],
                            "parse_allocs": res["parse_allocs"],
                            "exec_allocs": res["exec_allocs"],
                            "final_count": res["final_count"],
                        }
                        _apply_workload_meta(row, workload)
                        row.update(backend_fields)
                        for prefix, metrics in [("pre", res["pre_metrics"]), ("post", res["post_metrics"])]:
                            for k, v in metrics.items():
                                row[f"{prefix}_{k}"] = v
                        rows.append(row)

    _write_csv(args.out, rows)
    _summarize(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
