import argparse
import csv
import io
import random
import re
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


def _suppress_output():
    return redirect_stdout(io.StringIO())


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


def _compute_arena_metrics(arena: pv.Arena, block_size: Optional[int]) -> Dict[str, float]:
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

    return {
        "hot": hot,
        "warm": warm,
        "cold": cold,
        "free": free,
        "mean_edge_dist": mean_dist,
        "p95_edge_dist": p95_dist,
        "block_local_pct": block_local_pct if block_local_pct is not None else 0.0,
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
) -> Dict[str, object]:
    vm = pv.PrismVM_BSP()
    init_count = int(vm.arena.count)

    t0 = time.perf_counter()
    tokens = re.findall(r"\(|\)|[a-z]+", expr)
    root_ptr = vm.parse(tokens)
    vm.arena.opcode.block_until_ready()
    parse_ms = (time.perf_counter() - t0) * 1000
    start_count = int(vm.arena.count)

    pre_metrics = _compute_arena_metrics(vm.arena, metrics_block_size)

    t1 = time.perf_counter()
    for _ in range(cycles):
        vm.arena, root_ptr = pv.cycle(
            vm.arena,
            root_ptr,
            do_sort=do_sort,
            use_morton=use_morton,
            block_size=block_size,
        )
    vm.arena.opcode.block_until_ready()
    exec_ms = (time.perf_counter() - t1) * 1000

    end_count = int(vm.arena.count)
    post_metrics = _compute_arena_metrics(vm.arena, metrics_block_size)

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
) -> Dict[str, object]:
    pre_metrics = _compute_arena_metrics(arena, metrics_block_size)
    start_count = int(arena.count)

    t1 = time.perf_counter()
    for _ in range(cycles):
        arena, root_ptr = pv.cycle(
            arena,
            root_ptr,
            do_sort=do_sort,
            use_morton=use_morton,
            block_size=block_size,
        )
    arena.opcode.block_until_ready()
    exec_ms = (time.perf_counter() - t1) * 1000

    end_count = int(arena.count)
    post_metrics = _compute_arena_metrics(arena, metrics_block_size)

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


def _build_modes(block_sizes: List[int], include_baseline: bool) -> List[Dict[str, object]]:
    modes: List[Dict[str, object]] = []
    if include_baseline:
        modes.append({"name": "baseline", "kind": "baseline"})
    modes.append(
        {"name": "bsp-rank", "kind": "bsp", "do_sort": True, "use_morton": False, "block_size": None}
    )
    modes.append(
        {"name": "bsp-nosort", "kind": "bsp", "do_sort": False, "use_morton": False, "block_size": None}
    )
    modes.append(
        {"name": "bsp-morton", "kind": "bsp", "do_sort": True, "use_morton": True, "block_size": None}
    )
    for block_size in block_sizes:
        modes.append(
            {
                "name": f"bsp-block{block_size}",
                "kind": "bsp",
                "do_sort": True,
                "use_morton": False,
                "block_size": block_size,
            }
        )
        modes.append(
            {
                "name": f"bsp-morton-block{block_size}",
                "kind": "bsp",
                "do_sort": True,
                "use_morton": True,
                "block_size": block_size,
            }
        )
    return modes


def _build_workloads(seed: int) -> List[Workload]:
    random.seed(seed)
    return [
        Workload("expr_small", "expr", lambda: "(add (suc zero) (suc zero))"),
        Workload("expr_reuse", "expr", lambda: _reuse_tree(3)),
        Workload("expr_balanced", "expr", lambda: _balanced_add(6, 2)),
        Workload("expr_chain", "expr", lambda: _add_chain(64, 1)),
        Workload(
            "arena_dense",
            "arena",
            lambda: (
                _make_random_arena(seed + 1, 24000, 0.45, 0.35, 0.35),
                1,
            ),
        ),
        Workload(
            "arena_sparse",
            "arena",
            lambda: (
                _make_random_arena(seed + 2, 12000, 0.30, 0.25, 0.25),
                1,
            ),
        ),
        Workload(
            "arena_shatter",
            "arena",
            lambda: (
                _make_random_arena(seed + 3, 20000, 0.55, 0.30, 0.55),
                1,
            ),
        ),
    ]


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
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (row["workload"], row["mode"])
        grouped.setdefault(key, []).append(float(row["exec_ms"]))
    print("Summary (mean exec_ms):")
    for (workload, mode), values in sorted(grouped.items()):
        mean_ms = sum(values) / max(1, len(values))
        print(f"  {workload:14s} {mode:22s} {mean_ms:9.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Prism VM modes and BSP variants.")
    parser.add_argument("--out", default="bench_results.csv", help="CSV output path.")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per mode/workload.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per mode/workload.")
    parser.add_argument("--cycles", type=int, default=3, help="BSP cycles per run.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random arenas.")
    parser.add_argument("--block-sizes", default="256", help="Comma list for block sizes.")
    parser.add_argument("--workloads", default="", help="Comma list to filter workloads.")
    parser.add_argument("--modes", default="", help="Comma list to filter modes.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline mode.")
    parser.add_argument(
        "--metrics-block-size",
        type=int,
        default=256,
        help="Block size used for locality metrics.",
    )
    args = parser.parse_args()

    block_sizes: List[int] = []
    for raw in _parse_csv_list(args.block_sizes):
        if not raw.isdigit():
            continue
        value = int(raw)
        if value > 0:
            block_sizes.append(value)
    workloads = _build_workloads(args.seed)
    if args.workloads:
        wanted = set(_parse_csv_list(args.workloads))
        workloads = [w for w in workloads if w.name in wanted]

    modes = _build_modes(block_sizes, include_baseline=not args.no_baseline)
    if args.modes:
        wanted = set(_parse_csv_list(args.modes))
        modes = [m for m in modes if m["name"] in wanted]

    rows: List[Dict[str, object]] = []
    metrics_block_size = args.metrics_block_size if args.metrics_block_size > 0 else None

    for workload in workloads:
        for mode in modes:
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
                        )

                if workload.kind == "expr":
                    if mode["kind"] == "baseline":
                        row = {
                            "workload": workload.name,
                            "workload_kind": workload.kind,
                            "mode": mode["name"],
                            "do_sort": "",
                            "use_morton": "",
                            "block_size": "",
                            "metrics_block_size": metrics_block_size or 0,
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
                        rows.append(row)
                    else:
                        row = {
                            "workload": workload.name,
                            "workload_kind": workload.kind,
                            "mode": mode["name"],
                            "do_sort": mode["do_sort"],
                            "use_morton": mode["use_morton"],
                            "block_size": mode["block_size"] or 0,
                            "metrics_block_size": metrics_block_size or 0,
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
                        "metrics_block_size": metrics_block_size or 0,
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
                    for prefix, metrics in [("pre", res["pre_metrics"]), ("post", res["post_metrics"])]:
                        for k, v in metrics.items():
                            row[f"{prefix}_{k}"] = v
                    rows.append(row)

    _write_csv(args.out, rows)
    _summarize(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
