import os

import jax
import numpy as np

from prism_vm_core.domains import _host_int_value

# dataflow-bundle: changed, rewrite_child, wrap_emit

_damage_metrics_cycles = 0
_damage_metrics_hot_nodes = 0
_damage_metrics_edge_total = 0
_damage_metrics_edge_cross = 0
_cnf2_metrics_cycles = 0
_cnf2_metrics_rewrite_child = 0
_cnf2_metrics_changed = 0
_cnf2_metrics_wrap_emit = 0


def _damage_metrics_enabled():
    value = os.environ.get("PRISM_DAMAGE_METRICS", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _cnf2_metrics_enabled():
    value = os.environ.get("PRISM_CNF2_METRICS", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _damage_tile_size(block_size, l2_block_size, l1_block_size):
    value = os.environ.get("PRISM_DAMAGE_TILE_SIZE", "").strip()
    if value:
        if not value.isdigit():
            raise ValueError("PRISM_DAMAGE_TILE_SIZE must be an integer")
        return int(value)
    for candidate in (block_size, l1_block_size, l2_block_size):
        if candidate is not None:
            return int(candidate)
    return 0


def damage_metrics_reset():
    global _damage_metrics_cycles
    global _damage_metrics_hot_nodes
    global _damage_metrics_edge_total
    global _damage_metrics_edge_cross
    _damage_metrics_cycles = 0
    _damage_metrics_hot_nodes = 0
    _damage_metrics_edge_total = 0
    _damage_metrics_edge_cross = 0


def cnf2_metrics_reset():
    global _cnf2_metrics_cycles
    global _cnf2_metrics_rewrite_child
    global _cnf2_metrics_changed
    global _cnf2_metrics_wrap_emit
    _cnf2_metrics_cycles = 0
    _cnf2_metrics_rewrite_child = 0
    _cnf2_metrics_changed = 0
    _cnf2_metrics_wrap_emit = 0


def damage_metrics_get():
    if not _damage_metrics_enabled():
        return {
            "cycles": 0,
            "hot_nodes": 0,
            "edge_total": 0,
            "edge_cross": 0,
            "damage_rate": 0.0,
        }
    edge_total = int(_damage_metrics_edge_total)
    edge_cross = int(_damage_metrics_edge_cross)
    damage_rate = (edge_cross / edge_total) if edge_total else 0.0
    return {
        "cycles": int(_damage_metrics_cycles),
        "hot_nodes": int(_damage_metrics_hot_nodes),
        "edge_total": edge_total,
        "edge_cross": edge_cross,
        "damage_rate": float(damage_rate),
    }


def cnf2_metrics_get():
    if not _cnf2_metrics_enabled():
        return {
            "cycles": 0,
            "rewrite_child": 0,
            "changed": 0,
            "wrap_emit": 0,
        }
    return {
        "cycles": int(_cnf2_metrics_cycles),
        "rewrite_child": int(_cnf2_metrics_rewrite_child),
        "changed": int(_cnf2_metrics_changed),
        "wrap_emit": int(_cnf2_metrics_wrap_emit),
    }


def _cnf2_metrics_update(rewrite_child, changed, wrap_emit):
    global _cnf2_metrics_cycles
    global _cnf2_metrics_rewrite_child
    global _cnf2_metrics_changed
    global _cnf2_metrics_wrap_emit
    if not _cnf2_metrics_enabled():
        return
    _cnf2_metrics_cycles += 1
    _cnf2_metrics_rewrite_child += int(rewrite_child)
    _cnf2_metrics_changed += int(changed)
    _cnf2_metrics_wrap_emit += int(wrap_emit)


def _damage_metrics_update(arena, tile_size, rank_hot=0):
    global _damage_metrics_cycles
    global _damage_metrics_hot_nodes
    global _damage_metrics_edge_total
    global _damage_metrics_edge_cross
    if not _damage_metrics_enabled() or tile_size <= 0:
        return
    count = _host_int_value(arena.count)
    if count <= 0:
        return
    ranks = np.asarray(jax.device_get(arena.rank[:count]))
    arg1 = np.asarray(jax.device_get(arena.arg1[:count]))
    arg2 = np.asarray(jax.device_get(arena.arg2[:count]))
    idx = np.arange(count, dtype=np.int32)
    tile = idx // int(tile_size)
    hot_mask = ranks == rank_hot
    hot_nodes = int(hot_mask.sum())
    if hot_nodes <= 0:
        _damage_metrics_cycles += 1
        return
    hot_idx = idx[hot_mask]
    tile_hot = tile[hot_mask]
    a1_hot = arg1[hot_mask]
    a2_hot = arg2[hot_mask]
    valid1 = (a1_hot > 0) & (a1_hot < count)
    valid2 = (a2_hot > 0) & (a2_hot < count)
    a1_safe = np.where(valid1, a1_hot, 0)
    a2_safe = np.where(valid2, a2_hot, 0)
    cross1 = valid1 & (tile_hot != tile[a1_safe])
    cross2 = valid2 & (tile_hot != tile[a2_safe])
    edge_total = int(valid1.sum() + valid2.sum())
    edge_cross = int(cross1.sum() + cross2.sum())
    _damage_metrics_cycles += 1
    _damage_metrics_hot_nodes += hot_nodes
    _damage_metrics_edge_total += edge_total
    _damage_metrics_edge_cross += edge_cross


__all__ = [
    "damage_metrics_reset",
    "cnf2_metrics_reset",
    "damage_metrics_get",
    "cnf2_metrics_get",
    "_damage_metrics_enabled",
    "_cnf2_metrics_enabled",
    "_damage_tile_size",
    "_cnf2_metrics_update",
    "_damage_metrics_update",
]
