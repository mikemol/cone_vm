from dataclasses import dataclass
import jax
import jax.numpy as jnp

from prism_ledger.intern import intern_nodes_state
from prism_ledger.index import LedgerState, derive_ledger_state
from prism_vm_core.domains import _host_int_value, _host_raise_if_bad, _ledger_id
from prism_vm_core.ontology import OP_NULL
from prism_vm_core.structures import Ledger, NodeBatch

# dataflow-bundle: arg1, arg2, opcode


@dataclass(frozen=True)
class _ProjectArgs:
    opcode: object
    arg1: object
    arg2: object


def _node_batch(op, a1, a2):
    return NodeBatch(op=op, a1=a1, a2=a2)


def _project_graph_to_ledger(
    opcode,
    arg1,
    arg2,
    count,
    root_idx,
    ledger: Ledger | LedgerState,
    label,
    limit=None,
):
    if isinstance(ledger, LedgerState):
        ledger_state = ledger
    else:
        ledger_state = derive_ledger_state(ledger)
    bundle = _ProjectArgs(opcode=opcode, arg1=arg1, arg2=arg2)
    ops = jax.device_get(bundle.opcode[:count])
    a1s = jax.device_get(bundle.arg1[:count])
    a2s = jax.device_get(bundle.arg2[:count])
    mapping = {0: 0}
    visiting = set()

    def _project(idx):
        nonlocal ledger_state
        idx_i = int(idx)
        if idx_i in mapping:
            return mapping[idx_i]
        if idx_i < 0 or idx_i >= count:
            raise ValueError(
                f"{label}: root index {idx_i} out of range (count={count})"
            )
        if limit is not None and len(mapping) > int(limit):
            raise RuntimeError(
                f"{label}: projection exceeded limit={int(limit)}"
            )
        if idx_i in visiting:
            raise RuntimeError(f"{label}: cycle detected at id={idx_i}")
        visiting.add(idx_i)
        op = int(ops[idx_i])
        if op == OP_NULL:
            mapping[idx_i] = 0
            visiting.remove(idx_i)
            return 0
        child1 = _project(int(a1s[idx_i]))
        child2 = _project(int(a2s[idx_i]))
        ids, ledger_state = intern_nodes_state(
            ledger_state,
            _node_batch(
                jnp.array([op], dtype=jnp.int32),
                jnp.array([child1], dtype=jnp.int32),
                jnp.array([child2], dtype=jnp.int32),
            ),
        )
        mapping[idx_i] = int(ids[0])
        visiting.remove(idx_i)
        return mapping[idx_i]

    root_out = _project(int(root_idx))
    _host_raise_if_bad(
        ledger_state.ledger, f"{label}: projection exceeded ledger capacity"
    )
    return ledger_state.ledger, _ledger_id(root_out)


def project_manifest_to_ledger(manifest, root_ptr, ledger=None, limit=None):
    """Project a Manifest (baseline) root into a canonical Ledger via q."""
    if ledger is None:
        from prism_vm_core.graphs import init_ledger as _init_ledger
        from prism_vm_core.ledger_keys import _pack_key
        from prism_vm_core.constants import LEDGER_CAPACITY
        from prism_vm_core.ontology import OP_ZERO

        ledger = _init_ledger(_pack_key, LEDGER_CAPACITY, OP_NULL, OP_ZERO)
    root_idx = _host_int_value(root_ptr)
    count = _host_int_value(manifest.active_count)
    return _project_graph_to_ledger(
        manifest.opcode,
        manifest.arg1,
        manifest.arg2,
        count,
        root_idx,
        ledger,
        "q_manifest",
        limit=limit,
    )


def project_arena_to_ledger(arena, root_ptr, ledger=None, limit=None):
    """Project an Arena (BSPˢ microstate) root into a canonical Ledger via q."""
    if ledger is None:
        from prism_vm_core.graphs import init_ledger as _init_ledger
        from prism_vm_core.ledger_keys import _pack_key
        from prism_vm_core.constants import LEDGER_CAPACITY
        from prism_vm_core.ontology import OP_ZERO

        ledger = _init_ledger(_pack_key, LEDGER_CAPACITY, OP_NULL, OP_ZERO)
    root_idx = _host_int_value(root_ptr)
    count = _host_int_value(arena.count)
    return _project_graph_to_ledger(
        arena.opcode,
        arena.arg1,
        arena.arg2,
        count,
        root_idx,
        ledger,
        "q_arena",
        limit=limit,
    )


__all__ = [
    "_project_graph_to_ledger",
    "project_manifest_to_ledger",
    "project_arena_to_ledger",
]
