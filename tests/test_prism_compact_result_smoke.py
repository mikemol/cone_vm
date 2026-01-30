import jax.numpy as jnp

import prism_vm as pv


def test_prism_compact_candidates_result_smoke():
    enabled = jnp.array([1, 0, 1, 0], dtype=jnp.int32)
    opcode = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    arg1 = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    arg2 = jnp.array([5, 6, 7, 8], dtype=jnp.int32)
    candidates = pv.CandidateBuffer(enabled=enabled, opcode=opcode, arg1=arg1, arg2=arg2)

    compact_jit = pv.compact_candidates_result_jit_cfg()
    compacted, result = compact_jit(candidates)

    assert int(result.count) == 2
    assert int(result.idx[0]) == 0
    assert int(result.idx[1]) == 2
    assert int(compacted.opcode[0]) == 10
    assert int(compacted.opcode[1]) == 30


def test_prism_compact_candidates_with_index_result_smoke():
    enabled = jnp.array([0, 1, 1], dtype=jnp.int32)
    opcode = jnp.array([10, 20, 30], dtype=jnp.int32)
    arg1 = jnp.array([1, 2, 3], dtype=jnp.int32)
    arg2 = jnp.array([4, 5, 6], dtype=jnp.int32)
    candidates = pv.CandidateBuffer(enabled=enabled, opcode=opcode, arg1=arg1, arg2=arg2)

    compact_jit = pv.compact_candidates_with_index_result_jit_cfg()
    compacted, result, idx = compact_jit(candidates)

    assert int(result.count) == 2
    assert int(idx[0]) == 1
    assert int(idx[1]) == 2
    assert int(compacted.opcode[0]) == 20
    assert int(compacted.opcode[1]) == 30
