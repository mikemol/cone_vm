import jax
import jax.numpy as jnp
import pytest

from tests import harness
import prism_vm as pv

pytestmark = pytest.mark.m5


def test_servo_denotation_invariance(monkeypatch):
    exprs = [
        "zero",
        "(suc zero)",
        "(add (suc zero) (suc zero))",
        "(mul (suc (suc zero)) (suc zero))",
    ]
    for expr in exprs:
        monkeypatch.setenv("PRISM_ENABLE_SERVO", "0")
        base = harness.denote_pretty_arena(
            expr, steps=4, do_sort=True, use_morton=True
        )
        monkeypatch.setenv("PRISM_ENABLE_SERVO", "1")
        servo = harness.denote_pretty_arena(
            expr, steps=4, do_sort=True, use_morton=True
        )
        assert base == servo


def test_servo_sort_stable_tiebreaker():
    size = 32
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg1 = jnp.zeros(size, dtype=jnp.int32)
    arg2 = jnp.zeros(size, dtype=jnp.int32)
    rank = jnp.full(size, pv.RANK_HOT, dtype=jnp.int8)
    count = jnp.array(size, dtype=jnp.int32)
    servo = jnp.array([0, 0, 0], dtype=jnp.uint32)
    arena = pv.Arena(
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
        rank=rank,
        count=count,
        oom=jnp.array(False, dtype=jnp.bool_),
        servo=servo,
    )
    morton = jnp.zeros(size, dtype=jnp.uint32)
    _, inv_perm = pv.op_sort_and_swizzle_servo_with_perm(
        arena, morton, arena.servo[0]
    )
    perm = pv._invert_perm(inv_perm)
    perm_host = jax.device_get(perm)
    assert bool(jnp.array_equal(perm_host, jnp.arange(size, dtype=jnp.int32)))
