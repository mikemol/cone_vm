"""
servo_objects.py

Machine-actionable semantics for the mathematical objects in
"JAX Servo Update Logic Explained" (Google Docs export).

Focus:
- exact uint32 behavior for masks
- executable "finite-core" checks (exhaustive over k ∈ [0,31])
- reusable primitives for proofs-to-tests style

Authoring note:
This module avoids JAX; it models the logic deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict, Any, Optional

MASK32: int = (1 << 32) - 1


def u32(x: int) -> int:
    """Interpret x modulo 2^32."""
    return x & MASK32


def check_k(k: int) -> None:
    if not (0 <= k <= 31):
        raise ValueError("k must be in [0,31]")


def exp2(n: int) -> int:
    if n < 0:
        raise ValueError("exp2 expects n >= 0")
    return 1 << n


# ----------------------------
# Core mathematical objects
# ----------------------------

def low_bits(k: int) -> int:
    """
    low_bits = (1 << k) - 1, as uint32.
    For k=0: 0
    """
    check_k(k)
    return u32((1 << k) - 1)


def servo_mask_from_k(k: int) -> int:
    """
    mask_k = ~((1 << k) - 1)  (mod 2^32)
    This yields k low zeros followed by (32-k) ones.
    """
    check_k(k)
    return u32(~low_bits(k))


def lowbit(x: int) -> int:
    """
    Isolate lowest set bit: x & (-x) in uint32 arithmetic.
    """
    return u32(u32(x) & u32(-u32(x)))


def ilog2_pow2(x: int) -> int:
    """
    Integer log2 for x that is a power of two (and nonzero).
    Returns the bit index.
    """
    x = u32(x)
    if x == 0 or (x & (x - 1)) != 0:
        raise ValueError("ilog2_pow2 expects a nonzero power-of-two uint32")
    return x.bit_length() - 1


def servo_mask_to_k(mask: int) -> int:
    """
    Inverse map (on valid masks): recover k from the mask.

    For masks produced by servo_mask_from_k:
      mask = 111..1100..00 (k trailing zeros)
    The lowest set bit of mask is at position k, except for k=0 where it is bit 0.
    Thus k = ilog2(lowbit(mask)).
    """
    lb = lowbit(mask)
    return ilog2_pow2(lb)


def popcount32(x: int) -> int:
    """Population count of uint32."""
    return u32(x).bit_count()


# ----------------------------
# "Spectral" signals and metrics
# ----------------------------

def p_buffer(spectrum: Sequence[float], k: int) -> float:
    """
    Tail-sum pressure:
      start = max(k-1, 0)
      p_buffer = sum_{i=start..N-1} S[i]
    """
    check_k(k)
    start = max(k - 1, 0)
    return float(sum(spectrum[start:]))


def d_active(hot_count: int, k: int) -> float:
    """
    d_active = hot_count / 2^(k-1)
    with denom = 2^(k-1) for k>=1, denom = 1 for k=0 (because exp2(-1) is undefined).
    The source text uses exp2(k-1); implementations often guard k==0.
    """
    check_k(k)
    denom = exp2(max(k - 1, 0))
    return hot_count / float(denom)


# ----------------------------
# Servo update (branchless semantics)
# ----------------------------

@dataclass(frozen=True)
class ServoParams:
    spill_hi: float = 0.25
    vac_lo: float = 0.10
    min_density: float = 0.4


def servo_update_k(k: int, p_buf: float, d_act: float, params: ServoParams = ServoParams()) -> int:
    """
    Discrete-time update rule:
      spill  := p_buf > spill_hi
      vacuum := (p_buf < vac_lo) & (d_act < min_density)
      k_next := min(k+1,31) if spill else max(k-1,0) if vacuum else k
    """
    check_k(k)
    spill = p_buf > params.spill_hi
    vacuum = (p_buf < params.vac_lo) and (d_act < params.min_density)

    k_up = min(k + 1, 31)
    k_down = max(k - 1, 0)

    if spill:
        return k_up
    if vacuum:
        return k_down
    return k


def servo_update_mask(mask: int, p_buf: float, d_act: float, params: ServoParams = ServoParams()) -> int:
    """
    State stored as mask; update by decoding k, updating k, then encoding mask.
    """
    k = servo_mask_to_k(mask)
    k2 = servo_update_k(k, p_buf, d_act, params=params)
    return servo_mask_from_k(k2)


# ----------------------------
# Finite-core checks (tests as proofs)
# ----------------------------

def check_injective_mask_from_k() -> bool:
    seen: Dict[int, int] = {}
    for k in range(32):
        m = servo_mask_from_k(k)
        if m in seen and seen[m] != k:
            return False
        seen[m] = k
    return True


def check_left_inverse_mask_to_k() -> bool:
    for k in range(32):
        m = servo_mask_from_k(k)
        k2 = servo_mask_to_k(m)
        if k2 != k:
            return False
    return True


def check_popcount_property() -> bool:
    # For mask_k = ~((1<<k)-1), number of ones should be 32-k
    for k in range(32):
        m = servo_mask_from_k(k)
        if popcount32(m) != (32 - k):
            return False
    return True


def check_monotone_p_buffer_nonincreasing(spectrum: Sequence[float]) -> bool:
    """
    Empirical check of monotonicity for a provided spectrum.
    Requires spectrum[i] >= 0 for theorem to hold.
    """
    ps = [p_buffer(spectrum, k) for k in range(32)]
    return all(ps[k+1] <= ps[k] + 1e-12 for k in range(31))


def run_selftest() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["injective_mask_from_k"] = check_injective_mask_from_k()
    out["left_inverse_mask_to_k"] = check_left_inverse_mask_to_k()
    out["popcount_property"] = check_popcount_property()

    # A simple nonnegative spectrum sanity check
    spectrum = [float(i*i % 7) for i in range(64)]  # nonnegative
    out["monotone_p_buffer_example"] = check_monotone_p_buffer_nonincreasing(spectrum)

    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_selftest(), indent=2, sort_keys=True))
