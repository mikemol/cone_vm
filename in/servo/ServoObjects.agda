-- ServoObjects.agda
--
-- Backbone: postulates + proofs + bitvector ("bivector") providing cover.
--
-- Goal:
--   For each mathematical object in the servo document, provide:
--     (1) a typed Agda representation,
--     (2) a computational meaning (via Word32/BitVec32),
--     (3) explicit proof obligations (lemmas) that correspond to claims in the text.
--
-- Strategy:
--   * Use K = Fin 32 for the controller state k.
--   * Provide TWO interchangeable semantic carriers for 32-bit data:
--       - Word32 = Fin (2^32)  (modular arithmetic view)
--       - BitVec32 = Vec Bool 32 (bitvector view; "cover" for bitwise ops)
--   * Postulate a retraction/section pair between the carriers, and postulate the
--     bitwise operators on BitVec32 (or import/replace with a real library later).
--   * Prove (or initially postulate) laws once, then transport to Word32 via the bridge.

module ServoObjects where

open import Agda.Primitive using (Level; lzero)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong)
open import Data.Nat using (ℕ; _^_; _+_; _-_; suc; zero)
open import Data.Fin using (Fin; toℕ; fromℕ<)
open import Data.Vec using (Vec; [])
open import Data.Bool using (Bool; true; false)
open import Data.Nat.Properties using ()

-- ----------------------------
-- Core parameters / carriers
-- ----------------------------

two : ℕ
two = suc (suc zero)

pow2 : ℕ → ℕ
pow2 n = two ^ n

MOD32 : ℕ
MOD32 = pow2 32

MAX32 : ℕ
MAX32 = MOD32 - 1

-- Controller state: k ∈ {0..31}
K : Set
K = Fin 32

-- Modular carrier: uint32 as ℤ/(2^32)
Word32 : Set
Word32 = Fin MOD32

-- Bitvector carrier: 32 bits
BitVec32 : Set
BitVec32 = Vec Bool 32

-- ----------------------------
-- Bridge between carriers (cover)
-- ----------------------------

postulate
  encode : Word32 → BitVec32
  decode : BitVec32 → Word32

  -- Retraction law: decoding after encoding yields the same Word32
  decode∘encode : (w : Word32) → decode (encode w) ≡ w

-- (Optionally later) a section law encode∘decode ≡ id on a canonical subset of BitVec32
-- You can also postulate a normalization function on BitVec32 and prove section there.

-- ----------------------------
-- Bitvector operations (bitwise)
-- ----------------------------

postulate
  -- Bitwise NOT, AND on bitvectors
  notBV  : BitVec32 → BitVec32
  andBV  : BitVec32 → BitVec32 → BitVec32

  -- Two's complement negation on Word32 (mod 2^32)
  negW   : Word32 → Word32

  -- Derived: Word32 AND via the cover
  andW   : Word32 → Word32 → Word32

-- Definition via cover (kept as a postulate to avoid rewriting with the bridge each time)
postulate
  andW-def : (a b : Word32) → andW a b ≡ decode (andBV (encode a) (encode b))

-- ----------------------------
-- Nat-level helpers used in the text
-- ----------------------------

-- lowBitsNat(k) = (1<<k) - 1
lowBitsNat : K → ℕ
lowBitsNat k = (pow2 (toℕ k)) - 1

-- not32Nat(x) = (2^32 - 1) - x
not32Nat : ℕ → ℕ
not32Nat x = MAX32 - x

-- ----------------------------
-- Servo mask mapping (objects)
-- ----------------------------

postulate
  servoMaskFromK : K → Word32
  servoMaskToK   : Word32 → K

-- Obligations from the report:
postulate
  -- Left-inverse: mask_to_k(mask_from_k(k)) = k
  servoMaskToK-leftInv : (k : K) → servoMaskToK (servoMaskFromK k) ≡ k

  -- Popcount property (requires a popcount definition; expose as predicate first)
  Popcount32 : Word32 → ℕ
  popcount-maskFromK : (k : K) → Popcount32 (servoMaskFromK k) ≡ (32 - toℕ k)

-- ----------------------------
-- Lowest-set-bit trick (x & -x) (objects)
-- ----------------------------

postulate
  lowbitW : Word32 → Word32

  -- Definition: lowbitW x = x AND (-x)
  lowbitW-def : (x : Word32) → lowbitW x ≡ andW x (negW x)

  -- log2 for nonzero powers of two (returns a K)
  log2Pow2W : Word32 → K

  -- On valid servo masks, k is recovered by log2(lowbit(mask))
  servoMaskToK-via-lowbit :
    (k : K) → log2Pow2W (lowbitW (servoMaskFromK k)) ≡ k

-- ----------------------------
-- Spectral metrics (objects)
-- ----------------------------

-- We keep the scalar field abstract so you can pick ℚ or ℝ later.
postulate
  R : Set
  _≤R_ : R → R → Set
  _+R_ : R → R → R
  fromNatR : ℕ → R
  divR : R → R → R

  -- A tail sum operator over a spectrum S : ℕ → R
  sumTail : (S : ℕ → R) → (start : ℕ) → R

-- start(k) = max(k-1, 0) at the Nat level
startNat : K → ℕ
startNat k with toℕ k
... | zero  = zero
... | suc n = n

pBuffer : (S : ℕ → R) → K → R
pBuffer S k = sumTail S (startNat k)

dActive : (hotCount : ℕ) → K → R
dActive hot k = divR (fromNatR hot) (fromNatR (pow2 (startNat k)))

-- Nonnegativity assumption used in the monotonicity proof
postulate
  Nonneg : (S : ℕ → R) → Set

-- Monotonicity theorem obligation (shape; refine order later if you want a Fin-order relation)
postulate
  monotonePBuffer :
    (S : ℕ → R) → Nonneg S →
    (k₁ k₂ : K) → Set

-- ----------------------------
-- Servo update rule (objects)
-- ----------------------------

postulate
  ServoParams : Set
  spillHi vacLo minDensity : ServoParams → R

  -- Branchless update semantics (k-space and mask-space)
  servoUpdateK : ServoParams → K → R → R → K
  servoUpdateMask : ServoParams → Word32 → R → R → Word32

-- Consistency obligation: updating via mask equals decode/encode via k
postulate
  servoUpdateMask-consistent :
    (P : ServoParams) (k : K) (pb da : R) →
    servoUpdateMask P (servoMaskFromK k) pb da ≡ servoMaskFromK (servoUpdateK P k pb da)

-- ----------------------------
-- Meta objects (e.g., O(1))
-- ----------------------------

data BigO : Set where
  O1 : BigO

LOWBIT-COMPLEXITY : BigO
LOWBIT-COMPLEXITY = O1
