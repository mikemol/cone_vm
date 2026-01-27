module Prism.Signature where

open import Agda.Builtin.Nat
open import Agda.Builtin.Equality

-- Minimal signature scaffold (see in/in-26.md).

data Op : Set where
  add mul suc zero coordPair coordZero coordOne : Op

arity : Op -> Nat
arity add = 2
arity mul = 2
arity suc = 1
arity zero = 0
arity coordPair = 2
arity coordZero = 0
arity coordOne = 0

-- Simple arity equalities (first proofs; see in/in-26.md).
arity-add : arity add ≡ 2
arity-add = refl

arity-mul : arity mul ≡ 2
arity-mul = refl

arity-suc : arity suc ≡ 1
arity-suc = refl

arity-zero : arity zero ≡ 0
arity-zero = refl

arity-coordPair : arity coordPair ≡ 2
arity-coordPair = refl

arity-coordZero : arity coordZero ≡ 0
arity-coordZero = refl

arity-coordOne : arity coordOne ≡ 0
arity-coordOne = refl
