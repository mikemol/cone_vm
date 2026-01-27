module Prism.Signature where

open import Agda.Builtin.Nat

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
