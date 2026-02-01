module Prism.Spillway where

open import Agda.Builtin.Nat

import Prism.Novelty as N
import Prism.Vec as V

-- 2:1 spillway bound (per microstratum):
-- If each active item yields at most one net new item, total spillway usage
-- is bounded by the active count. This is the arithmetic core of the
-- spillway witness (see in/in-29.md).

data AtMost1 : Nat -> Set where
  ≤0 : AtMost1 zero
  ≤1 : AtMost1 (suc zero)

data AllAtMost1 : {n : Nat} -> V.Vec Nat n -> Set where
  all[] : AllAtMost1 V.[]
  all::_ : {n : Nat} {x : Nat} {xs : V.Vec Nat n} ->
           AtMost1 x -> AllAtMost1 xs -> AllAtMost1 (x V.:: xs)

sum : {n : Nat} -> V.Vec Nat n -> Nat
sum V.[] = zero
sum (x V.:: xs) = x + sum xs

n<=sucn : {n : Nat} -> n N.<= suc n
n<=sucn {zero} = N.z<=n
n<=sucn {suc n} = N.s<=s n<=sucn

spillway-bound : {n : Nat} {xs : V.Vec Nat n} ->
                 AllAtMost1 xs -> sum xs N.<= n
spillway-bound {zero} all[] = N.z<=n
spillway-bound {suc n} (all::_ bound rest) with bound
... | ≤0 = N.<=-trans (spillway-bound rest) n<=sucn
... | ≤1 = N.s<=s (spillway-bound rest)
