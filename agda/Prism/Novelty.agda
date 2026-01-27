module Prism.Novelty where

open import Agda.Builtin.Nat

import Prism.Key as K
import Prism.Vec as V

data _<=_ : Nat -> Nat -> Set where
  z<=n : {n : Nat} -> zero <= n
  s<=s : {m n : Nat} -> m <= n -> suc m <= suc n

<=-refl : {n : Nat} -> n <= n
<=-refl {zero} = z<=n
<=-refl {suc n} = s<=s <=-refl

<=-trans : {a b c : Nat} -> a <= b -> b <= c -> a <= c
<=-trans z<=n _ = z<=n
<=-trans (s<=s ab) (s<=s bc) = s<=s (<=-trans ab bc)

max : Nat -> Nat -> Nat
max zero n = n
max (suc m) zero = suc m
max (suc m) (suc n) = suc (max m n)

mutual
  depth : K.Key -> Nat
  depth (K.node _ args) = suc (depthVec args)

  depthVec : {n : Nat} -> V.Vec K.Key n -> Nat
  depthVec V.[] = zero
  depthVec (k V.:: ks) = max (depth k) (depthVec ks)

Execution : Set
Execution = Nat

Novelty : Execution -> K.Key -> Set
Novelty n k = depth k <= n

monotone-novelty : {n m : Execution} {k : K.Key} -> n <= m -> Novelty n k -> Novelty m k
monotone-novelty n<=m depth<=n = <=-trans depth<=n n<=m
