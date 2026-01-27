module Prism.MinPrism where

open import Agda.Builtin.Nat
open import Agda.Builtin.List

import Prism.Key as K
import Prism.Signature as Sig
import Prism.Vec as V

Finite : Set -> Set
Finite A = List A

append : {A : Set} -> List A -> List A -> List A
append [] ys = ys
append (x ∷ xs) ys = x ∷ append xs ys

map : {A B : Set} -> (A -> B) -> List A -> List B
map f [] = []
map f (x ∷ xs) = f x ∷ map f xs

concatMap : {A B : Set} -> (A -> List B) -> List A -> List B
concatMap f [] = []
concatMap f (x ∷ xs) = append (f x) (concatMap f xs)

vec0 : {A : Set} -> V.Vec A zero
vec0 = V.[]

vec1 : {A : Set} -> A -> V.Vec A (suc zero)
vec1 a = a V.:: V.[]

vec2 : {A : Set} -> A -> A -> V.Vec A (suc (suc zero))
vec2 a b = a V.:: (b V.:: V.[])

allOps : List Sig.Op
allOps =
  Sig.add ∷ Sig.mul ∷ Sig.suc ∷ Sig.zero ∷
  Sig.coordPair ∷ Sig.coordZero ∷ Sig.coordOne ∷ []

nodesForOp : Sig.Op -> List K.Key -> List K.Key
nodesForOp Sig.zero _ = K.node Sig.zero vec0 ∷ []
nodesForOp Sig.coordZero _ = K.node Sig.coordZero vec0 ∷ []
nodesForOp Sig.coordOne _ = K.node Sig.coordOne vec0 ∷ []
nodesForOp Sig.suc ks =
  map (\k -> K.node Sig.suc (vec1 k)) ks
nodesForOp Sig.add ks =
  concatMap (\k1 -> map (\k2 -> K.node Sig.add (vec2 k1 k2)) ks) ks
nodesForOp Sig.mul ks =
  concatMap (\k1 -> map (\k2 -> K.node Sig.mul (vec2 k1 k2)) ks) ks
nodesForOp Sig.coordPair ks =
  concatMap (\k1 -> map (\k2 -> K.node Sig.coordPair (vec2 k1 k2)) ks) ks

keysDepth : Nat -> List K.Key
keysDepth zero =
  append (nodesForOp Sig.zero [])
    (append (nodesForOp Sig.coordZero []) (nodesForOp Sig.coordOne []))
keysDepth (suc n) =
  let prev = keysDepth n in
  append prev (concatMap (\op -> nodesForOp op prev) allOps)

closure-enumerable : (n : Nat) -> List K.Key
closure-enumerable = keysDepth

finite-closure : (n : Nat) -> Finite K.Key
finite-closure = keysDepth
