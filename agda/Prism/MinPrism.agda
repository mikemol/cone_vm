module Prism.MinPrism where

open import Agda.Builtin.Nat
open import Agda.Builtin.List

import Prism.Key as K
import Prism.Signature as Sig
import Prism.Vec as V
import Prism.Novelty as N

Finite : Set -> Set
Finite A = List A

data _∈_ {A : Set} (x : A) : List A -> Set where
  here : {xs : List A} -> x ∈ (x ∷ xs)
  there : {y : A} {xs : List A} -> x ∈ xs -> x ∈ (y ∷ xs)

append : {A : Set} -> List A -> List A -> List A
append [] ys = ys
append (x ∷ xs) ys = x ∷ append xs ys

in-append-left : {A : Set} {x : A} {xs ys : List A} -> x ∈ xs -> x ∈ append xs ys
in-append-left here = here
in-append-left (there p) = there (in-append-left p)

in-append-right : {A : Set} {x : A} {xs ys : List A} -> x ∈ ys -> x ∈ append xs ys
in-append-right {xs = []} p = p
in-append-right {xs = _ ∷ xs} p = there (in-append-right {xs = xs} p)

map : {A B : Set} -> (A -> B) -> List A -> List B
map f [] = []
map f (x ∷ xs) = f x ∷ map f xs

in-map : {A B : Set} {x : A} {xs : List A} -> (f : A -> B) -> x ∈ xs -> f x ∈ map f xs
in-map f here = here
in-map f (there p) = there (in-map f p)

concatMap : {A B : Set} -> (A -> List B) -> List A -> List B
concatMap f [] = []
concatMap f (x ∷ xs) = append (f x) (concatMap f xs)

in-concatMap :
  {A B : Set}
  {x : A}
  {y : B}
  {xs : List A}
  -> (f : A -> List B)
  -> x ∈ xs
  -> y ∈ f x
  -> y ∈ concatMap f xs
in-concatMap f here y∈fx = in-append-left y∈fx
in-concatMap f (there p) y∈fx = in-append-right (in-concatMap f p y∈fx)

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

op-in-allOps : (op : Sig.Op) -> op ∈ allOps
op-in-allOps Sig.add = here
op-in-allOps Sig.mul = there here
op-in-allOps Sig.suc = there (there here)
op-in-allOps Sig.zero = there (there (there here))
op-in-allOps Sig.coordPair = there (there (there (there here)))
op-in-allOps Sig.coordZero = there (there (there (there (there here))))
op-in-allOps Sig.coordOne = there (there (there (there (there (there here)))))

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

data AllVec {A : Set} (P : A -> Set) : {n : Nat} -> V.Vec A n -> Set where
  allNil : AllVec P V.[]
  allCons : {n : Nat} {x : A} {xs : V.Vec A n} -> P x -> AllVec P xs -> AllVec P (x V.:: xs)

allVecMap : {A : Set} {P Q : A -> Set} {n : Nat} {xs : V.Vec A n} ->
  (∀ {x} -> P x -> Q x) -> AllVec P xs -> AllVec Q xs
allVecMap f allNil = allNil
allVecMap f (allCons px rest) = allCons (f px) (allVecMap f rest)

pred<= : {m n : Nat} -> suc m N.<= suc n -> m N.<= n
pred<= (N.s<=s p) = p

max-left : {a b : Nat} -> a N.<= N.max a b
max-left {zero} {b} = N.z<=n
max-left {suc a} {zero} = N.<=-refl
max-left {suc a} {suc b} = N.s<=s (max-left {a} {b})

max-right : {a b : Nat} -> b N.<= N.max a b
max-right {zero} {b} = N.<=-refl
max-right {suc a} {zero} = N.z<=n
max-right {suc a} {suc b} = N.s<=s (max-right {a} {b})

depthVec-upper :
  {n : Nat} -> (xs : V.Vec K.Key n) -> AllVec (λ k -> N.depth k N.<= N.depthVec xs) xs
depthVec-upper V.[] = allNil
depthVec-upper (k V.:: ks) =
  let tail = allVecMap (λ {x} dx -> N.<=-trans dx (max-right {a = N.depth k} {b = N.depthVec ks})) (depthVec-upper ks)
  in allCons (max-left {a = N.depth k} {b = N.depthVec ks}) tail

depthVec-bound :
  {n : Nat}
  -> (xs : V.Vec K.Key n)
  -> (m : Nat)
  -> N.depthVec xs N.<= m
  -> AllVec (λ k -> N.depth k N.<= m) xs
depthVec-bound xs m bound =
  allVecMap (λ {x} dx -> N.<=-trans dx bound) (depthVec-upper xs)

keysDepth : Nat -> List K.Key
keysDepth zero =
  []
keysDepth (suc n) =
  let prev = keysDepth n in
  append prev (concatMap (\op -> nodesForOp op prev) allOps)

closure-enumerable : (n : Nat) -> List K.Key
closure-enumerable = keysDepth

finite-closure : (n : Nat) -> Finite K.Key
finite-closure = keysDepth

inNodesForOp :
  (op : Sig.Op)
  -> {prev : List K.Key}
  -> (args : V.Vec K.Key (Sig.arity op))
  -> AllVec (λ k -> k ∈ prev) args
  -> K.node op args ∈ nodesForOp op prev
inNodesForOp Sig.zero V.[] allNil = here
inNodesForOp Sig.coordZero V.[] allNil = here
inNodesForOp Sig.coordOne V.[] allNil = here
inNodesForOp Sig.suc (k V.:: V.[]) (allCons k∈ allNil) =
  in-map (\x -> K.node Sig.suc (vec1 x)) k∈
inNodesForOp Sig.add {prev} (k1 V.:: k2 V.:: V.[]) (allCons k1∈ (allCons k2∈ allNil)) =
  in-concatMap
    (\x -> map (\y -> K.node Sig.add (vec2 x y)) prev)
    k1∈
    (in-map (\y -> K.node Sig.add (vec2 k1 y)) k2∈)
inNodesForOp Sig.mul {prev} (k1 V.:: k2 V.:: V.[]) (allCons k1∈ (allCons k2∈ allNil)) =
  in-concatMap
    (\x -> map (\y -> K.node Sig.mul (vec2 x y)) prev)
    k1∈
    (in-map (\y -> K.node Sig.mul (vec2 k1 y)) k2∈)
inNodesForOp Sig.coordPair {prev} (k1 V.:: k2 V.:: V.[]) (allCons k1∈ (allCons k2∈ allNil)) =
  in-concatMap
    (\x -> map (\y -> K.node Sig.coordPair (vec2 x y)) prev)
    k1∈
    (in-map (\y -> K.node Sig.coordPair (vec2 k1 y)) k2∈)

coverage :
  (n : Nat)
  -> (k : K.Key)
  -> N.depth k N.<= n
  -> k ∈ keysDepth n
coverage zero (K.node _ _) ()
coverage (suc n) (K.node op args) depth<= =
  let prev = keysDepth n
      argsDepth = depthVec-bound args n (pred<= depth<=)
      argsInPrev = allVecMap (λ {k} dk -> coverage n k dk) argsDepth
      inNodes = inNodesForOp op args argsInPrev
      inOps = op-in-allOps op
      inNew = in-concatMap (\o -> nodesForOp o prev) inOps inNodes
  in in-append-right {xs = prev} inNew
