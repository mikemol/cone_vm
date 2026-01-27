module Prism.Boundaries where

open import Agda.Builtin.Bool
open import Agda.Builtin.Equality
open import Agda.Builtin.Nat
open import Agda.Builtin.Sigma

import Prism.Key as K
import Prism.Novelty as N
import Prism.Signature as Sig
import Prism.Vec as V

data ⊥ : Set where

¬ : Set -> Set
¬ A = A -> ⊥

cong :
  {A B : Set}
  {x y : A}
  -> (f : A -> B)
  -> x ≡ y
  -> f x ≡ f y
cong f refl = refl

sym : {A : Set} {x y : A} -> x ≡ y -> y ≡ x
sym refl = refl

if_then_else_ : {A : Set} -> Bool -> A -> A -> A
if true then x else _ = x
if false then _ else y = y

even : Nat -> Bool
even zero = true
even (suc n) = if even n then false else true

opOf : K.Key -> Sig.Op
opOf (K.node op _) = op

zero≠suc : Sig.zero ≡ Sig.suc -> ⊥
zero≠suc ()

vec0 : {A : Set} -> V.Vec A zero
vec0 = V.[]

vec1 : {A : Set} -> A -> V.Vec A (suc zero)
vec1 a = a V.:: V.[]

k0 : K.Key
k0 = K.node Sig.zero vec0

k1 : K.Key
k1 = K.node Sig.suc (vec1 k0)

k0≠k1 : k0 ≡ k1 -> ⊥
k0≠k1 eq = zero≠suc (cong opOf eq)

k1≠k0 : k1 ≡ k0 -> ⊥
k1≠k0 eq = k0≠k1 (sym eq)

Execution : Set
Execution = Nat -> K.Key

terminates : Execution -> Set
terminates exec =
  Σ Nat (λ n -> (m : Nat) -> n N.<= m -> exec m ≡ exec n)

bounded : Execution -> Set
bounded exec =
  Σ Nat (λ b -> (t : Nat) -> N.depth (exec t) N.<= b)

alternating : Execution
alternating n = if even n then k0 else k1

alternating-suc : (n : Nat) -> alternating (suc n) ≡ if even n then k1 else k0
alternating-suc n with even n
... | true = refl
... | false = refl

one : Nat
one = suc zero

two : Nat
two = suc one

depth-k0≤2 : N.depth k0 N.<= two
depth-k0≤2 = N.s<=s N.z<=n

depth-k1≤2 : N.depth k1 N.<= two
depth-k1≤2 = N.<=-refl

bounded-alternating : bounded alternating
bounded-alternating = two , witness
  where
    witness : (t : Nat) -> N.depth (alternating t) N.<= two
    witness t with even t
    ... | true = depth-k0≤2
    ... | false = depth-k1≤2

n<=sucn : {n : Nat} -> n N.<= suc n
n<=sucn {zero} = N.z<=n
n<=sucn {suc n} = N.s<=s n<=sucn

alternating-step-neq : (n : Nat) -> alternating n ≡ alternating (suc n) -> ⊥
alternating-step-neq n eq rewrite alternating-suc n = step eq
  where
    step : alternating n ≡ if even n then k1 else k0 -> ⊥
    step eq' with even n
    ... | true = k0≠k1 eq'
    ... | false = k1≠k0 eq'

not-terminates-alternating : ¬ (terminates alternating)
not-terminates-alternating (n , stable) =
  alternating-step-neq n (sym (stable (suc n) n<=sucn))

novelty-not-termination : Set
novelty-not-termination =
  Σ Execution (λ exec -> Σ (bounded exec) (λ _ -> ¬ (terminates exec)))

novelty-not-termination-proof : novelty-not-termination
novelty-not-termination-proof =
  alternating , (bounded-alternating , not-terminates-alternating)
