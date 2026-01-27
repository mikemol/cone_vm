module Prism.Gauge where

open import Agda.Builtin.Equality
open import Agda.Primitive using (Level; lzero; lsuc)

-- BSPˢ gauge scaffold (minimal, see in/in-26.md).

data Arena : Set₁ where
  arena : Set -> Arena

_≈s_ : Arena -> Arena -> Set₁
_≈s_ = _≡_

q : Arena -> Set
q (arena X) = X

cong :
  {ℓ ℓ' : Level}
  {A : Set ℓ}
  {B : Set ℓ'}
  {x y : A}
  -> (f : A -> B)
  -> x ≡ y
  -> f x ≡ f y
cong f refl = refl

q-invariant : (A A' : Arena) -> A ≈s A' -> q A ≡ q A'
q-invariant _ _ eq = cong q eq
