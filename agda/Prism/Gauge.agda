module Prism.Gauge where

open import Agda.Builtin.Equality

-- BSPˢ gauge scaffold (minimal, see in/in-26.md).

data Arena : Set where
  arena : Set -> Arena

_≈s_ : Arena -> Arena -> Set
_≈s_ = _≡_

q : Arena -> Set
q (arena X) = X

cong : {A B : Set} {x y : A} -> (f : A -> B) -> x ≡ y -> f x ≡ f y
cong f refl = refl

q-invariant : (A A' : Arena) -> A ≈s A' -> q A ≡ q A'
q-invariant _ _ eq = cong q eq
