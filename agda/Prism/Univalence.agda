module Prism.Univalence where

open import Agda.Builtin.Equality

import Prism.Key as K

-- Minimal univalence: semantic equality is definitional equality.
semanticEq : K.Key -> K.Key -> Set
semanticEq k1 k2 = k1 ≡ k2

univalence : (k1 k2 : K.Key) -> semanticEq k1 k2 ≡ (k1 ≡ k2)
univalence _ _ = refl
