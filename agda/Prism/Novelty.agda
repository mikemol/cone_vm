module Prism.Novelty where

open import Agda.Builtin.Nat

import Prism.Key as K

data Top : Set where
  tt : Top

Execution : Set
Execution = Nat

_<=_ : Execution -> Execution -> Set
_<=_ _ _ = Top

Novelty : Execution -> K.Key -> Set
Novelty _ _ = Top

monotone-novelty : (E1 E2 : Execution) -> E1 <= E2 -> Top
monotone-novelty _ _ _ = tt
