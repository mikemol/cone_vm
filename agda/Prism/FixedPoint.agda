module Prism.FixedPoint where

open import Agda.Builtin.Equality
open import Agda.Builtin.Sigma

import Prism.Novelty as N
import Prism.Key as K

stable-after : N.Execution -> K.Key -> Set
stable-after n k = (m : N.Execution) -> n N.<= m -> N.Novelty m k

eventual-novelty : (k : K.Key) -> Σ N.Execution (λ n -> stable-after n k)
eventual-novelty k = N.depth k , witness
  where
    witness : (m : N.Execution) -> N.depth k N.<= m -> N.Novelty m k
    witness m depth<=m =
      N.monotone-novelty depth<=m (N.<=-refl {n = N.depth k})
