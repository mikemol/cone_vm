module Prism.FixedPoint where

open import Agda.Builtin.Equality

import Prism.Novelty as N

postulate
  step : N.Execution -> N.Execution

fixed-point : (E : N.Execution) -> Set
fixed-point E = step E ≡ E
