module Prism.FixedPoint where

open import Agda.Builtin.Equality

import Prism.Novelty as N

-- Toy execution step for proof scaffold (identity).
step : N.Execution -> N.Execution
step E = E

fixed-point : (E : N.Execution) -> Set
fixed-point E = step E ≡ E

fixed-point-all : (E : N.Execution) -> fixed-point E
fixed-point-all _ = refl
