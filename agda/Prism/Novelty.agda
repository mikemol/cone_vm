module Prism.Novelty where

import Prism.Key as K

postulate
  Execution : Set
  Novelty : Execution -> K.Key -> Set
  _<=_ : Execution -> Execution -> Set
  monotone-novelty : (E1 E2 : Execution) -> E1 <= E2 -> Set
