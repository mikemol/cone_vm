module Prism.FixedPoint where

import Prism.Key as K
import Prism.Novelty as N

postulate
  step : N.Execution -> N.Execution
  fixed-point : (E : N.Execution) -> Set
