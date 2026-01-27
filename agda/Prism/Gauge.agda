module Prism.Gauge where

-- BSP^s gauge scaffold (see in/in-26.md).
postulate
  Arena : Set
  _≈s_ : Arena -> Arena -> Set
  q : Arena -> Set
  q-invariant : (A A' : Arena) -> A ≈s A' -> Set
