module Prism.Univalence where

import Prism.Key as K

-- Placeholder univalence statement (see in/in-26.md).
postulate
  semanticEq : K.Key -> K.Key -> Set
  univalence : (k1 k2 : K.Key) -> Set
