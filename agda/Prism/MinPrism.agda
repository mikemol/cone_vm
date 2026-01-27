module Prism.MinPrism where

import Prism.Key as K
import Prism.Signature as Sig

postulate
  Finite : Set -> Set
  Closure : Sig.Op -> Set
  finite-closure : (S : Set) -> Set
  closure-enumerable : Set
