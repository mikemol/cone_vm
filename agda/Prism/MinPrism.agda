module Prism.MinPrism where

open import Agda.Builtin.Unit
open import Agda.Builtin.List

import Prism.Key as K
import Prism.Signature as Sig

-- Minimal finiteness scaffold: any set admits a finite witness.
Finite : Set -> Set
Finite _ = ⊤

Closure : Sig.Op -> Set
Closure _ = K.Key

finite-closure : (op : Sig.Op) -> Finite (Closure op)
finite-closure _ = tt

closure-enumerable : (op : Sig.Op) -> List K.Key
closure-enumerable _ = []
