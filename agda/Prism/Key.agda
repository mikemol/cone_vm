module Prism.Key where

open import Agda.Builtin.Vec

import Prism.Signature as Sig

data Key : Set where
  node : (op : Sig.Op) -> Vec Key (Sig.arity op) -> Key
