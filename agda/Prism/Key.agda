module Prism.Key where

open import Agda.Builtin.List

import Prism.Signature as Sig

-- Placeholder for canonical semantic objects (see in/in-26.md).
-- TODO: replace List with Vec indexed by arity.

data Key : Set where
  node : Sig.Op -> List Key -> Key
