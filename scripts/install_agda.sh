#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT/.tools/agda/bin"

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required (https://mise.jdx.dev)" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"

echo "Installing Agda via cabal (downloads from Hackage)..."
mise exec -- cabal update
mise exec -- cabal install Agda \
  --installdir "$BIN_DIR" \
  --install-method=copy \
  --overwrite-policy=always

echo "Agda installed:"
"$BIN_DIR/agda" --version
