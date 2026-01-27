#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT/.tools/agda/bin"
VERSION_FILE="${AGDA_VERSION_FILE:-$ROOT/agda/AGDA_VERSION}"

if [[ -z "${AGDA_VERSION:-}" ]]; then
  if [[ -f "$VERSION_FILE" ]]; then
    AGDA_VERSION="$(tr -d '[:space:]' < "$VERSION_FILE")"
  else
    echo "AGDA_VERSION not set and $VERSION_FILE not found." >&2
    exit 1
  fi
fi

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required (https://mise.jdx.dev)" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"

echo "Installing Agda ${AGDA_VERSION} via cabal (downloads from Hackage)..."
mise exec -- cabal update
mise exec -- cabal install "Agda-${AGDA_VERSION}" \
  --installdir "$BIN_DIR" \
  --install-method=symlink \
  --overwrite-policy=always

echo "Agda installed:"
"$BIN_DIR/agda" --version

AGDA_DIR="$("$BIN_DIR/agda" --print-agda-dir 2>/dev/null || true)"
if [[ -n "$AGDA_DIR" && -f "$AGDA_DIR/Agda/Builtin/Vec.agda" ]]; then
  echo "Agda prim dir: $AGDA_DIR"
else
  echo "Warning: Agda prim dir not found (Agda.Builtin.Vec missing)" >&2
  echo "AGDA_DIR resolved to: ${AGDA_DIR:-<empty>}" >&2
fi
