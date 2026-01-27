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
  --install-method=copy \
  --overwrite-policy=always

echo "Agda installed:"
"$BIN_DIR/agda" --version
