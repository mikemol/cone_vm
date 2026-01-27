#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT/.tools"
BIN_DIR="$TOOLS_DIR/agda/bin"
VERSION_FILE="${AGDA_VERSION_FILE:-$ROOT/agda/AGDA_VERSION}"

# Keep cabal state/config under the repo to avoid touching $HOME.
export CABAL_DIR="${CABAL_DIR:-$TOOLS_DIR/cabal}"
export CABAL_CONFIG="${CABAL_CONFIG:-$CABAL_DIR/config}"
export CABAL_REPO_CACHE="${CABAL_REPO_CACHE:-$CABAL_DIR/packages}"
export CABAL_STORE_DIR="${CABAL_STORE_DIR:-$CABAL_DIR/store}"
export CABAL_LOGS_DIR="${CABAL_LOGS_DIR:-$CABAL_DIR/logs}"

mkdir -p "$CABAL_DIR" "$CABAL_REPO_CACHE" "$CABAL_STORE_DIR" "$CABAL_LOGS_DIR"

if [[ ! -f "$CABAL_CONFIG" ]]; then
  mise exec -- cabal user-config init --config-file "$CABAL_CONFIG"
fi

# Ensure repo-local cache/store/logs in the cabal config.
if grep -q "^remote-repo-cache:" "$CABAL_CONFIG"; then
  sed -i "s|^remote-repo-cache:.*|remote-repo-cache: ${CABAL_REPO_CACHE}|" "$CABAL_CONFIG"
else
  printf "\nremote-repo-cache: %s\n" "$CABAL_REPO_CACHE" >> "$CABAL_CONFIG"
fi
if grep -q "^store-dir:" "$CABAL_CONFIG"; then
  sed -i "s|^store-dir:.*|store-dir: ${CABAL_STORE_DIR}|" "$CABAL_CONFIG"
else
  printf "store-dir: %s\n" "$CABAL_STORE_DIR" >> "$CABAL_CONFIG"
fi
if grep -q "^logs-dir:" "$CABAL_CONFIG"; then
  sed -i "s|^logs-dir:.*|logs-dir: ${CABAL_LOGS_DIR}|" "$CABAL_CONFIG"
else
  printf "logs-dir: %s\n" "$CABAL_LOGS_DIR" >> "$CABAL_CONFIG"
fi

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
mise exec -- cabal --config-file "$CABAL_CONFIG" update
mise exec -- cabal --config-file "$CABAL_CONFIG" install "Agda-${AGDA_VERSION}" \
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
