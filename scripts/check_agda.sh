#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGDA_BIN_DEFAULT="$ROOT/.tools/agda/bin/agda"

if [[ -n "${AGDA_BIN:-}" ]]; then
  AGDA_BIN="$AGDA_BIN"
elif [[ -x "$AGDA_BIN_DEFAULT" ]]; then
  AGDA_BIN="$AGDA_BIN_DEFAULT"
elif command -v agda >/dev/null 2>&1; then
  AGDA_BIN="$(command -v agda)"
else
  echo "Agda not installed. Use the container helper (scripts/check_agda_container.sh) or install agda in PATH." >&2
  exit 1
fi

"$AGDA_BIN" -i "$ROOT/agda" "$ROOT/agda/Prism/Prism.agda"
