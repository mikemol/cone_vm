#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-src}"
OUT_DIR="${2:-artifacts/dataflow_grammar}"
DOT_FILE="${OUT_DIR}/dataflow.dot"
PNG_FILE="${OUT_DIR}/dataflow.png"

mkdir -p "${OUT_DIR}"

python scripts/dataflow_grammar_audit.py "${ROOT}" --dot "${DOT_FILE}"

if command -v dot >/dev/null 2>&1; then
  dot -Tpng "${DOT_FILE}" -o "${PNG_FILE}"
  echo "Wrote ${PNG_FILE}"
else
  echo "Graphviz 'dot' not found; wrote ${DOT_FILE} only."
fi
