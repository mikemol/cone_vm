#!/usr/bin/env bash
set -euo pipefail

# Wrapper for pytest with durable logs. Intended for sandboxed runs too.
# Note: if running under Codex, increase the sandbox command timeout.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/artifacts/test_runs"
mkdir -p "${RUN_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${RUN_DIR}/pytest_${STAMP}.log"

echo "==> Logging to ${LOG}"
echo "==> Command: mise exec -- python -m pytest $*" | tee -a "${LOG}"

mise exec -- python -m pytest "$@" 2>&1 | tee -a "${LOG}"
