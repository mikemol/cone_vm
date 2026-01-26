#!/usr/bin/env bash
set -euo pipefail

exec python3 scripts/ci_watch.py "$@"
