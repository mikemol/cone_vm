#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hooks_dir="${repo_root}/.git/hooks"

mkdir -p "${hooks_dir}"
install -m 0755 "${repo_root}/scripts/hooks/pre-commit" "${hooks_dir}/pre-commit"
install -m 0755 "${repo_root}/scripts/hooks/pre-push" "${hooks_dir}/pre-push"

echo "Installed policy hooks into ${hooks_dir}"
