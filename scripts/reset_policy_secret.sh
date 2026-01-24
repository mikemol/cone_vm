#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/reset_policy_secret.sh [options]

Prompts for a token, prints metadata (length/prefix/newline), and sets a GitHub
Actions secret.

Options:
  --repo <owner/name>   Repository (default: gh repo view)
  --secret <name>       Secret name (default: POLICY_GITHUB_TOKEN)
  --help                Show this help
USAGE
}

repo=""
secret="POLICY_GITHUB_TOKEN"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      repo="$2"
      shift 2
      ;;
    --secret)
      secret="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
 done

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required" >&2
  exit 1
fi

if [[ -z "$repo" ]]; then
  repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
fi

printf "%s: " "$secret" >&2
IFS= read -rs token
printf '\n' >&2

if [[ -z "$token" ]]; then
  echo "token is empty" >&2
  exit 1
fi

token_len=${#token}
if [[ "$token" == github_pat_* ]]; then
  token_prefix="github_pat_"
else
  token_prefix="(non github_pat)"
fi
token_newline=$(POLICY_TOKEN="$token" python - <<'PY'
import os
token = os.environ["POLICY_TOKEN"]
has_newline = ("\n" in token) or ("\r" in token)
print("yes" if has_newline else "no")
PY
)
token_fingerprint=$(POLICY_TOKEN="$token" python - <<'PY'
import os
import hashlib
token = os.environ["POLICY_TOKEN"]
print(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8])
PY
)

echo "token length: ${token_len}"
echo "token prefix: ${token_prefix}"
echo "token contains newline: ${token_newline}"
echo "token sha256 prefix: ${token_fingerprint}"

printf %s "$token" | gh secret set "$secret" -R "$repo"
unset token

echo "secret updated: ${secret} (${repo})"
