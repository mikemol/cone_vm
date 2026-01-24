#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/ci_watch.sh [options]

Push, locate the latest workflow run, watch it, and show failed logs.

Options:
  --workflow <name|path>  Workflow name or path (default: .github/workflows/ci-milestones.yml)
  --branch <branch>       Branch to filter runs (default: current branch)
  --remote <remote>       Git remote (default: origin)
  --run-id <id>           Use specific run id (skip lookup)
  --no-push               Skip git push
  --no-watch              Skip gh run watch
  --no-logs               Skip gh run view --log-failed
  --help                  Show this help

Examples:
  scripts/ci_watch.sh
  scripts/ci_watch.sh --workflow .github/workflows/ci-milestones.yml --branch main
  scripts/ci_watch.sh --run-id 123456789 --no-push
USAGE
}

workflow=".github/workflows/ci-milestones.yml"
branch=""
remote="origin"
run_id=""
do_push=1
do_watch=1
do_logs=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workflow)
      workflow="$2"
      shift 2
      ;;
    --branch)
      branch="$2"
      shift 2
      ;;
    --remote)
      remote="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --no-push)
      do_push=0
      shift
      ;;
    --no-watch)
      do_watch=0
      shift
      ;;
    --no-logs)
      do_logs=0
      shift
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

run_unbuffered() {
  local env_vars=(
    PAGER=cat
    GH_PAGER=cat
    GIT_PAGER=cat
    LESS=FRX
    PYTHONUNBUFFERED=1
  )
  if command -v stdbuf >/dev/null 2>&1; then
    env "${env_vars[@]}" stdbuf -o0 -e0 "$@"
  else
    env "${env_vars[@]}" "$@"
  fi
}

if [[ -z "$branch" ]]; then
  branch=$(git rev-parse --abbrev-ref HEAD)
fi

if [[ $do_push -eq 1 ]]; then
  git push "$remote" "$branch"
fi

if [[ -z "$run_id" ]]; then
  run_id=$(gh run list \
    --workflow "$workflow" \
    --branch "$branch" \
    --event push \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId | tostring')
fi

if [[ -z "$run_id" ]]; then
  echo "No runs found for workflow=$workflow branch=$branch" >&2
  exit 1
fi

echo "Run id: $run_id"

if [[ $do_watch -eq 1 ]]; then
  run_unbuffered gh run watch "$run_id" --exit-status
fi

if [[ $do_logs -eq 1 ]]; then
  run_unbuffered gh run view "$run_id" --log-failed
fi
