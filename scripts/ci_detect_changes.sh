#!/usr/bin/env bash
set -euo pipefail

base=""
head=""
output=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      base="${2:-}"
      shift 2
      ;;
    --head)
      head="${2:-}"
      shift 2
      ;;
    --output)
      output="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$base" || -z "$head" || -z "$output" ]]; then
  echo "usage: $0 --base <sha> --head <sha> --output <github_output>" >&2
  exit 2
fi

if [[ "$base" == "0000000000000000000000000000000000000000" ]]; then
  {
    echo "code_changed=true"
    echo "changed_files=initial"
  } >> "$output"
  exit 0
fi

git diff --name-only "$base" "$head" > /tmp/changed.txt || true

code_changed=false
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  case "$path" in
    *.md) ;;
    *) code_changed=true; break ;;
  esac
done < /tmp/changed.txt

{
  echo "code_changed=$code_changed"
  echo "changed_files<<CHANGED_FILES_EOF"
  cat /tmp/changed.txt
  echo "CHANGED_FILES_EOF"
} >> "$output"
