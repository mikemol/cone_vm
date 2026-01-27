#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${AGDA_IMAGE:-}"
IMAGE_FILE="${AGDA_IMAGE_FILE:-$ROOT/agda/AGDA_IMAGE}"
RUNTIME="${AGDA_CONTAINER_RUNTIME:-}"
VOLUME_SUFFIX="${AGDA_VOLUME_SUFFIX:-}"

if [[ -z "$IMAGE" && -f "$IMAGE_FILE" ]]; then
  IMAGE="$(tr -d ' \t\r\n' < "$IMAGE_FILE")"
fi

if [[ -z "$IMAGE" ]]; then
  echo "AGDA_IMAGE is required (digest-pinned, e.g. ghcr.io/mikemol/act-ubuntu-agda@sha256:...)." >&2
  echo "Set AGDA_IMAGE or write the digest to agda/AGDA_IMAGE." >&2
  exit 1
fi

if [[ -z "$RUNTIME" ]]; then
  if command -v docker >/dev/null 2>&1; then
    RUNTIME="docker"
  elif command -v podman >/dev/null 2>&1; then
    RUNTIME="podman"
  else
    echo "No container runtime found (docker or podman)." >&2
    exit 1
  fi
fi

USER_ARGS=()
if [[ "$RUNTIME" == "docker" ]]; then
  USER_ARGS=(-u "$(id -u):$(id -g)")
elif [[ "$RUNTIME" == "podman" ]]; then
  USER_ARGS=(--userns=keep-id)
else
  echo "Unsupported runtime: $RUNTIME" >&2
  exit 1
fi

"$RUNTIME" run --rm \
  "${USER_ARGS[@]}" \
  -v "$ROOT:/work${VOLUME_SUFFIX}" \
  -w /work \
  "$IMAGE" \
  scripts/check_agda.sh
