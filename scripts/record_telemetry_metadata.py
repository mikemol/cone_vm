#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ENV_KEYS = [
    "PRISM_MILESTONE",
    "PRISM_ENABLE_CNF2",
    "PRISM_ENABLE_CNF2_SLOT1",
    "PRISM_ENABLE_SERVO",
    "PRISM_DAMAGE_METRICS",
    "PRISM_DAMAGE_TILE_SIZE",
    "PRISM_SWIZZLE_BACKEND",
    "JAX_PLATFORM_NAME",
    "XLA_FLAGS",
]


def _git_value(args: list[str]) -> str:
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _collect_git() -> dict:
    sha = os.environ.get("GITHUB_SHA") or _git_value(["git", "rev-parse", "HEAD"])
    ref = (
        os.environ.get("GITHUB_REF_NAME")
        or os.environ.get("GITHUB_REF")
        or _git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    )
    return {"sha": sha, "ref": ref}


def _collect_python() -> dict:
    return {
        "version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
    }


def _collect_platform() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _collect_jax() -> dict:
    try:
        import jax  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        return {"error": str(exc)}

    try:
        import jaxlib  # type: ignore

        jaxlib_version = getattr(jaxlib, "__version__", "")
    except Exception:
        jaxlib_version = ""

    devices = jax.devices()
    platforms = sorted({d.platform for d in devices})
    kinds = sorted({getattr(d, "device_kind", "") for d in devices if hasattr(d, "device_kind")})
    default_backend = jax.default_backend()
    summary_parts = [f"{default_backend}:{len(devices)}"]
    if kinds:
        summary_parts.append(", ".join(k for k in kinds if k))
    device_summary = " ".join(summary_parts)

    return {
        "version": getattr(jax, "__version__", ""),
        "jaxlib_version": jaxlib_version,
        "default_backend": default_backend,
        "device_platforms": platforms,
        "device_count": len(devices),
        "device_kinds": kinds,
        "device_summary": device_summary,
    }


def _collect_env() -> dict:
    return {key: os.environ.get(key, "") for key in ENV_KEYS}


def _parse_extra(values: list[str]) -> dict:
    extras = {}
    for item in values:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        extras[key.strip()] = value.strip()
    return extras


def main() -> int:
    parser = argparse.ArgumentParser(description="Record telemetry metadata snapshot.")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--label", default="", help="Human-friendly label")
    parser.add_argument("--engine", default="", help="Engine identifier (optional)")
    parser.add_argument("--milestone", default="", help="Milestone identifier (optional)")
    parser.add_argument("--backend", default="", help="Backend identifier (optional)")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra key=value metadata (repeatable)",
    )
    args = parser.parse_args()

    payload = {
        "label": args.label,
        "engine": args.engine,
        "milestone": args.milestone,
        "backend": args.backend,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _collect_git(),
        "python": _collect_python(),
        "platform": _collect_platform(),
        "jax": _collect_jax(),
        "env": _collect_env(),
        "extra": _parse_extra(args.extra),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
