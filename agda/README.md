# Agda Proof Kernel (Scaffold)

This directory hosts the initial Agda module scaffolding for the semantic
kernel described in `in/in-26.md`. The intent is to capture the proof roadmap
with minimal, syntactically valid placeholders so work can begin without
blocking on full formalization.

Status: scaffold + basic lemmas (no CI integration).

Suggested entrypoint: `agda/Prism/Prism.agda`.

## Local setup (container)

Agda runs via the pinned container image for fast, reproducible checks.

1. Ensure the image digest is set (defaults to `agda/AGDA_IMAGE`).
2. Run the checker:
   - `scripts/check_agda_container.sh`

CI note: Agda checks run inside the same pinned container image.

Agda version pin:
- The pinned version lives in `agda/AGDA_VERSION`.
- The container image digest lives in `agda/AGDA_IMAGE`.
- Keep the workflow image digest and `agda/AGDA_IMAGE` in sync.
