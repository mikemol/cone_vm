# Agda Proof Kernel (Scaffold)

This directory hosts the initial Agda module scaffolding for the semantic
kernel described in `in/in-26.md`. The intent is to capture the proof roadmap
with minimal, syntactically valid placeholders so work can begin without
blocking on full formalization.

Status: scaffold + basic lemmas (no CI integration).

Suggested entrypoint: `agda/Prism/Prism.agda`.

## Local setup (mise)

Agda is installed via `cabal` under `mise` to keep toolchain versions pinned
in `mise.toml` without committing binaries.

1. Install toolchain:
   - `mise install`
2. Install Agda (downloads from Hackage):
   - `scripts/install_agda.sh`
3. Run the checker:
   - `scripts/check_agda.sh`

CI note: the checker will also use a system `agda` on PATH (or `AGDA_BIN` if
set), so GitHub-hosted runners can install Agda via apt without `mise`.

Agda version pin:
- The pinned version lives in `agda/AGDA_VERSION`.
- `scripts/install_agda.sh` respects `AGDA_VERSION` or that file.
