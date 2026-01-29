# Contributing

Thanks for contributing. This repo enforces a strict execution policy to protect
self-hosted runners. Please read `POLICY_SEED.md` before making changes.

## Policy requirements (summary)
- Self-hosted workflows must trigger only on `push` to trusted branches.
- Self-hosted jobs must include `self-hosted`, `gpu`, and `local` labels.
- Self-hosted jobs must be guarded with
  `if: github.actor == github.repository_owner`.
- Workflow actions must be pinned to full commit SHAs and allow-listed.
- Workflows must declare `permissions: contents: read`.

## Guardrails
Install the advisory hooks:
```
scripts/install_policy_hooks.sh
```

Install the policy-check dependency (once):
```
mise exec -- python -m pip install pyyaml
```

Run the policy checks manually when editing workflows:
```
mise exec -- python scripts/policy_check.py --workflows
```

CI also runs `scripts/policy_check.py --workflows --posture`, which checks the
GitHub Actions settings for this repository.

## GPU tests and sandboxed environments
Some tests rely on CUDA/JAX GPU backends. If you are running in a sandboxed
environment, GPU access may require explicit sandbox escalation/privileged
execution. Without GPU access, CUDA backend init can fail. Do not mask these
failures; rerun with GPU access enabled, or explicitly select a CPU-only path
when that is the intent of the test run.

## Agda proofs
Agda checks run in a pinned container image. See `agda/README.md` for details.
Local run:
```
scripts/check_agda_container.sh
```
