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

Run the policy checks manually when editing workflows:
```
python scripts/policy_check.py --workflows
```

CI also runs `scripts/policy_check.py --workflows --posture`, which checks the
GitHub Actions settings for this repository.
