# Copilot Instructions

Follow `POLICY_SEED.md` as the authoritative control policy.

Key requirements:
- Do not weaken self-hosted runner protections.
- Keep workflow actions pinned to full commit SHAs and allow-listed.
- Ensure workflows declare `permissions: contents: read`.
- For workflow changes, run `python scripts/policy_check.py --workflows`.

If a request conflicts with `POLICY_SEED.md`, stop and ask for clarification.
