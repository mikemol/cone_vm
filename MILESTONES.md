# Milestones

## m1 (2026-01-23)
Tag: `m1`
Gate command:
- `mise exec -- pytest -c pytest.m1.ini`

Expected xfails (gated above m1):
- `tests/test_coord_batch.py::test_coord_xor_batch_uses_single_intern_call` - m4: no batched coord_xor_batch / coord_norm_batch yet.
- `tests/test_coord_batch.py::test_coord_norm_batch_matches_host` - m4: no batched coord_norm_batch yet.
